"""Tests for optbench.param_groups.

These test behaviour and properties, not one specific implementation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from optbench.model import ModelConfig, Transformer
from optbench.param_groups import (
    GroupConfig,
    build_param_groups,
    classify,
    group_summary,
    split_by_optimizer,
)
from optbench.schedule import apply_lr

# A 2-layer GQA model: 14 matrix tensors, 2 embed, 5 scalar (21 total).
N_MATRIX, N_EMBED, N_SCALAR = 14, 2, 5
P_MATRIX, P_EMBED, P_SCALAR = 4608, 4096, 80


def make_model(**kw) -> Transformer:
    base = dict(
        vocab_size=128, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=32, max_seq_len=8
    )
    base.update(kw)
    return Transformer(ModelConfig(**base))


@pytest.fixture
def model() -> Transformer:
    return make_model()


def by_kind(groups) -> dict[str, dict]:
    return {g["kind"]: g for g in groups}


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,shape,expected",
    [
        ("blocks.0.attn.wq.weight", (16, 16), "matrix"),
        ("blocks.1.attn.wk.weight", (8, 16), "matrix"),
        ("blocks.0.mlp.w2.weight", (16, 32), "matrix"),
        ("tok_emb.weight", (128, 16), "embed"),
        ("lm_head.weight", (128, 16), "embed"),
        ("blocks.0.attn_norm.weight", (16,), "scalar"),
        ("norm.weight", (16,), "scalar"),
        # 1-D wins over the embed patterns: a vector has no matrix structure.
        ("lm_head.bias", (128,), "scalar"),
        ("tok_emb.bias", (16,), "scalar"),
        # A 0-d parameter (a learned scalar gate) is also "scalar".
        ("gate", (), "scalar"),
        # A 3-D tensor (e.g. a fused MoE expert stack) is not 1-D, so "matrix".
        ("experts.w", (4, 16, 32), "matrix"),
    ],
)
def test_classify_rules(name, shape, expected):
    assert classify(name, torch.zeros(shape), GroupConfig()) == expected


def test_classify_honours_custom_patterns():
    cfg = GroupConfig(embed_patterns=("wte", "output_proj"))
    assert classify("wte.weight", torch.zeros(128, 16), cfg) == "embed"
    assert classify("decoder.output_proj.weight", torch.zeros(128, 16), cfg) == "embed"
    # The defaults no longer apply.
    assert classify("tok_emb.weight", torch.zeros(128, 16), cfg) == "matrix"


# ---------------------------------------------------------------------------
# build_param_groups: structure
# ---------------------------------------------------------------------------


def test_group_order_and_keys(model):
    groups = build_param_groups(model, GroupConfig())
    assert [g["kind"] for g in groups] == ["matrix", "embed", "scalar"]
    for g in groups:
        assert set(g) == {"params", "base_lr", "weight_decay", "kind", "names"}
        assert len(g["params"]) == len(g["names"])
        assert all(isinstance(p, nn.Parameter) for p in g["params"])
        assert all(isinstance(n, str) for n in g["names"])


def test_tensor_and_parameter_counts(model):
    g = by_kind(build_param_groups(model, GroupConfig()))
    assert len(g["matrix"]["params"]) == N_MATRIX
    assert len(g["embed"]["params"]) == N_EMBED
    assert len(g["scalar"]["params"]) == N_SCALAR
    assert sum(p.numel() for p in g["matrix"]["params"]) == P_MATRIX
    assert sum(p.numel() for p in g["embed"]["params"]) == P_EMBED
    assert sum(p.numel() for p in g["scalar"]["params"]) == P_SCALAR


def test_every_trainable_parameter_appears_exactly_once(model):
    groups = build_param_groups(model, GroupConfig())
    seen = [id(p) for g in groups for p in g["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(seen) == len(set(seen)), "a parameter is in two groups: it would be updated twice"
    assert set(seen) == set(expected), "a parameter is in no group: it would never be trained"
    assert sum(p.numel() for g in groups for p in g["params"]) == model.num_params(
        non_embedding=False
    )


def test_names_match_the_model(model):
    groups = build_param_groups(model, GroupConfig())
    real = dict(model.named_parameters())
    for g in groups:
        for name, param in zip(g["names"], g["params"]):
            assert name in real, name
            assert real[name] is param


def test_matrix_group_is_exactly_the_hidden_weights(model):
    g = by_kind(build_param_groups(model, GroupConfig()))
    names = g["matrix"]["names"]
    assert all(("attn." in n) or ("mlp." in n) for n in names)
    assert all("norm" not in n for n in names)
    assert all(p.ndim == 2 for p in g["matrix"]["params"])
    for leaf in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        assert sum(f".{leaf}." in n for n in names) == 2, leaf  # one per layer


def test_embed_group_holds_the_vocabulary_matrices(model):
    g = by_kind(build_param_groups(model, GroupConfig()))
    assert sorted(g["embed"]["names"]) == ["lm_head.weight", "tok_emb.weight"]


def test_scalar_group_is_every_vector(model):
    g = by_kind(build_param_groups(model, GroupConfig()))
    assert all(p.ndim <= 1 for p in g["scalar"]["params"])
    assert sorted(g["scalar"]["names"]) == [
        "blocks.0.attn_norm.weight",
        "blocks.0.mlp_norm.weight",
        "blocks.1.attn_norm.weight",
        "blocks.1.mlp_norm.weight",
        "norm.weight",
    ]


def test_order_within_a_group_follows_named_parameters(model):
    groups = build_param_groups(model, GroupConfig())
    order = [n for n, _ in model.named_parameters()]
    for g in groups:
        positions = [order.index(n) for n in g["names"]]
        assert positions == sorted(positions), f"{g['kind']} group is out of model order"


def test_build_is_deterministic(model):
    a = build_param_groups(model, GroupConfig())
    b = build_param_groups(model, GroupConfig())
    assert [g["names"] for g in a] == [g["names"] for g in b]


# ---------------------------------------------------------------------------
# build_param_groups: hyperparameters
# ---------------------------------------------------------------------------


def test_lrs_come_from_the_config(model):
    cfg = GroupConfig(matrix_lr=0.05, embed_lr=1e-3, scalar_lr=2e-4)
    g = by_kind(build_param_groups(model, cfg))
    assert g["matrix"]["base_lr"] == pytest.approx(0.05)
    assert g["embed"]["base_lr"] == pytest.approx(1e-3)
    assert g["scalar"]["base_lr"] == pytest.approx(2e-4)
    assert all("lr" not in grp for grp in g.values()), "groups carry base_lr, the schedule sets lr"


def test_scalars_never_get_weight_decay(model):
    g = by_kind(build_param_groups(model, GroupConfig(weight_decay=0.123)))
    assert g["matrix"]["weight_decay"] == pytest.approx(0.123)
    assert g["embed"]["weight_decay"] == pytest.approx(0.123)
    assert g["scalar"]["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_tied_embeddings_appear_once(model):
    tied = make_model(tie_embeddings=True)
    g = by_kind(build_param_groups(tied, GroupConfig()))
    assert len(g["embed"]["params"]) == 1
    ids = [id(p) for grp in g.values() for p in grp["params"]]
    assert len(ids) == len(set(ids)) == 20


def test_frozen_parameters_are_skipped(model):
    model.tok_emb.weight.requires_grad_(False)
    for block in model.blocks:
        block.attn.wq.weight.requires_grad_(False)
    g = by_kind(build_param_groups(model, GroupConfig()))
    assert len(g["matrix"]["params"]) == N_MATRIX - 2
    assert len(g["embed"]["params"]) == N_EMBED - 1
    assert all(p.requires_grad for grp in g.values() for p in grp["params"])


def test_empty_groups_are_omitted():
    only_vectors = nn.Module()
    only_vectors.a = nn.Parameter(torch.zeros(4))
    only_vectors.b = nn.Parameter(torch.zeros(7))
    groups = build_param_groups(only_vectors, GroupConfig())
    assert [g["kind"] for g in groups] == ["scalar"]
    assert len(groups[0]["params"]) == 2


def test_a_model_with_no_trainable_parameters_gives_no_groups():
    m = nn.Linear(4, 4, bias=False)
    m.weight.requires_grad_(False)
    assert build_param_groups(m, GroupConfig()) == []


# ---------------------------------------------------------------------------
# split_by_optimizer
# ---------------------------------------------------------------------------


def test_split_routes_matrix_and_adamw(model):
    groups = build_param_groups(model, GroupConfig())
    matrix, adamw = split_by_optimizer(groups)
    assert [g["kind"] for g in matrix] == ["matrix"]
    assert [g["kind"] for g in adamw] == ["embed", "scalar"]
    # The same dict objects, so there is one source of truth for "lr".
    assert matrix[0] is groups[0]
    assert all(any(g is orig for orig in groups) for g in adamw)


def test_split_covers_every_parameter_once(model):
    groups = build_param_groups(model, GroupConfig())
    matrix, adamw = split_by_optimizer(groups)
    ids = [id(p) for g in matrix + adamw for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}


def test_split_handles_a_model_with_no_matrices():
    only_vectors = nn.Module()
    only_vectors.a = nn.Parameter(torch.zeros(4))
    matrix, adamw = split_by_optimizer(build_param_groups(only_vectors, GroupConfig()))
    assert matrix == []
    assert [g["kind"] for g in adamw] == ["scalar"]


# ---------------------------------------------------------------------------
# Integration: a real optimizer and the schedule
# ---------------------------------------------------------------------------


def test_groups_drive_a_real_optimizer(model):
    groups = build_param_groups(model, GroupConfig(matrix_lr=0.05, embed_lr=1e-3, scalar_lr=2e-4))
    opt = torch.optim.AdamW([{**g, "lr": g["base_lr"]} for g in groups], betas=(0.9, 0.95))
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.05, 1e-3, 2e-4])
    assert [g["weight_decay"] for g in opt.param_groups] == pytest.approx([0.1, 0.1, 0.0])

    before = [p.detach().clone() for g in groups for p in g["params"]]
    idx = torch.randint(0, 128, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()
    opt.step()
    after = [p.detach().clone() for g in groups for p in g["params"]]
    assert all(not torch.equal(a, b) for a, b in zip(before, after)), "a group was never updated"


def test_schedule_can_drive_the_groups(model):
    cfg = GroupConfig(matrix_lr=0.02, embed_lr=1e-3, scalar_lr=1e-3)
    groups = build_param_groups(model, cfg)
    opt = torch.optim.SGD([{**g, "lr": g["base_lr"]} for g in groups])
    apply_lr(opt, 0.5)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 5e-4, 5e-4])
    apply_lr(opt, 0.25)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.005, 2.5e-4, 2.5e-4])
    assert [g["base_lr"] for g in opt.param_groups] == pytest.approx([0.02, 1e-3, 1e-3])


# ---------------------------------------------------------------------------
# Logging and validation
# ---------------------------------------------------------------------------


def test_group_summary_is_one_line_per_group(model):
    groups = build_param_groups(model, GroupConfig())
    lines = group_summary(groups).splitlines()
    assert len(lines) == 3
    for line, g in zip(lines, groups):
        assert line.startswith(g["kind"])
        assert f"{len(g['params'])} tensor" in line
        assert f"{sum(p.numel() for p in g['params']):,}" in line
    assert group_summary([]) == ""


@pytest.mark.parametrize(
    "cfg",
    [
        GroupConfig(matrix_lr=0.0),
        GroupConfig(matrix_lr=-0.1),
        GroupConfig(embed_lr=0.0),
        GroupConfig(scalar_lr=-1e-5),
        GroupConfig(weight_decay=-0.1),
        GroupConfig(embed_patterns=()),
        GroupConfig(embed_patterns=("tok_emb", "")),
    ],
)
def test_invalid_configs_raise(cfg):
    with pytest.raises(ValueError):
        cfg.validate()


def test_build_validates_the_config(model):
    with pytest.raises(ValueError):
        build_param_groups(model, GroupConfig(matrix_lr=-1.0))


@pytest.mark.parametrize("cfg", [GroupConfig(), GroupConfig(weight_decay=0.0)])
def test_valid_configs_pass(cfg):
    cfg.validate()
