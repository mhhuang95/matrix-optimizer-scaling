"""Tests for optbench.optimizers — TASK 3b."""

from __future__ import annotations

import pytest
import torch

from optbench.checkpoint import optimizer_group_names
from optbench.model import ModelConfig, Transformer
from optbench.muon import Muon
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
from optbench.schedule import apply_lr

SEED = 7


def tiny_model(**kw) -> Transformer:
    torch.manual_seed(SEED)
    base = dict(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=32,
    )
    base.update(kw)
    return Transformer(ModelConfig(**base))


def tiny_configs(**opt_kw) -> tuple[GroupConfig, OptimizerConfig]:
    return GroupConfig(), OptimizerConfig(**opt_kw)


# ---------------------------------------------------------------------------
# OptimizerConfig.validate
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults_are_valid(self):
        OptimizerConfig().validate()
        OptimizerConfig(arm="adamw").validate()

    def test_rejects_unknown_arm(self):
        with pytest.raises(ValueError):
            OptimizerConfig(arm="shampoo").validate()  # not wired up yet

    def test_rejects_bad_betas(self):
        with pytest.raises(ValueError):
            OptimizerConfig(adamw_betas=(0.9, 1.0)).validate()
        with pytest.raises(ValueError):
            OptimizerConfig(adamw_betas=(-0.1, 0.95)).validate()

    def test_rejects_bad_eps(self):
        with pytest.raises(ValueError):
            OptimizerConfig(adamw_eps=0.0).validate()

    def test_rejects_bad_muon_hparams(self):
        with pytest.raises(ValueError):
            OptimizerConfig(muon_momentum=1.0).validate()
        with pytest.raises(ValueError):
            OptimizerConfig(muon_ns_steps=0).validate()


# ---------------------------------------------------------------------------
# the muon arm
# ---------------------------------------------------------------------------


class TestMuonArm:
    def test_returns_muon_then_adamw(self):
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm="muon"))
        assert len(opts) == 2
        assert isinstance(opts[0], Muon)
        assert isinstance(opts[1], torch.optim.AdamW)

    def test_muon_gets_exactly_the_matrix_group(self):
        model = tiny_model()
        muon, _ = build_optimizers(model, *tiny_configs(arm="muon"))
        kinds = [g["kind"] for g in muon.param_groups]
        assert kinds == ["matrix"]
        for p in muon.param_groups[0]["params"]:
            assert p.ndim == 2

    def test_adamw_gets_embed_and_scalar(self):
        model = tiny_model()
        _, adamw = build_optimizers(model, *tiny_configs(arm="muon"))
        assert [g["kind"] for g in adamw.param_groups] == ["embed", "scalar"]

    def test_every_trainable_param_is_covered_exactly_once(self):
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm="muon"))
        seen = [id(p) for opt in opts for g in opt.param_groups for p in g["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert len(seen) == len(set(seen)), "a parameter appears in two groups"
        assert set(seen) == expected

    def test_lr_is_seeded_from_base_lr(self):
        model = tiny_model()
        gcfg = GroupConfig(matrix_lr=0.03, embed_lr=1e-4, scalar_lr=2e-4)
        muon, adamw = build_optimizers(model, gcfg, OptimizerConfig())
        assert muon.param_groups[0]["lr"] == 0.03
        assert [g["lr"] for g in adamw.param_groups] == [1e-4, 2e-4]

    def test_muon_hparams_are_plumbed(self):
        model = tiny_model()
        ocfg = OptimizerConfig(muon_momentum=0.8, muon_nesterov=False, muon_ns_steps=3)
        muon, _ = build_optimizers(model, GroupConfig(), ocfg)
        g = muon.param_groups[0]
        assert g["momentum"] == 0.8
        assert g["nesterov"] is False
        assert g["ns_steps"] == 3

    def test_adamw_hparams_are_plumbed(self):
        model = tiny_model()
        ocfg = OptimizerConfig(adamw_betas=(0.85, 0.92), adamw_eps=1e-10)
        _, adamw = build_optimizers(model, GroupConfig(), ocfg)
        for g in adamw.param_groups:
            assert g["betas"] == (0.85, 0.92)
            assert g["eps"] == 1e-10

    def test_scalar_group_has_zero_weight_decay(self):
        model = tiny_model()
        gcfg = GroupConfig(weight_decay=0.1)
        muon, adamw = build_optimizers(model, gcfg, OptimizerConfig())
        assert muon.param_groups[0]["weight_decay"] == 0.1
        by_kind = {g["kind"]: g["weight_decay"] for g in adamw.param_groups}
        assert by_kind == {"embed": 0.1, "scalar": 0.0}

    def test_checkpoint_fingerprint_works(self):
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm="muon"))
        names = optimizer_group_names(opts)
        assert len(names) == 2
        flat = [n for per_opt in names for group in per_opt for n in group]
        assert "blocks.0.attn.wq.weight" in flat
        assert "tok_emb.weight" in flat

    def test_validation_failure_builds_nothing(self):
        model = tiny_model()
        with pytest.raises(ValueError):
            build_optimizers(model, GroupConfig(), OptimizerConfig(arm="sgd"))


# ---------------------------------------------------------------------------
# the adamw arm
# ---------------------------------------------------------------------------


class TestAdamWArm:
    def test_single_adamw_with_all_three_groups(self):
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm="adamw"))
        assert len(opts) == 1
        assert isinstance(opts[0], torch.optim.AdamW)
        assert [g["kind"] for g in opts[0].param_groups] == ["matrix", "embed", "scalar"]

    def test_matrix_group_keeps_its_own_lr(self):
        # The baseline arm still uses per-kind LRs: matrix_lr is a *sweepable*
        # hyperparameter of the AdamW baseline, not a Muon-only concept.
        model = tiny_model()
        gcfg = GroupConfig(matrix_lr=0.01, embed_lr=3e-4, scalar_lr=3e-4)
        (adamw,) = build_optimizers(model, gcfg, OptimizerConfig(arm="adamw"))
        assert [g["lr"] for g in adamw.param_groups] == [0.01, 3e-4, 3e-4]


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.parametrize("arm", ["muon", "adamw"])
    def test_one_schedule_drives_all_optimizers(self, arm):
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm=arm))
        for opt in opts:
            apply_lr(opt, 0.25)
        for opt in opts:
            for g in opt.param_groups:
                assert g["lr"] == pytest.approx(0.25 * g["base_lr"])

    @pytest.mark.parametrize("arm", ["muon", "adamw"])
    def test_few_steps_reduce_loss(self, arm):
        torch.manual_seed(SEED)
        model = tiny_model()
        opts = build_optimizers(model, *tiny_configs(arm=arm))
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        _, first = model(x, y)
        for _ in range(20):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            for opt in opts:
                opt.step()
        _, last = model(x, y)
        assert last.item() < first.item()
