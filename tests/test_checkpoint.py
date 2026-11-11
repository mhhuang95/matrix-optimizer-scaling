"""Tests for optbench.checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from optbench.checkpoint import (
    CHECKPOINT_VERSION,
    atomic_save,
    latest_checkpoint,
    load_checkpoint,
    optimizer_group_names,
    save_checkpoint,
)
from optbench.model import ModelConfig, Transformer
from optbench.param_groups import GroupConfig, build_param_groups, split_by_optimizer
from optbench.train_step import TrainStepConfig, train_step

V, T = 64, 8


def make_model(seed: int = 0, **overrides) -> Transformer:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=V,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=T,
        **overrides,
    )
    return Transformer(cfg)


def make_optimizers(model: nn.Module, lr: float = 0.1):
    """Two optimizers over disjoint groups, as the real training loop has."""
    groups = build_param_groups(model, GroupConfig(matrix_lr=lr, embed_lr=lr, scalar_lr=lr))
    matrix_groups, adamw_groups = split_by_optimizer(groups)
    matrix_opt = torch.optim.SGD(matrix_groups, lr=lr, momentum=0.9)
    adamw_opt = torch.optim.AdamW(adamw_groups, lr=lr)
    return [matrix_opt, adamw_opt]


def make_batches(n: int, seed: int = 1234, batch: int = 2):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, V, (batch, T), generator=g)
        y = torch.randint(0, V, (batch, T), generator=g)
        out.append([(x, y)])
    return out


def run_steps(model, optimizers, batches, start: int = 0):
    cfg = TrainStepConfig(accum_steps=1, grad_clip=1.0)
    for micro in batches:
        train_step(model, micro, optimizers, cfg)
    return model


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in sorted(model.named_parameters())])


# ---------------------------------------------------------------- atomic_save


def test_atomic_save_writes_a_loadable_file(tmp_path: Path):
    p = tmp_path / "x.pt"
    atomic_save({"a": torch.tensor([1.0, 2.0])}, p)
    assert p.is_file()
    got = torch.load(p, weights_only=False)
    assert torch.equal(got["a"], torch.tensor([1.0, 2.0]))


def test_atomic_save_leaves_no_temp_file(tmp_path: Path):
    p = tmp_path / "x.pt"
    atomic_save({"a": 1}, p)
    assert sorted(q.name for q in tmp_path.iterdir()) == ["x.pt"]


def test_atomic_save_creates_missing_parent_directories(tmp_path: Path):
    p = tmp_path / "runs" / "exp1" / "x.pt"
    atomic_save({"a": 1}, p)
    assert p.is_file()


def test_atomic_save_accepts_a_string_path(tmp_path: Path):
    p = tmp_path / "x.pt"
    atomic_save({"a": 1}, str(p))
    assert p.is_file()


def test_atomic_save_overwrites_an_existing_file(tmp_path: Path):
    p = tmp_path / "x.pt"
    atomic_save({"a": 1}, p)
    atomic_save({"a": 2}, p)
    assert torch.load(p, weights_only=False)["a"] == 2


class _SaveBoom(Exception):
    """Distinct from NotImplementedError so the stub cannot pass these."""


class _Unpicklable:
    def __reduce__(self):
        raise _SaveBoom("boom")


def test_a_failed_save_leaves_the_previous_file_intact(tmp_path: Path):
    """The whole point: a crash mid-save must not destroy good state."""
    p = tmp_path / "x.pt"
    atomic_save({"a": 1}, p)
    with pytest.raises(_SaveBoom):
        atomic_save({"bad": _Unpicklable()}, p)
    assert torch.load(p, weights_only=False)["a"] == 1


def test_a_failed_save_leaves_no_temp_file(tmp_path: Path):
    p = tmp_path / "x.pt"
    with pytest.raises(_SaveBoom):
        atomic_save({"bad": _Unpicklable()}, p)
    assert list(tmp_path.iterdir()) == []


def test_the_temp_file_is_a_sibling_not_in_tmpdir(tmp_path: Path, monkeypatch):
    """os.replace is atomic only within one filesystem, so the temp file must
    sit next to its destination."""
    seen: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst, **kw):
        seen.append((str(src), str(dst)))
        return real_replace(src, dst, **kw)

    monkeypatch.setattr(os, "replace", spy)
    p = tmp_path / "x.pt"
    atomic_save({"a": 1}, p)
    assert len(seen) == 1
    src, dst = seen[0]
    assert Path(src).parent == Path(dst).parent == tmp_path
    assert Path(src).name == "x.pt.tmp"


def test_atomic_save_fsyncs(tmp_path: Path, monkeypatch):
    calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])
    atomic_save({"a": 1}, tmp_path / "x.pt")
    assert calls, "expected os.fsync before the rename"


# ------------------------------------------------------- optimizer_group_names


def test_group_names_shape_and_content():
    model = make_model()
    opts = make_optimizers(model)
    names = optimizer_group_names(opts)
    assert len(names) == 2
    assert len(names[0]) == 1  # matrix
    assert len(names[1]) == 2  # embed, scalar
    flat = [n for per_opt in names for g in per_opt for n in g]
    assert sorted(flat) == sorted(n for n, _ in model.named_parameters())


def test_group_names_preserves_order():
    model = make_model()
    opts = make_optimizers(model)
    names = optimizer_group_names(opts)
    assert names[0][0] == list(opts[0].param_groups[0]["names"])


def test_group_names_rejects_a_group_without_names():
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="names"):
        optimizer_group_names([opt])


def test_group_names_message_locates_the_group():
    model = make_model()
    good = build_param_groups(model, GroupConfig())
    opt = torch.optim.SGD([{"params": good[0]["params"], "base_lr": 0.1}], lr=0.1)
    with pytest.raises(ValueError) as e:
        optimizer_group_names([opt])
    assert "0" in str(e.value)


def test_group_names_returns_copies_not_the_live_lists():
    model = make_model()
    opts = make_optimizers(model)
    names = optimizer_group_names(opts)
    names[0][0].append("injected")
    assert "injected" not in opts[0].param_groups[0]["names"]


# -------------------------------------------------------------- save and load


def test_save_writes_the_expected_keys(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    ckpt = torch.load(p, weights_only=False)
    assert set(ckpt) == {
        "version",
        "step",
        "model",
        "optimizers",
        "group_names",
        "scaler",
        "extra",
    }
    assert ckpt["version"] == CHECKPOINT_VERSION
    assert ckpt["step"] == 0
    assert ckpt["scaler"] is None
    assert ckpt["extra"] == {}
    assert len(ckpt["optimizers"]) == 2


def test_save_rejects_a_negative_step(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "x.pt", step=-1, model=model, optimizers=opts)


def test_save_is_atomic(tmp_path: Path, monkeypatch):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    assert sorted(q.name for q in tmp_path.iterdir()) == ["step_0.pt"]


def test_extra_is_copied_not_referenced(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    extra = {"run_name": "a"}
    save_checkpoint(p, step=0, model=model, optimizers=opts, extra=extra)
    extra["run_name"] = "b"
    assert torch.load(p, weights_only=False)["extra"]["run_name"] == "a"


def test_load_returns_the_step_and_extra(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_7.pt"
    save_checkpoint(p, step=7, model=model, optimizers=opts, extra={"k": 3})
    model2 = make_model(seed=1)
    opts2 = make_optimizers(model2)
    out = load_checkpoint(p, model=model2, optimizers=opts2)
    assert out == {"step": 7, "extra": {"k": 3}}


def test_load_restores_the_model_weights(tmp_path: Path):
    model = make_model(seed=0)
    opts = make_optimizers(model)
    run_steps(model, opts, make_batches(3))
    p = tmp_path / "step_3.pt"
    save_checkpoint(p, step=3, model=model, optimizers=opts)

    model2 = make_model(seed=99)
    opts2 = make_optimizers(model2)
    assert not torch.allclose(flat_params(model), flat_params(model2))
    load_checkpoint(p, model=model2, optimizers=opts2)
    assert torch.allclose(flat_params(model), flat_params(model2))


def test_load_restores_optimizer_momentum(tmp_path: Path):
    """SGD momentum and Adam's exp_avg are the state that makes resume differ
    from a cold start at the same weights."""
    model = make_model()
    opts = make_optimizers(model)
    run_steps(model, opts, make_batches(3))
    p = tmp_path / "step_3.pt"
    save_checkpoint(p, step=3, model=model, optimizers=opts)

    model2 = make_model(seed=99)
    opts2 = make_optimizers(model2)
    load_checkpoint(p, model=model2, optimizers=opts2)

    before = opts[0].state_dict()["state"]
    after = opts2[0].state_dict()["state"]
    assert set(before) == set(after) and before
    for k in before:
        assert torch.allclose(before[k]["momentum_buffer"], after[k]["momentum_buffer"])


def test_load_accepts_a_string_path(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_1.pt"
    save_checkpoint(p, step=1, model=model, optimizers=opts)
    assert load_checkpoint(str(p), model=model, optimizers=opts)["step"] == 1


# -------------------------------------------------------------- the guard


def test_load_rejects_a_wrong_version(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    ckpt = torch.load(p, weights_only=False)
    ckpt["version"] = CHECKPOINT_VERSION + 1
    torch.save(ckpt, p)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(p, model=model, optimizers=opts)


def test_load_rejects_a_different_optimizer_count(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    with pytest.raises(ValueError):
        load_checkpoint(p, model=model, optimizers=opts[:1])


def test_load_rejects_reordered_parameters(tmp_path: Path):
    """The headline failure mode. Counts are unchanged, so PyTorch's own
    validation passes and wq's momentum lands on wk."""
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)

    model2 = make_model()
    opts2 = make_optimizers(model2)
    g = opts2[0].param_groups[0]
    g["params"] = list(reversed(g["params"]))
    g["names"] = list(reversed(g["names"]))

    with pytest.raises(ValueError):
        load_checkpoint(p, model=model2, optimizers=opts2)


def test_the_reorder_message_names_the_optimizer(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)

    model2 = make_model()
    opts2 = make_optimizers(model2)
    g = opts2[1].param_groups[0]
    g["names"] = list(reversed(g["names"]))
    g["params"] = list(reversed(g["params"]))

    with pytest.raises(ValueError) as e:
        load_checkpoint(p, model=model2, optimizers=opts2)
    assert "1" in str(e.value)


def test_load_rejects_swapped_optimizer_order(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    with pytest.raises(ValueError):
        load_checkpoint(p, model=model, optimizers=list(reversed(opts)))


def test_load_rejects_a_renamed_parameter(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    opts[0].param_groups[0]["names"][0] = "blocks.0.attn.wq.weight_RENAMED"
    with pytest.raises(ValueError):
        load_checkpoint(p, model=model, optimizers=opts)


def test_nothing_is_mutated_when_validation_fails(tmp_path: Path):
    """Validate before loading: a half-restored model is worse than a refusal."""
    model = make_model(seed=0)
    opts = make_optimizers(model)
    run_steps(model, opts, make_batches(2))
    p = tmp_path / "step_2.pt"
    save_checkpoint(p, step=2, model=model, optimizers=opts)

    model2 = make_model(seed=99)
    opts2 = make_optimizers(model2)
    g = opts2[0].param_groups[0]
    g["params"] = list(reversed(g["params"]))
    g["names"] = list(reversed(g["names"]))
    before = flat_params(model2).clone()

    with pytest.raises(ValueError):
        load_checkpoint(p, model=model2, optimizers=opts2)
    assert torch.equal(before, flat_params(model2))


def test_load_rejects_a_scaler_the_checkpoint_does_not_have(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts)
    with pytest.raises(ValueError, match="scaler"):
        load_checkpoint(p, model=model, optimizers=opts, scaler=_FakeScaler())


def test_load_rejects_a_missing_scaler_the_checkpoint_has(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(p, step=0, model=model, optimizers=opts, scaler=_FakeScaler())
    with pytest.raises(ValueError, match="scaler"):
        load_checkpoint(p, model=model, optimizers=opts)


class _FakeScaler:
    """GradScaler's state_dict/load_state_dict surface, usable on CPU."""

    def __init__(self, scale: float = 65536.0, growth_tracker: int = 0):
        self._scale = scale
        self._growth_tracker = growth_tracker

    def state_dict(self):
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        self._scale = sd["scale"]
        self._growth_tracker = sd["growth_tracker"]


def test_scaler_state_round_trips(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    p = tmp_path / "step_0.pt"
    save_checkpoint(
        p, step=0, model=model, optimizers=opts, scaler=_FakeScaler(1024.0, 3)
    )
    target = _FakeScaler()
    load_checkpoint(p, model=model, optimizers=opts, scaler=target)
    assert target.state_dict() == {"scale": 1024.0, "growth_tracker": 3}


# ------------------------------------------------------------ the real property


def test_resume_matches_an_uninterrupted_run(tmp_path: Path):
    """10 steps, save, resume, 10 more == 20 steps in one go.

    This is the test the whole module exists for. It passes only if the model
    weights, BOTH optimizers' state, and the step counter all survive.
    """
    batches = make_batches(20)

    cold = make_model(seed=0)
    cold_opts = make_optimizers(cold)
    run_steps(cold, cold_opts, batches)

    warm = make_model(seed=0)
    warm_opts = make_optimizers(warm)
    run_steps(warm, warm_opts, batches[:10])
    p = tmp_path / "step_10.pt"
    save_checkpoint(p, step=10, model=warm, optimizers=warm_opts)

    resumed = make_model(seed=123)  # different init, fully overwritten by load
    resumed_opts = make_optimizers(resumed)
    out = load_checkpoint(p, model=resumed, optimizers=resumed_opts)
    assert out["step"] == 10
    run_steps(resumed, resumed_opts, batches[10:])

    torch.testing.assert_close(flat_params(cold), flat_params(resumed))


def test_resume_without_optimizer_state_diverges(tmp_path: Path):
    """Control for the test above: if momentum were dropped, it would differ.
    Guards against a resume test that passes because the model is too simple."""
    batches = make_batches(20)

    cold = make_model(seed=0)
    cold_opts = make_optimizers(cold)
    run_steps(cold, cold_opts, batches)

    warm = make_model(seed=0)
    warm_opts = make_optimizers(warm)
    run_steps(warm, warm_opts, batches[:10])
    p = tmp_path / "step_10.pt"
    save_checkpoint(p, step=10, model=warm, optimizers=warm_opts)

    resumed = make_model(seed=0)
    fresh_opts = make_optimizers(resumed)
    load_checkpoint(p, model=resumed, optimizers=fresh_opts)
    # deliberately throw the restored state away
    cleared = make_optimizers(resumed)
    run_steps(resumed, cleared, batches[10:])

    assert not torch.allclose(flat_params(cold), flat_params(resumed))


def test_two_saves_at_different_steps_are_independent(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    run_steps(model, opts, make_batches(2))
    save_checkpoint(tmp_path / "step_2.pt", step=2, model=model, optimizers=opts)
    first = flat_params(model).clone()
    run_steps(model, opts, make_batches(2, seed=7))
    save_checkpoint(tmp_path / "step_4.pt", step=4, model=model, optimizers=opts)

    back = make_model(seed=5)
    back_opts = make_optimizers(back)
    load_checkpoint(tmp_path / "step_2.pt", model=back, optimizers=back_opts)
    assert torch.allclose(first, flat_params(back))


# ------------------------------------------------------- latest_checkpoint


def test_latest_checkpoint_picks_the_largest_step(tmp_path: Path):
    for n in (1, 5, 3):
        (tmp_path / f"step_{n}.pt").write_bytes(b"")
    assert latest_checkpoint(tmp_path).name == "step_5.pt"


def test_latest_checkpoint_compares_numerically_not_as_strings(tmp_path: Path):
    """step_9 sorts after step_10 as a string. This is the bug that resumes a
    multi-thousand-step run from step 9."""
    for n in (9, 10, 100, 2000):
        (tmp_path / f"step_{n}.pt").write_bytes(b"")
    assert latest_checkpoint(tmp_path).name == "step_2000.pt"


def test_latest_checkpoint_returns_none_for_an_empty_directory(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None


def test_latest_checkpoint_returns_none_for_a_missing_directory(tmp_path: Path):
    assert latest_checkpoint(tmp_path / "nope") is None


def test_latest_checkpoint_ignores_unrelated_files(tmp_path: Path):
    (tmp_path / "config.yaml").write_bytes(b"")
    (tmp_path / "train.log").write_bytes(b"")
    (tmp_path / "step_4.pt").write_bytes(b"")
    (tmp_path / "step_9.pt.tmp").write_bytes(b"")
    (tmp_path / "step_abc.pt").write_bytes(b"")
    (tmp_path / "final_step_99.pt").write_bytes(b"")
    assert latest_checkpoint(tmp_path).name == "step_4.pt"


def test_latest_checkpoint_returns_a_path(tmp_path: Path):
    (tmp_path / "step_0.pt").write_bytes(b"")
    got = latest_checkpoint(tmp_path)
    assert isinstance(got, Path) and got.parent == tmp_path


def test_latest_checkpoint_round_trips_with_save(tmp_path: Path):
    model = make_model()
    opts = make_optimizers(model)
    for n in (0, 10, 20):
        save_checkpoint(tmp_path / f"step_{n}.pt", step=n, model=model, optimizers=opts)
    p = latest_checkpoint(tmp_path)
    assert load_checkpoint(p, model=model, optimizers=opts)["step"] == 20


def test_latest_checkpoint_accepts_a_string_path(tmp_path: Path):
    (tmp_path / "step_3.pt").write_bytes(b"")
    assert latest_checkpoint(str(tmp_path)).name == "step_3.pt"
