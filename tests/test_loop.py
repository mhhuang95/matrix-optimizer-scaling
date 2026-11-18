"""Tests for optbench.loop — TASK 5."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from optbench.loop import LoopConfig, is_due, run_loop
from optbench.model import ModelConfig, Transformer
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
from optbench.schedule import ScheduleConfig, lr_multiplier
from optbench.shards import write_shard
from optbench.train_step import TrainStepConfig

SEED = 13
V = 64
T = 16


def make_windows(tmp_path, n_tokens: int = 1500):
    from optbench.data import TokenWindows

    arr = (np.arange(n_tokens) * 7 + 3) % V
    path = tmp_path / "shard_000.bin"
    write_shard(path, arr, vocab_size=V)
    return TokenWindows([path], T)


def tiny_model() -> Transformer:
    torch.manual_seed(SEED)
    cfg = ModelConfig(
        vocab_size=V,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=T,
    )
    return Transformer(cfg)


def build(tmp_path, arm: str = "adamw", total_steps: int = 6, **loop_kw):
    model = tiny_model()
    opts = build_optimizers(model, GroupConfig(), OptimizerConfig(arm=arm))
    loop_kw.setdefault("global_batch", 4)
    loop_kw.setdefault("checkpoint_dir", tmp_path / "ckpts")
    loop_kw.setdefault("seed", SEED)
    loop_cfg = LoopConfig(**loop_kw)
    sched_cfg = ScheduleConfig(kind="cosine", total_steps=total_steps, warmup_steps=2)
    step_cfg = TrainStepConfig(accum_steps=1)
    return model, opts, loop_cfg, sched_cfg, step_cfg


def run(tmp_path, windows, log=None, arm="adamw", total_steps=6, **kw):
    stop_after = kw.pop("stop_after", None)
    model, opts, loop_cfg, sched_cfg, step_cfg = build(
        tmp_path, arm=arm, total_steps=total_steps, **kw
    )
    out = run_loop(
        model,
        windows,
        opts,
        loop_cfg,
        sched_cfg,
        step_cfg,
        stop_after=stop_after,
        log_fn=None if log is None else log.append,
    )
    return model, out


# ---------------------------------------------------------------------------
# LoopConfig.validate and is_due
# ---------------------------------------------------------------------------


class TestConfigAndCadence:
    def test_defaults_are_valid(self):
        LoopConfig().validate()

    def test_rejects_bad_fields(self):
        bad = [
            dict(global_batch=0),
            dict(checkpoint_every=-1),
            dict(log_every=-1),
            dict(spectral_every=-1),
            dict(meter_warmup=-1),
        ]
        for kw in bad:
            with pytest.raises(ValueError):
                LoopConfig(**kw).validate()

    def test_every_zero_never_fires(self):
        assert not any(is_due(step, 0) for step in range(50))

    def test_cadence_counts_completed_updates(self):
        fired = [step for step in range(6) if is_due(step, 2)]
        assert fired == [1, 3, 5]
        assert is_due(0, 1) and is_due(1, 1)


# ---------------------------------------------------------------------------
# one run, from scratch
# ---------------------------------------------------------------------------


class TestFreshRun:
    def test_runs_to_total_steps(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        _, out = run(tmp_path, windows, log=log, total_steps=6)
        assert out == {"final_step": 6, "resumed_from": 0, "steps_run": 6}
        assert [row["step"] for row in log] == list(range(6))

    def test_log_rows_have_the_contract_keys(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        run(tmp_path, windows, log=log, total_steps=3)
        for row in log:
            assert {"step", "loss", "grad_norm", "lr_mult", "tokens_per_sec", "skipped"} <= set(row)
            assert np.isfinite(row["loss"])
            assert row["skipped"] is False
        assert log[-1]["tokens_per_sec"] > 0

    def test_lr_mult_matches_the_schedule(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        model, opts, loop_cfg, sched_cfg, step_cfg = build(tmp_path, total_steps=6)
        run_loop(model, windows, opts, loop_cfg, sched_cfg, step_cfg, log_fn=log.append)
        for row in log:
            assert row["lr_mult"] == pytest.approx(lr_multiplier(row["step"], sched_cfg))
        final_mult = lr_multiplier(5, sched_cfg)
        for opt in opts:
            for g in opt.param_groups:
                assert g["lr"] == pytest.approx(g["base_lr"] * final_mult)

    def test_loss_goes_down(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        run(tmp_path, windows, log=log, total_steps=25)
        first = np.mean([row["loss"] for row in log[:5]])
        last = np.mean([row["loss"] for row in log[-5:]])
        assert last < first

    def test_log_every_zero_logs_nothing(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        run(tmp_path, windows, log=log, total_steps=4, log_every=0)
        assert log == []

    @pytest.mark.parametrize("arm", ["muon", "shampoo"])
    def test_matrix_arms_complete(self, tmp_path, arm):
        windows = make_windows(tmp_path)
        log = []
        _, out = run(tmp_path, windows, log=log, arm=arm, total_steps=3)
        assert out["steps_run"] == 3
        assert all(np.isfinite(row["loss"]) for row in log)

    def test_rejects_bad_stop_after(self, tmp_path):
        windows = make_windows(tmp_path)
        with pytest.raises(ValueError):
            run(tmp_path, windows, stop_after=0)


# ---------------------------------------------------------------------------
# checkpoints and resume
# ---------------------------------------------------------------------------


class TestCheckpointsAndResume:
    def test_cadence_plus_final_checkpoint(self, tmp_path):
        windows = make_windows(tmp_path)
        run(tmp_path, windows, total_steps=5, checkpoint_every=2)
        names = sorted(p.name for p in (tmp_path / "ckpts").glob("step_*.pt"))
        assert names == ["step_2.pt", "step_4.pt", "step_5.pt"]

    def test_only_final_when_cadence_disabled(self, tmp_path):
        windows = make_windows(tmp_path)
        run(tmp_path, windows, total_steps=5, checkpoint_every=0)
        names = [p.name for p in (tmp_path / "ckpts").glob("step_*.pt")]
        assert names == ["step_5.pt"]

    def test_interrupted_run_resumes_bit_exact(self, tmp_path):
        windows = make_windows(tmp_path)
        gold, _ = run(tmp_path, windows, total_steps=6, checkpoint_dir=tmp_path / "gold")

        part, out1 = run(
            tmp_path, windows, total_steps=6, stop_after=3, checkpoint_dir=tmp_path / "r"
        )
        assert out1 == {"final_step": 3, "resumed_from": 0, "steps_run": 3}
        resumed, out2 = run(tmp_path, windows, total_steps=6, checkpoint_dir=tmp_path / "r")
        assert out2 == {"final_step": 6, "resumed_from": 3, "steps_run": 3}

        for p_gold, p_res in zip(gold.parameters(), resumed.parameters()):
            assert torch.equal(p_gold, p_res)

    def test_completed_run_is_a_noop(self, tmp_path):
        windows = make_windows(tmp_path)
        run(tmp_path, windows, total_steps=4)
        log = []
        _, out = run(tmp_path, windows, log=log, total_steps=4)
        assert out == {"final_step": 4, "resumed_from": 4, "steps_run": 0}
        assert log == []

    def test_same_seed_same_weights(self, tmp_path):
        windows = make_windows(tmp_path)
        m1, _ = run(tmp_path, windows, total_steps=4, checkpoint_dir=tmp_path / "a")
        m2, _ = run(tmp_path, windows, total_steps=4, checkpoint_dir=tmp_path / "b")
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_resume_with_different_data_order_is_loud(self, tmp_path):
        windows = make_windows(tmp_path)
        run(tmp_path, windows, total_steps=6, stop_after=2)
        with pytest.raises(ValueError):
            run(tmp_path, windows, total_steps=6, seed=SEED + 1)
        with pytest.raises(ValueError):
            run(tmp_path, windows, total_steps=6, global_batch=8)


# ---------------------------------------------------------------------------
# spectral cadence
# ---------------------------------------------------------------------------


class TestSpectralCadence:
    def test_spectral_lands_on_its_cadence_only(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        run(tmp_path, windows, log=log, total_steps=6, spectral_every=3)
        with_spec = [row["step"] for row in log if "spectral" in row]
        assert with_spec == [2, 5]
        stats = next(row["spectral"] for row in log if "spectral" in row)
        assert "blocks.0.attn.wq.weight" in stats
        assert "tok_emb.weight" not in stats

    def test_spectral_due_emits_row_even_when_log_not_due(self, tmp_path):
        windows = make_windows(tmp_path)
        log = []
        run(tmp_path, windows, log=log, total_steps=6, log_every=4, spectral_every=3)
        steps = [row["step"] for row in log]
        assert steps == [2, 3, 5]
        assert [s for s in steps if "spectral" in log[steps.index(s)]] == [2, 5]
