"""Tests for MFU in the log row — TASK 8."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from optbench.data import TokenWindows
from optbench.ddp import DistContext
from optbench.loop import LoopConfig, run_loop
from optbench.model import ModelConfig, Transformer
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
from optbench.schedule import ScheduleConfig
from optbench.shards import write_shard
from optbench.train_step import TrainStepConfig

SEED = 13
V = 64
T = 16
FPT = 6000  # flops per token: any positive int works, the loop just plumbs it
PEAK = 1.0e12


def make_shard(dirpath) -> Path:
    arr = (np.arange(1500) * 7 + 3) % V
    path = Path(dirpath) / "shard_000.bin"
    write_shard(path, arr, vocab_size=V)
    return path


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


def build_adamw(model):
    return build_optimizers(model, GroupConfig(), OptimizerConfig(arm="adamw"))


def run_logged(tmp_path, *, total_steps=4, dist_ctx=None, **loop_kw):
    loop_kw.setdefault("global_batch", 4)
    loop_kw.setdefault("seed", SEED)
    loop_kw.setdefault("log_every", 1)
    windows = TokenWindows([make_shard(tmp_path)], T)
    model = tiny_model()
    opts = build_adamw(model)
    log = []
    run_loop(
        model,
        windows,
        opts,
        LoopConfig(checkpoint_dir=tmp_path / "ckpts", **loop_kw),
        ScheduleConfig(kind="cosine", total_steps=total_steps, warmup_steps=2),
        TrainStepConfig(accum_steps=1),
        log_fn=log.append,
        dist_ctx=dist_ctx,
    )
    return log


class TestMfuConfig:
    def test_flops_without_peak_is_loud(self):
        with pytest.raises(ValueError):
            LoopConfig(flops_per_token=FPT).validate()

    def test_peak_without_flops_is_loud(self):
        with pytest.raises(ValueError):
            LoopConfig(peak_flops_per_gpu=PEAK).validate()

    def test_bad_values_are_loud(self):
        with pytest.raises(ValueError):
            LoopConfig(flops_per_token=0, peak_flops_per_gpu=PEAK).validate()
        with pytest.raises(ValueError):
            LoopConfig(flops_per_token=FPT, peak_flops_per_gpu=0.0).validate()
        with pytest.raises(ValueError):
            LoopConfig(flops_per_token=FPT, peak_flops_per_gpu=-1.0).validate()

    def test_both_set_validates(self):
        LoopConfig(flops_per_token=FPT, peak_flops_per_gpu=PEAK).validate()


class TestMfuRows:
    def test_unconfigured_rows_have_no_mfu_key(self, tmp_path):
        log = run_logged(tmp_path)
        assert len(log) == 4
        for row in log:
            assert "mfu" not in row

    def test_mfu_matches_the_formula(self, tmp_path):
        log = run_logged(tmp_path, flops_per_token=FPT, peak_flops_per_gpu=PEAK)
        assert len(log) == 4
        measured = 0
        for row in log:
            assert "mfu" in row
            if row["tokens_per_sec"] is None:
                assert row["mfu"] is None
            else:
                measured += 1
                assert row["mfu"] == pytest.approx(
                    row["tokens_per_sec"] * FPT / PEAK
                )
        assert measured >= 1  # the clock on any real machine moves within 4 steps

    def test_warmup_rows_report_none_not_zero(self, tmp_path):
        log = run_logged(
            tmp_path,
            flops_per_token=FPT,
            peak_flops_per_gpu=PEAK,
            meter_warmup=10,  # > total_steps: the whole run is warmup
        )
        assert len(log) == 4
        for row in log:
            assert row["tokens_per_sec"] is None
            assert "mfu" in row
            assert row["mfu"] is None

    def test_world_size_scales_the_denominator(self, tmp_path):
        log = run_logged(
            tmp_path,
            flops_per_token=FPT,
            peak_flops_per_gpu=PEAK,
            dist_ctx=DistContext(rank=0, world_size=2),
        )
        for row in log:
            if row["tokens_per_sec"] is not None:
                assert row["mfu"] == pytest.approx(
                    row["tokens_per_sec"] * FPT / (2 * PEAK)
                )
        assert any(row["tokens_per_sec"] is not None for row in log)
