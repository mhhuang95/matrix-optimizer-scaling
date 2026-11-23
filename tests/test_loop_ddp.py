"""Tests for the distributed run_loop — TASK 7.

TestTwoRankLoop spawns real 2-rank gloo jobs running the FULL loop
(data sharding, DDP grad sync, rank-0 checkpoints, resume), so this file
is one of the slower ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from optbench.checkpoint import load_checkpoint
from optbench.data import TokenWindows, get_batch
from optbench.ddp import (
    DistContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    wrap_ddp,
)
from optbench.loop import LoopConfig, run_loop
from optbench.model import ModelConfig, Transformer
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
from optbench.schedule import ScheduleConfig, apply_lr, lr_multiplier
from optbench.shards import write_shard
from optbench.train_step import TrainStepConfig, train_step

SEED = 13
V = 64
T = 16


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


def make_cfgs(ckpt_dir, *, total_steps=4, **loop_kw):
    loop_kw.setdefault("global_batch", 4)
    loop_kw.setdefault("seed", SEED)
    loop_cfg = LoopConfig(checkpoint_dir=ckpt_dir, **loop_kw)
    sched_cfg = ScheduleConfig(kind="cosine", total_steps=total_steps, warmup_steps=2)
    return loop_cfg, sched_cfg, TrainStepConfig(accum_steps=1)


# ---------------------------------------------------------------------------
# single-process: ctx semantics that need no process group
# ---------------------------------------------------------------------------


class TestSingleProcessCtx:
    def test_explicit_default_ctx_matches_none(self, tmp_path):
        windows = TokenWindows([make_shard(tmp_path)], T)

        model_a = tiny_model()
        opts_a = build_adamw(model_a)
        out_a = run_loop(model_a, windows, opts_a, *make_cfgs(tmp_path / "a"))

        model_b = tiny_model()
        opts_b = build_adamw(model_b)
        out_b = run_loop(
            model_b, windows, opts_b, *make_cfgs(tmp_path / "b"),
            dist_ctx=DistContext(),
        )

        assert out_a == out_b
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(pa, pb)

    def test_non_main_rank_saves_and_logs_nothing(self, tmp_path):
        windows = TokenWindows([make_shard(tmp_path)], T)
        model = tiny_model()
        opts = build_adamw(model)
        log = []
        out = run_loop(
            model, windows, opts,
            *make_cfgs(tmp_path / "c", total_steps=3, checkpoint_every=1),
            log_fn=log.append,
            dist_ctx=DistContext(rank=1, world_size=2),
        )
        assert out == {"final_step": 3, "resumed_from": 0, "steps_run": 3}
        assert log == []
        assert list((tmp_path / "c").glob("step_*.pt")) == []

    def test_main_rank_of_two_trains_on_its_shard(self, tmp_path):
        windows = TokenWindows([make_shard(tmp_path)], T)

        model_loop = tiny_model()
        opts_loop = build_adamw(model_loop)
        loop_cfg, sched_cfg, step_cfg = make_cfgs(tmp_path / "d", total_steps=3)
        run_loop(
            model_loop, windows, opts_loop, loop_cfg, sched_cfg, step_cfg,
            dist_ctx=DistContext(rank=0, world_size=2),
        )
        assert [p.name for p in (tmp_path / "d").glob("step_*.pt")] == ["step_3.pt"]

        model_man = tiny_model()
        opts_man = build_adamw(model_man)
        for step in range(3):
            mult = lr_multiplier(step, sched_cfg)
            for opt in opts_man:
                apply_lr(opt, mult)
            micros = [get_batch(windows, step, 4, SEED, rank=0, world_size=2)]
            train_step(model_man, micros, opts_man, step_cfg, None)

        for p1, p2 in zip(model_loop.parameters(), model_man.parameters()):
            assert torch.equal(p1, p2)

    def test_bad_dist_ctx_is_loud(self, tmp_path):
        windows = TokenWindows([make_shard(tmp_path)], T)
        model = tiny_model()
        opts = build_adamw(model)
        with pytest.raises(ValueError):
            run_loop(
                model, windows, opts, *make_cfgs(tmp_path / "e"),
                dist_ctx=DistContext(rank=3, world_size=2),
            )

    def test_barrier_is_safe_without_a_group(self):
        barrier(DistContext())
        barrier(DistContext(rank=1, world_size=2))


# ---------------------------------------------------------------------------
# real 2-rank jobs running the full loop
# ---------------------------------------------------------------------------


def _worker_loop(rank, world, init_file, shard, out_dir):
    ctx = DistContext(rank=rank, world_size=world)
    init_distributed(ctx, backend="gloo", init_method=f"file://{init_file}")
    try:
        windows = TokenWindows([Path(shard)], T)
        model = tiny_model()
        opts = build_adamw(model)
        wrapped = wrap_ddp(model, ctx)
        log = []
        loop_cfg, sched_cfg, step_cfg = make_cfgs(
            Path(out_dir) / "ckpts", total_steps=4, checkpoint_every=2, log_every=1
        )
        out = run_loop(
            wrapped, windows, opts, loop_cfg, sched_cfg, step_cfg,
            log_fn=log.append, dist_ctx=ctx,
        )
        assert out == {"final_step": 4, "resumed_from": 0, "steps_run": 4}
        if rank == 0:
            assert [row["step"] for row in log] == [0, 1, 2, 3]
            with open(Path(out_dir) / "rank0_log.json", "w") as f:
                json.dump(log, f)
        else:
            assert log == []  # log_fn was given, is_main gating must ignore it
    finally:
        cleanup_distributed()


def _worker_loop_resume(rank, world, init_file, shard, out_dir):
    ctx = DistContext(rank=rank, world_size=world)
    init_distributed(ctx, backend="gloo", init_method=f"file://{init_file}")
    try:
        windows = TokenWindows([Path(shard)], T)

        model = tiny_model()
        opts = build_adamw(model)
        wrapped = wrap_ddp(model, ctx)
        loop_cfg, sched_cfg, step_cfg = make_cfgs(Path(out_dir) / "r", log_every=0)
        out1 = run_loop(
            wrapped, windows, opts, loop_cfg, sched_cfg, step_cfg,
            stop_after=2, dist_ctx=ctx,
        )
        assert out1 == {"final_step": 2, "resumed_from": 0, "steps_run": 2}

        model2 = tiny_model()
        opts2 = build_adamw(model2)
        wrapped2 = wrap_ddp(model2, ctx)
        out2 = run_loop(
            wrapped2, windows, opts2, loop_cfg, sched_cfg, step_cfg,
            dist_ctx=ctx,
        )
        assert out2 == {"final_step": 4, "resumed_from": 2, "steps_run": 2}
    finally:
        cleanup_distributed()


class TestTwoRankLoop:
    def _single_process_gold(self, shard, ckpt_dir, log=None):
        windows = TokenWindows([shard], T)
        model = tiny_model()
        opts = build_adamw(model)
        loop_cfg, sched_cfg, step_cfg = make_cfgs(ckpt_dir, total_steps=4, log_every=1)
        run_loop(
            model, windows, opts, loop_cfg, sched_cfg, step_cfg,
            log_fn=None if log is None else log.append,
        )
        return model

    def test_two_rank_loop_matches_single_process(self, tmp_path):
        shard = make_shard(tmp_path)
        mp.spawn(
            _worker_loop,
            args=(2, str(tmp_path / "init_a"), str(shard), str(tmp_path)),
            nprocs=2,
            join=True,
        )

        # one writer: only rank 0's files, cadence 2 plus final
        names = sorted(p.name for p in (tmp_path / "ckpts").glob("step_*.pt"))
        assert names == ["step_2.pt", "step_4.pt"]

        gold_log = []
        gold = self._single_process_gold(shard, tmp_path / "gold", log=gold_log)

        # the 2-rank checkpoint loads into an UNWRAPPED single-process model
        model2 = tiny_model()
        opts2 = build_adamw(model2)
        load_checkpoint(tmp_path / "ckpts" / "step_4.pt", model=model2, optimizers=opts2)
        for p1, p2 in zip(gold.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)

        # rank 0 logged the all-reduced global loss, step by step
        rows = json.loads((tmp_path / "rank0_log.json").read_text())
        for two_rank_row, gold_row in zip(rows, gold_log):
            assert two_rank_row["loss"] == pytest.approx(gold_row["loss"], rel=1e-4)

    def test_two_rank_interrupted_resume_matches_single_process(self, tmp_path):
        shard = make_shard(tmp_path)
        mp.spawn(
            _worker_loop_resume,
            args=(2, str(tmp_path / "init_b"), str(shard), str(tmp_path)),
            nprocs=2,
            join=True,
        )

        gold = self._single_process_gold(shard, tmp_path / "gold")

        model2 = tiny_model()
        opts2 = build_adamw(model2)
        load_checkpoint(tmp_path / "r" / "step_4.pt", model=model2, optimizers=opts2)
        for p1, p2 in zip(gold.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)
