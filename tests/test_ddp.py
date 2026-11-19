"""Tests for optbench.ddp — TASK 6.

The TestMultiProcess cases spawn real 2-rank gloo jobs (separate python
processes), so this file takes noticeably longer than the others.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from optbench.data import TokenWindows, get_batch
from optbench.ddp import (
    DistContext,
    all_reduce_mean,
    cleanup_distributed,
    detect_dist,
    init_distributed,
    maybe_no_sync,
    unwrap_ddp,
    wrap_ddp,
)
from optbench.model import ModelConfig, Transformer
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
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


def run_steps(step_model, windows, opts, *, rank=0, world=1, accum=1, n_steps=3):
    cfg = TrainStepConfig(accum_steps=accum)
    for step in range(n_steps):
        micros = [
            get_batch(
                windows, step, 8, SEED,
                rank=rank, world_size=world,
                micro_step=m, accum_steps=accum,
            )
            for m in range(accum)
        ]
        train_step(step_model, micros, opts, cfg, None)


# ---------------------------------------------------------------------------
# DistContext and environment parsing (no torch.distributed state involved)
# ---------------------------------------------------------------------------


class TestDistContext:
    def test_defaults_are_a_single_process(self):
        ctx = DistContext()
        ctx.validate()
        assert ctx.is_main
        assert not ctx.is_distributed

    def test_nonzero_rank_in_a_job(self):
        ctx = DistContext(rank=1, world_size=4, local_rank=1)
        ctx.validate()
        assert not ctx.is_main
        assert ctx.is_distributed

    def test_validate_rejects_bad_fields(self):
        bad = [
            dict(world_size=0),
            dict(rank=2, world_size=2),
            dict(rank=-1, world_size=2),
            dict(local_rank=-1),
        ]
        for kw in bad:
            with pytest.raises(ValueError):
                DistContext(**kw).validate()


class TestDetect:
    def test_empty_env_is_single_process(self):
        assert detect_dist(env={}) == DistContext()

    def test_torchrun_env_is_parsed(self):
        env = {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}
        assert detect_dist(env=env) == DistContext(rank=1, world_size=2, local_rank=1)

    def test_local_rank_falls_back_to_zero(self):
        env = {"RANK": "0", "WORLD_SIZE": "2"}
        assert detect_dist(env=env).local_rank == 0

    def test_half_configured_env_is_loud(self):
        with pytest.raises(ValueError):
            detect_dist(env={"RANK": "0"})
        with pytest.raises(ValueError):
            detect_dist(env={"WORLD_SIZE": "2"})

    def test_out_of_range_rank_is_loud(self):
        with pytest.raises(ValueError):
            detect_dist(env={"RANK": "2", "WORLD_SIZE": "2"})


# ---------------------------------------------------------------------------
# single-process paths: everything must work with no process group at all
# ---------------------------------------------------------------------------


class TestSingleProcessPaths:
    def test_init_single_process_creates_no_group(self):
        init_distributed(DistContext())
        assert not tdist.is_initialized()

    def test_cleanup_without_a_group_is_safe(self):
        cleanup_distributed()
        cleanup_distributed()

    def test_wrap_ddp_returns_the_model_unchanged(self):
        m = torch.nn.Linear(2, 2)
        assert wrap_ddp(m, DistContext()) is m

    def test_unwrap_plain_module_is_identity(self):
        m = torch.nn.Linear(2, 2)
        assert unwrap_ddp(m) is m

    def test_all_reduce_mean_without_a_group_is_identity(self):
        out = all_reduce_mean(3.5)
        assert out == 3.5
        assert isinstance(out, float)

    def test_maybe_no_sync_plain_module_is_nullcontext(self):
        m = torch.nn.Linear(2, 2)
        assert isinstance(maybe_no_sync(m, is_last=False), nullcontext)
        assert isinstance(maybe_no_sync(m, is_last=True), nullcontext)
        with maybe_no_sync(m, is_last=False):
            pass


# ---------------------------------------------------------------------------
# with a real (single-rank) gloo group: DDP objects without multiprocessing
# ---------------------------------------------------------------------------


@pytest.fixture
def solo_group(tmp_path):
    tdist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'solo_init'}", rank=0, world_size=1
    )
    yield
    tdist.destroy_process_group()


class _CountingDDP(DDP):
    """A DDP whose no_sync() calls are counted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counted_no_syncs = 0

    def no_sync(self):
        self.counted_no_syncs += 1
        return super().no_sync()


class TestWithSoloGroup:
    def test_unwrap_returns_the_inner_module(self, solo_group):
        lin = torch.nn.Linear(2, 2)
        wrapped = DDP(lin)
        assert unwrap_ddp(wrapped) is lin

    def test_maybe_no_sync_ddp_contexts(self, solo_group):
        wrapped = DDP(torch.nn.Linear(2, 2))
        assert not isinstance(maybe_no_sync(wrapped, is_last=False), nullcontext)
        assert isinstance(maybe_no_sync(wrapped, is_last=True), nullcontext)

    def test_train_step_no_syncs_all_but_the_last_micro(self, solo_group, tmp_path):
        windows = TokenWindows([make_shard(tmp_path)], T)
        model = tiny_model()
        opts = build_adamw(model)
        wrapped = _CountingDDP(model)

        micros = [
            get_batch(windows, 0, 6, SEED, micro_step=m, accum_steps=3)
            for m in range(3)
        ]
        train_step(wrapped, micros, opts, TrainStepConfig(accum_steps=3), None)
        assert wrapped.counted_no_syncs == 2

        wrapped.counted_no_syncs = 0
        micros = [get_batch(windows, 1, 6, SEED)]
        train_step(wrapped, micros, opts, TrainStepConfig(accum_steps=1), None)
        assert wrapped.counted_no_syncs == 0


# ---------------------------------------------------------------------------
# real 2-rank jobs
# ---------------------------------------------------------------------------


def _worker_all_reduce_and_grad_sync(rank, world, init_file, shard):
    init_distributed(
        DistContext(rank=rank, world_size=world),
        backend="gloo",
        init_method=f"file://{init_file}",
    )
    try:
        assert all_reduce_mean(float(rank + 1)) == pytest.approx(1.5)

        windows = TokenWindows([Path(shard)], T)
        model = tiny_model()
        opts = build_adamw(model)
        wrapped = wrap_ddp(model, DistContext(rank=rank, world_size=world))
        assert isinstance(wrapped, DDP)

        micros = [get_batch(windows, 0, 8, SEED, rank=rank, world_size=world)]
        train_step(wrapped, micros, opts, TrainStepConfig(), None)
        for p in model.parameters():
            gathered = [torch.zeros_like(p.grad) for _ in range(world)]
            tdist.all_gather(gathered, p.grad)
            assert torch.equal(gathered[0], gathered[1])
    finally:
        cleanup_distributed()


def _worker_equivalence(rank, world, init_file, shard, out_dir):
    init_distributed(
        DistContext(rank=rank, world_size=world),
        backend="gloo",
        init_method=f"file://{init_file}",
    )
    try:
        windows = TokenWindows([Path(shard)], T)
        model = tiny_model()
        opts = build_adamw(model)
        wrapped = wrap_ddp(model, DistContext(rank=rank, world_size=world))
        run_steps(wrapped, windows, opts, rank=rank, world=world, n_steps=3)
        if rank == 0:
            torch.save(unwrap_ddp(wrapped).state_dict(), Path(out_dir) / "ddp_final.pt")
    finally:
        cleanup_distributed()


def _worker_accum_equivalence(rank, world, init_file, shard):
    init_distributed(
        DistContext(rank=rank, world_size=world),
        backend="gloo",
        init_method=f"file://{init_file}",
    )
    try:
        windows = TokenWindows([Path(shard)], T)
        ctx = DistContext(rank=rank, world_size=world)

        model_a = tiny_model()
        opts_a = build_adamw(model_a)
        run_steps(wrap_ddp(model_a, ctx), windows, opts_a,
                  rank=rank, world=world, accum=2, n_steps=2)

        model_b = tiny_model()
        opts_b = build_adamw(model_b)
        run_steps(wrap_ddp(model_b, ctx), windows, opts_b,
                  rank=rank, world=world, accum=1, n_steps=2)

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb, atol=1e-6)
    finally:
        cleanup_distributed()


class TestMultiProcess:
    def test_all_reduce_mean_and_grad_sync_across_ranks(self, tmp_path):
        shard = make_shard(tmp_path)
        mp.spawn(
            _worker_all_reduce_and_grad_sync,
            args=(2, str(tmp_path / "init_a"), str(shard)),
            nprocs=2,
            join=True,
        )

    def test_two_rank_ddp_matches_single_process(self, tmp_path):
        shard = make_shard(tmp_path)
        mp.spawn(
            _worker_equivalence,
            args=(2, str(tmp_path / "init_b"), str(shard), str(tmp_path)),
            nprocs=2,
            join=True,
        )

        windows = TokenWindows([shard], T)
        model = tiny_model()
        opts = build_adamw(model)
        run_steps(model, windows, opts, n_steps=3)

        ddp_state = torch.load(tmp_path / "ddp_final.pt")
        single = model.state_dict()
        assert set(single) == set(ddp_state)  # no "module." prefixes
        for key in single:
            assert torch.allclose(single[key], ddp_state[key], atol=1e-5), key

    def test_accum_with_no_sync_matches_one_micro(self, tmp_path):
        shard = make_shard(tmp_path)
        mp.spawn(
            _worker_accum_equivalence,
            args=(2, str(tmp_path / "init_c"), str(shard)),
            nprocs=2,
            join=True,
        )
