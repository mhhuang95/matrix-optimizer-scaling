"""Data parallelism: the process context, DDP wrapping, and the sync idioms.

TASK 6 — implement everything marked `raise NotImplementedError`, then make
the one marked edit in train_step.py (see maybe_no_sync below).
Run `uv run pytest tests/test_ddp.py -v` until all tests pass. The
multi-process tests spawn real 2-rank gloo jobs, so the file takes ~a minute.

The DDP mental model, in one paragraph: one process per GPU, each holding a
FULL replica of the model. Data is sharded by rank (get_batch already does
this — rank-major slices of the same global batch), each replica runs its
own forward/backward, and DDP averages the gradients across ranks DURING
backward — all-reduce fires bucket-by-bucket as gradients become ready, so
communication overlaps with the rest of the backward compute. After backward
every rank holds identical averaged gradients, steps its own optimizer
identically, and the replicas never drift. The math: the average over ranks
of (mean loss over the rank's tokens) equals the mean over the global batch
— TRUE only because every rank sees the same number of tokens, which our
fixed-shape loader guarantees (the same invariant accumulation relied on).

Ordering rule the tests enforce: **build optimizers on the raw model, THEN
wrap.** DDP wraps forward/backward; the parameters inside are the same
tensors, so optimizers built first keep working. Built after, every name in
param_groups would grow a "module." prefix — breaking the checkpoint
fingerprint and the spectral stats names, which must stay identical between
single- and multi-GPU runs.

The same prefix problem is why checkpoints save the UNWRAPPED model:
DDP.state_dict() prefixes every key with "module.", and a checkpoint written
that way cannot be read by a single-process run. unwrap_ddp exists so the
call site reads `save_checkpoint(..., model=unwrap_ddp(model), ...)`.

torchrun's environment contract: it launches one process per GPU and sets
RANK, LOCAL_RANK, WORLD_SIZE (and MASTER_ADDR / MASTER_PORT for
rendezvous). detect_dist is a pure parse of that environment — no torch
calls — so it is trivially testable; init_distributed does the stateful
part. Backend: nccl on CUDA (GPU-to-GPU transport), gloo on CPU (and in our
tests).

Logging discipline under DDP: every rank computes stats, but only rank 0
should write them (is_main) — W&B does not want world_size copies of every
row. A rank's local loss is a legitimate sample, but the honest global
number is all_reduce_mean of the local losses.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistContext:
    """Who am I in the job? Pure data — no torch state.

    Defaults describe a single-process run, so code written against a
    DistContext needs no special case for "not distributed at all".
    """

    rank: int = 0  # my index in the job, 0-based
    world_size: int = 1  # total processes in the job
    local_rank: int = 0  # my index on THIS machine == my GPU index

    @property
    def is_main(self) -> bool:
        """True exactly on rank 0 — the only rank that logs/saves."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True when there is more than one process."""
        return self.world_size > 1

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * world_size >= 1
        * 0 <= rank < world_size
        * local_rank >= 0
        """
        if not self.world_size >= 1 or not 0 <= self.rank < self.world_size or not self.local_rank >= 0:
            raise ValueError(f"world_size {self.world_size} must be >= 1, rank {self.rank} must be in [0, world_size), local_rank {self.local_rank} must be >= 0.") 


def detect_dist(env: Mapping[str, str] | None = None) -> DistContext:
    """Parse the torchrun environment into a DistContext (validated).

    * `env` defaults to os.environ; it is injectable so tests never have to
      mutate the real environment.
    * Neither RANK nor WORLD_SIZE set: a plain single-process run — return
      the default DistContext.
    * Both set: parse them as ints; LOCAL_RANK falls back to 0 when absent.
    * Exactly ONE of RANK / WORLD_SIZE set: raise ValueError. A
      half-configured environment means a broken launcher, and guessing
      either value trains a silently wrong job.
    """
    if env is None:
        env = os.environ
    if "RANK" not in env and "WORLD_SIZE" not in env:
        ctx = DistContext()
    elif "RANK" in env and "WORLD_SIZE" in env:
        ctx = DistContext(rank=int(env["RANK"]), world_size=int(env["WORLD_SIZE"]), local_rank=int(env.get("LOCAL_RANK", 0)))
    else:
        raise ValueError("Exactly ONE of RANK / WORLD_SIZE exists.")
    ctx.validate()
    return ctx


def init_distributed(
    ctx: DistContext,
    *,
    backend: str | None = None,
    init_method: str | None = None,
) -> None:
    """Create the process group — or do nothing for a single-process ctx.

    * ctx.validate() first.
    * Not distributed: return without touching torch.distributed. A
      single-process run must stay runnable on a machine with no rendezvous
      environment at all.
    * Distributed: `dist.init_process_group(backend=..., init_method=...,
      rank=ctx.rank, world_size=ctx.world_size)`. Backend default: "nccl"
      when CUDA is available, else "gloo". init_method None lets torch use
      its default env:// rendezvous (MASTER_ADDR/PORT — exactly what
      torchrun provides); tests pass file:// URLs instead.
    """
    raise NotImplementedError


def cleanup_distributed() -> None:
    """Destroy the process group if one exists; safe to call when none does.

    (Safe-when-absent matters: the single-process path never created one,
    and crash-path cleanup code should not itself crash.)
    """
    raise NotImplementedError


def wrap_ddp(model: nn.Module, ctx: DistContext) -> nn.Module:
    """Wrap the model in DistributedDataParallel — iff ctx is distributed.

    * Not distributed: return the model unchanged (same object).
    * Distributed on CUDA: `DDP(model, device_ids=[ctx.local_rank])`.
    * Distributed on CPU (gloo): `DDP(model)` — no device_ids.

    Call this AFTER build_optimizers (module docstring says why). DDP's
    constructor broadcasts rank 0's parameters to every rank, so even
    differently-initialized replicas start in lockstep.
    """
    raise NotImplementedError


def unwrap_ddp(model: nn.Module) -> nn.Module:
    """The raw module inside a DDP wrapper; a plain module passes through.

    Checkpoints must always save the unwrapped model — DDP.state_dict()
    prefixes keys with "module.", unreadable by a single-process resume.
    """
    raise NotImplementedError


def maybe_no_sync(model: nn.Module, is_last: bool) -> ContextManager[Any]:
    """The context for ONE micro-batch's forward+backward.

    * DDP-wrapped model and NOT the last micro-batch: `model.no_sync()` —
      gradients accumulate locally, no all-reduce.
    * Anything else (plain module, or the last micro-batch): nullcontext().

    Why: DDP all-reduces during EVERY backward. With accum_steps micro
    batches that is accum_steps all-reduces per optimizer step, all but one
    wasted — the result of averaging each micro-grad equals averaging once
    at the end (the all-reduce is linear). no_sync suppresses the hooks;
    the first synced backward after the context reduces the ACCUMULATED
    gradient. Bandwidth drops by accum_steps with identical math.

    THE EDIT IN train_step.py: wrap each micro-batch's forward+backward in
    `maybe_no_sync(model, is_last=...)` where is_last is true only for the
    final micro-batch. (Import it from optbench.ddp — train_step gains its
    first awareness of DDP, and exactly this much.) The TestNoSyncWiring
    tests fail until the edit lands.
    """
    raise NotImplementedError


def all_reduce_mean(value: float) -> float:
    """Average a python float across ranks; identity when not distributed.

    * No process group initialized (or world size 1): return float(value).
    * Else: put the value in a float64 tensor, all_reduce(SUM), divide by
      world size, return the float. (float64 so that averaging many ranks'
      float32-ish losses does not lose digits in the sum.)

    For logging only — never on the training path (it is a sync point).
    """
    raise NotImplementedError
