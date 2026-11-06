"""Deterministic, resumable token data loader.

TASK 2b — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_data.py -v` until all tests pass.

Shards are read with `optbench.shards.read_shard` (already written).

Conventions (the tests rely on these):
  * Windows stay inside a shard. A shard with n tokens holds
        (n - 1) // seq_len
    windows; window j of that shard starts at token j * seq_len and spans
    seq_len + 1 tokens (consecutive windows overlap by one token, so every
    token except the shard's first is a target exactly once). Leftover tail
    tokens are dropped.
  * Global window indices run shard by shard in the order of `paths`:
    shard 0's windows are 0 .. c0-1, shard 1's are c0 .. c0+c1-1, and so on.
  * The data for a step is a pure function of (seed, step, global_batch):
    it must not depend on world size, micro-batch size, or call history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from optbench.shards import read_shard  # noqa: F401  (you will need it)


class TokenWindows:
    """Fixed-length training windows over a list of token shards.

    Args:
        paths:   shard files, in order
        seq_len: T, the number of input tokens per window

    Attributes / methods:
        n_windows (int): total number of windows across all shards
        window(k) -> (x, y): two np.int64 arrays of shape (seq_len,), where
            x = tokens[start : start + T] and y = tokens[start + 1 : start + T + 1]
            of the shard that owns global window k. Raise IndexError unless
            0 <= k < n_windows.
    """

    def __init__(self, paths: list[str | Path], seq_len: int) -> None:
        self.paths = paths
        self.seq_len = seq_len
        self.shards = [read_shard(path) for path in self.paths]
        self.ends = np.cumsum([(len(shard)-1) // self.seq_len for shard in self.shards])


    @property
    def n_windows(self) -> int:
        return int(self.ends[-1])

    def window(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= k < self.n_windows:
            raise IndexError(k)
        i = int(np.searchsorted(self.ends, k, side="right"))   # which shard owns k
        before = 0 if i == 0 else self.ends[i - 1]
        local = k - before
        start = local * self.seq_len
        w = np.asarray(self.shards[i][start: start + self.seq_len + 1], dtype=np.int64)
        return w[:-1], w[1:]


def step_window_indices(step: int, global_batch: int, n_windows: int, seed: int) -> np.ndarray:
    """Global window indices (np.int64, shape (global_batch,)) used at `step`.

    Think of an endless stream of sample positions p = 0, 1, 2, ...; step s
    uses positions s * global_batch ... (s + 1) * global_batch - 1. Position p
    belongs to epoch p // n_windows, and maps to
        perm_epoch[p % n_windows]
    where perm_epoch is a random permutation of range(n_windows) drawn from a
    LOCAL generator seeded by (seed, epoch). A step may straddle two epochs.

    Each epoch's permutation must be computed once and reused (calling this
    for consecutive steps must not redo an O(n_windows) shuffle every time).
    """
    


def get_batch(
    windows: TokenWindows,
    step: int,
    global_batch: int,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    micro_step: int = 0,
    accum_steps: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (x, y) micro-batch for this rank and micro-step.

    The step's global batch (from step_window_indices) is split rank-major:
    rank r owns the contiguous block of global_batch // world_size windows
    starting at r * (global_batch // world_size); within it, micro-step m owns
    the contiguous block of micro = global_batch // (world_size * accum_steps)
    windows starting at m * micro.

    Returns two torch.int64 tensors of shape (micro, seq_len) on `device`.
    On CUDA, use pinned memory and a non-blocking copy.

    Raise ValueError if global_batch is not divisible by world_size * accum_steps,
    or if rank / micro_step are out of range.
    """
    raise NotImplementedError
