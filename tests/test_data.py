"""Tests for optbench.shards and optbench.data.

These test behaviour and properties, not one specific implementation.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from optbench.data import TokenWindows, get_batch, step_window_indices
from optbench.shards import read_shard, write_shard

V = 50_000
T = 16
SHARD_LENS = [1000, 517, 10, 2049]  # the 10-token shard is too short for a window


@pytest.fixture
def shard_data(tmp_path):
    """Shards whose token values encode their global position (mod V), so any
    slice can be checked against the source."""
    paths, arrays, pos = [], [], 0
    for i, n in enumerate(SHARD_LENS):
        arr = (np.arange(pos, pos + n) * 7 + 3) % V
        p = tmp_path / f"shard_{i:03d}.bin"
        write_shard(p, arr, vocab_size=V)
        paths.append(p)
        arrays.append(arr)
        pos += n
    return paths, arrays


def expected_windows(arrays, seq_len):
    out = []
    for arr in arrays:
        for j in range((len(arr) - 1) // seq_len):
            s = j * seq_len
            out.append((arr[s : s + seq_len], arr[s + 1 : s + seq_len + 1]))
    return out


# ---------------------------------------------------------------------------
# Shard format (boilerplate, sanity only)
# ---------------------------------------------------------------------------


def test_shard_roundtrip_and_dtype(tmp_path):
    p = tmp_path / "a.bin"
    write_shard(p, np.array([0, 1, 65535]), vocab_size=65536)
    m = read_shard(p)
    assert m.dtype == np.uint16 and list(m) == [0, 1, 65535]
    write_shard(p, np.array([70_000]), vocab_size=128_256)
    m = read_shard(p)
    assert m.dtype == np.uint32 and list(m) == [70_000]


def test_shard_rejects_out_of_vocab_and_bad_files(tmp_path):
    with pytest.raises(ValueError):
        write_shard(tmp_path / "a.bin", np.array([5, 50_000]), vocab_size=50_000)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        read_shard(bad)


# ---------------------------------------------------------------------------
# TokenWindows
# ---------------------------------------------------------------------------


def test_window_count(shard_data):
    paths, arrays = shard_data
    tw = TokenWindows(paths, T)
    assert tw.n_windows == sum((n - 1) // T for n in SHARD_LENS)
    assert isinstance(tw.n_windows, int)


def test_every_window_is_the_exact_slice(shard_data):
    paths, arrays = shard_data
    tw = TokenWindows(paths, T)
    exp = expected_windows(arrays, T)
    assert len(exp) == tw.n_windows
    for k, (ex, ey) in enumerate(exp):
        x, y = tw.window(k)
        assert x.dtype == np.int64 and y.dtype == np.int64
        assert x.shape == y.shape == (T,)
        np.testing.assert_array_equal(x, ex)
        np.testing.assert_array_equal(y, ey)


def test_y_is_x_shifted_by_one(shard_data):
    paths, _ = shard_data
    tw = TokenWindows(paths, T)
    for k in range(tw.n_windows):
        x, y = tw.window(k)
        np.testing.assert_array_equal(x[1:], y[:-1])


def test_every_target_appears_once_per_shard(shard_data):
    # Overlap of one token: within a shard, targets are tokens 1 .. n_w*T, each once.
    paths, arrays = shard_data
    tw = TokenWindows([paths[0]], T)
    ys = np.concatenate([tw.window(k)[1] for k in range(tw.n_windows)])
    np.testing.assert_array_equal(ys, arrays[0][1 : 1 + tw.n_windows * T])


@pytest.mark.parametrize("n_tokens,expected", [(32, 1), (33, 2), (17, 1), (16, 0)])
def test_window_needs_seq_len_plus_one_tokens(tmp_path, n_tokens, expected):
    # Boundary case: a window spans T + 1 tokens, so a shard of exactly k*T
    # tokens holds only k - 1 full windows.
    p = tmp_path / "edge.bin"
    write_shard(p, np.arange(n_tokens), vocab_size=V)
    tw = TokenWindows([p], T)
    assert tw.n_windows == expected
    for k in range(tw.n_windows):
        x, y = tw.window(k)
        assert x.shape == y.shape == (T,)


@pytest.mark.parametrize("k", [-1, "end"])
def test_window_index_out_of_range(shard_data, k):
    paths, _ = shard_data
    tw = TokenWindows(paths, T)
    with pytest.raises(IndexError):
        tw.window(tw.n_windows if k == "end" else k)


# ---------------------------------------------------------------------------
# step_window_indices
# ---------------------------------------------------------------------------


def test_indices_shape_dtype_range():
    idx = step_window_indices(3, global_batch=8, n_windows=50, seed=0)
    assert idx.shape == (8,) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < 50


def test_indices_are_deterministic_and_seed_dependent():
    a = step_window_indices(5, 8, 1000, seed=1)
    b = step_window_indices(5, 8, 1000, seed=1)
    c = step_window_indices(5, 8, 1000, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_one_epoch_is_a_permutation():
    n, gb = 60, 6
    epoch0 = np.concatenate([step_window_indices(s, gb, n, seed=0) for s in range(10)])
    np.testing.assert_array_equal(np.sort(epoch0), np.arange(n))
    assert not np.array_equal(epoch0, np.arange(n)), "order must be shuffled"
    epoch1 = np.concatenate([step_window_indices(s, gb, n, seed=0) for s in range(10, 20)])
    np.testing.assert_array_equal(np.sort(epoch1), np.arange(n))
    assert not np.array_equal(epoch0, epoch1), "each epoch needs its own permutation"


def test_steps_can_straddle_epochs():
    n, gb = 50, 8  # 50 is not a multiple of 8
    stream = np.concatenate([step_window_indices(s, gb, n, seed=3) for s in range(13)])  # 104 positions
    np.testing.assert_array_equal(np.sort(stream[:50]), np.arange(n))
    np.testing.assert_array_equal(np.sort(stream[50:100]), np.arange(n))
    assert len(set(stream[100:].tolist())) == 4


def test_indices_do_not_depend_on_call_history():
    fresh = step_window_indices(7, 8, 50, seed=4)
    for s in range(20):
        step_window_indices(s, 8, 50, seed=4)
    step_window_indices(3, 8, 77, seed=9)  # a different dataset/seed in between
    np.testing.assert_array_equal(step_window_indices(7, 8, 50, seed=4), fresh)


def test_epoch_permutation_is_reused():
    # 2M windows: an uncached permutation per call costs ~tens of ms each.
    n = 2_000_000
    step_window_indices(0, 256, n, seed=0)  # warm up epoch 0
    t0 = time.perf_counter()
    for s in range(1, 201):
        step_window_indices(s, 256, n, seed=0)
    assert time.perf_counter() - t0 < 1.0, "recomputing the permutation every step?"


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------


def test_get_batch_shapes_dtype_and_content(shard_data):
    paths, _ = shard_data
    tw = TokenWindows(paths, T)
    x, y = get_batch(tw, step=2, global_batch=8, seed=0)
    assert x.shape == y.shape == (8, T)
    assert x.dtype == y.dtype == torch.int64
    idx = step_window_indices(2, 8, tw.n_windows, seed=0)
    for row, k in enumerate(idx):
        ex, ey = tw.window(int(k))
        np.testing.assert_array_equal(x[row].numpy(), ex)
        np.testing.assert_array_equal(y[row].numpy(), ey)


@pytest.mark.parametrize("world_size,accum_steps", [(2, 1), (1, 4), (2, 2), (4, 2), (8, 2)])
def test_global_batch_is_invariant_to_the_split(shard_data, world_size, accum_steps):
    paths, _ = shard_data
    tw = TokenWindows(paths, T)
    gb, step = 16, 5
    ref_x, ref_y = get_batch(tw, step, gb, seed=0)
    parts_x, parts_y = [], []
    for r in range(world_size):
        for m in range(accum_steps):
            x, y = get_batch(tw, step, gb, seed=0, rank=r, world_size=world_size, micro_step=m, accum_steps=accum_steps)
            assert x.shape == (gb // (world_size * accum_steps), T)
            parts_x.append(x)
            parts_y.append(y)
    torch.testing.assert_close(torch.cat(parts_x), ref_x)
    torch.testing.assert_close(torch.cat(parts_y), ref_y)


def test_resume_gives_the_same_batch(shard_data):
    paths, _ = shard_data
    a = get_batch(TokenWindows(paths, T), step=9, global_batch=8, seed=1)
    b = get_batch(TokenWindows(paths, T), step=9, global_batch=8, seed=1)  # "fresh process"
    torch.testing.assert_close(a, b)
    c = get_batch(TokenWindows(paths, T), step=10, global_batch=8, seed=1)
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize(
    "kw",
    [
        dict(world_size=3),  # 8 not divisible by 3
        dict(world_size=2, accum_steps=3),
        dict(world_size=2, rank=2),
        dict(rank=-1),
        dict(accum_steps=2, micro_step=2),
    ],
)
def test_get_batch_validates_split(shard_data, kw):
    paths, _ = shard_data
    tw = TokenWindows(paths, T)
    with pytest.raises(ValueError):
        get_batch(tw, step=0, global_batch=8, seed=0, **kw)
