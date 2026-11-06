"""Token shard file format (boilerplate, not part of the exercise).

A shard is a flat array of token ids with a small fixed header:

    offset 0   int64  MAGIC
    offset 8   int64  VERSION
    offset 16  int64  itemsize of the token dtype (2 = uint16, 4 = uint32)
    offset 24  int64  number of tokens
    offset 32  tokens...

The dtype is chosen from the vocab size when writing and read back from the
header, so a reader can never interpret a file with the wrong dtype.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAGIC = 0x4F50544245_4E4348  # "OPTBENCH"-ish
VERSION = 1
HEADER_BYTES = 32
_DTYPES = {2: np.uint16, 4: np.uint32}


def dtype_for_vocab(vocab_size: int) -> type[np.unsignedinteger]:
    if vocab_size <= 2**16:
        return np.uint16
    if vocab_size <= 2**32:
        return np.uint32
    raise ValueError(f"vocab_size {vocab_size} too large")


def write_shard(path: str | Path, tokens: np.ndarray, vocab_size: int) -> None:
    """Write `tokens` (any integer array) to `path`. Atomic: tmp file + rename."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("tokens must be 1-D")
    if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab_size):
        raise ValueError(f"token ids must be in [0, {vocab_size})")
    dtype = dtype_for_vocab(vocab_size)
    header = np.array([MAGIC, VERSION, np.dtype(dtype).itemsize, tokens.size], dtype=np.int64)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(dtype, copy=False).tobytes())
    tmp.replace(path)


def read_shard(path: str | Path) -> np.memmap:
    """Memory-map a shard written by `write_shard`. Returns a read-only 1-D array."""
    header = np.fromfile(path, dtype=np.int64, count=4)
    if header.size != 4 or header[0] != MAGIC:
        raise ValueError(f"{path}: not an optbench shard (bad magic)")
    if header[1] != VERSION:
        raise ValueError(f"{path}: unsupported shard version {header[1]}")
    itemsize, n_tokens = int(header[2]), int(header[3])
    if itemsize not in _DTYPES:
        raise ValueError(f"{path}: unsupported itemsize {itemsize}")
    return np.memmap(path, dtype=_DTYPES[itemsize], mode="r", offset=HEADER_BYTES, shape=(n_tokens,))
