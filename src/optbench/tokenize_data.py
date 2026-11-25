"""FineWeb-Edu -> shard_*.bin: the one-time pass that feeds every run.

TASK 10 — implement `pack_documents` (the one `raise NotImplementedError`).
Run `uv run pytest tests/test_tokenize_data.py -v` until all tests pass.

The glue around it (HF streaming iterator, tiktoken encoding, the CLI) is
boilerplate — third-party integration, already written below. The core is
yours because it IS the data contract of the whole study:

  * **Packing, not padding.** Documents are concatenated into one endless
    token stream with an EOS token APPENDED after every document; fixed-size
    shards are cut from that stream wherever the boundary falls, so a
    document may span two shards. No padding tokens, no dropped documents —
    every FLOP in the budget lands on a real token, and the EOS is what lets
    the model learn where documents end.

  * **The filename IS the data order.** train.py does
    sorted(glob("shard_*.bin")), and step_window_indices makes the batch a
    pure function of (seed, step). That determinism is only as good as the
    sort: shard_999 < shard_1000 is FALSE as strings, so names are
    zero-padded to 5 digits (shard_00000.bin ...) and a resumed or re-run
    job sees the identical stream.

  * **Loud, not clobbering.** An out_dir that already holds shard_*.bin
    files is a ValueError naming the directory. Two tokenization runs mixed
    into one directory would train on silently duplicated data — the same
    "half-configured is a lie" rule as everywhere else.

pack_documents(docs, *, out_dir, vocab_size, eos_id,
               shard_tokens=1 << 24, max_tokens=None) -> dict

Steps:
  1. Validate: shard_tokens < 1 is a ValueError; eos_id outside
     [0, vocab_size) is a ValueError. mkdir out_dir (parents, exist_ok),
     then raise ValueError naming out_dir if it already contains any
     shard_*.bin.
  2. Keep the buffer as a LIST of numpy arrays plus a running token count —
     never one giant Python list of ints (billions of boxed ints is the
     classic way this script eats all the RAM). Concatenate only at write
     time. For each doc: append np.asarray(doc) and a 1-element EOS array;
     an empty doc still contributes its EOS.
  3. WHILE the buffered count >= shard_tokens (while, not if — one huge
     document can fill several shards): concatenate, write exactly the
     first shard_tokens tokens as shard_{index:05d}.bin via
     optbench.shards.write_shard (atomic tmp+rename, validates ids against
     vocab_size), keep the remainder as the new buffer. Indices count up
     from 0.
  4. Budget: after each doc lands in the buffer, if max_tokens is not None
     and the running TOTAL (written + buffered, EOS included) is
     >= max_tokens, stop pulling docs. The stream is lazy — stopping is
     what keeps a 2B-token job from downloading all 10B.
  5. After the doc loop: a total of 0 tokens (empty docs iterable) is a
     ValueError — a dataset of nothing must die here, not as an empty glob
     at train time. Otherwise flush whatever is buffered as one final,
     smaller shard (TokenWindows gives a short shard (n-1)//seq_len
     windows, so a tiny tail is harmless, never corrupting).
  6. Return {"docs": ..., "tokens": ..., "shards": ...} — the numbers the
     run log and the grant arithmetic both want.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from optbench.shards import write_shard

# GPT-2 BPE via tiktoken: 50257 real ids, <|endoftext|> = 50256. The MODEL
# vocab in the run TOMLs will be padded to 50304 (a multiple of 128) for
# matmul efficiency; the shards only ever hold ids < 50257, and write_shard's
# dtype/bounds come from the real vocab here.
GPT2_VOCAB = 50257
GPT2_EOS = 50256


def pack_documents(
    docs: Iterable[Sequence[int] | np.ndarray],
    *,
    out_dir: str | Path,
    vocab_size: int,
    eos_id: int,
    shard_tokens: int = 1 << 24,
    max_tokens: int | None = None,
) -> dict[str, int]:
    """Pack token-id documents into fixed-size shards. See module docstring."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Boilerplate from here down: HF streaming + tiktoken + CLI. Not the exercise.
# ----------------------------------------------------------------------------


def iter_fineweb_docs(
    dataset: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    split: str = "train",
) -> Iterator[str]:
    """Stream document texts from the hub; nothing is downloaded up front."""
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "datasets is not installed; install with `uv sync --extra data`"
        ) from e
    ds = datasets.load_dataset(dataset, name=subset, split=split, streaming=True)
    for example in ds:
        yield example["text"]


def encode_docs(texts: Iterable[str], batch: int = 1024) -> Iterator[np.ndarray]:
    """GPT-2 BPE encode, in batches (tiktoken's batch API releases the GIL)."""
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "tiktoken is not installed; install with `uv sync --extra data`"
        ) from e
    enc = tiktoken.get_encoding("gpt2")
    pending: list[str] = []

    def drain() -> Iterator[np.ndarray]:
        for ids in enc.encode_ordinary_batch(pending):
            yield np.asarray(ids, dtype=np.uint32)
        pending.clear()

    for text in texts:
        pending.append(text)
        if len(pending) == batch:
            yield from drain()
    if pending:
        yield from drain()


def _progress(docs: Iterator[np.ndarray], every: int) -> Iterator[np.ndarray]:
    n = tokens = 0
    for arr in docs:
        n += 1
        tokens += arr.size + 1  # + its EOS
        if n % every == 0:
            print(f"  {n:>10,d} docs  {tokens:>14,d} tokens", flush=True)
        yield arr


def main(argv: list[str] | None = None) -> dict[str, int]:
    p = argparse.ArgumentParser(description="tokenize FineWeb-Edu into optbench shards")
    p.add_argument("--out", required=True, help="output directory for shard_*.bin")
    p.add_argument(
        "--max-tokens",
        type=lambda s: int(float(s)),  # accepts 2e9
        default=None,
        help="stop after roughly this many tokens (default: the whole subset)",
    )
    p.add_argument("--shard-tokens", type=lambda s: int(float(s)), default=1 << 24)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--progress-every", type=int, default=20_000)
    args = p.parse_args(argv)

    docs = encode_docs(iter_fineweb_docs(args.dataset, args.subset))
    summary = pack_documents(
        _progress(docs, args.progress_every),
        out_dir=args.out,
        vocab_size=GPT2_VOCAB,
        eos_id=GPT2_EOS,
        shard_tokens=args.shard_tokens,
        max_tokens=args.max_tokens,
    )
    print(summary)
    return summary


if __name__ == "__main__":
    main()
