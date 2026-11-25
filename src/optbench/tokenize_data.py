"""FineWeb-Edu -> shard_*.bin: the one-time pass that feeds every run.

    python -m optbench.tokenize_data --out data/fineweb_edu --max-tokens 2e9

pack_documents is the data contract of the whole study:

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

  * **Lazy budget.** max_tokens stops PULLING documents once
     written + buffered crosses it — the HF stream is lazy, so this is what
     keeps a 2B-token job from downloading all 10B.
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
    if shard_tokens < 1:
        raise ValueError(f"shard_tokens {shard_tokens} must be >= 1!")
    if not 0 <= eos_id < vocab_size:
        raise ValueError(f"eos_id {eos_id} must be in the range [0, vocab_size).")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("shard_*.bin")):
        raise ValueError(f"{out} already contains shard_*.bin files; refusing to mix datasets")

    eos = np.array([eos_id], dtype=np.uint32)
    buf: list[np.ndarray] = []  # list of arrays, concatenated only at write time
    buffered = total = n_docs = n_shards = 0

    for doc in docs:
        arr = np.asarray(doc)
        buf.append(arr)
        buf.append(eos)
        buffered += arr.size + 1
        total += arr.size + 1
        n_docs += 1
        while buffered >= shard_tokens:  # while: one huge doc can fill several shards
            flat = np.concatenate(buf)
            write_shard(out / f"shard_{n_shards:05d}.bin", flat[:shard_tokens], vocab_size)
            n_shards += 1
            rest = flat[shard_tokens:]
            buf = [rest] if rest.size else []
            buffered = int(rest.size)
        if max_tokens is not None and total >= max_tokens:
            break  # the stream is lazy: stop pulling, don't keep downloading

    if total == 0:
        raise ValueError("no tokens to pack: the docs iterable was empty")
    if buffered:
        write_shard(out / f"shard_{n_shards:05d}.bin", np.concatenate(buf), vocab_size)
        n_shards += 1
    return {"docs": n_docs, "tokens": total, "shards": n_shards}


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
