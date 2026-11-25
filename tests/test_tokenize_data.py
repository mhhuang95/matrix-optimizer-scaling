"""Tests for pack_documents and the tokenize CLI — TASK 10.

The core tests feed synthetic token docs; the CLI test fakes tiktoken and the
HF iterator, so nothing here touches the network or needs the data extras.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from optbench.shards import read_shard
from optbench.tokenize_data import GPT2_EOS, GPT2_VOCAB, main, pack_documents

V = 100
EOS = 99


def stream(out_dir) -> np.ndarray:
    """All written tokens, concatenated in sorted-filename order."""
    paths = sorted(out_dir.glob("shard_*.bin"))
    return np.concatenate([np.asarray(read_shard(p)) for p in paths])


def expected(docs) -> np.ndarray:
    parts = []
    for d in docs:
        parts.append(np.asarray(d))
        parts.append(np.array([EOS]))
    return np.concatenate(parts).astype(np.int64)


class TestPackDocuments:
    def test_stream_is_docs_with_eos_appended(self, tmp_path):
        docs = [[1, 2, 3], [], [4, 5]]  # empty doc still contributes its EOS
        out = pack_documents(docs, out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=4)
        got = stream(tmp_path)
        assert np.array_equal(got.astype(np.int64), expected(docs))
        assert out == {"docs": 3, "tokens": 8, "shards": 2}

    def test_all_shards_full_except_last(self, tmp_path):
        docs = [[1, 2]] * 5  # 3 tokens per doc with EOS -> 15 total
        pack_documents(docs, out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=4)
        sizes = [read_shard(p).size for p in sorted(tmp_path.glob("shard_*.bin"))]
        assert sizes == [4, 4, 4, 3]

    def test_doc_spans_shard_boundary(self, tmp_path):
        docs = [list(range(10))]  # 11 tokens with EOS, shard_tokens 6
        pack_documents(docs, out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=6)
        first = np.asarray(read_shard(tmp_path / "shard_00000.bin"))
        assert np.array_equal(first.astype(np.int64), np.arange(6))  # split mid-doc
        assert np.array_equal(stream(tmp_path).astype(np.int64), expected(docs))

    def test_one_huge_doc_fills_several_shards(self, tmp_path):
        docs = [list(range(50)) + [0] * 49]  # 100 tokens with EOS
        out = pack_documents(docs, out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=8)
        assert out["shards"] == 13  # 12 full + tail of 4
        assert np.array_equal(stream(tmp_path).astype(np.int64), expected(docs))

    def test_filenames_sort_as_data_order(self, tmp_path):
        docs = [[7]] * 12  # 24 tokens -> 12 shards of 2
        pack_documents(docs, out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=2)
        names = [p.name for p in sorted(tmp_path.glob("shard_*.bin"))]
        assert names[0] == "shard_00000.bin"
        assert names == [f"shard_{i:05d}.bin" for i in range(12)]  # zero-padded

    def test_max_tokens_stops_the_stream(self, tmp_path):
        pulled = []

        def docs():
            for i in range(1000):
                pulled.append(i)
                yield [1, 2, 3, 4, 5]  # 6 tokens per doc with EOS

        out = pack_documents(
            docs(), out_dir=tmp_path, vocab_size=V, eos_id=EOS, shard_tokens=4, max_tokens=14
        )
        # doc 3 pushes the total to 18 >= 14; doc 4 is never pulled
        assert len(pulled) == 3
        assert out["docs"] == 3 and out["tokens"] == 18

    def test_existing_shards_are_loud(self, tmp_path):
        pack_documents([[1]], out_dir=tmp_path, vocab_size=V, eos_id=EOS)
        with pytest.raises(ValueError, match=str(tmp_path)):
            pack_documents([[2]], out_dir=tmp_path, vocab_size=V, eos_id=EOS)

    def test_empty_docs_iterable_is_loud(self, tmp_path):
        with pytest.raises(ValueError):
            pack_documents([], out_dir=tmp_path, vocab_size=V, eos_id=EOS)

    def test_bad_eos_and_bad_shard_tokens_are_loud(self, tmp_path):
        with pytest.raises(ValueError):
            pack_documents([[1]], out_dir=tmp_path / "a", vocab_size=V, eos_id=V)
        with pytest.raises(ValueError):
            pack_documents([[1]], out_dir=tmp_path / "b", vocab_size=V, eos_id=EOS, shard_tokens=0)

    def test_out_dir_is_created(self, tmp_path):
        out = tmp_path / "deep" / "data"
        pack_documents([[1, 2]], out_dir=out, vocab_size=V, eos_id=EOS)
        assert (out / "shard_00000.bin").exists()


class TestCli:
    def test_main_end_to_end_with_fake_tokenizer(self, tmp_path, monkeypatch):
        import optbench.tokenize_data as td

        # fake HF stream: two documents
        monkeypatch.setattr(td, "iter_fineweb_docs", lambda *a, **k: iter(["ab", "c"]))
        # fake tiktoken: one id per character
        enc = types.SimpleNamespace(
            encode_ordinary_batch=lambda texts: [[ord(ch) % 100 for ch in t] for t in texts]
        )
        monkeypatch.setitem(
            sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=lambda name: enc)
        )

        out_dir = tmp_path / "data"
        summary = main(["--out", str(out_dir), "--shard-tokens", "3"])
        assert summary == {"docs": 2, "tokens": 5, "shards": 2}
        got = stream(out_dir).astype(np.int64)
        want = [ord("a") % 100, ord("b") % 100, GPT2_EOS, ord("c") % 100, GPT2_EOS]
        assert got.tolist() == want
        # GPT-2 ids fit uint16: the real-vocab dtype rule
        assert read_shard(out_dir / "shard_00000.bin").dtype == np.uint16
        assert GPT2_VOCAB <= 2**16
