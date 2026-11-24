"""Tests for the train.py composition root — TASK 9.

The two-rank test launches main() under a simulated torchrun environment
(env vars + env:// rendezvous), so this file exercises the real entry path
end to end.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from optbench.checkpoint import load_checkpoint
from optbench.config import load_config
from optbench.data import TokenWindows
from optbench.loop import run_loop
from optbench.model import Transformer
from optbench.optimizers import build_optimizers
from optbench.shards import write_shard
from optbench.throughput import flops_per_token
from optbench.train import main

SEED = 13
V = 64
T = 16


def write_run(tmp_path, *, extra_toml="", loop_extra="", total_steps=4) -> Path:
    """A complete tiny run directory: shards + TOML. Returns the TOML path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    arr = (np.arange(1500) * 7 + 3) % V
    write_shard(data_dir / "shard_000.bin", arr, vocab_size=V)

    toml = f"""
[run]
data_dir = "{data_dir}"
seq_len = {T}

[model]
vocab_size = {V}
d_model = 16
n_layers = 2
n_heads = 4
n_kv_heads = 2
d_ff = 32
max_seq_len = {T}

[loop]
global_batch = 4
seed = {SEED}
checkpoint_dir = "{tmp_path / 'ckpts'}"
log_every = 1
{loop_extra}

[schedule]
kind = "cosine"
total_steps = {total_steps}
warmup_steps = 2

[optimizer]
arm = "adamw"
""" + extra_toml
    p = tmp_path / "run.toml"
    p.write_text(toml)
    return p


def gold_run(cfg_path):
    """The same run built by hand, straight against the library."""
    cfg = load_config(cfg_path)
    torch.manual_seed(cfg.loop.seed)
    model = Transformer(cfg.model)
    opts = build_optimizers(model, cfg.groups, cfg.opt)
    shards = sorted(Path(cfg.data_dir).glob("shard_*.bin"))
    windows = TokenWindows(shards, cfg.seq_len)
    import dataclasses

    loop_cfg = dataclasses.replace(cfg.loop, checkpoint_dir=str(cfg.loop.checkpoint_dir) + "_gold")
    run_loop(model, windows, opts, loop_cfg, cfg.sched, cfg.step)
    return model


class TestSingleProcess:
    def test_end_to_end(self, tmp_path):
        cfg_path = write_run(tmp_path)
        out = main(["--config", str(cfg_path)])
        assert out == {"final_step": 4, "resumed_from": 0, "steps_run": 4}

        assert (tmp_path / "ckpts" / "step_4.pt").exists()
        rows = [
            json.loads(line)
            for line in (tmp_path / "ckpts" / "log.jsonl").read_text().splitlines()
        ]
        assert [r["step"] for r in rows] == [0, 1, 2, 3]
        for r in rows:
            assert {"loss", "grad_norm", "lr_mult", "tokens_per_sec", "skipped"} <= set(r)

    def test_matches_a_hand_built_run(self, tmp_path):
        cfg_path = write_run(tmp_path)
        main(["--config", str(cfg_path)])
        gold = gold_run(cfg_path)

        cfg = load_config(cfg_path)
        torch.manual_seed(cfg.loop.seed)
        fresh = Transformer(cfg.model)
        opts = build_optimizers(fresh, cfg.groups, cfg.opt)
        load_checkpoint(tmp_path / "ckpts" / "step_4.pt", model=fresh, optimizers=opts)
        for p1, p2 in zip(gold.parameters(), fresh.parameters()):
            assert torch.equal(p1, p2)

    def test_completed_run_resumes_as_noop(self, tmp_path):
        cfg_path = write_run(tmp_path)
        main(["--config", str(cfg_path)])
        out = main(["--config", str(cfg_path)])
        assert out == {"final_step": 4, "resumed_from": 4, "steps_run": 0}

    def test_mfu_flows_from_config_to_rows(self, tmp_path):
        cfg_path = write_run(tmp_path, loop_extra="peak_flops_per_gpu = 1.0e12")
        main(["--config", str(cfg_path)])
        rows = [
            json.loads(line)
            for line in (tmp_path / "ckpts" / "log.jsonl").read_text().splitlines()
        ]
        cfg = load_config(cfg_path)
        fpt = flops_per_token(cfg.model, cfg.seq_len)
        for r in rows:
            assert "mfu" in r
            if r["tokens_per_sec"] is not None:
                assert r["mfu"] == pytest.approx(r["tokens_per_sec"] * fpt / 1.0e12)
        assert any(r["tokens_per_sec"] is not None for r in rows)

    def test_log_file_none_disables_logging(self, tmp_path):
        cfg_path = write_run(tmp_path)
        out = main(["--config", str(cfg_path), "--log-file", "none"])
        assert out["final_step"] == 4
        assert not (tmp_path / "ckpts" / "log.jsonl").exists()

    def test_no_shards_is_loud(self, tmp_path):
        cfg_path = write_run(tmp_path)
        for shard in (tmp_path / "data").glob("shard_*.bin"):
            shard.unlink()
        with pytest.raises(ValueError):
            main(["--config", str(cfg_path)])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_main(rank, world, port, cfg_path):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    out = main(["--config", cfg_path])
    assert out == {"final_step": 4, "resumed_from": 0, "steps_run": 4}


class TestTwoRankTorchrunEnv:
    def test_two_rank_main_matches_single_process(self, tmp_path):
        cfg_path = write_run(tmp_path)
        mp.spawn(
            _worker_main,
            args=(2, _free_port(), str(cfg_path)),
            nprocs=2,
            join=True,
        )

        # one writer: rank 0's cadence files only
        assert sorted(p.name for p in (tmp_path / "ckpts").glob("step_*.pt")) == [
            "step_4.pt"
        ]

        gold = gold_run(cfg_path)
        cfg = load_config(cfg_path)
        torch.manual_seed(cfg.loop.seed)
        fresh = Transformer(cfg.model)
        opts = build_optimizers(fresh, cfg.groups, cfg.opt)
        load_checkpoint(tmp_path / "ckpts" / "step_4.pt", model=fresh, optimizers=opts)
        for p1, p2 in zip(gold.parameters(), fresh.parameters()):
            assert torch.allclose(p1, p2, atol=1e-5)
