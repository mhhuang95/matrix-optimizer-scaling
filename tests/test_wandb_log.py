"""Tests for the W&B adapter — all boilerplate, like the module.

wandb itself is never imported: a fake module is injected into sys.modules,
so the suite stays hermetic and runs on machines without wandb installed.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from optbench.wandb_log import flatten_row, tee, wandb_logger


class FakeRun:
    def __init__(self, **kw):
        self.init_kwargs = kw
        self.logged: list[tuple[int, dict]] = []
        self.finished = False

    def log(self, data, step=None):
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


def install_fake_wandb(monkeypatch) -> list[FakeRun]:
    runs: list[FakeRun] = []

    def init(**kw):
        run = FakeRun(**kw)
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    return runs


class TestFlattenRow:
    def test_nones_are_dropped_not_logged(self):
        row = {"step": 0, "loss": 2.5, "tokens_per_sec": None, "mfu": None}
        assert flatten_row(row) == {"step": 0, "loss": 2.5}

    def test_spectral_flattens_to_slash_keys(self):
        row = {
            "step": 3,
            "loss": 1.0,
            "spectral": {
                "blocks.0.attn.wq": {"top_sv": 2.0, "stable_rank": 5.5},
                "blocks.1.mlp.w1": {"top_sv": 1.5},
            },
        }
        assert flatten_row(row) == {
            "step": 3,
            "loss": 1.0,
            "spectral/blocks.0.attn.wq/top_sv": 2.0,
            "spectral/blocks.0.attn.wq/stable_rank": 5.5,
            "spectral/blocks.1.mlp.w1/top_sv": 1.5,
        }


class TestTee:
    def test_every_sink_sees_every_row(self):
        a, b = [], []
        log = tee(a.append, b.append)
        log({"step": 0})
        log({"step": 1})
        assert [r["step"] for r in a] == [0, 1]
        assert [r["step"] for r in b] == [0, 1]

    def test_sinks_get_independent_copies(self):
        a, b = [], []
        tee(lambda r: (r.pop("loss"), a.append(r)), b.append)({"step": 0, "loss": 2.0})
        assert b == [{"step": 0, "loss": 2.0}]  # first sink's mutation invisible


class TestWandbLogger:
    def test_missing_wandb_is_loud(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "wandb", None)  # import raises
        with pytest.raises(ImportError, match="--wandb"):
            wandb_logger(project="p", checkpoint_dir=str(tmp_path))

    def test_rows_flow_flattened_with_step(self, monkeypatch, tmp_path):
        runs = install_fake_wandb(monkeypatch)
        log = wandb_logger(project="p", checkpoint_dir=str(tmp_path))
        log({"step": 0, "loss": 2.0, "tokens_per_sec": None})
        log({"step": 1, "loss": 1.5, "tokens_per_sec": 100.0})
        (run,) = runs
        assert run.logged == [
            (0, {"step": 0, "loss": 2.0}),
            (1, {"step": 1, "loss": 1.5, "tokens_per_sec": 100.0}),
        ]

    def test_run_id_is_stable_per_checkpoint_dir(self, monkeypatch, tmp_path):
        runs = install_fake_wandb(monkeypatch)
        wandb_logger(project="p", checkpoint_dir=str(tmp_path / "a"))
        wandb_logger(project="p", checkpoint_dir=str(tmp_path / "a"))
        wandb_logger(project="p", checkpoint_dir=str(tmp_path / "b"))
        ids = [r.init_kwargs["id"] for r in runs]
        assert ids[0] == ids[1] != ids[2]  # relaunch resumes, new dir is new run
        assert all(r.init_kwargs["resume"] == "allow" for r in runs)

    def test_finish_is_exposed(self, monkeypatch, tmp_path):
        runs = install_fake_wandb(monkeypatch)
        log = wandb_logger(project="p", checkpoint_dir=str(tmp_path))
        log.finish()
        assert runs[0].finished


class TestMainWiring:
    def test_wandb_flag_tees_alongside_jsonl(self, monkeypatch, tmp_path):
        from test_train import write_run
        from optbench.train import main

        runs = install_fake_wandb(monkeypatch)
        cfg_path = write_run(tmp_path)
        main(["--config", str(cfg_path), "--wandb", "optbench", "--wandb-run-name", "tiny"])

        (run,) = runs
        assert run.init_kwargs["project"] == "optbench"
        assert run.init_kwargs["name"] == "tiny"
        assert run.init_kwargs["config"]["loop"]["seed"] == 13  # cfg forwarded
        assert run.finished  # finished in the finally

        # same rows in both sinks (modulo flattening/None-dropping)
        jsonl_rows = [
            json.loads(line)
            for line in (tmp_path / "ckpts" / "log.jsonl").read_text().splitlines()
        ]
        assert [s for s, _ in run.logged] == [r["step"] for r in jsonl_rows] == [0, 1, 2, 3]
        for (_, flat), row in zip(run.logged, jsonl_rows):
            assert flat == flatten_row(row)

    def test_wandb_with_log_file_none_is_wandb_only(self, monkeypatch, tmp_path):
        from test_train import write_run
        from optbench.train import main

        runs = install_fake_wandb(monkeypatch)
        cfg_path = write_run(tmp_path)
        main(["--config", str(cfg_path), "--log-file", "none", "--wandb", "p"])
        assert not (tmp_path / "ckpts" / "log.jsonl").exists()
        assert [s for s, _ in runs[0].logged] == [0, 1, 2, 3]

    def test_no_flag_never_touches_wandb(self, monkeypatch, tmp_path):
        from test_train import write_run
        from optbench.train import main

        monkeypatch.setitem(sys.modules, "wandb", None)  # any import would raise
        cfg_path = write_run(tmp_path)
        out = main(["--config", str(cfg_path)])
        assert out["final_step"] == 4
