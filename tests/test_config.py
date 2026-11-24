"""Tests for optbench.config — TOML into typed configs. All boilerplate."""

from __future__ import annotations

import pytest
import torch

from optbench.config import load_config


def write_toml(tmp_path, text):
    p = tmp_path / "run.toml"
    p.write_text(text)
    return p


MINIMAL = """
[run]
data_dir = "data"
"""


class TestLoadConfig:
    def test_minimal_gets_defaults(self, tmp_path):
        cfg = load_config(write_toml(tmp_path, MINIMAL))
        assert cfg.data_dir == "data"
        assert cfg.seq_len == cfg.model.max_seq_len
        assert cfg.opt.arm == "muon"
        assert cfg.step.amp_dtype is None
        assert cfg.loop.flops_per_token is None

    def test_values_land_in_the_right_dataclasses(self, tmp_path):
        cfg = load_config(
            write_toml(
                tmp_path,
                """
[run]
data_dir = "shards"
seq_len = 128

[model]
d_model = 256
n_layers = 4
max_seq_len = 256

[loop]
global_batch = 32
seed = 7
checkpoint_dir = "out/ckpts"
peak_flops_per_gpu = 3.12e14

[schedule]
kind = "wsd"
total_steps = 500
warmup_steps = 50

[train_step]
accum_steps = 4
amp_dtype = "bf16"

[groups]
matrix_lr = 0.01

[optimizer]
arm = "adamw"
adamw_betas = [0.9, 0.99]
""",
            )
        )
        assert cfg.seq_len == 128
        assert cfg.model.d_model == 256
        assert cfg.loop.global_batch == 32
        assert cfg.loop.peak_flops_per_gpu == 3.12e14
        assert cfg.sched.kind == "wsd"
        assert cfg.step.accum_steps == 4
        assert cfg.step.amp_dtype is torch.bfloat16
        assert cfg.groups.matrix_lr == 0.01
        assert cfg.opt.arm == "adamw"
        assert cfg.opt.adamw_betas == (0.9, 0.99)  # list became tuple

    def test_missing_data_dir_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="data_dir"):
            load_config(write_toml(tmp_path, "[run]\nseq_len = 64\n"))

    def test_unknown_key_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="warmup_step"):
            load_config(
                write_toml(
                    tmp_path, MINIMAL + "[schedule]\nwarmup_step = 100\n"
                )
            )

    def test_unknown_section_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="modle"):
            load_config(write_toml(tmp_path, MINIMAL + "[modle]\nd_model = 8\n"))

    def test_bad_amp_name_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="fp32"):
            load_config(
                write_toml(tmp_path, MINIMAL + "[train_step]\namp_dtype = \"fp32\"\n")
            )
