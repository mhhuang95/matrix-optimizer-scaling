"""Run configuration: one TOML file feeds every config dataclass in the stack.

This module is deliberately dumb plumbing: it maps TOML tables onto the
existing dataclasses field-for-field and converts the two things TOML cannot
express directly (tuples arrive as lists; torch dtypes arrive as strings).
It does NOT call validate() on anything — run_loop validates every config at
startup, and keeping a single validation point means a config error is
reported the same way no matter how the config was built.

Loudness rule: an unknown section or key is a ValueError, never silently
ignored. A typo like `warmup_step = 100` that is dropped on the floor trains
a wrong run that looks healthy — the same "half-configured is a lie" rule as
detect_dist and the MFU pair.

TOML layout (every section optional except [run], every key optional —
dataclass defaults fill the gaps):

    [run]
    data_dir = "data/fineweb"      # REQUIRED: holds shard_*.bin files
    seq_len = 1024                 # default: model.max_seq_len

    [model]      # ModelConfig fields: d_model, n_layers, n_heads, ...
    [loop]       # LoopConfig fields: global_batch, seed, checkpoint_dir, ...
    [schedule]   # ScheduleConfig fields: kind, total_steps, warmup_steps, ...
    [train_step] # TrainStepConfig fields; amp_dtype = "bf16" | "fp16"
    [groups]     # GroupConfig fields: matrix_lr, embed_lr, ...
    [optimizer]  # OptimizerConfig fields: arm = "muon" | "shampoo" | "adamw"
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from optbench.loop import LoopConfig
from optbench.model import ModelConfig
from optbench.optimizers import OptimizerConfig
from optbench.param_groups import GroupConfig
from optbench.schedule import ScheduleConfig
from optbench.train_step import TrainStepConfig

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

_SECTIONS = ("run", "model", "loop", "schedule", "train_step", "groups", "optimizer")


@dataclass
class RunConfig:
    """Everything one training run needs, parsed and typed."""

    data_dir: str
    seq_len: int
    model: ModelConfig
    loop: LoopConfig
    sched: ScheduleConfig
    step: TrainStepConfig
    groups: GroupConfig
    opt: OptimizerConfig


def _build(cls: type, table: dict[str, Any], section: str) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = set(table) - allowed
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [{section}]; "
            f"allowed: {sorted(allowed)}"
        )
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in table.items()}
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    """Parse a TOML file into a RunConfig. Loud on anything unrecognized."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            f"unknown section(s) {sorted(unknown)}; allowed: {list(_SECTIONS)}"
        )

    step_table = dict(raw.get("train_step", {}))
    if "amp_dtype" in step_table:
        name = step_table["amp_dtype"]
        if name not in _AMP_DTYPES:
            raise ValueError(
                f"amp_dtype {name!r} must be one of {sorted(_AMP_DTYPES)}"
            )
        step_table["amp_dtype"] = _AMP_DTYPES[name]

    model = _build(ModelConfig, raw.get("model", {}), "model")

    run_table = raw.get("run", {})
    unknown = set(run_table) - {"data_dir", "seq_len"}
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [run]; "
            f"allowed: ['data_dir', 'seq_len']"
        )
    if "data_dir" not in run_table:
        raise ValueError("[run] data_dir is required: where are the shards?")

    return RunConfig(
        data_dir=run_table["data_dir"],
        seq_len=run_table.get("seq_len", model.max_seq_len),
        model=model,
        loop=_build(LoopConfig, raw.get("loop", {}), "loop"),
        sched=_build(ScheduleConfig, raw.get("schedule", {}), "schedule"),
        step=_build(TrainStepConfig, step_table, "train_step"),
        groups=_build(GroupConfig, raw.get("groups", {}), "groups"),
        opt=_build(OptimizerConfig, raw.get("optimizer", {}), "optimizer"),
    )
