"""A W&B log_fn: the same rows jsonl_logger writes, mirrored to wandb.

Boilerplate (third-party integration) — nothing here to implement.

Design decisions, since they encode the project's logging contract:

  * **jsonl stays the source of truth.** The analysis notebooks read log.jsonl;
    W&B is live monitoring only. train.py therefore TEES rows to both sinks
    instead of replacing one with the other — losing the jsonl file because
    someone wanted dashboards would corrupt the study.

  * **None values are dropped per-row, not logged.** The loop's no-fake-values
    rule puts None in a row to mean "no measurement yet" (warmup
    tokens_per_sec, and mfu with it). jsonl keeps that honestly as `null`;
    W&B charts treat null as a point, which draws misleading zeros/gaps.
    Dropping the KEY for that step preserves the meaning: no point plotted.

  * **The nested "spectral" dict is flattened** to `spectral/{param}/{stat}`
    keys — W&B's namespace separator is "/", and flat keys are what make each
    stat its own chart.

  * **Resume continues the same W&B run.** The run id is derived from the
    resolved checkpoint_dir, so a relaunch after preemption appends to the
    same dashboard — the same append-not-truncate semantics jsonl_logger has.
    Two different runs must therefore never share a checkpoint_dir, which the
    checkpoint collision check already enforces.

  * **wandb is an optional dependency** (`uv sync --extra wandb`): the import
    happens inside wandb_logger, so a Mac dev box without wandb installed can
    run everything except `--wandb`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

LogFn = Callable[[dict[str, Any]], None]


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """One W&B-ready flat dict: spectral/{param}/{stat} keys, Nones dropped."""
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if key == "spectral":
            for name, stats in value.items():
                for stat, v in stats.items():
                    flat[f"spectral/{name}/{stat}"] = v
        else:
            flat[key] = value
    return flat


def tee(*log_fns: LogFn) -> LogFn:
    """One log_fn that feeds every sink. Each sink gets its own dict copy."""

    def log(row: dict[str, Any]) -> None:
        for fn in log_fns:
            fn(dict(row))

    return log


def wandb_logger(
    *,
    project: str,
    checkpoint_dir: str,
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> LogFn:
    """A log_fn mirroring rows to W&B. The returned closure carries `.finish`."""
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is not installed; install with `uv sync --extra wandb` "
            "or `pip install wandb`, or drop the --wandb flag"
        ) from e

    # Stable id <- checkpoint_dir: a preempted-and-relaunched job resumes the
    # same W&B run instead of littering the project with fragments.
    run_id = hashlib.sha1(
        str(Path(checkpoint_dir).resolve()).encode()
    ).hexdigest()[:16]
    run = wandb.init(
        project=project,
        name=run_name,
        id=run_id,
        resume="allow",
        config=config,
    )

    def log(row: dict[str, Any]) -> None:
        run.log(flatten_row(row), step=row["step"])

    log.finish = run.finish  # type: ignore[attr-defined]
    return log
