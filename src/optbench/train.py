"""The composition root: a config file in, trained checkpoints out.

TASK 9 — implement main() (the one `raise NotImplementedError`).
Run `uv run pytest tests/test_train.py -v` until all tests pass.

This is the file the cluster actually launches:

    single process:  python -m optbench.train --config runs/tiny.toml
    distributed:     torchrun --nproc_per_node=8 -m optbench.train \
                         --config runs/tiny.toml

There is no new algorithm here — main() is pure composition, and its entire
value is doing eight known things in the one order that is correct. Every
ordering rule the previous tasks taught appears exactly once:

    detect -> init -> seed -> model -> optimizers -> wrap -> loop -> cleanup

parse_args and jsonl_logger below are provided; config parsing lives in
optbench.config (load_config).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from optbench.config import load_config
from optbench.data import TokenWindows
from optbench.ddp import cleanup_distributed, detect_dist, init_distributed, wrap_ddp
from optbench.loop import run_loop
from optbench.model import Transformer
from optbench.optimizers import build_optimizers
from optbench.throughput import flops_per_token


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """--config is required; --log-file optional (see main step 8)."""
    p = argparse.ArgumentParser(description="optbench training entry point")
    p.add_argument("--config", required=True, help="path to the run's TOML file")
    p.add_argument(
        "--log-file",
        default=None,
        help="JSONL metrics path; default <checkpoint_dir>/log.jsonl; "
        "'none' disables logging",
    )
    return p.parse_args(argv)


def jsonl_logger(path: str | Path) -> Callable[[dict[str, Any]], None]:
    """A log_fn that appends one JSON object per row to `path`.

    Line-buffered append: rows survive a crash, and a resumed run keeps
    appending to the same file instead of truncating its history.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", buffering=1)

    def log(row: dict[str, Any]) -> None:
        f.write(json.dumps(row) + "\n")

    return log


def main(argv: list[str] | None = None) -> dict[str, int]:
    """Build everything in the right order and run the loop.

    Steps:
      1. args = parse_args(argv); cfg = load_config(args.config).
      2. ctx = detect_dist() — under torchrun the env vars are there, in a
         plain launch they are not and ctx is the single-process default.
         init_distributed(ctx). EVERYTHING from here on runs inside
         try/finally with cleanup_distributed() in the finally: a rank that
         crashes mid-build must still tear down the process group, or its
         peers hang in rendezvous instead of dying loudly.
      3. Resolve the device THIS rank uses: "cuda:{ctx.local_rank}" when
         torch.cuda.is_available(), else "cpu"; on cuda also call
         torch.cuda.set_device(...) so stray `.cuda()` calls land on the
         right card. The config cannot know this — the same TOML file is
         launched on a laptop and an 8-GPU node — so the config's
         loop.device and train_step.device_type are OVERRIDDEN here:
         loop_cfg = replace(cfg.loop, device=device),
         step_cfg = replace(cfg.step, device_type="cuda"/"cpu").
         (dataclasses.replace, not mutation: configs stay values.)
      4. torch.manual_seed(cfg.loop.seed) BEFORE constructing the model:
         every rank builds identical weights by construction. DDP's
         broadcast would paper over a divergence, but identical-by-
         construction also makes single-process and distributed runs
         bit-comparable — which the equivalence tests rely on.
      5. model = Transformer(cfg.model).to(device).
      6. optimizers = build_optimizers(model, cfg.groups, cfg.opt) — on the
         RAW model, BEFORE wrapping (the param-group names rule).
      7. wrapped = wrap_ddp(model, ctx).
      8. shards = sorted(Path(cfg.data_dir).glob("shard_*.bin"));
         an empty list is a ValueError naming the directory — a run that
         starts with zero shards must die before it burns a GPU-hour.
         windows = TokenWindows(shards, cfg.seq_len).
      9. MFU wiring: when loop_cfg.peak_flops_per_gpu is set and
         loop_cfg.flops_per_token is None, fill it:
         replace(loop_cfg, flops_per_token=flops_per_token(cfg.model,
         cfg.seq_len)) — the config names the hardware number it knows,
         main computes the model number it can derive.
     10. log_fn: None unless ctx.is_main (don't open files you will never
         write — emission is is_main-gated inside the loop anyway) and
         args.log_file != "none". Path: args.log_file, defaulting to
         Path(loop_cfg.checkpoint_dir) / "log.jsonl". Build it with
         jsonl_logger.
     11. out = run_loop(wrapped, windows, optimizers, loop_cfg, cfg.sched,
         step_cfg, log_fn=log_fn, dist_ctx=ctx); return out.

    What main() does NOT do: validate configs (run_loop's first act),
    choose checkpoint naming (the loop's contract), or special-case
    world_size == 1 anywhere — the ctx defaults make one process just the
    degenerate case of N.
    """
    args = parse_args(argv)
    cfg = load_config(args.config)
    ctx = detect_dist()
    try:
        init_distributed(ctx)
        if torch.cuda.is_available():
            device = f"cuda:{ctx.local_rank}"
            torch.cuda.set_device(device)
            device_type = "cuda"
        else:
            device = "cpu"
            device_type = "cpu"
        loop_cfg = replace(cfg.loop, device=device)
        step_cfg = replace(cfg.step, device_type=device_type)
        torch.manual_seed(cfg.loop.seed)
        model = Transformer(cfg.model).to(device)
        optimizers = build_optimizers(model, cfg.groups, cfg.opt)
        wrapped = wrap_ddp(model, ctx)
        shards = sorted(Path(cfg.data_dir).glob("shard_*.bin"))
        if not shards:
            raise ValueError(f"{cfg.data_dir} -- empty shards!")
        windows = TokenWindows(shards, cfg.seq_len)
        if loop_cfg.peak_flops_per_gpu is not None and loop_cfg.flops_per_token is None:
            loop_cfg = replace(loop_cfg, flops_per_token=flops_per_token(cfg.model, cfg.seq_len))
        log_fn = None
        if ctx.is_main and args.log_file != "none":
            path = args.log_file if args.log_file is not None else Path(loop_cfg.checkpoint_dir) / "log.jsonl"
            log_fn = jsonl_logger(path)
        out = run_loop(wrapped, windows, optimizers, loop_cfg, cfg.sched, step_cfg, log_fn=log_fn, dist_ctx=ctx)
        return out
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
