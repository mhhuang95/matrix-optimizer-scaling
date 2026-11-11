"""Checkpoint and resume: atomic writes, and a guard against silent misloads.

TASK 2e — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_checkpoint.py -v` until all tests pass.

A checkpoint has one job that is easy and one that is hard. The easy job is
recording the numbers. The hard job is failing loudly when the thing you load
them back into is not the thing you saved them from.

  * **Atomicity.** A 1000-H100-hour sweep will be preempted. If the process
    dies partway through `torch.save`, the file on disk is a truncated
    prefix — and if you wrote it over the previous checkpoint, you have
    destroyed the only good state you had. Write to a temporary name in the
    SAME directory, fsync it, then `os.replace` onto the real name.
    `os.replace` is atomic on POSIX and Windows, but only within one
    filesystem, which is why the temp file cannot live in /tmp.

  * **Optimizer state is keyed by integer index, not by name.** PyTorch
    stores `optimizer.state` as `{0: {...}, 1: {...}}`, where the integers
    count parameters in `param_groups` order. `load_state_dict` validates the
    number of groups and the number of parameters in each group, so adding a
    tensor raises. But a REORDERING with unchanged counts loads silently: wq's
    momentum buffer lands on wk. The model still trains, slightly worse, and
    nothing in the logs says why. The fix is to save the per-group `names`
    list that `build_param_groups` already produces, and refuse to load when
    it does not match.

  * **What resume means here.** We want data-stream reproducibility: the data
    for a step is a pure function of (seed, step), so the checkpoint only has
    to record `step` — not an iterator, not a position in a shuffle. That is
    what makes `10 steps + resume + 10 steps == 20 steps` a property we can
    actually test, and it survives a change of world size. Bitwise
    reproducibility additionally needs the torch/CUDA RNG state, deterministic
    kernels and a fixed world size; out of scope, see the notes doc.

  * **The scaler has state too.** `GradScaler` holds the current loss scale
    and its growth counter. Resume an fp16 run without it and the scaler
    restarts at 65536 and burns several skipped steps rediscovering the scale
    it already knew — steps that consume your LR schedule while the weights
    do not move.

Conventions (the tests rely on these):
  * Filenames are `step_<n>.pt`. `latest_checkpoint` picks the largest <n>
    NUMERICALLY. Sorting the names as strings puts step_9 after step_10, which
    is the classic way to resume from the wrong checkpoint.
  * Every mismatch raises ValueError with a message naming what differed.
    A checkpoint that loads into the wrong tensors is worse than one that
    refuses to load.
  * Keep-last-N rotation is deliberately not here: it is a policy decision for
    the launch script, and deleting files is not something a library should do
    as a side effect of saving.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

CHECKPOINT_VERSION = 1


def atomic_save(obj: Any, path: str | Path) -> None:
    """`torch.save(obj, path)`, but the file at `path` is never half-written.

    Steps:
      1. Create `path`'s parent directory if it does not exist.
      2. Save to a temporary file in the SAME directory. Name it
         `<path.name>.tmp` — the tests look for exactly that name, and a
         sibling is what keeps `os.replace` on one filesystem.
      3. Flush and `os.fsync(f.fileno())` before closing, so the bytes are on
         the device and not just in the page cache. Without this, a machine
         that loses power after `os.replace` can come back with a correctly
         named, empty file.
      4. `os.replace(tmp, path)` — atomic rename.

    If the save raises, remove the temporary file and re-raise. Any existing
    file at `path` must be left exactly as it was.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_name(dest.name + ".tmp")

    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    
    os.replace(tmp, dest)


def optimizer_group_names(
    optimizers: Sequence[torch.optim.Optimizer],
) -> list[list[list[str]]]:
    """The "names" list of every param group, nested by optimizer.

    Shape: `result[i][j]` is the list of parameter names in group j of
    optimizer i, in order. This is the fingerprint the checkpoint compares.

    Raise ValueError if any group has no "names" key — the guard only works
    if the names are there, so a group built by hand rather than by
    `build_param_groups` must fail at save time, not at load time three days
    later. Name the optimizer index and the group index in the message.
    """
    result = []
    for i, optimizer in enumerate(optimizers):
        names = []
        for k, group in enumerate(optimizer.param_groups):
            if "names" not in group:
                raise ValueError(f"optimizer {i}, group {k} has no 'names' key; build the optimizer's groups with optbench.param_groups.build_param_groups")
            names.append(list(group["names"]))
        result.append(names)
    return result


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    scaler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write everything needed to resume training at `step`.

    Raise ValueError if step < 0.

    The payload is a dict with exactly these keys:

        "version":     CHECKPOINT_VERSION
        "step":        int(step) — the number of optimizer updates ALREADY
                       done, so training resumes by running step, step+1, ...
        "model":       model.state_dict()
        "optimizers":  [opt.state_dict() for opt in optimizers], in order
        "group_names": optimizer_group_names(optimizers)
        "scaler":      scaler.state_dict(), or None if scaler is None
        "extra":       dict(extra), or {} if extra is None — a copy, so a
                       later mutation of the caller's dict cannot change what
                       was written

    Written with atomic_save.

    Note what is NOT here: no data iterator, no RNG state, no config. The
    config belongs in the run directory as the YAML you launched with; a
    checkpoint that carries its own config invites the two to disagree.
    """
    if step < 0:
        raise ValueError(f"Step {step} < 0 is not allowed in checkpoint saving!")
    ckpt = {
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "group_names": optimizer_group_names(optimizers),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": dict(extra) if extra is not None else {},
    }
    atomic_save(ckpt, path)
    

def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    scaler: Any | None = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Restore model, optimizer and scaler state in place.

    Returns `{"step": int, "extra": dict}` — the caller resumes its loop at
    that step.

    Load with `torch.load(path, map_location=map_location, weights_only=False)`.
    `map_location="cpu"` is the default on purpose: a checkpoint saved from
    cuda:3 carries that device in its tensors, and loading it on a box where
    cuda:3 does not exist raises. Load to CPU, then let the model's own
    `.to(device)` place things.

    Validate BEFORE mutating anything — a half-loaded model is harder to
    debug than a refusal. In this order, each raising ValueError:
      1. ckpt["version"] != CHECKPOINT_VERSION
      2. len(ckpt["optimizers"]) != len(optimizers)
      3. ckpt["group_names"] != optimizer_group_names(optimizers)
         — when they differ, say WHICH optimizer differs and show both lists
      4. the checkpoint has scaler state but no scaler was passed, or vice
         versa (an fp16 run resumed as bf16, or the reverse)

    Then load model, each optimizer, and the scaler if there is one.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Ckpt loading: ckpt version {ckpt["version"]} != CHECKPOINT_VERSION {CHECKPOINT_VERSION}!")
    if len(ckpt["optimizers"]) != len(optimizers):
        raise ValueError(f"Ckpt loading: Number of optimizers in ckpt {len(ckpt["optimizers"])} not equal to Number of optimizers in model {len(optimizers)}!")
    for i, (ckpt_gnames, opt_gnames) in enumerate(zip(ckpt["group_names"], optimizer_group_names(optimizers))):
        if ckpt_gnames != opt_gnames:
            raise ValueError(f"Ckpt loading: in optimizer {i}, there is a mismatch on the group names in optimizer {opt_gnames} from the ckpt {ckpt_gnames}!")
    if (ckpt["scaler"] is not None and scaler is None) or (ckpt["scaler"] is None and scaler is not None):
        raise ValueError(f"Ckpt loading: there is mismatch on the scaler from checkpoint!")
    model.load_state_dict(ckpt["model"])
    for opt, ckpt_opt in zip(optimizers, ckpt["optimizers"]):
        opt.load_state_dict(ckpt_opt)
    if ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {"step": int(ckpt["step"]), "extra": ckpt["extra"]}


def latest_checkpoint(dirpath: str | Path) -> Path | None:
    """The `step_<n>.pt` file in `dirpath` with the largest n, or None.

    Returns None if the directory does not exist or holds no matching file.
    Files whose names do not match the pattern are ignored, not an error — a
    run directory also holds logs, configs and `.tmp` leftovers.

    `n` is compared as an integer. The regex below is given; you need the
    rest.
    """
    _step_re = re.compile(r"^step_(\d+)\.pt$")

    folder = Path(dirpath)
    if not folder.is_dir():
        return None

    best_step = -1
    best_path = None
    for ckpt_path in folder.iterdir():
        m = _step_re.match(ckpt_path.name)
        if not m: 
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = ckpt_path
    return best_path