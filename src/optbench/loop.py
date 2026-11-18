"""The training loop: every module in the stack, wired into one resumable run.

TASK 5 — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_loop.py -v` until all tests pass.

This is deliberately the LAST module: it owns no algorithm and no learnable
state — it is pure orchestration, and every bug it can have is an off-by-one
or an ordering mistake. The design that makes it small:

  * The schedule is a pure function of `step` (lr_multiplier).
  * The data is a pure function of `(step, seed)` (get_batch).
  * Everything else that evolves lives in model/optimizer state_dicts.

  Therefore a checkpoint is just (step, states): resuming needs no data
  iterator position, no RNG snapshot, no schedule bookkeeping. Kill the run
  at any checkpoint and restart it with the same configs, and the weights
  it produces are BIT-IDENTICAL to the uninterrupted run — the tests check
  exactly that, down to torch.equal. (Our models run dropout-free, standard
  for LLM pretraining; a model with dropout would need the torch RNG state
  in the checkpoint too, and this loop would refuse to be this simple.)

Step indexing, fixed once for every consumer:

  * `step` is 0-indexed inside the loop; a saved checkpoint records
    `step + 1` — "optimizer updates ALREADY done" — and is named
    `step_{step+1}.pt`, which is exactly what latest_checkpoint parses and
    load_checkpoint returns, so a resumed loop continues at `range(done,
    total)` with no correction anywhere.
  * "every N" cadences (logging, checkpoints, spectral) count COMPLETED
    updates: an event fires on the steps where (step + 1) % N == 0.
    N == 0 disables the cadence entirely.

Config collisions must be loud: the checkpoint stores the data-order
contract (seed, global_batch) in `extra`, and resuming under a different
value raises instead of silently training on reshuffled data — the same
"checkpoint beats config, loudly" rule the optimizer tests taught.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from optbench.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from optbench.data import TokenWindows, get_batch
from optbench.schedule import ScheduleConfig, apply_lr, lr_multiplier
from optbench.spectral import collect_spectral_stats
from optbench.throughput import ThroughputMeter, step_was_skipped
from optbench.train_step import TrainStepConfig, train_step


@dataclass
class LoopConfig:
    """What the loop itself owns: data order, device placement, cadences.

    Everything algorithmic lives in the other configs — if a field here can
    change the final weights, it is in the wrong place (seed and
    global_batch are the deliberate exceptions: they define the data order,
    and the checkpoint pins them for exactly that reason).
    """

    global_batch: int = 8
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | Path = "checkpoints"
    checkpoint_every: int = 0  # 0 = only the end-of-run checkpoint
    log_every: int = 1  # 0 = never log
    spectral_every: int = 0  # 0 = never collect spectral stats
    meter_warmup: int = 0  # ThroughputMeter warmup_steps

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * global_batch >= 1
        * checkpoint_every >= 0, log_every >= 0, spectral_every >= 0
        * meter_warmup >= 0
        """
        if not self.global_batch >= 1:
            raise ValueError(f"global batch {self.global_batch} must be >= 1!")
        if not self.checkpoint_every >= 0 or not self.log_every >= 0 or not self.spectral_every >= 0:
            raise ValueError(f"checkpoint_every {self.checkpoint_every}, log_every {self.log_every}, spectral_every {self.spectral_every} all must be nonnegative!")
        if not self.meter_warmup >= 0:
            raise ValueError(f"meter_warmup {self.meter_warmup} must be nonnegative!")


def is_due(step: int, every: int) -> bool:
    """True when a cadence of `every` fires on 0-indexed `step`.

    Counts completed updates: fires when (step + 1) % every == 0, so
    every=2 fires on steps 1, 3, 5, ... and a checkpoint cadence lines up
    with the `step_{n}.pt` naming (n completed updates). every == 0 means
    "never" — False for all steps, and NOT a ZeroDivisionError.
    """
    if every == 0:
        return False
    return (step + 1) % every == 0


def run_loop(
    model: nn.Module,
    windows: TokenWindows,
    optimizers: Sequence[torch.optim.Optimizer],
    loop_cfg: LoopConfig,
    sched_cfg: ScheduleConfig,
    step_cfg: TrainStepConfig,
    *,
    scaler: Any | None = None,
    stop_after: int | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Train from the latest checkpoint (or scratch) toward total_steps.

    Steps:
      1. Validate all three configs. Raise ValueError if stop_after is not
         None and stop_after < 1 (stop_after = how many steps THIS CALL may
         run — a time-budget knob for spot instances, not a global cap).
      2. Resume: `latest_checkpoint(loop_cfg.checkpoint_dir)`; if one
         exists, `load_checkpoint` into model/optimizers/scaler and take
         start_step from it, then check the checkpoint's extra dict against
         loop_cfg: extra["seed"] != seed or extra["global_batch"] !=
         global_batch is a ValueError (silently reshuffled data is a lie).
         No checkpoint: start_step = 0.
      3. end_step = total_steps, or min(total_steps, start_step +
         stop_after) when stop_after is given. model.train(). Build a
         ThroughputMeter(warmup_steps=loop_cfg.meter_warmup).
      4. For step in range(start_step, end_step):
           a. mult = lr_multiplier(step, sched_cfg); apply_lr on EVERY
              optimizer — before the step, so warmup applies from step 0.
           b. micro_batches: get_batch(windows, step, global_batch, seed,
              micro_step=m, accum_steps=step_cfg.accum_steps,
              device=loop_cfg.device) for m in range(accum_steps).
           c. If a scaler is present, read scaler.get_scale() BEFORE the
              step and after it, and let step_was_skipped decide `skipped`;
              no scaler means skipped=False. stats = train_step(...).
           d. meter.update(stats.n_tokens, skipped=skipped).
           e. When log or spectral is due (either one) and log_fn is given,
              emit ONE row: {"step", "loss", "grad_norm", "lr_mult",
              "tokens_per_sec", "skipped"}, plus "spectral":
              collect_spectral_stats(optimizers, source="grad") only when
              spectral is due (grads are intact here — train_step zeroes
              at the START of the next step, not the end of this one).
           f. When the checkpoint cadence is due: save_checkpoint to
              checkpoint_dir / f"step_{step+1}.pt" with step=step+1 and
              extra={"seed": ..., "global_batch": ...}.
      5. After the loop, if any step ran and the last completed step was
         not already saved by the cadence, save the end-of-run checkpoint
         (same naming). A call that ran zero steps saves nothing.
      6. Return {"final_step": the step count now completed,
                 "resumed_from": start_step,
                 "steps_run": end_step - start_step clamped to >= 0}.

    What this function does NOT do: move the model (the caller placed it);
    build optimizers (the factory's job); own total_steps (the schedule
    already does — a second copy here would drift, the same argument that
    kept `lr` out of build_optimizers).
    """
    raise NotImplementedError
