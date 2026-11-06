"""Learning-rate schedules.

TASK 2a — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_schedule.py -v` until all tests pass.

Design: the schedule is a *pure function of the step* that returns a
multiplier in [min_lr_ratio, 1]. Each optimizer param group stores its own
peak LR under the key "base_lr", and `apply_lr` sets
    group["lr"] = group["base_lr"] * multiplier
for every group. This lets Muon groups and AdamW groups (very different peak
LRs) share one schedule shape, and makes resume trivial: the LR at any step is
recomputed from the step alone, with no scheduler state to checkpoint.

Conventions (the tests rely on these):
  * `step` is 0-indexed: the multiplier for the optimizer update number `step`.
  * Warmup (both kinds): for step < warmup_steps,
        multiplier = (step + 1) / warmup_steps
    so step 0 is already non-zero and step warmup_steps - 1 reaches 1.0.
    warmup_steps == 0 means no warmup.
  * For step >= total_steps the multiplier stays at min_lr_ratio.
  * "cosine": after warmup, cosine from 1.0 down to min_lr_ratio, with
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
    so progress is 0 at the first post-warmup step and reaches 1 at
    step == total_steps.
  * "wsd" (warmup-stable-decay): decay_steps = round(decay_frac * total_steps),
    decay_start = total_steps - decay_steps. Constant 1.0 on
    [warmup_steps, decay_start), then LINEAR from 1.0 to min_lr_ratio, with
        progress = (step - decay_start) / decay_steps.
  * Invalid configs raise ValueError (see ScheduleConfig.validate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import math


@dataclass
class ScheduleConfig:
    kind: Literal["cosine", "wsd"] = "cosine"
    total_steps: int = 1000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # final multiplier
    decay_frac: float = 0.2  # WSD only: fraction of total_steps spent decaying

    @property
    def decay_steps(self) -> int:
        return round(self.decay_frac * self.total_steps)
    
    @property
    def decay_start(self) -> int:
        return self.total_steps - self.decay_steps

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * kind is "cosine" or "wsd"
        * total_steps > 0 and 0 <= warmup_steps < total_steps
        * 0 <= min_lr_ratio <= 1
        * for "wsd": 0 < decay_frac <= 1, decay_steps >= 1, and the warmup
          ends no later than the decay starts (warmup_steps <= decay_start)
        """
        if self.kind not in ["cosine", "wsd"]:
            raise ValueError("kind must be either cosine or wsd!")
        if self.total_steps <=0 or self.warmup_steps < 0 or self.warmup_steps >= self.total_steps:
            raise ValueError("Check total_steps and warmup_steps! We must have total_steps > 0 and 0 <= warmup_steps < total_steps!")
        if self.min_lr_ratio < 0 or self.min_lr_ratio > 1:
            raise ValueError("Check min_lr_ratio! We must have 0 <= min_lr_ratio <= 1!")
        if self.kind == "wsd":
            if self.decay_frac <= 0 or self.decay_frac > 1 or self.decay_steps < 1 or self.warmup_steps > self.decay_start:
                raise ValueError("In wsd, the decay_frac must be within (0, 1], the decay_steps must be >= 1, the decay_start must be >= warmup_steps.")
        


def lr_multiplier(step: int, cfg: ScheduleConfig) -> float:
    """Return the LR multiplier (a Python float) for optimizer update `step`."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    elif step >=   cfg.total_steps:
        return cfg.min_lr_ratio
    elif cfg.kind == "cosine":
        progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
        ratio = (math.cos(progress * math.pi) + 1 )/ 2
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * ratio
    elif cfg.kind == "wsd":
        if step <= cfg.decay_start:
            return 1.0
        else:
            progress = (step - cfg.decay_start) / cfg.decay_steps
            ratio = 1 - progress
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * ratio
    else:
        raise ValueError(f"unknown kind {cfg.kind!r}")



def apply_lr(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    """Set group["lr"] = group["base_lr"] * multiplier for every param group.

    Raise KeyError if a group has no "base_lr" (fail loudly rather than
    silently scheduling from whatever "lr" happens to be).
    """
    for group in optimizer.param_groups:
        if "base_lr" not in group.keys():
            raise KeyError("Group has no key `base_lr`!")
        group["lr"] = group["base_lr"] * multiplier
    
    
