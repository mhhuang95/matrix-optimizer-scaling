"""The optimizer factory: one config in, the run's optimizer list out.

TASK 3b — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_optimizers.py -v` until all tests pass.

This is a small module on purpose. Every arm of the study — Muon vs AdamW at
every model size — must be constructed through this one function, so that a
YAML config fully determines the optimizers and nothing is wired by hand in a
launch script. When the Shampoo arm lands, it is one new branch HERE, and
every other module (train_step, checkpoint, schedule) is already generic over
"a list of optimizers".

The two arms:

  * "muon":  [Muon(matrix groups), AdamW(embed + scalar groups)] — the
    matrix method under test plus the AdamW sidecar for everything a matrix
    method cannot touch. Most steps this list has two optimizers; a model
    with no matrix parameters (or no embed/scalar parameters) just omits the
    empty side rather than constructing an optimizer with zero groups
    (PyTorch refuses an empty params list).
  * "adamw": [AdamW(ALL groups)] — the baseline. One optimizer, three
    groups, still with per-group LRs and per-group weight decay (the scalar
    group's 0.0 survives, because AdamW reads weight_decay from each group).

Conventions (the tests rely on these):
  * Groups come from build_param_groups + split_by_optimizer — never
    re-classified here. One classification, one source of truth.
  * Before constructing anything, set group["lr"] = group["base_lr"] on every
    group: the optimizer needs an initial LR and the schedule owns "lr" from
    step 0, so the factory seeds it rather than letting each optimizer's
    constructor default leak in.
  * Muon's lr/weight_decay are NOT in OptimizerConfig — they live on the
    groups (GroupConfig.matrix_lr and weight_decay). The config split is:
    GroupConfig says WHAT each parameter kind gets, OptimizerConfig says HOW
    each algorithm runs (betas, momentum, ns_steps).
  * The returned list order is [matrix optimizer, AdamW] for the muon arm —
    train_step and the checkpoint fingerprint both depend on a stable order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from optbench.muon import Muon
from optbench.param_groups import GroupConfig, build_param_groups, split_by_optimizer

Arm = Literal["muon", "adamw"]


@dataclass
class OptimizerConfig:
    """How each algorithm runs. What each parameter kind gets is GroupConfig's job.

    adamw_betas: (0.9, 0.95) — beta2 = 0.95, not torch's default 0.999:
        LLM training sees rare large gradient bursts, and a 0.999 second
        moment remembers them for ~1000 steps, muting the LR long after the
        burst. 0.95 is the standard LLM choice (GPT-3, Llama, Chinchilla).
    """

    arm: Arm = "muon"
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * arm is "muon" or "adamw"
        * both adamw_betas are in [0, 1)
        * adamw_eps > 0
        * 0 <= muon_momentum < 1
        * muon_ns_steps >= 1
        """
        raise NotImplementedError


def build_optimizers(
    model: nn.Module,
    group_cfg: GroupConfig,
    opt_cfg: OptimizerConfig,
) -> list[torch.optim.Optimizer]:
    """Build the run's optimizers from the two configs.

    Steps:
      1. opt_cfg.validate() (build_param_groups validates group_cfg itself).
      2. groups = build_param_groups(model, group_cfg); seed every group's
         "lr" from its "base_lr".
      3. arm "adamw": one AdamW over ALL groups, betas/eps from opt_cfg.
         Return [that].
      4. arm "muon": split_by_optimizer, then Muon over the matrix groups
         (momentum/nesterov/ns_steps from opt_cfg; lr and weight_decay ride
         on the groups) and AdamW over the rest. Skip a side whose group
         list is empty. Return [muon, adamw] in that order.

    Note what this function does NOT take: a learning rate. LRs enter through
    GroupConfig and leave through the schedule; a bare lr argument here would
    be a second source of truth that drifts from the first.
    """
    raise NotImplementedError
