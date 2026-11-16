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

from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo

from optbench.muon import Muon
from optbench.param_groups import GroupConfig, build_param_groups, split_by_optimizer

Arm = Literal["muon", "adamw", "shampoo"]


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
    # Shampoo (TASK 3d) — plumbed into the vendored DistributedShampoo.
    # betas[1] may be exactly 1.0: that is AdaGrad-style accumulation of the
    # factor matrices (Meta's own default) rather than an EMA.
    shampoo_betas: tuple[float, float] = (0.9, 0.999)
    shampoo_epsilon: float = 1e-12
    shampoo_max_preconditioner_dim: int = 1024
    shampoo_precondition_frequency: int = 10
    # -1 = Meta's default: start at precondition_frequency.
    shampoo_start_preconditioning_step: int = -1
    # Grafting: direction from Shampoo, per-layer magnitude from Adam —
    # keeps Shampoo's lr on Adam's scale, same fairness story as everywhere.
    shampoo_graft_beta2: float = 0.999
    shampoo_graft_epsilon: float = 1e-8

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * arm is "muon", "adamw" or "shampoo"
        * both adamw_betas are in [0, 1)
        * adamw_eps > 0
        * 0 <= muon_momentum < 1
        * muon_ns_steps >= 1
        * shampoo_betas: 0 <= betas[0] < 1 and 0 < betas[1] <= 1
          (note the asymmetry — beta2 == 1.0 is legal, see the field comment)
        * shampoo_epsilon > 0
        * shampoo_max_preconditioner_dim >= 1
        * shampoo_precondition_frequency >= 1
        * shampoo_start_preconditioning_step is -1 or >= 1
          (DistributedShampoo itself additionally enforces
          start >= precondition_frequency at construction — we do not
          duplicate a check the constructor already owns)
        * 0 < shampoo_graft_beta2 < 1
        * shampoo_graft_epsilon > 0
        """
        if self.arm not in ["muon", "adamw", "shampoo"]:
            raise ValueError(f"Arm {self.arm} must be in [muon, adamw, shampoo]!")
        if not 0 <= self.adamw_betas[0] < 1 or not 0 <= self.adamw_betas[1] < 1:
            raise ValueError(f"both adamw_betas {self.adamw_betas} are in [0, 1)!")
        if not self.adamw_eps > 0:
            raise ValueError(f"adamw eps {self.adamw_eps} must be > 0!")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError(f"muon momentum {self.muon_momentum} must be within [0, 1)!")
        if not self.muon_ns_steps >= 1:
            raise ValueError(f"muon ne steps {self.muon_ns_steps} must be >= 1!")
        if not 0 <= self.shampoo_betas[0] < 1 or not 0 < self.shampoo_betas[1] <= 1:
            raise ValueError(f"shampoo beta1 {self.shampoo_betas[0]} should be in range [0, 1), shampoo beta2 {self.shampoo_betas[1]} should be in range (0, 1].")
        if not self.shampoo_epsilon > 0:
            raise ValueError(f"shampoo epsilon {self.shampoo_epsilon} should be larger than 0!")
        if not self.shampoo_max_preconditioner_dim >= 1:
            raise ValueError(f"shampoo max preconditioner dim {self.shampoo_max_preconditioner_dim} should be larger than or equal to 1!")
        if not self.shampoo_precondition_frequency >= 1:
            raise ValueError(f"shampoo precondition frequency {self.shampoo_precondition_frequency} must be >= 1!")
        if not (self.shampoo_start_preconditioning_step  == -1 or self.shampoo_start_preconditioning_step >= 1):
            raise ValueError(f"shampoo start preconditioning step {self.shampoo_start_preconditioning_step} must be -1 or >= 1.")
        if not 0 < self.shampoo_graft_beta2 < 1:
            raise ValueError(f"shampoo grafting beta2 {self.shampoo_graft_beta2} must be in the range (0, 1).")
        if not self.shampoo_graft_epsilon > 0:
            raise ValueError(f"shampoo graft epsilon {self.shampoo_graft_epsilon} must be positive.")
        

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
      5. arm "shampoo" (TASK 3d): same shape as the muon arm —
         split_by_optimizer, DistributedShampoo over the matrix groups,
         AdamW over the rest, skip empty sides, [shampoo, adamw] order.
         DistributedShampoo kwargs, all from opt_cfg: betas, epsilon,
         max_preconditioner_dim, precondition_frequency,
         start_preconditioning_step, and grafting_config =
         AdamPreconditionerConfig(beta2=shampoo_graft_beta2,
         epsilon=shampoo_graft_epsilon). As with the other arms, lr and
         weight_decay ride on the groups (its decoupled weight decay is the
         default), so pass neither.

    Note what this function does NOT take: a learning rate. LRs enter through
    GroupConfig and leave through the schedule; a bare lr argument here would
    be a second source of truth that drifts from the first.
    """
    opt_cfg.validate()
    groups = build_param_groups(model, group_cfg)
    for group in groups:
        group["lr"] = group["base_lr"]
    opt_list = []
    if opt_cfg.arm == "adamw":
        group_opt = torch.optim.AdamW(params=groups, betas=opt_cfg.adamw_betas, eps=opt_cfg.adamw_eps)
        opt_list.append(group_opt)
    elif opt_cfg.arm == "muon":
        matrix_groups, adamw_groups = split_by_optimizer(groups)
        if matrix_groups:
            group_opt = Muon(params=matrix_groups, momentum=opt_cfg.muon_momentum, nesterov=opt_cfg.muon_nesterov, ns_steps=opt_cfg.muon_ns_steps)
            opt_list.append(group_opt)
        if adamw_groups:
            group_opt = torch.optim.AdamW(params=adamw_groups, betas=opt_cfg.adamw_betas, eps=opt_cfg.adamw_eps)
            opt_list.append(group_opt)
    elif opt_cfg.arm == "shampoo":
        matrix_groups, adamw_groups = split_by_optimizer(groups)
        if matrix_groups:
            group_opt = DistributedShampoo(params=matrix_groups, betas=opt_cfg.shampoo_betas, epsilon=opt_cfg.shampoo_epsilon, max_preconditioner_dim=opt_cfg.shampoo_max_preconditioner_dim, precondition_frequency=opt_cfg.shampoo_precondition_frequency, start_preconditioning_step=opt_cfg.shampoo_start_preconditioning_step, grafting_config=AdamPreconditionerConfig(beta2=opt_cfg.shampoo_graft_beta2, epsilon=opt_cfg.shampoo_graft_epsilon))
            opt_list.append(group_opt)
        if adamw_groups:
            group_opt = torch.optim.AdamW(params=adamw_groups, betas=opt_cfg.adamw_betas, eps=opt_cfg.adamw_eps)
            opt_list.append(group_opt)
        
    
    return opt_list
