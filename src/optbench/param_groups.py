"""Optimizer parameter groups.

TASK 2c — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_param_groups.py -v` until all tests pass.

Why this file exists at all. Muon, Shampoo and SOAP are *matrix* methods:
they precondition a 2-D gradient using its own structure (an orthogonalized
update for Muon, Kronecker-factored second moments for Shampoo/SOAP). That
only makes sense for a weight matrix mapping one hidden space to another.
Three kinds of tensor are therefore routed differently:

  * "matrix"  — 2-D hidden weights (attention wq/wk/wv/wo, MLP w1/w2/w3).
                These go to the matrix optimizer under test.
  * "embed"   — the token embedding and the LM head. Also 2-D, but one axis
                is the vocabulary: a row is a token, not a feature direction,
                and the gradient is extremely sparse (only the tokens in the
                batch get a nonzero row). Orthogonalizing across 32k rows is
                both semantically wrong and the single most expensive matrix
                in the model. Every public Muon implementation excludes them;
                they go to AdamW.
  * "scalar"  — everything with ndim <= 1: RMSNorm gains, any bias. A matrix
                method has no structure to exploit on a vector. AdamW, and
                NO weight decay (shrinking a norm gain toward zero fights the
                normalization it exists to perform).

Keeping this in its own tested function matters because the two classic bugs
are both silent: a parameter listed in two groups is updated twice per step
(an effective 2x LR on exactly the layers you care about), and a parameter
listed in no group is never trained at all. Neither raises; both just produce
a worse curve that you would blame on the optimizer.

Conventions (the tests rely on these):
  * Groups carry "base_lr", not "lr", so `optbench.schedule.apply_lr` can
    drive them. Build the optimizer with lr=group["base_lr"] as the initial
    value, then let the schedule own "lr" from step 0 on.
  * Group order is always matrix, embed, scalar. Empty groups are omitted.
  * Within a group, parameters stay in `model.named_parameters()` order.
    This is not cosmetic: optimizer state is keyed by position in some
    distributed wrappers, and a nondeterministic order makes checkpoints
    load into the wrong tensors.
  * Parameters with requires_grad=False are skipped entirely.
  * Tied weights appear exactly once (`named_parameters()` already dedupes,
    but the count assertions depend on it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

Kind = Literal["matrix", "embed", "scalar"]
_ORDER: tuple[Kind, ...] = ("matrix", "embed", "scalar")


@dataclass
class GroupConfig:
    """Peak learning rates and decay for the three kinds of parameter.

    The LRs differ by an order of magnitude because the optimizers do: Muon's
    orthogonalized update has unit spectral norm by construction, so it
    tolerates (and needs) a much larger step than AdamW's per-coordinate
    normalized update.
    """

    matrix_lr: float = 0.02  # peak LR for the matrix optimizer (Muon/Shampoo/SOAP)
    embed_lr: float = 3e-4  # peak LR for AdamW on embeddings + head
    scalar_lr: float = 3e-4  # peak LR for AdamW on norms/biases
    weight_decay: float = 0.1  # applied to "matrix" and "embed", never "scalar"
    embed_patterns: tuple[str, ...] = ("tok_emb", "lm_head")

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * matrix_lr, embed_lr, scalar_lr are all > 0
        * weight_decay >= 0
        * embed_patterns is non-empty and every entry is a non-empty string
        """
        if not self.matrix_lr > 0:
            raise ValueError(f"matrix_lr {self.matrix_lr} should be larger than 0!")
        if not self.embed_lr > 0:
            raise ValueError(f"embed_lr {self.embed_lr} should be larger than 0!")
        if not self. scalar_lr > 0:
            raise ValueError(f"scalar_lr {self.scalar_lr} should be larger than 0!")
        if not self.weight_decay >=  0:
            raise ValueError(f"weight_decay {self.weight_decay} should be larger than or equal to 0!")
        if  not self.embed_patterns or any(not isinstance(p, str) or not p for p in self.embed_patterns) :
            raise ValueError("embed_patterns must be non-empty and every entry must be a non-empty string")


def classify(name: str, param: torch.Tensor, cfg: GroupConfig) -> Kind:
    """Which of the three kinds `param` belongs to.

    Rules, in this order:
      1. ndim <= 1                                     -> "scalar"
      2. any pattern in cfg.embed_patterns is a
         substring of `name`                           -> "embed"
      3. otherwise                                     -> "matrix"

    Note the order: a 1-D parameter is "scalar" even if its name matches an
    embed pattern. (A hypothetical `lm_head.bias` is a vector; a matrix
    method and weight decay are both wrong for it regardless of where it sits.)
    """
    if param.ndim <= 1:
        return "scalar"
    elif any(emb_pattern in name for emb_pattern in cfg.embed_patterns):
        return "embed"
    else:
        return "matrix"


def build_param_groups(model: nn.Module, cfg: GroupConfig) -> list[dict[str, Any]]:
    """Split `model`'s trainable parameters into optimizer param groups.

    Returns a list of dicts, in the order matrix, embed, scalar, omitting any
    group that ended up empty. Each dict has exactly these keys:

        "params":       list[nn.Parameter], in named_parameters() order
        "base_lr":      float, from cfg (matrix_lr / embed_lr / scalar_lr)
        "weight_decay": float, cfg.weight_decay for matrix and embed, 0.0 for scalar
        "kind":         the Kind string, for logging and for routing to the
                        right optimizer
        "names":        list[str], parallel to "params"; purely for logging,
                        but it makes a misrouted tensor obvious in a run log

    Calls cfg.validate() first.
    """
    cfg.validate()
    buckets: dict[Kind, list[tuple[str, nn.Parameter]]] = {k: [] for k in _ORDER}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        buckets[classify(name, param, cfg)].append((name, param))
    
    base_lr = {"matrix": cfg.matrix_lr, "embed": cfg.embed_lr, "scalar": cfg.scalar_lr}
    weight_decay = {"matrix": cfg.weight_decay, "embed": cfg.weight_decay, "scalar": 0.0}

    groups: list[dict[str, Any]] = []
    for kind in _ORDER:
        items = buckets[kind]
        if not items:
            continue
        groups.append(
            {
                "params": [param for _, param in items],
                "names": [name for name, _ in items],
                "base_lr": float(base_lr[kind]),
                "weight_decay": float(weight_decay[kind]),
                "kind": kind,
            }
        )
    return groups


def split_by_optimizer(
    groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition groups into (matrix_groups, adamw_groups).

    The matrix optimizer gets the "matrix" group; AdamW gets "embed" and
    "scalar". Returns the same dict objects, not copies, so both optimizers
    and any logging see one source of truth. Either list may be empty.
    """
    matrix_groups = [g for g in groups if g["kind"] == "matrix"]
    adamw_groups = [g for g in groups if g["kind"] in ("embed", "scalar")]
    return matrix_groups, adamw_groups


def group_summary(groups: list[dict[str, Any]]) -> str:
    """One line per group: kind, tensor count, parameter count, LR, decay.

    Printed once at startup. This is the cheapest possible guard against a
    misrouted tensor: if `matrix` holds 14 tensors in a 2-layer model you can
    see it, where a silently doubled LR on wq you cannot.

    Format (one line per group, no trailing newline):
        matrix: 14 tensors, 12,288 params, base_lr 2.0e-02, wd 0.1
    """
    lines = []
    for g in groups:
        n_tensors = len(g["params"])
        n_params = sum(p.numel() for p in g["params"])
        lines.append(
            f"{g['kind']}: {n_tensors} tensors, {n_params:,} params, "
            f"base_lr {g['base_lr']:.1e}, wd {g['weight_decay']:g}"
        )
    return "\n".join(lines)