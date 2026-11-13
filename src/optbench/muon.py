"""Muon: momentum + Newton-Schulz orthogonalization for matrix parameters.

TASK 3a — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_muon.py -v` until all tests pass.

The idea in one paragraph. SGD's update is the gradient; Adam's is the
gradient normalized per-coordinate. Muon's is the gradient normalized per
SINGULAR DIRECTION: take the (momentum-averaged) gradient matrix G with
SVD U S Vᵀ and step along U Vᵀ — every singular direction gets step size 1,
however lopsided S was. Gradients of dense layers are dominated by a few
large singular values, so most directions barely move under SGD/Adam; Muon
moves them all equally. An exact SVD per step is too slow, so Muon
approximates U Vᵀ with a few matmuls (Newton-Schulz iteration) — GPU-friendly
and good enough: we need "roughly orthogonal", not machine precision.

Why this is THE optimizer this project studies: orthogonalizing is only
meaningful for 2-D hidden weights (see param_groups.py for why embeddings
and 1-D tensors are excluded), its cost is extra matmuls per step (in step
TIME, never in MFU's numerator — see throughput.py), and whether its
advantage survives scale is exactly the question of the study.

Newton-Schulz, concretely. We want the polynomial iteration
    X_{k+1} = a·X_k + b·(X_k X_kᵀ)·X_k + c·(X_k X_kᵀ)²·X_k
to drive every singular value of X toward 1 while leaving the singular
VECTORS untouched (odd polynomials in X do exactly that: they act on the
singular values alone, s ← a·s + b·s³ + c·s⁵). The tuned coefficients below
are Keller Jordan's: they maximize the slope at 0 so small singular values
rise fast, at the cost of landing in ~[0.7, 1.2] instead of exactly 1 —
which is fine for an optimizer step and why ns_steps=5 suffices.

Conventions (the tests rely on these):
  * The iteration needs X Xᵀ to be the SMALL Gram matrix: when the input is
    tall (rows > cols), work on the transpose and transpose back at the end.
  * Normalize X to unit Frobenius norm first (plus eps for the zero matrix):
    the polynomial only converges for singular values in (0, ~1.12], and
    ||X||_F = 1 guarantees every s <= 1.
  * A consequence worth internalizing: the result is SCALE-INVARIANT —
    newton_schulz(c·G) == newton_schulz(G) for any c > 0. Muon's step size
    is set entirely by lr, never by the gradient's magnitude.
  * Everything stays in the input's dtype. (Production Muon runs the
    iteration in bfloat16 for speed; on this Mac we test in float32.)
  * Muon is an `torch.optim.Optimizer`: param groups, `self.state` keyed by
    parameter, `state_dict()`/`load_state_dict()` for free — which is what
    makes our checkpoint module work on it unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

# Keller Jordan's quintic coefficients: max slope at 0 subject to
# a·s + b·s³ + c·s⁵ staying in [~0.7, ~1.2] on s ∈ (0, 1].
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize G: return ~U Vᵀ where G = U S Vᵀ.

    Steps:
      1. Raise ValueError unless G.ndim == 2 and steps >= 1.
      2. X = G / (||G||_F + eps).
      3. If X is tall (rows > cols), transpose it — X @ X.mT must be the
         small side's Gram matrix, both for cost and convergence.
      4. Iterate `steps` times with (a, b, c) = _NS_COEFFS:
             A = X @ X.mT
             B = b·A + c·(A @ A)
             X = a·X + B @ X
      5. Transpose back if you transposed, and return X.

    The zero matrix comes back as zeros (eps prevents 0/0) — relevant
    because a freshly-zeroed momentum buffer can produce exactly that.
    """
    if not G.ndim == 2:
        raise ValueError(f"Muon only works for G.ndim {G.ndim} == 2!")
    if not steps >= 1:
        raise ValueError(f"GS Steps must be >= 1!")
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(0, 1)
    (a, b, c) = _NS_COEFFS
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(0, 1)
    return X

class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2-D hidden-weight matrices.

    Per parameter p with gradient g, one step does:

        buf = momentum * buf + g                     # momentum accumulation
        eff = g + momentum * buf    if nesterov      # Nesterov lookahead
              buf                   otherwise
        O   = newton_schulz(eff, ns_steps)           # orthogonalize
        p  *= (1 - lr * weight_decay)                # decoupled decay (AdamW-style)
        p  -= lr * scale * O
          where scale = sqrt(max(1, rows/cols))      # tall matrices get a boost:
                                                     # UVᵀ has unit spectral norm, but
                                                     # its ROWS shrink as 1/sqrt(rows);
                                                     # the boost keeps per-row update
                                                     # RMS comparable across shapes

    Constructor:
      * Signature: (params, lr=0.02, momentum=0.95, nesterov=True,
        ns_steps=5, weight_decay=0.0).
      * Validate (ValueError): lr > 0, 0 <= momentum < 1, ns_steps >= 1,
        weight_decay >= 0.
      * Build `defaults` from all five hyperparameters and call
        `super().__init__(params, defaults)`. Param groups built by
        `build_param_groups` carry their own "weight_decay" (overrides the
        default) and "base_lr" plus "names" (extra keys ride along — the
        checkpoint fingerprint depends on "names" surviving).
      * AFTER super().__init__, walk every group and raise ValueError if any
        parameter has ndim != 2 — a misrouted embedding or norm gain must
        fail at construction, not orthogonalize garbage for three days.
        Name the group index and the offending ndim.

    step():
      * Decorate with @torch.no_grad(); accept the standard optional
        `closure` (call it under torch.enable_grad(), return its loss).
      * Read lr/momentum/nesterov/ns_steps/weight_decay from EACH group —
        never from self.defaults; the schedule mutates group["lr"].
      * Skip parameters whose .grad is None.
      * The momentum buffer lives in self.state[p]["momentum_buffer"],
        created as zeros_like(g) on first use. Update it IN PLACE
        (mul_/add_) — it must be the same tensor across steps so
        state_dict() checkpoints it.
      * The Nesterov `eff` must NOT alias g or buf (g.add(buf, alpha=m) is
        out of place; g.add_ would corrupt p.grad, which the spectral
        diagnostics read after the step).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError
