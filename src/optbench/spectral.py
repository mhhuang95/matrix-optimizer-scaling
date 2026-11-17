"""Per-layer spectral diagnostics: the study's novelty lives here.

TASK 4 — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_spectral.py -v` until all tests pass.

The research question (Q3 of the plan): matrix optimizers' speedup over a
tuned AdamW shrinks with scale — can per-layer spectral statistics of the
gradients explain and predict where the remaining speedup lives? That makes
these numbers first-class training telemetry, logged for EVERY arm (the
AdamW baseline included — the comparison across arms is the whole point),
not a post-hoc analysis script.

Why these statistics:

  * Muon orthogonalizes the update: after Newton-Schulz every singular
    value is ~1, i.e. the update is spectrally FLAT. The more non-flat the
    raw gradient/momentum spectrum, the more orthogonalization changes the
    step — so "how far from flat" is the natural candidate predictor of
    where Muon buys something.
  * nuclear/Frobenius ratio: ||A||_* / ||A||_F lies in [1, sqrt(r)] — it is
    1 when one singular value dominates (rank-1, maximally non-flat) and
    sqrt(r) when all r singular values are equal (flat). A normalized
    flatness dial, scale-invariant by construction.
  * stable rank: ||A||_F^2 / ||A||_2^2, also in [1, r], also
    scale-invariant. Cheap, robust (no log), standard in the matrix
    concentration literature.
  * effective rank (Roy & Vetterli): exp(entropy of the normalized singular
    value distribution p_i = s_i / sum(s)). Equals r for a flat spectrum,
    1 for rank-1; unlike stable rank it weighs the whole tail, not just
    the top.

Conventions (the tests rely on these):

  * spectral_stats computes on `mat.detach().float()`: diagnostics must
    never build graph state, and SVD in bf16 is either unsupported or
    garbage — promote first, always.
  * torch.linalg.svdvals, never full torch.linalg.svd: we need singular
    values only, and skipping U/V is both cheaper and numerically safer.
  * The zero matrix returns ALL stats 0.0 — ranks are undefined there, and
    0.0 is a loud sentinel (every legal nonzero matrix has ranks >= 1).
    No NaN ever leaves this module: a NaN logged to W&B poisons every
    downstream aggregation silently.
  * Entropy uses the 0 * log(0) = 0 convention (drop zero-mass entries
    before taking logs — log(0) is -inf and 0 * -inf is NaN in fp land).
  * Cost discipline: these run on matrix params only, every N steps, from
    the training loop. An SVD of d x 4d at our scales is milliseconds; the
    API still never computes anything the caller did not ask for.
"""

from __future__ import annotations

from typing import Literal

import torch

Source = Literal["grad", "param"]

STAT_KEYS = (
    "frobenius_norm",
    "spectral_norm",
    "nuclear_norm",
    "stable_rank",
    "effective_rank",
    "nuclear_frobenius_ratio",
)


def spectral_stats(mat: torch.Tensor) -> dict[str, float]:
    """Return the six STAT_KEYS statistics of a 2-D tensor, as plain floats.

    Steps:
      1. Raise ValueError unless mat.ndim == 2 (a diagnostics module that
         silently flattens a conv kernel or a bias would lie in the logs).
      2. s = singular values of mat.detach().float().
      3. frobenius_norm = sqrt(sum s_i^2)   (== ||mat||_F)
         spectral_norm  = max s_i
         nuclear_norm   = sum s_i
      4. Zero matrix (spectral_norm == 0): return every stat 0.0.
      5. stable_rank = frobenius^2 / spectral^2
         effective_rank = exp(-sum p_i ln p_i), p_i = s_i / sum(s),
         zero-mass entries dropped first
         nuclear_frobenius_ratio = nuclear / frobenius
      6. Every value is a Python float (never a 0-d tensor — these go
         straight into logging dicts and JSON).
    """
    if not mat.ndim == 2:
        raise ValueError(f"tensor ndim {mat.ndim} must be 2.")
    s = torch.linalg.svdvals(mat.detach().float())
    frobenius_norm = torch.sqrt(torch.sum(s**2))
    spectral_norm = s.max()
    nuclear_norm = torch.sum(s)
    if spectral_norm == 0:
        return {key: 0.0 for key in STAT_KEYS}
    stable_rank = (frobenius_norm / spectral_norm)**2
    p = s / nuclear_norm
    p = p[p > 0]
    effective_rank = torch.exp(- torch.sum(p * torch.log(p)))
    nuclear_frobenius_ratio = nuclear_norm / frobenius_norm
    return {
        "frobenius_norm": float(frobenius_norm),
        "spectral_norm": float(spectral_norm),
        "nuclear_norm": float(nuclear_norm),
        "stable_rank": float(stable_rank),
        "effective_rank": float(effective_rank),
        "nuclear_frobenius_ratio": float(nuclear_frobenius_ratio),
    }


def collect_spectral_stats(
    optimizers: list[torch.optim.Optimizer],
    source: Source = "grad",
) -> dict[str, dict[str, float]]:
    """Walk the matrix groups of a run's optimizers; stats per param name.

    Steps:
      1. Raise ValueError unless source is "grad" or "param".
      2. For every optimizer, every param_group with kind == "matrix"
         (embed/scalar groups are never diagnosed — 1-D tensors have no
         spectrum and the embedding rows are a different object of study),
         walk zip(group["names"], group["params"]).
      3. source "grad": the tensor is p.grad; a param whose grad is None is
         SKIPPED, not an error — before the first backward, or for frozen
         params, there is simply nothing to measure.
         source "param": the tensor is p itself (weight spectra — how far
         training has moved the weights from init is its own diagnostic).
      4. Returns {name: spectral_stats(tensor)} in traversal order. Names
         come from the groups — the same names the checkpoint fingerprint
         uses, so a W&B panel and a checkpoint row agree forever.
    """
    raise NotImplementedError
