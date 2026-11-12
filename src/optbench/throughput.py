"""Throughput and MFU: the number that tells you whether a run is worth its GPUs.

TASK 2f — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_throughput.py -v` until all tests pass.

MFU (model FLOPs utilization) is the fraction of the hardware's peak FLOP/s
that your training loop actually turns into model FLOPs:

    MFU = (tokens/sec x FLOPs/token) / (n_gpus x peak FLOP/s per GPU)

Two reasons this module matters for THIS project specifically:

  * **The grant is a fixed number of GPU-hours.** A run at 25% MFU costs twice
    as much as the same run at 50%. The scaling study has a token budget per
    model size; MFU is what converts that budget into a dollar figure you can
    put in a proposal and a wall-clock figure you can put in a schedule.

  * **Matrix optimizers are the thing being measured.** Muon's
    Newton-Schulz orthogonalization and Shampoo's preconditioner updates are
    real FLOPs that do NOT appear in the numerator: MFU counts MODEL FLOPs
    only. So "Muon reaches the target loss in 20% fewer steps" and "Muon
    reaches it in 20% less wall-clock" are different claims, and the honest
    version of this study has to report both. A per-step time that includes
    optimizer overhead, divided into model FLOPs, is exactly the quantity that
    makes the two differ — which is why the meter times whole steps and never
    just the forward/backward.

Conventions (the tests rely on these):
  * FLOPs/token is the standard 6N + attention correction, counting forward
    and backward: 2 FLOPs per MAC, x3 for fwd+bwd. N here is MATMUL
    parameters, not `model.num_params()` — see `matmul_params`.
  * We do NOT halve the attention term for causality. The matmul is dense;
    the mask is applied to its output. Halving it is a common way to report
    an MFU that is too high.
  * Times come from an injected clock, never `time.time()` read inline.
    That is what makes the meter testable, and on CUDA it is also what lets a
    caller pass a clock that calls `torch.cuda.synchronize()` first — without
    a sync, the CPU races ahead of the GPU and you time the queueing, not the
    work.
  * Every rate is None until there is a real measurement window. Returning
    0.0 or inf for "I do not know yet" puts a lie in the log.
"""

from __future__ import annotations

import time
from typing import Callable

from optbench.model import ModelConfig

# Dense bf16/fp16 peak FLOP/s per GPU, WITHOUT sparsity. Vendor slides quote
# the 2x sparsity number; using it halves every MFU you report.
PEAK_BF16_FLOPS: dict[str, float] = {
    "h100-sxm": 989e12,
    "a100": 312e12,
}


def matmul_params(cfg: ModelConfig) -> int:
    """Parameters that take part in a matmul, per forward pass.

    This is the N in "6N FLOPs per token". It is NOT `model.num_params()`:

      * The token embedding is a LOOKUP, not a matmul — excluded.
      * The lm_head IS a matmul (D x V) — included, even when
        `tie_embeddings` makes it the same tensor as the embedding. Tying
        saves memory, not FLOPs.
      * RMSNorm weights are elementwise — excluded (and negligible).

    Per layer, counting weight matrices only (no biases anywhere):
        wq: D x (H*Dh)      wk: D x (KV*Dh)     wv: D x (KV*Dh)
        wo: (H*Dh) x D
        w1: D x F           w3: D x F           w2: F x D

    Note GQA shrinks this: with KV < H, wk and wv are smaller, so a GQA model
    does fewer FLOPs per token than the MHA model with the same d_model.
    """
    attn = cfg.d_model * cfg.n_heads * cfg.head_dim * 2 + cfg.d_model * cfg.n_kv_heads * cfg.head_dim * 2
    mlp = cfg.d_model * cfg.d_ff * 3
    return cfg.n_layers * (attn + mlp) + cfg.d_model * cfg.vocab_size


def flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """Forward+backward FLOPs for one token at sequence length `seq_len`.

        6 * matmul_params(cfg)  +  12 * n_layers * n_heads * head_dim * seq_len

    Where the two terms come from:
      * `6N`: each parameter does one multiply-accumulate per token (2 FLOPs)
        in the forward pass, and the backward pass costs twice the forward
        (one gradient w.r.t. the input, one w.r.t. the weight). 2 * 3 = 6.
      * The attention term is the two batched matmuls that have no parameters:
        scores = Q @ K^T and out = probs @ V. Per layer, per token, each is
        2 * seq_len * n_heads * head_dim FLOPs forward, so 4 * ... forward and
        12 * ... with backward.

    The attention term grows with seq_len while 6N does not, which is why MFU
    at a fixed model size drifts as you change the context length — and why a
    scaling study must hold seq_len fixed or report it alongside every number.

    Raise ValueError unless 1 <= seq_len <= cfg.max_seq_len.
    """
    if not 1 <= seq_len <= cfg.max_seq_len:
        raise ValueError(f"seq_len {seq_len} must be in the range [1, {cfg.max_seq_len}]!")
    matmul_flops = 6 * matmul_params(cfg)
    attn_flops = 12 * cfg.n_layers * cfg.n_heads * cfg.head_dim * seq_len
    return matmul_flops + attn_flops


def mfu(
    tokens_per_sec: float,
    flops_per_tok: int,
    n_gpus: int,
    peak_flops_per_gpu: float,
) -> float:
    """Model FLOPs utilization as a fraction (0.45, not 45).

        tokens_per_sec * flops_per_tok / (n_gpus * peak_flops_per_gpu)

    `tokens_per_sec` is the whole job's rate across all ranks, not per-GPU.

    Raise ValueError if n_gpus < 1 or peak_flops_per_gpu <= 0.

    The result is NOT clamped to 1.0. An MFU above 1 means the inputs are
    wrong — a sparsity peak, a halved attention term, or tokens counted twice
    under DDP — and clamping would hide exactly the bug worth seeing.
    """
    raise NotImplementedError


def step_was_skipped(scale_before: float, scale_after: float) -> bool:
    """Did GradScaler skip the step whose `update()` just ran?

    Call `scaler.get_scale()` immediately before `scaler.update()` and again
    immediately after. The scaler halves the scale when it finds an inf/nan
    in the unscaled gradients, having already discarded that step's update; it
    leaves the scale alone or multiplies it up otherwise. So: skipped exactly
    when the scale went DOWN.

    Why count these at all: a skipped step still consumes a step of the LR
    schedule while the weights do not move. A run that silently skips 5% of
    its steps has quietly shortened its own training, and the loss curve shows
    it as "this optimizer is slightly worse".
    """
    raise NotImplementedError


class ThroughputMeter:
    """Accumulates step timings into tokens/sec, skipping a warmup window.

    Warmup exists because the first steps are not representative: CUDA context
    creation, autotuning for the chosen matmul shapes, allocator growth and
    (later) torch.compile all land there. Including them drags the reported
    rate down by a chunk that has nothing to do with the optimizer.

    The measurement window runs from the END of warmup to the most recent
    `update`. That is deliberate: it means n warmup steps consume n steps and
    contribute no time, rather than leaving a gap that inflates the rate.

    Attributes (public, the tests read them):
        warmup_steps:  int, how many leading steps are excluded
        steps:         int, every update() ever, warmup included
        skipped_steps: int, updates passed with skipped=True, warmup included
    """

    def __init__(
        self,
        warmup_steps: int = 0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Raise ValueError if warmup_steps < 0.

        With warmup_steps == 0 the window opens NOW, at construction, so the
        first step's own time is counted. With warmup_steps > 0 the window
        opens when the warmup-th update arrives.

        `clock` returns seconds as a float. With warmup_steps == 0 it is read
        once here (the window's start); every other read happens in `update`.
        Default `time.perf_counter` — a monotonic clock, unlike `time.time()`,
        which can step backwards when ntp adjusts it mid-run.
        """
        raise NotImplementedError

    def update(self, n_tokens: int, skipped: bool = False) -> None:
        """Record one completed optimizer step.

        `n_tokens` is that step's target-token count — `StepStats.n_tokens`
        from `train_step`, summed over all ranks if you are logging the job
        rather than the rank.

        Always increments `steps` (and `skipped_steps` when skipped), even
        during warmup: those counters describe the run, not the window.

        Tokens from a skipped step count toward `tokens_per_sec` — the FLOPs
        were really spent — but not toward `goodput_tokens_per_sec`.

        Raise ValueError if n_tokens < 0.
        """
        raise NotImplementedError

    @property
    def tokens_per_sec(self) -> float | None:
        """Tokens per second over the measurement window, or None.

        None when no update has landed after warmup, or when the window has
        zero width (a clock that has not moved — common on a fast machine with
        a coarse clock, and a division by zero if you do not check).
        """
        raise NotImplementedError

    @property
    def goodput_tokens_per_sec(self) -> float | None:
        """Like `tokens_per_sec`, but counting only steps that actually
        updated the weights. Same None conditions.

        The gap between the two is the cost of fp16 overflow. If it is more
        than a percent or so, that is an argument for bf16, not for a bigger
        scaler patience.
        """
        raise NotImplementedError
