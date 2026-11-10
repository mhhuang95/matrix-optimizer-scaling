"""One optimizer step: accumulation, autocast, clipping, sync-free stats.

TASK 2d — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_train_step.py -v` until all tests pass.

This is the function every number in the study comes out of, so the two
things it must get exactly right are both invisible when wrong:

  * **Accumulation normalization.** Splitting a batch of 8 into 4 micro-steps
    of 2 must produce the same gradient as one batch of 8. Each micro-batch's
    loss is already a mean over ITS tokens, so summing four of them gives 4x
    the gradient: each must be divided by accum_steps before backward. The
    late-2024 HF Trainer / Unsloth bug was exactly this, and the symptom was
    not a crash but a slightly worse loss curve at larger accum_steps.
    (Dividing by accum_steps is only correct when every micro-batch has the
    same number of target tokens. Our loader guarantees that — fixed-length
    windows — so this file asserts it rather than silently weighting wrong.)

  * **Global-norm clipping across BOTH optimizers.** Muon holds the matrices
    and AdamW holds everything else, so clipping each optimizer's parameters
    separately computes two norms over two halves. That is a different
    algorithm from clipping the one global norm, and it changes with the
    matrix/scalar split — i.e. with model scale. The norm must span the union.

Conventions (the tests rely on these):
  * `zero_grad(set_to_none=True)` happens at the START of the step, so when
    `train_step` returns, `p.grad` still holds this step's gradients. The
    spectral diagnostics read them.
  * The returned grad norm is the norm BEFORE clipping. That is the quantity
    worth logging: once clipping bites, the post-clip norm is just max_norm.
  * Exactly one `.item()` per step, at the very end. Every `.item()` inside
    the micro-batch loop is a CPU-GPU sync that stalls the pipeline; losses
    accumulate as a tensor on-device instead.
  * fp16 needs a GradScaler; bf16 must not have one (it has fp32's exponent
    range, so there is nothing to rescale, and a scaler would only add a sync).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class StepStats:
    """What one optimizer step reports. All plain Python scalars."""

    loss: float  # mean over the step's micro-batches
    grad_norm: float  # global 2-norm BEFORE clipping
    n_tokens: int  # target tokens in the whole step


@dataclass
class TrainStepConfig:
    accum_steps: int = 1
    grad_clip: float = 1.0  # 0.0 disables clipping (the norm is still reported)
    amp_dtype: torch.dtype | None = None  # None = fp32, else bfloat16 or float16
    device_type: str = "cpu"  # "cpu" or "cuda", for torch.autocast

    def validate(self) -> None:
        """Raise ValueError unless all of these hold:
        * accum_steps >= 1
        * grad_clip >= 0
        * amp_dtype is None, torch.bfloat16 or torch.float16
        * device_type is "cpu" or "cuda"
        """
        if not self.accum_steps >= 1:
            raise ValueError(f"accum_steps {self.accum_steps} must be >= 1!")
        if not self.grad_clip >= 0:
            raise ValueError(f"grad_clip {self.grad_clip} must be non-negative!")
        if not self.amp_dtype in [None, torch.bfloat16, torch.float16]:
            raise ValueError(f"amp_dtype must be selected from [None, torch.bfloat16, torch.float16].")
        if not self.device_type in ["cpu", "cuda"]:
            raise ValueError(f"device_type {self.device_type} must be selected from ['cpu', 'cuda'].")

    def autocast(self) -> ContextManager[Any]:
        """The context manager to run the forward pass under.

        fp32 (amp_dtype is None) -> `contextlib.nullcontext()`; otherwise
        `torch.autocast(device_type=..., dtype=...)`.

        Why a context manager rather than casting the model: autocast keeps
        the master weights in fp32 and casts only the ops that benefit
        (matmuls, convs), leaving reductions, softmax and the loss in fp32.
        Casting the model itself would put the optimizer state in low
        precision too, which is where mixed-precision training falls apart.
        """
        if self.amp_dtype is None:
            return nullcontext()
        else:
            return torch.autocast(device_type=self.device_type, dtype=torch.float16)
        


def all_params(optimizers: Sequence[torch.optim.Optimizer]) -> list[nn.Parameter]:
    """Every parameter held by every optimizer, in group order.

    This is the set the global norm must span. Keep it a function so the
    clipping call site cannot accidentally be handed one optimizer's share.
    """
    all_params_list = []
    for optimizer in optimizers:
      for group in optimizer.param_groups:
          all_params_list += group["params"]
    return all_params_list


def clip_global_norm(params: Iterable[nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip gradients to a global 2-norm, returning the norm BEFORE clipping.

    Equivalent to `torch.nn.utils.clip_grad_norm_`, written out because the
    call site must own which parameters are in the norm:

        total = sqrt( sum over params of ||p.grad||^2 )
        if max_norm > 0 and total > max_norm:
            scale every p.grad by max_norm / (total + 1e-6)

    Parameters whose grad is None are skipped entirely (not treated as zero —
    they may simply not have been used this step). With no gradients at all,
    return a zero tensor.

    `max_norm <= 0` means "report the norm, clip nothing".

    Returns a 0-dim tensor, not a float: the caller decides when to sync.
    """
    total = torch.sqrt(torch.stack([(p.grad**2).sum() for p in params if p.grad is not None]).sum())
    if max_norm > 0 and total > max_norm:
        scale_factor = max_norm / (total + 1e-6)
        for p in params:
            p.grad.mul(scale_factor)
    return total


def train_step(
    model: nn.Module,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    optimizers: Sequence[torch.optim.Optimizer],
    cfg: TrainStepConfig,
    scaler: Any | None = None,
) -> StepStats:
    """Run one full optimizer step over `cfg.accum_steps` micro-batches.

    Order of operations (the tests check several of these):
      1. cfg.validate(); check len(micro_batches) == cfg.accum_steps; check
         every micro-batch has the same number of target tokens. ValueError
         otherwise.
      2. Reject the impossible AMP combinations: float16 without a scaler,
         or any scaler alongside bfloat16 / fp32.
      3. `opt.zero_grad(set_to_none=True)` for every optimizer.
      4. For each (x, y): forward under `cfg.autocast()`, divide the loss by
         accum_steps, backward (through `scaler.scale(loss)` if scaling), and
         add the detached loss to a running tensor. No `.item()` in here.
      5. If scaling: `scaler.unscale_(opt)` for every optimizer, BEFORE
         clipping — the gradients still carry the scale factor and clipping a
         scaled gradient clips at the wrong threshold.
      6. Clip over `all_params(optimizers)`, keeping the pre-clip norm.
      7. Step every optimizer: `scaler.step(opt)` if scaling else `opt.step()`,
         then `scaler.update()` once at the end.
      8. Return StepStats, converting to Python scalars here and nowhere else.

    `model(x, y)` returns `(logits, loss)`; the loss is a mean over the
    micro-batch's tokens.

    Note on DDP: a real multi-GPU step wraps micro-batches 0..n-2 in
    `model.no_sync()` so the gradient all-reduce fires once, on the last one.
    Out of scope here — the wrapper lands in the DDP module.
    """
    
    
