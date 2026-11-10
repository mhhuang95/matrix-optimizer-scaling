"""Tests for optbench.train_step.

These test behaviour and properties, not one specific implementation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from optbench.model import ModelConfig, Transformer
from optbench.param_groups import GroupConfig, build_param_groups, split_by_optimizer
from optbench.train_step import (
    StepStats,
    TrainStepConfig,
    all_params,
    clip_global_norm,
    train_step,
)

V, T = 64, 8


def make_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(
        ModelConfig(
            vocab_size=V, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=32, max_seq_len=T
        )
    )


def make_optimizers(model: nn.Module, lr: float = 0.0) -> list[torch.optim.Optimizer]:
    """Two optimizers over disjoint parameter sets, as the real study will run.

    lr=0 leaves parameters untouched, so gradients can be compared across runs.
    """
    groups = build_param_groups(model, GroupConfig())
    matrix, adamw = split_by_optimizer(groups)
    return [
        torch.optim.SGD([{**g, "lr": lr} for g in matrix]),
        torch.optim.AdamW([{**g, "lr": lr} for g in adamw]),
    ]


def batch(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, V, (n, T + 1), generator=g)
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()


def split(x: torch.Tensor, y: torch.Tensor, n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return list(zip(x.chunk(n), y.chunk(n)))


def grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


# ---------------------------------------------------------------------------
# TrainStepConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(accum_steps=0),
        dict(accum_steps=-1),
        dict(grad_clip=-0.5),
        dict(amp_dtype=torch.float64),
        dict(amp_dtype=torch.int8),
        dict(device_type="tpu"),
    ],
)
def test_invalid_configs_raise(kw):
    with pytest.raises(ValueError):
        TrainStepConfig(**kw).validate()


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(accum_steps=4, grad_clip=0.0),
        dict(amp_dtype=torch.bfloat16),
        dict(amp_dtype=torch.float16, device_type="cuda"),
    ],
)
def test_valid_configs_pass(kw):
    TrainStepConfig(**kw).validate()


def test_fp32_autocast_is_a_noop():
    cfg = TrainStepConfig()
    with cfg.autocast():
        x = torch.zeros(2, 2) @ torch.zeros(2, 2)
    assert x.dtype == torch.float32


def test_bf16_autocast_lowers_matmul_precision():
    cfg = TrainStepConfig(amp_dtype=torch.bfloat16, device_type="cpu")
    with cfg.autocast():
        x = torch.zeros(2, 2) @ torch.zeros(2, 2)
    assert x.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# all_params
# ---------------------------------------------------------------------------


def test_all_params_spans_every_optimizer():
    model = make_model()
    opts = make_optimizers(model)
    collected = all_params(opts)
    assert {id(p) for p in collected} == {id(p) for p in model.parameters()}
    assert len(collected) == len(list(model.parameters())), "no duplicates"


def test_all_params_of_nothing_is_empty():
    assert all_params([]) == []


# ---------------------------------------------------------------------------
# clip_global_norm
# ---------------------------------------------------------------------------


def _params_with_grads(scale: float = 1.0, seed: int = 0) -> list[nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for shape in [(4, 4), (8,), (3, 5)]:
        p = nn.Parameter(torch.zeros(shape))
        p.grad = torch.randn(shape, generator=g) * scale
        out.append(p)
    return out


def test_clip_matches_torch_reference():
    mine, theirs = _params_with_grads(), _params_with_grads()
    got = clip_global_norm(mine, 0.5)
    want = torch.nn.utils.clip_grad_norm_(theirs, 0.5)
    assert torch.allclose(got, want)
    for a, b in zip(mine, theirs):
        assert torch.allclose(a.grad, b.grad)


def test_clip_returns_a_tensor_not_a_float():
    assert isinstance(clip_global_norm(_params_with_grads(), 1.0), torch.Tensor)


def test_clip_caps_the_norm():
    params = _params_with_grads(scale=10.0)
    before = clip_global_norm(params, 1.0)
    assert before > 1.0
    after = torch.sqrt(sum((p.grad**2).sum() for p in params))
    assert after == pytest.approx(1.0, rel=1e-5)


def test_clip_leaves_small_gradients_alone():
    params = _params_with_grads(scale=1e-3)
    before = [p.grad.clone() for p in params]
    norm = clip_global_norm(params, 1.0)
    assert norm < 1.0
    for p, b in zip(params, before):
        assert torch.equal(p.grad, b)


def test_zero_max_norm_reports_without_clipping():
    params = _params_with_grads(scale=10.0)
    before = [p.grad.clone() for p in params]
    norm = clip_global_norm(params, 0.0)
    assert norm > 1.0
    for p, b in zip(params, before):
        assert torch.equal(p.grad, b)


def test_clip_skips_parameters_without_gradients():
    params = _params_with_grads()
    params.append(nn.Parameter(torch.zeros(6)))  # grad is None
    expected = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    assert clip_global_norm(params, 0.0) == pytest.approx(expected.item(), rel=1e-6)


def test_clip_of_nothing_is_zero():
    assert clip_global_norm([], 1.0) == pytest.approx(0.0)
    assert clip_global_norm([nn.Parameter(torch.zeros(3))], 1.0) == pytest.approx(0.0)


def test_global_norm_is_not_the_two_halves_clipped_separately():
    """The reason clipping must span both optimizers.

    Two halves each of norm 3 have a global norm of sqrt(18) = 4.24. Clipping
    the union to 4 scales everything by 0.94; clipping each half to 4 does
    nothing at all. Those are different algorithms.
    """
    left, right = _params_with_grads(seed=1), _params_with_grads(seed=2)
    for group in (left, right):
        n = torch.sqrt(sum((p.grad**2).sum() for p in group))
        for p in group:
            p.grad *= 3.0 / n

    union = left + right
    assert clip_global_norm(union, 0.0) == pytest.approx(18**0.5, rel=1e-5)
    clip_global_norm(union, 4.0)
    after = torch.sqrt(sum((p.grad**2).sum() for p in union))
    assert after == pytest.approx(4.0, rel=1e-5)
    # Each half alone would have been under the threshold, so untouched.
    assert clip_global_norm(left, 4.0) < 4.0


# ---------------------------------------------------------------------------
# train_step: validation
# ---------------------------------------------------------------------------


def test_micro_batch_count_must_match_accum_steps():
    model, cfg = make_model(), TrainStepConfig(accum_steps=4)
    x, y = batch(8)
    with pytest.raises(ValueError):
        train_step(model, split(x, y, 2), make_optimizers(model), cfg)


def test_micro_batches_must_have_equal_token_counts():
    """Mean-of-means is only the true mean when the batches are the same size."""
    model, cfg = make_model(), TrainStepConfig(accum_steps=2)
    x, y = batch(6)
    uneven = [(x[:4], y[:4]), (x[4:], y[4:])]
    with pytest.raises(ValueError):
        train_step(model, uneven, make_optimizers(model), cfg)


def test_fp16_without_a_scaler_is_rejected():
    model = make_model()
    x, y = batch(4)
    cfg = TrainStepConfig(amp_dtype=torch.float16, device_type="cuda")
    with pytest.raises(ValueError):
        train_step(model, [(x, y)], make_optimizers(model), cfg)


@pytest.mark.parametrize("amp", [None, torch.bfloat16])
def test_a_scaler_without_fp16_is_rejected(amp):
    model = make_model()
    x, y = batch(4)
    cfg = TrainStepConfig(amp_dtype=amp)
    with pytest.raises(ValueError):
        train_step(model, [(x, y)], make_optimizers(model), cfg, scaler=_RecordingScaler())


# ---------------------------------------------------------------------------
# train_step: the accumulation invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accum", [1, 2, 4])
def test_accumulation_matches_one_big_batch(accum):
    x, y = batch(8, seed=7)

    ref = make_model()
    train_step(ref, [(x, y)], make_optimizers(ref), TrainStepConfig(accum_steps=1, grad_clip=0.0))
    want = grads(ref)

    got_model = make_model()
    train_step(
        got_model,
        split(x, y, accum),
        make_optimizers(got_model),
        TrainStepConfig(accum_steps=accum, grad_clip=0.0),
    )
    got = grads(got_model)

    assert set(got) == set(want)
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=1e-4, atol=1e-6)


def test_reported_loss_matches_the_full_batch_loss():
    x, y = batch(8, seed=3)
    model = make_model()
    with torch.no_grad():
        _, full = model(x, y)

    stats = train_step(
        model, split(x, y, 4), make_optimizers(model), TrainStepConfig(accum_steps=4)
    )
    assert stats.loss == pytest.approx(full.item(), rel=1e-5)


def test_gradients_do_not_leak_between_steps():
    """zero_grad at the start of the step, not the end."""
    x, y = batch(4, seed=5)
    model = make_model()
    opts = make_optimizers(model, lr=0.0)  # parameters never move
    cfg = TrainStepConfig(accum_steps=1, grad_clip=0.0)

    first = train_step(model, [(x, y)], opts, cfg)
    g1 = grads(model)
    second = train_step(model, [(x, y)], opts, cfg)
    g2 = grads(model)

    assert second.grad_norm == pytest.approx(first.grad_norm, rel=1e-5)
    for name in g1:
        torch.testing.assert_close(g2[name], g1[name])


def test_gradients_are_still_available_after_the_step():
    """The spectral diagnostics read p.grad once the step returns."""
    x, y = batch(4)
    model = make_model()
    train_step(model, [(x, y)], make_optimizers(model), TrainStepConfig())
    assert any(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------------------
# train_step: stats, clipping, stepping
# ---------------------------------------------------------------------------


def test_stats_are_plain_python_scalars():
    x, y = batch(4)
    model = make_model()
    stats = train_step(model, [(x, y)], make_optimizers(model), TrainStepConfig())
    assert isinstance(stats, StepStats)
    assert type(stats.loss) is float
    assert type(stats.grad_norm) is float
    assert type(stats.n_tokens) is int


def test_n_tokens_counts_the_whole_step():
    x, y = batch(8)
    model = make_model()
    stats = train_step(
        model, split(x, y, 4), make_optimizers(model), TrainStepConfig(accum_steps=4)
    )
    assert stats.n_tokens == 8 * T


def test_reported_norm_is_before_clipping():
    x, y = batch(4, seed=11)
    unclipped = make_model()
    a = train_step(unclipped, [(x, y)], make_optimizers(unclipped), TrainStepConfig(grad_clip=0.0))

    clipped = make_model()
    b = train_step(clipped, [(x, y)], make_optimizers(clipped), TrainStepConfig(grad_clip=1e-4))

    assert b.grad_norm == pytest.approx(a.grad_norm, rel=1e-5)
    assert b.grad_norm > 1e-4
    after = torch.sqrt(sum((p.grad**2).sum() for p in clipped.parameters() if p.grad is not None))
    assert after == pytest.approx(1e-4, rel=1e-3)


def test_every_optimizer_actually_steps():
    x, y = batch(4)
    model = make_model()
    opts = make_optimizers(model, lr=0.1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train_step(model, [(x, y)], opts, TrainStepConfig())
    moved = [not torch.equal(p, before[n]) for n, p in model.named_parameters()]
    assert all(moved), "a parameter did not move: one optimizer was never stepped"


def test_bf16_autocast_runs_and_keeps_the_loss_finite():
    x, y = batch(4)
    model = make_model()
    cfg = TrainStepConfig(amp_dtype=torch.bfloat16, device_type="cpu")
    stats = train_step(model, [(x, y)], make_optimizers(model), cfg)
    assert torch.isfinite(torch.tensor(stats.loss))
    assert all(p.grad.dtype == torch.float32 for p in model.parameters() if p.grad is not None)


def test_step_is_deterministic():
    x, y = batch(4, seed=13)
    a = make_model()
    b = make_model()
    cfg = TrainStepConfig(accum_steps=2)
    sa = train_step(a, split(x, y, 2), make_optimizers(a), cfg)
    sb = train_step(b, split(x, y, 2), make_optimizers(b), cfg)
    assert sa.loss == pytest.approx(sb.loss)
    assert sa.grad_norm == pytest.approx(sb.grad_norm)


# ---------------------------------------------------------------------------
# train_step: the GradScaler protocol
# ---------------------------------------------------------------------------


class _RecordingScaler:
    """A stand-in for torch.cuda.amp.GradScaler, usable on CPU.

    GradScaler is CUDA-only, but the *protocol* train_step must follow is
    testable anywhere: scale the loss, unscale before clipping, step through
    the scaler, update once.
    """

    def __init__(self, factor: float = 128.0) -> None:
        self.factor = factor
        self.calls: list[str] = []

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self.calls.append("scale")
        return loss * self.factor

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.calls.append("unscale_")
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.div_(self.factor)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.calls.append("step")
        optimizer.step()

    def update(self) -> None:
        self.calls.append("update")


def _fp16_cfg(**kw) -> TrainStepConfig:
    # device_type "cpu" so autocast is a no-op here; only the scaler protocol
    # is under test, and CPU fp16 autocast is not supported on torch 2.2.
    return TrainStepConfig(amp_dtype=torch.float16, device_type="cuda", **kw)


def test_scaler_protocol_order():
    x, y = batch(8)
    model = make_model()
    scaler = _RecordingScaler()
    train_step(
        model, split(x, y, 2), make_optimizers(model), _fp16_cfg(accum_steps=2), scaler=scaler
    )
    assert scaler.calls.count("scale") == 2, "the loss is scaled once per micro-batch"
    assert scaler.calls.count("update") == 1, "update is called once per step"
    assert scaler.calls.count("step") == 2, "one step per optimizer, through the scaler"
    first_unscale = scaler.calls.index("unscale_")
    assert all(c == "scale" for c in scaler.calls[:first_unscale])
    assert scaler.calls.index("step") > scaler.calls.index("unscale_")
    assert scaler.calls[-1] == "update"


def test_unscaling_happens_before_the_norm_is_measured():
    """A scaled gradient has a scaled norm; clipping it would use the wrong threshold."""
    x, y = batch(4, seed=17)
    plain = make_model()
    ref = train_step(plain, [(x, y)], make_optimizers(plain), TrainStepConfig(grad_clip=0.0))

    scaled = make_model()
    got = train_step(
        scaled,
        [(x, y)],
        make_optimizers(scaled),
        _fp16_cfg(grad_clip=0.0),
        scaler=_RecordingScaler(factor=1024.0),
    )
    assert got.grad_norm == pytest.approx(ref.grad_norm, rel=1e-3)


def test_optimizer_step_goes_through_the_scaler():
    x, y = batch(4)
    model = make_model()

    class _Tripwire(torch.optim.SGD):
        def step(self, *a, **kw):  # noqa: D102
            raise AssertionError("optimizer.step() called directly while scaling")

    groups = build_param_groups(model, GroupConfig())
    opts = [_Tripwire([{**g, "lr": 0.0} for g in groups])]

    class _NoStepScaler(_RecordingScaler):
        def step(self, optimizer):
            self.calls.append("step")  # swallows it, as a real scaler may on overflow

    train_step(model, [(x, y)], opts, _fp16_cfg(), scaler=_NoStepScaler())
