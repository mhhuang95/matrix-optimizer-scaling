"""Tests for optbench.schedule.

These test behaviour and properties, not one specific implementation.
"""

from __future__ import annotations

import math

import pytest
import torch

from optbench.schedule import ScheduleConfig, apply_lr, lr_multiplier


def cosine(**kw) -> ScheduleConfig:
    base = dict(kind="cosine", total_steps=1000, warmup_steps=100, min_lr_ratio=0.1)
    base.update(kw)
    return ScheduleConfig(**base)


def wsd(**kw) -> ScheduleConfig:
    base = dict(kind="wsd", total_steps=1000, warmup_steps=100, min_lr_ratio=0.1, decay_frac=0.2)
    base.update(kw)
    return ScheduleConfig(**base)


def curve(cfg: ScheduleConfig, steps) -> list[float]:
    return [lr_multiplier(s, cfg) for s in steps]


# ---------------------------------------------------------------------------
# Warmup (shared by both kinds)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [cosine, wsd])
def test_warmup_is_linear_and_starts_nonzero(make):
    cfg = make()
    w = cfg.warmup_steps
    assert lr_multiplier(0, cfg) == pytest.approx(1 / w)
    assert lr_multiplier(w - 1, cfg) == pytest.approx(1.0)
    diffs = [b - a for a, b in zip(curve(cfg, range(w - 1)), curve(cfg, range(1, w)))]
    assert all(d == pytest.approx(1 / w) for d in diffs)


@pytest.mark.parametrize("make", [cosine, wsd])
def test_no_warmup_starts_at_peak(make):
    cfg = make(warmup_steps=0)
    assert lr_multiplier(0, cfg) == pytest.approx(1.0)


@pytest.mark.parametrize("make", [cosine, wsd])
def test_returns_python_float(make):
    assert type(lr_multiplier(10, make())) is float


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


def test_cosine_peak_mid_and_end_values():
    cfg = cosine()
    w, T, m = cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio
    assert lr_multiplier(w, cfg) == pytest.approx(1.0)
    mid = w + (T - w) // 2  # progress = 0.5
    assert lr_multiplier(mid, cfg) == pytest.approx((1 + m) / 2)
    # One step before the end: just above the floor.
    assert m < lr_multiplier(T - 1, cfg) < m + 1e-3
    assert lr_multiplier(T, cfg) == pytest.approx(m)
    assert lr_multiplier(T + 500, cfg) == pytest.approx(m)


def test_cosine_matches_formula_at_quarter():
    cfg = cosine()
    w, T, m = cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio
    s = w + (T - w) // 4  # progress = 0.25
    expected = m + 0.5 * (1 - m) * (1 + math.cos(math.pi * 0.25))
    assert lr_multiplier(s, cfg) == pytest.approx(expected)


def test_cosine_is_monotone_after_warmup_and_bounded():
    cfg = cosine()
    vals = curve(cfg, range(cfg.warmup_steps, cfg.total_steps + 50))
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))
    assert all(cfg.min_lr_ratio - 1e-12 <= v <= 1.0 + 1e-12 for v in vals)


def test_cosine_can_decay_to_zero():
    cfg = cosine(min_lr_ratio=0.0)
    assert lr_multiplier(cfg.total_steps, cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WSD
# ---------------------------------------------------------------------------


def test_wsd_stable_phase_is_flat():
    cfg = wsd()
    decay_start = cfg.total_steps - round(cfg.decay_frac * cfg.total_steps)  # 800
    vals = curve(cfg, range(cfg.warmup_steps, decay_start + 1))
    assert all(v == pytest.approx(1.0) for v in vals)
    assert lr_multiplier(decay_start + 1, cfg) < 1.0


def test_wsd_decay_is_linear_to_floor():
    cfg = wsd()
    T, m = cfg.total_steps, cfg.min_lr_ratio
    n_decay = round(cfg.decay_frac * T)
    start = T - n_decay
    vals = curve(cfg, range(start, T + 1))
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    assert all(d == pytest.approx(-(1 - m) / n_decay) for d in diffs)
    assert lr_multiplier(start + n_decay // 2, cfg) == pytest.approx((1 + m) / 2)
    assert lr_multiplier(T, cfg) == pytest.approx(m)
    assert lr_multiplier(T + 500, cfg) == pytest.approx(m)


def test_wsd_runs_of_different_length_share_the_stable_phase():
    # The property that makes WSD cheap for multi-length studies: a short run
    # is a branch of a long one until the short run starts decaying.
    short, long = wsd(total_steps=1000), wsd(total_steps=4000)
    short_decay_start = 1000 - round(0.2 * 1000)
    for s in range(0, short_decay_start + 1):
        assert lr_multiplier(s, short) == pytest.approx(lr_multiplier(s, long))


def test_wsd_full_decay():
    # decay_frac = 1 with no warmup: linear from step 0 to the end.
    cfg = wsd(warmup_steps=0, decay_frac=1.0, min_lr_ratio=0.0, total_steps=10)
    assert curve(cfg, range(11)) == pytest.approx([1 - s / 10 for s in range(11)])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        ScheduleConfig(kind="linear"),
        cosine(total_steps=0),
        cosine(warmup_steps=-1),
        cosine(warmup_steps=1000),  # warmup must end before total_steps
        cosine(min_lr_ratio=-0.1),
        cosine(min_lr_ratio=1.5),
        wsd(decay_frac=0.0),
        wsd(decay_frac=1.5),
        wsd(warmup_steps=900, decay_frac=0.2),  # warmup overlaps the decay
        wsd(total_steps=4, warmup_steps=0, decay_frac=0.1),  # rounds to 0 decay steps
    ],
    ids=lambda c: f"{c.kind}-T{c.total_steps}-w{c.warmup_steps}-m{c.min_lr_ratio}-d{c.decay_frac}",
)
def test_invalid_configs_raise(cfg):
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("cfg", [cosine(), wsd(), cosine(warmup_steps=0), wsd(warmup_steps=800)])
def test_valid_configs_pass(cfg):
    cfg.validate()


# ---------------------------------------------------------------------------
# apply_lr
# ---------------------------------------------------------------------------


def _two_group_optimizer() -> torch.optim.Optimizer:
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD(
        [{"params": [a], "lr": 0.02, "base_lr": 0.02}, {"params": [b], "lr": 3e-4, "base_lr": 3e-4}]
    )


def test_apply_lr_scales_each_group_from_its_own_base():
    opt = _two_group_optimizer()
    apply_lr(opt, 0.5)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 1.5e-4])
    # Applying twice must not compound: it is always relative to base_lr.
    apply_lr(opt, 0.5)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 1.5e-4])
    assert [g["base_lr"] for g in opt.param_groups] == pytest.approx([0.02, 3e-4])


def test_apply_lr_requires_base_lr():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=0.1)
    with pytest.raises(KeyError):
        apply_lr(opt, 0.5)
