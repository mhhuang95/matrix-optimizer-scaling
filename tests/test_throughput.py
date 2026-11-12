"""Tests for optbench.throughput — TASK 2f.

The FLOP counts are hand-computed for a tiny config; the meter is driven by a
fake clock so every rate is exact. No test here sleeps.
"""

from __future__ import annotations

import pytest

from optbench.model import ModelConfig, Transformer
from optbench.throughput import (
    PEAK_BF16_FLOPS,
    ThroughputMeter,
    flops_per_token,
    matmul_params,
    mfu,
    step_was_skipped,
)

# ---------------------------------------------------------------------------
# A tiny config, small enough to count by hand
# ---------------------------------------------------------------------------
#
#   D=16, L=2, H=4, KV=2, Dh=4, F=32, V=64
#
#   per layer:  wq 16*16=256, wk 16*8=128, wv 16*8=128, wo 16*16=256  -> 768
#               w1+w3+w2 = 3 * 16*32                                 -> 1536
#   2 layers:   2 * 2304 = 4608
#   lm_head:    16 * 64  = 1024
#   total:      5632


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=32,
    )
    base.update(kw)
    return ModelConfig(**base)


class FakeClock:
    """A clock the test sets by hand: `clk.t = 3.0`."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# matmul_params
# ---------------------------------------------------------------------------


class TestMatmulParams:
    def test_hand_computed_value(self):
        assert matmul_params(tiny_cfg()) == 5632

    def test_mha_counts_more_than_gqa(self):
        # KV=H=4: wk and wv double from 128 to 256 -> +256/layer -> 6144
        assert matmul_params(tiny_cfg(n_kv_heads=None)) == 6144

    def test_tying_embeddings_changes_nothing(self):
        # Tying saves memory, not FLOPs: the lm_head matmul still happens.
        assert matmul_params(tiny_cfg(tie_embeddings=True)) == matmul_params(
            tiny_cfg(tie_embeddings=False)
        )

    def test_embedding_is_not_counted_but_lm_head_is(self):
        # Growing the vocab grows the count by exactly D*dV (lm_head only).
        # If the embedding were counted too, the gap would be 2*D*dV.
        small, big = tiny_cfg(vocab_size=64), tiny_cfg(vocab_size=128)
        assert matmul_params(big) - matmul_params(small) == 16 * 64

    def test_agrees_with_the_real_model(self):
        # num_params(non_embedding=True) = matmuls-in-blocks + all RMSNorm
        # weights; swap the norms (2 per block + final) for the lm_head.
        cfg = tiny_cfg()
        model = Transformer(cfg)
        norm_weights = cfg.d_model * (2 * cfg.n_layers + 1)
        lm_head = cfg.d_model * cfg.vocab_size
        assert matmul_params(cfg) == model.num_params(non_embedding=True) - norm_weights + lm_head


# ---------------------------------------------------------------------------
# flops_per_token
# ---------------------------------------------------------------------------


class TestFlopsPerToken:
    def test_hand_computed_value(self):
        # 6*5632 + 12 * L*H*Dh * T = 33792 + 12*2*4*4*8 = 33792 + 3072
        assert flops_per_token(tiny_cfg(), seq_len=8) == 36864

    def test_attention_term_scales_linearly_with_seq_len(self):
        cfg = tiny_cfg()
        f8, f16 = flops_per_token(cfg, 8), flops_per_token(cfg, 16)
        assert f16 - f8 == 12 * cfg.n_layers * cfg.n_heads * cfg.head_dim * 8

    def test_seq_len_bounds(self):
        cfg = tiny_cfg()
        flops_per_token(cfg, 1)
        flops_per_token(cfg, cfg.max_seq_len)
        with pytest.raises(ValueError):
            flops_per_token(cfg, 0)
        with pytest.raises(ValueError):
            flops_per_token(cfg, cfg.max_seq_len + 1)


# ---------------------------------------------------------------------------
# mfu
# ---------------------------------------------------------------------------


class TestMfu:
    def test_arithmetic(self):
        # 100 tok/s * 50 FLOPs/tok / (2 GPUs * 1000 FLOP/s) = 2.5
        assert mfu(100.0, 50, n_gpus=2, peak_flops_per_gpu=1000.0) == pytest.approx(2.5)

    def test_realistic_number(self):
        # 1M tok/s on 8 H100s at ~3e14 FLOPs/tok would be fantasy; this one
        # lands near 0.42, a plausible dense-model MFU.
        got = mfu(1.0e6, 3.3e9, n_gpus=8, peak_flops_per_gpu=PEAK_BF16_FLOPS["h100-sxm"])
        assert got == pytest.approx(1.0e6 * 3.3e9 / (8 * 989e12))
        assert 0.3 < got < 0.5

    def test_not_clamped_above_one(self):
        assert mfu(10.0, 10, n_gpus=1, peak_flops_per_gpu=1.0) == pytest.approx(100.0)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            mfu(1.0, 1, n_gpus=0, peak_flops_per_gpu=1.0)
        with pytest.raises(ValueError):
            mfu(1.0, 1, n_gpus=1, peak_flops_per_gpu=0.0)
        with pytest.raises(ValueError):
            mfu(1.0, 1, n_gpus=1, peak_flops_per_gpu=-5.0)


# ---------------------------------------------------------------------------
# step_was_skipped
# ---------------------------------------------------------------------------


class TestStepWasSkipped:
    def test_scale_halved_means_skipped(self):
        assert step_was_skipped(65536.0, 32768.0) is True

    def test_scale_unchanged_means_stepped(self):
        assert step_was_skipped(32768.0, 32768.0) is False

    def test_scale_grown_means_stepped(self):
        assert step_was_skipped(32768.0, 65536.0) is False


# ---------------------------------------------------------------------------
# ThroughputMeter
# ---------------------------------------------------------------------------


class TestMeterValidation:
    def test_negative_warmup_rejected(self):
        with pytest.raises(ValueError):
            ThroughputMeter(warmup_steps=-1)

    def test_negative_tokens_rejected(self):
        meter = ThroughputMeter(clock=FakeClock())
        with pytest.raises(ValueError):
            meter.update(-1)


class TestMeterRates:
    def test_rate_with_no_warmup(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=0, clock=clk)  # window opens at t=0
        clk.t = 1.0
        meter.update(100)
        clk.t = 2.0
        meter.update(100)
        assert meter.tokens_per_sec == pytest.approx(100.0)

    def test_none_before_any_measurement(self):
        meter = ThroughputMeter(warmup_steps=0, clock=FakeClock())
        assert meter.tokens_per_sec is None
        assert meter.goodput_tokens_per_sec is None

    def test_none_while_still_in_warmup(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=2, clock=clk)
        clk.t = 1.0
        meter.update(100)
        clk.t = 2.0
        meter.update(100)  # warmup ends HERE; window has zero steps so far
        assert meter.tokens_per_sec is None

    def test_none_on_zero_width_window(self):
        clk = FakeClock(5.0)
        meter = ThroughputMeter(warmup_steps=0, clock=clk)
        meter.update(100)  # clock has not moved: dt == 0
        assert meter.tokens_per_sec is None

    def test_warmup_time_and_tokens_are_excluded(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=2, clock=clk)
        clk.t = 10.0
        meter.update(1_000_000)  # glacial warmup step, huge token count
        clk.t = 20.0
        meter.update(1_000_000)  # last warmup step: window opens at t=20
        clk.t = 21.0
        meter.update(100)
        clk.t = 22.0
        meter.update(100)
        # Only the two post-warmup steps count: 200 tokens over 2 seconds.
        assert meter.tokens_per_sec == pytest.approx(100.0)

    def test_skipped_tokens_count_for_throughput_not_goodput(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=0, clock=clk)
        clk.t = 1.0
        meter.update(100, skipped=False)
        clk.t = 2.0
        meter.update(100, skipped=True)
        assert meter.tokens_per_sec == pytest.approx(100.0)  # 200 tok / 2 s
        assert meter.goodput_tokens_per_sec == pytest.approx(50.0)  # 100 tok / 2 s

    def test_goodput_equals_throughput_when_nothing_skipped(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=0, clock=clk)
        clk.t = 1.0
        meter.update(64)
        clk.t = 3.0
        meter.update(64)
        assert meter.goodput_tokens_per_sec == meter.tokens_per_sec


class TestMeterCounters:
    def test_steps_count_warmup_too(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=2, clock=clk)
        for t in (1.0, 2.0, 3.0):
            clk.t = t
            meter.update(10)
        assert meter.steps == 3

    def test_skipped_steps_count_warmup_too(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=1, clock=clk)
        clk.t = 1.0
        meter.update(10, skipped=True)  # during warmup
        clk.t = 2.0
        meter.update(10, skipped=True)
        clk.t = 3.0
        meter.update(10, skipped=False)
        assert meter.steps == 3
        assert meter.skipped_steps == 2

    def test_zero_tokens_is_allowed(self):
        clk = FakeClock(0.0)
        meter = ThroughputMeter(warmup_steps=0, clock=clk)
        clk.t = 1.0
        meter.update(0)
        assert meter.tokens_per_sec == pytest.approx(0.0)


class TestDefaultClock:
    def test_default_clock_smoke(self):
        # No fake clock: just prove the default wiring does not crash and
        # never reports a negative rate.
        meter = ThroughputMeter(warmup_steps=0)
        meter.update(100)
        rate = meter.tokens_per_sec
        assert rate is None or rate >= 0.0
