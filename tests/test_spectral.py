"""Tests for optbench.spectral — TASK 4."""

from __future__ import annotations

import math

import pytest
import torch

from optbench.model import ModelConfig, Transformer
from optbench.optimizers import OptimizerConfig, build_optimizers
from optbench.param_groups import GroupConfig
from optbench.spectral import STAT_KEYS, collect_spectral_stats, spectral_stats

SEED = 11


def tiny_model() -> Transformer:
    torch.manual_seed(SEED)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        max_seq_len=32,
    )
    return Transformer(cfg)


# ---------------------------------------------------------------------------
# spectral_stats
# ---------------------------------------------------------------------------


class TestSpectralStats:
    def test_identity_matrix_known_values(self):
        stats = spectral_stats(torch.eye(4))
        assert stats["frobenius_norm"] == pytest.approx(2.0)
        assert stats["spectral_norm"] == pytest.approx(1.0)
        assert stats["nuclear_norm"] == pytest.approx(4.0)
        assert stats["stable_rank"] == pytest.approx(4.0)
        assert stats["effective_rank"] == pytest.approx(4.0)
        assert stats["nuclear_frobenius_ratio"] == pytest.approx(2.0)

    def test_diagonal_spectrum_exact(self):
        stats = spectral_stats(torch.diag(torch.tensor([3.0, 4.0])))
        assert stats["frobenius_norm"] == pytest.approx(5.0)
        assert stats["spectral_norm"] == pytest.approx(4.0)
        assert stats["nuclear_norm"] == pytest.approx(7.0)
        assert stats["stable_rank"] == pytest.approx(25.0 / 16.0)
        p = [4.0 / 7.0, 3.0 / 7.0]
        entropy = -sum(q * math.log(q) for q in p)
        assert stats["effective_rank"] == pytest.approx(math.exp(entropy), rel=1e-5)
        assert stats["nuclear_frobenius_ratio"] == pytest.approx(7.0 / 5.0)

    def test_rank_one_matrix(self):
        torch.manual_seed(SEED)
        mat = torch.outer(torch.randn(6), torch.randn(9))
        stats = spectral_stats(mat)
        assert stats["stable_rank"] == pytest.approx(1.0, rel=1e-4)
        assert stats["effective_rank"] == pytest.approx(1.0, rel=1e-3)
        assert stats["nuclear_frobenius_ratio"] == pytest.approx(1.0, rel=1e-4)
        assert stats["spectral_norm"] == pytest.approx(stats["frobenius_norm"], rel=1e-4)

    def test_ranks_are_scale_invariant_norms_are_not(self):
        torch.manual_seed(SEED)
        mat = torch.randn(8, 5)
        a, b = spectral_stats(mat), spectral_stats(7.5 * mat)
        for key in ("stable_rank", "effective_rank", "nuclear_frobenius_ratio"):
            assert b[key] == pytest.approx(a[key], rel=1e-4)
        for key in ("frobenius_norm", "spectral_norm", "nuclear_norm"):
            assert b[key] == pytest.approx(7.5 * a[key], rel=1e-4)

    def test_transpose_invariant(self):
        torch.manual_seed(SEED)
        mat = torch.randn(12, 4)
        a, b = spectral_stats(mat), spectral_stats(mat.T)
        for key in STAT_KEYS:
            assert b[key] == pytest.approx(a[key], rel=1e-4)

    def test_zero_matrix_all_zeros_no_nan(self):
        stats = spectral_stats(torch.zeros(5, 7))
        assert set(stats) == set(STAT_KEYS)
        for key in STAT_KEYS:
            assert stats[key] == 0.0

    def test_bounds_on_random_matrices(self):
        torch.manual_seed(SEED)
        for rows, cols in [(16, 16), (4, 32), (32, 4)]:
            stats = spectral_stats(torch.randn(rows, cols))
            r = min(rows, cols)
            assert 1.0 <= stats["stable_rank"] <= r + 1e-5
            assert 1.0 <= stats["effective_rank"] <= r + 1e-5
            assert 1.0 <= stats["nuclear_frobenius_ratio"] <= math.sqrt(r) + 1e-5

    def test_bf16_input_promoted_not_garbage(self):
        torch.manual_seed(SEED)
        mat = torch.randn(8, 16)
        lo = spectral_stats(mat.to(torch.bfloat16))
        hi = spectral_stats(mat)
        for key in STAT_KEYS:
            assert math.isfinite(lo[key])
            assert lo[key] == pytest.approx(hi[key], rel=0.05)

    def test_requires_grad_tensor_is_safe(self):
        mat = torch.randn(4, 4, requires_grad=True)
        stats = spectral_stats(mat)
        assert mat.grad_fn is None  # leaf untouched
        assert all(isinstance(v, float) for v in stats.values())

    def test_values_are_python_floats(self):
        stats = spectral_stats(torch.eye(3))
        for v in stats.values():
            assert type(v) is float

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            spectral_stats(torch.randn(8))
        with pytest.raises(ValueError):
            spectral_stats(torch.randn(2, 3, 4))


# ---------------------------------------------------------------------------
# collect_spectral_stats
# ---------------------------------------------------------------------------


def _built(arm: str):
    model = tiny_model()
    opts = build_optimizers(model, GroupConfig(), OptimizerConfig(arm=arm))
    return model, opts


def _backward(model):
    torch.manual_seed(SEED)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    _, loss = model(x, y)
    loss.backward()


class TestCollect:
    @pytest.mark.parametrize("arm", ["muon", "adamw", "shampoo"])
    def test_covers_exactly_the_matrix_names_for_every_arm(self, arm):
        model, opts = _built(arm)
        _backward(model)
        stats = collect_spectral_stats(opts, source="grad")
        matrix_names = {
            name
            for opt in opts
            for g in opt.param_groups
            if g["kind"] == "matrix"
            for name in g["names"]
        }
        assert set(stats) == matrix_names
        assert "blocks.0.attn.wq.weight" in stats
        assert "tok_emb.weight" not in stats

    def test_each_entry_has_all_stat_keys(self):
        model, opts = _built("muon")
        _backward(model)
        stats = collect_spectral_stats(opts)
        for per_param in stats.values():
            assert set(per_param) == set(STAT_KEYS)

    def test_no_grads_means_empty_dict_not_error(self):
        _, opts = _built("muon")
        assert collect_spectral_stats(opts, source="grad") == {}

    def test_param_source_needs_no_grads(self):
        _, opts = _built("muon")
        stats = collect_spectral_stats(opts, source="param")
        assert "blocks.0.attn.wq.weight" in stats
        for per_param in stats.values():
            assert per_param["frobenius_norm"] > 0.0

    def test_grad_source_measures_grads_not_weights(self):
        model, opts = _built("muon")
        _backward(model)
        name = "blocks.0.attn.wq.weight"
        grad_stats = collect_spectral_stats(opts, source="grad")[name]
        param_stats = collect_spectral_stats(opts, source="param")[name]
        assert grad_stats["frobenius_norm"] != pytest.approx(
            param_stats["frobenius_norm"]
        )

    def test_rejects_unknown_source(self):
        _, opts = _built("muon")
        with pytest.raises(ValueError):
            collect_spectral_stats(opts, source="update")
