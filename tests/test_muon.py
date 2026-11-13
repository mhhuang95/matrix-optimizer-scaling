"""Tests for optbench.muon — TASK 3a.

Numerical notes baked into these tests:
  * After 5 Newton-Schulz steps the singular values of the output land in
    roughly [0.68, 1.2] for RECTANGULAR Gaussian inputs. A SQUARE Gaussian
    matrix has near-zero smallest singular values that 5 steps cannot lift,
    so the square-input test checks direction (against exact U Vᵀ), not the
    full singular-value range.
  * newton_schulz is scale-invariant, which several Muon tests exploit: the
    first step's direction is NS(g) no matter what Nesterov multiplies g by.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from optbench.checkpoint import (
    load_checkpoint,
    optimizer_group_names,
    save_checkpoint,
)
from optbench.muon import Muon, newton_schulz
from optbench.schedule import apply_lr

SEED = 1234


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _matrix_group(p: nn.Parameter, name: str = "w", **overrides) -> dict:
    g = {"params": [p], "names": [name], "kind": "matrix"}
    g.update(overrides)
    return g


class _TinyNet(nn.Module):
    """One matrix parameter; loss = ||w - target||² for deterministic grads."""

    def __init__(self, rows: int = 8, cols: int = 16) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(rows, cols))

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return ((self.w - target) ** 2).sum()


def _manual_muon_step(p, g, buf, *, lr, momentum, nesterov, ns_steps, wd):
    """The update rule, written independently of the optimizer."""
    buf = momentum * buf + g
    eff = g + momentum * buf if nesterov else buf
    update = newton_schulz(eff, steps=ns_steps)
    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
    p = p * (1 - lr * wd) - lr * scale * update
    return p, buf


# ---------------------------------------------------------------------------
# newton_schulz
# ---------------------------------------------------------------------------


class TestNewtonSchulz:
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            newton_schulz(torch.randn(5))
        with pytest.raises(ValueError):
            newton_schulz(torch.randn(2, 3, 4))

    def test_rejects_bad_steps(self):
        with pytest.raises(ValueError):
            newton_schulz(torch.randn(4, 4), steps=0)

    def test_shape_and_dtype_preserved(self):
        torch.manual_seed(SEED)
        for shape in [(8, 8), (8, 32), (32, 8)]:
            G = torch.randn(*shape)
            X = newton_schulz(G)
            assert X.shape == G.shape
            assert X.dtype == G.dtype

    def test_singular_values_near_one_rectangular(self):
        torch.manual_seed(SEED)
        for shape in [(32, 128), (128, 32), (16, 48)]:
            for _ in range(3):
                s = torch.linalg.svdvals(newton_schulz(torch.randn(*shape)))
                assert s.min() > 0.55, f"{shape}: smallest sv {s.min():.3f}"
                assert s.max() < 1.35, f"{shape}: largest sv {s.max():.3f}"

    def test_approximates_uvt_direction(self):
        torch.manual_seed(SEED)
        G = torch.randn(64, 64)
        X = newton_schulz(G)
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        UVt = U @ Vh
        cos = (X * UVt).sum() / (X.norm() * UVt.norm())
        assert cos > 0.95, f"cosine to exact UVᵀ is {cos:.4f}"

    def test_scale_invariance(self):
        torch.manual_seed(SEED)
        G = torch.randn(16, 32)
        for c in (37.5, 1e-3):
            assert torch.allclose(newton_schulz(G), newton_schulz(c * G), atol=1e-4)

    def test_transpose_consistency(self):
        torch.manual_seed(SEED)
        G = torch.randn(24, 96)
        assert torch.allclose(newton_schulz(G).mT, newton_schulz(G.mT), atol=1e-5)

    def test_zero_matrix_is_finite_zeros(self):
        X = newton_schulz(torch.zeros(8, 8))
        assert torch.isfinite(X).all()
        assert X.abs().max() == 0.0

    def test_input_not_mutated(self):
        torch.manual_seed(SEED)
        G = torch.randn(8, 16)
        G_copy = G.clone()
        newton_schulz(G)
        assert torch.equal(G, G_copy)


# ---------------------------------------------------------------------------
# Muon constructor
# ---------------------------------------------------------------------------


class TestMuonInit:
    def test_rejects_bad_hyperparameters(self):
        p = nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError):
            Muon([p], lr=0.0)
        with pytest.raises(ValueError):
            Muon([p], momentum=1.0)
        with pytest.raises(ValueError):
            Muon([p], momentum=-0.1)
        with pytest.raises(ValueError):
            Muon([p], ns_steps=0)
        with pytest.raises(ValueError):
            Muon([p], weight_decay=-0.01)

    def test_rejects_non_2d_parameter_at_construction(self):
        bias = nn.Parameter(torch.randn(16))
        with pytest.raises(ValueError):
            Muon([bias])
        emb3d = nn.Parameter(torch.randn(2, 3, 4))
        with pytest.raises(ValueError):
            Muon([{"params": [emb3d]}])

    def test_group_weight_decay_overrides_default(self):
        p = nn.Parameter(torch.randn(4, 8))
        opt = Muon([_matrix_group(p, weight_decay=0.1)], weight_decay=0.0)
        assert opt.param_groups[0]["weight_decay"] == 0.1

    def test_default_lr_fills_missing_group_lr(self):
        p = nn.Parameter(torch.randn(4, 8))
        opt = Muon([_matrix_group(p)], lr=0.05)
        assert opt.param_groups[0]["lr"] == 0.05

    def test_names_key_survives_for_checkpoint_fingerprint(self):
        p = nn.Parameter(torch.randn(4, 8))
        opt = Muon([_matrix_group(p, name="blocks.0.attn.wq.weight")])
        assert optimizer_group_names([opt]) == [[["blocks.0.attn.wq.weight"]]]


# ---------------------------------------------------------------------------
# Muon.step
# ---------------------------------------------------------------------------


class TestMuonStep:
    def test_first_step_is_orthogonalized_gradient(self):
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(16, 32))
        p0 = p.detach().clone()
        g = torch.randn(16, 32)
        p.grad = g.clone()
        Muon([p], lr=0.02, momentum=0.95, nesterov=True).step()
        # Nesterov makes eff = (1 + m)·g on step one, but NS is scale-invariant,
        # so the update direction is NS(g) regardless.
        expected = p0 - 0.02 * newton_schulz(g)
        assert torch.allclose(p.detach(), expected, atol=1e-5)

    def test_tall_matrix_gets_sqrt_scale_boost(self):
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(64, 16))  # rows/cols = 4 -> scale 2
        p0 = p.detach().clone()
        g = torch.randn(64, 16)
        p.grad = g.clone()
        Muon([p], lr=0.01, momentum=0.9).step()
        expected = p0 - 0.01 * 2.0 * newton_schulz(g)
        assert torch.allclose(p.detach(), expected, atol=1e-5)

    def test_wide_matrix_scale_is_one(self):
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(16, 64))
        p0 = p.detach().clone()
        g = torch.randn(16, 64)
        p.grad = g.clone()
        Muon([p], lr=0.01, momentum=0.9).step()
        expected = p0 - 0.01 * 1.0 * newton_schulz(g)
        assert torch.allclose(p.detach(), expected, atol=1e-5)

    @pytest.mark.parametrize("nesterov", [True, False])
    def test_three_steps_match_manual_simulation(self, nesterov):
        torch.manual_seed(SEED)
        lr, m, ns, wd = 0.02, 0.9, 5, 0.1
        p = nn.Parameter(torch.randn(8, 16))
        opt = Muon([p], lr=lr, momentum=m, nesterov=nesterov, ns_steps=ns, weight_decay=wd)

        ref_p = p.detach().clone()
        ref_buf = torch.zeros_like(ref_p)
        grads = [torch.randn(8, 16) for _ in range(3)]

        for g in grads:
            p.grad = g.clone()
            opt.step()
            ref_p, ref_buf = _manual_muon_step(
                ref_p, g, ref_buf, lr=lr, momentum=m, nesterov=nesterov, ns_steps=ns, wd=wd
            )
        assert torch.allclose(p.detach(), ref_p, atol=1e-4)

    def test_weight_decay_is_decoupled(self):
        # Zero gradient: NS(0) = 0, so the step is pure decay.
        p = nn.Parameter(torch.ones(4, 8))
        p.grad = torch.zeros(4, 8)
        Muon([p], lr=0.1, weight_decay=0.5).step()
        assert torch.allclose(p.detach(), torch.full((4, 8), 1 - 0.1 * 0.5), atol=1e-6)

    def test_param_without_grad_is_untouched(self):
        torch.manual_seed(SEED)
        a = nn.Parameter(torch.randn(4, 8))
        b = nn.Parameter(torch.randn(4, 8))
        b0 = b.detach().clone()
        a.grad = torch.randn(4, 8)
        Muon([a, b], lr=0.02).step()
        assert torch.equal(b.detach(), b0)

    def test_gradient_not_mutated_by_step(self):
        # Spectral diagnostics read p.grad after the step; Nesterov must not
        # have clobbered it.
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(8, 16))
        g = torch.randn(8, 16)
        p.grad = g.clone()
        Muon([p], lr=0.02, nesterov=True).step()
        assert torch.equal(p.grad, g)

    def test_momentum_buffer_is_persistent_state(self):
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(8, 16))
        opt = Muon([p], lr=0.02, momentum=0.9, nesterov=False)
        p.grad = torch.randn(8, 16)
        opt.step()
        buf_id = id(opt.state[p]["momentum_buffer"])
        p.grad = torch.randn(8, 16)
        opt.step()
        assert id(opt.state[p]["momentum_buffer"]) == buf_id, "buffer must be updated in place"

    def test_closure_is_called_and_loss_returned(self):
        p = nn.Parameter(torch.randn(4, 8))
        p.grad = torch.randn(4, 8)
        sentinel = torch.tensor(7.5)
        loss = Muon([p], lr=0.02).step(closure=lambda: sentinel)
        assert loss is sentinel

    def test_respects_schedule_driven_lr(self):
        torch.manual_seed(SEED)
        p = nn.Parameter(torch.randn(8, 16))
        g = torch.randn(8, 16)
        p.grad = g.clone()
        opt = Muon([_matrix_group(p, base_lr=0.04)], lr=999.0)  # default lr is a decoy
        apply_lr(opt, 0.5)  # schedule sets group["lr"] = base_lr * 0.5 = 0.02
        p0 = p.detach().clone()
        opt.step()
        expected = p0 - 0.02 * newton_schulz(g)
        assert torch.allclose(p.detach(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


class TestMuonIntegration:
    def test_optimization_makes_progress(self):
        torch.manual_seed(SEED)
        net = _TinyNet(16, 16)
        target = torch.randn(16, 16)
        opt = Muon(net.parameters(), lr=0.05, momentum=0.9)
        first = net.loss(target).item()
        for _ in range(50):
            opt.zero_grad()
            loss = net.loss(target)
            loss.backward()
            opt.step()
        assert net.loss(target).item() < 0.5 * first

    def test_checkpoint_resume_matches_uninterrupted_run(self, tmp_path):
        def make():
            torch.manual_seed(SEED)
            net = _TinyNet(8, 16)
            opt = Muon(
                [_matrix_group(net.w, base_lr=0.02)], lr=0.02, momentum=0.9
            )
            return net, opt

        torch.manual_seed(0)
        target = torch.randn(8, 16)

        def run_steps(net, opt, n):
            for _ in range(n):
                opt.zero_grad()
                net.loss(target).backward()
                opt.step()

        # Uninterrupted: 4 steps.
        net_a, opt_a = make()
        run_steps(net_a, opt_a, 4)

        # Interrupted: 2 steps, checkpoint, fresh objects, resume, 2 more.
        net_b, opt_b = make()
        run_steps(net_b, opt_b, 2)
        save_checkpoint(tmp_path / "step_2.pt", step=2, model=net_b, optimizers=[opt_b])

        net_c, opt_c = make()
        meta = load_checkpoint(tmp_path / "step_2.pt", model=net_c, optimizers=[opt_c])
        assert meta["step"] == 2
        run_steps(net_c, opt_c, 2)

        assert torch.allclose(net_a.w.detach(), net_c.w.detach(), atol=1e-6), (
            "resume must restore the momentum buffer, not just the weights"
        )
