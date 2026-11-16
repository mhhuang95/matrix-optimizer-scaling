"""Integration tests for the VENDORED DistributedShampoo (third_party/meta-optimizers).

These are infrastructure tests, not a coaching task: they pin down that
Meta's current main, running through our torch-2.2 compat shims, honors every
contract the rest of this stack depends on — param-group ride-along keys, the
standard state_dict()/load_state_dict() protocol, our generic checkpoint
module, apply_lr, and determinism. If a future vendored update breaks one of
these, this file says so before a training run does.
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo

from optbench.checkpoint import (
    load_checkpoint,
    optimizer_group_names,
    save_checkpoint,
)
from optbench.schedule import apply_lr

SEED = 1234


def _matrix_group(p: nn.Parameter, name: str = "w", **overrides) -> dict:
    g = {"params": [p], "names": [name], "kind": "matrix"}
    g.update(overrides)
    return g


def _shampoo(groups, lr=0.01, **overrides) -> DistributedShampoo:
    kwargs = dict(
        lr=lr,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        max_preconditioner_dim=256,
        precondition_frequency=2,
        start_preconditioning_step=2,
        grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
    )
    kwargs.update(overrides)
    return DistributedShampoo(groups, **kwargs)


class _TinyNet(nn.Module):
    """One matrix parameter; loss = ||w - target||² for deterministic grads."""

    def __init__(self, rows: int = 8, cols: int = 16) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(rows, cols))

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return ((self.w - target) ** 2).sum()


def _fixed_grads(n: int, shape=(8, 16)) -> list[torch.Tensor]:
    torch.manual_seed(SEED)
    return [torch.randn(*shape) for _ in range(n)]


def _run(w: nn.Parameter, opt, grads) -> None:
    for g in grads:
        opt.zero_grad(set_to_none=True)
        w.grad = g.clone()
        opt.step()


class TestVendoredShampoo:
    def test_is_a_torch_optimizer(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt = _shampoo([_matrix_group(p)])
        assert isinstance(opt, torch.optim.Optimizer)

    def test_extra_group_keys_survive(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt = _shampoo([_matrix_group(p, base_lr=0.01, weight_decay=0.1)])
        g = opt.param_groups[0]
        assert g["kind"] == "matrix"
        assert g["base_lr"] == 0.01
        assert g["weight_decay"] == 0.1
        assert g["lr"] == 0.01  # constructor default did not clobber the dict

    def test_names_fingerprint_works(self):
        p = nn.Parameter(torch.randn(4, 4))
        opt = _shampoo([_matrix_group(p, name="blocks.0.attn.wq.weight")])
        assert optimizer_group_names([opt]) == [[["blocks.0.attn.wq.weight"]]]

    def test_deterministic_given_same_grads(self):
        grads = _fixed_grads(3)

        def make():
            torch.manual_seed(SEED)
            w = nn.Parameter(torch.randn(8, 16))
            return w, _shampoo([_matrix_group(w)])

        w1, opt1 = make()
        _run(w1, opt1, grads)
        w2, opt2 = make()
        _run(w2, opt2, grads)
        assert torch.equal(w1.detach(), w2.detach())

    def test_standard_state_dict_roundtrip_resumes_bit_exact(self):
        grads = _fixed_grads(4)

        def make():
            torch.manual_seed(SEED)
            w = nn.Parameter(torch.randn(8, 16))
            return w, _shampoo([_matrix_group(w)])

        w_a, opt_a = make()
        _run(w_a, opt_a, grads)

        w_b, opt_b = make()
        _run(w_b, opt_b, grads[:2])
        buf = io.BytesIO()
        torch.save(opt_b.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf)

        w_c, opt_c = make()
        opt_c.load_state_dict(sd)
        with torch.no_grad():
            w_c.copy_(w_b)
        _run(w_c, opt_c, grads[2:])
        assert torch.allclose(w_a.detach(), w_c.detach(), atol=1e-7)

    def test_checkpoint_beats_config_on_lr(self):
        w1 = nn.Parameter(torch.randn(8, 16))
        opt1 = _shampoo([_matrix_group(w1)], lr=0.01)
        _run(w1, opt1, _fixed_grads(1))
        sd = opt1.state_dict()

        w2 = nn.Parameter(torch.randn(8, 16))
        opt2 = _shampoo([_matrix_group(w2)], lr=0.77)
        opt2.load_state_dict(sd)
        assert opt2.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_resume_through_generic_checkpoint_module(self, tmp_path):
        grads = _fixed_grads(4)

        def make():
            torch.manual_seed(SEED)
            net = _TinyNet()
            opt = _shampoo([_matrix_group(net.w, name="w", base_lr=0.01)])
            return net, opt

        net_a, opt_a = make()
        _run(net_a.w, opt_a, grads)

        net_b, opt_b = make()
        _run(net_b.w, opt_b, grads[:2])
        save_checkpoint(tmp_path / "ckpt.pt", step=2, model=net_b, optimizers=[opt_b])

        net_c, opt_c = make()
        load_checkpoint(tmp_path / "ckpt.pt", model=net_c, optimizers=[opt_c])
        _run(net_c.w, opt_c, grads[2:])

        assert torch.allclose(net_a.w.detach(), net_c.w.detach(), atol=1e-7)

    def test_respects_schedule_driven_lr(self):
        grads = _fixed_grads(3)

        def make(mult):
            torch.manual_seed(SEED)
            w = nn.Parameter(torch.randn(8, 16))
            opt = _shampoo([_matrix_group(w, base_lr=0.01)], lr=999.0)
            apply_lr(opt, mult)
            return w, opt

        w_full, opt_full = make(1.0)
        assert opt_full.param_groups[0]["lr"] == pytest.approx(0.01)
        _run(w_full, opt_full, grads)
        w_half, opt_half = make(0.5)
        _run(w_half, opt_half, grads)
        assert not torch.allclose(w_full.detach(), w_half.detach())

    def test_optimization_makes_progress(self):
        torch.manual_seed(SEED)
        net = _TinyNet()
        target = torch.randn(8, 16)
        opt = _shampoo([_matrix_group(net.w)], lr=0.05)
        first = net.loss(target).item()
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            loss = net.loss(target)
            loss.backward()
            opt.step()
        assert net.loss(target).item() < 0.5 * first
