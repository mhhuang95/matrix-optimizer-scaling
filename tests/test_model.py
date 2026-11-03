"""Tests for TASK 1 (src/optbench/model.py).

These test behaviour and properties, not one specific implementation.
Run: uv run pytest tests/test_model.py -v
"""

import math

import pytest
import torch

from optbench.model import (
    Attention,
    ModelConfig,
    RMSNorm,
    Transformer,
    apply_rope,
    precompute_rope,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=97,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=None,
        d_ff=128,
        max_seq_len=32,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ---------------------------------------------------------------- RMSNorm


def test_rmsnorm_output_has_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(3, 5, 64) * 7.0 + 2.0
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=0)


def test_rmsnorm_is_scale_invariant():
    norm = RMSNorm(64)
    x = torch.randn(4, 64)
    torch.testing.assert_close(norm(x), norm(5.0 * x), atol=1e-4, rtol=1e-4)


def test_rmsnorm_gain_is_learnable_and_applied():
    norm = RMSNorm(64)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.shape == (64,)
    x = torch.randn(4, 64)
    base = norm(x)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    torch.testing.assert_close(norm(x), 2.0 * base)


def test_rmsnorm_preserves_dtype_and_is_accurate_in_bf16():
    norm = RMSNorm(64)
    x32 = torch.randn(8, 64) * 100.0
    y16 = norm(x32.to(torch.bfloat16))
    assert y16.dtype == torch.bfloat16
    torch.testing.assert_close(y16.float(), norm(x32), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------- RoPE


def test_rope_table_shapes_and_dtype():
    cos, sin = precompute_rope(head_dim=16, max_seq_len=10)
    assert cos.shape == sin.shape == (10, 8)
    assert cos.dtype == sin.dtype == torch.float32
    torch.testing.assert_close(cos.pow(2) + sin.pow(2), torch.ones(10, 8))


def test_rope_position_zero_is_identity():
    cos, sin = precompute_rope(16, 10)
    x = torch.randn(2, 3, 10, 16)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y[:, :, 0], x[:, :, 0])


def test_rope_preserves_norm_shape_and_dtype():
    cos, sin = precompute_rope(16, 10)
    x = torch.randn(2, 3, 10, 16)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape and y.dtype == x.dtype
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_rope_uses_rotate_half_pairing():
    # Only pair (0, Dh/2) is non-zero -> rotation must stay inside that pair.
    Dh, T = 8, 6
    cos, sin = precompute_rope(Dh, T)
    x = torch.zeros(1, 1, T, Dh)
    x[..., 0] = 1.0
    y = apply_rope(x, cos, sin)
    others = [d for d in range(Dh) if d not in (0, Dh // 2)]
    assert y[..., others].abs().max() < 1e-6
    # Pair 0 has frequency 1, so position t is rotated by angle t.
    t = torch.arange(T, dtype=torch.float32)
    torch.testing.assert_close(y[0, 0, :, 0], torch.cos(t))
    torch.testing.assert_close(y[0, 0, :, Dh // 2], torch.sin(t))


def test_rope_scores_depend_only_on_relative_position():
    # Same q at every position, same k at every position:
    # <rope(q, m), rope(k, n)> must depend only on m - n.
    Dh, T = 16, 12
    cos, sin = precompute_rope(Dh, T)
    q = torch.randn(Dh).expand(1, 1, T, Dh).contiguous()
    k = torch.randn(Dh).expand(1, 1, T, Dh).contiguous()
    S = apply_rope(q, cos, sin)[0, 0] @ apply_rope(k, cos, sin)[0, 0].T  # (T, T)
    torch.testing.assert_close(S[1:, 1:], S[:-1, :-1], atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- Attention


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_attention_projection_shapes(n_kv_heads):
    cfg = tiny_cfg(n_kv_heads=n_kv_heads)
    attn = Attention(cfg)
    D, Dh = cfg.d_model, cfg.head_dim
    assert attn.wq.weight.shape == (cfg.n_heads * Dh, D)
    assert attn.wk.weight.shape == (n_kv_heads * Dh, D)
    assert attn.wv.weight.shape == (n_kv_heads * Dh, D)
    assert attn.wo.weight.shape == (D, cfg.n_heads * Dh)
    for lin in (attn.wq, attn.wk, attn.wv, attn.wo):
        assert lin.bias is None


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
def test_attention_is_causal(n_kv_heads):
    cfg = tiny_cfg(n_kv_heads=n_kv_heads)
    attn = Attention(cfg).eval()
    T, t = 10, 6
    cos, sin = precompute_rope(cfg.head_dim, T)
    x = torch.randn(2, T, cfg.d_model)
    x2 = x.clone()
    x2[:, t] += 1.0
    y, y2 = attn(x, cos, sin), attn(x2, cos, sin)
    assert y.shape == (2, T, cfg.d_model)
    torch.testing.assert_close(y[:, :t], y2[:, :t])  # past unaffected by the future
    assert (y[:, t:] - y2[:, t:]).abs().max() > 1e-4  # present/future do see it


def test_gqa_with_shared_kv_heads_equals_mha():
    # If every query head in a group uses identical K/V weights, GQA must equal
    # MHA whose K/V weights are those weights repeated per head.
    gqa_cfg, mha_cfg = tiny_cfg(n_kv_heads=2), tiny_cfg(n_kv_heads=4)
    gqa, mha = Attention(gqa_cfg).eval(), Attention(mha_cfg).eval()
    Dh, group = gqa_cfg.head_dim, gqa_cfg.n_heads // gqa_cfg.n_kv_heads
    with torch.no_grad():
        mha.wq.weight.copy_(gqa.wq.weight)
        mha.wo.weight.copy_(gqa.wo.weight)
        for name in ("wk", "wv"):
            w = getattr(gqa, name).weight.view(gqa_cfg.n_kv_heads, Dh, -1)
            getattr(mha, name).weight.copy_(w.repeat_interleave(group, dim=0).reshape(-1, w.shape[-1]))
    cos, sin = precompute_rope(Dh, 8)
    x = torch.randn(2, 8, gqa_cfg.d_model)
    torch.testing.assert_close(gqa(x, cos, sin), mha(x, cos, sin), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- Transformer


def test_forward_shapes_and_logits_dtype():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, loss = model(idx)
    assert logits.shape == (3, 16, cfg.vocab_size)
    assert logits.dtype == torch.float32
    assert loss is None


def test_initial_loss_is_close_to_uniform():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, 32))
    _, loss = model(idx, targets=torch.randint(0, cfg.vocab_size, (8, 32)))
    assert loss.ndim == 0
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1


def test_loss_ignores_minus_100_targets():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    tgt = torch.randint(0, cfg.vocab_size, (2, 8))
    masked = tgt.clone()
    masked[:, 4:] = -100
    logits, loss_masked = model(idx, masked)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :4].reshape(-1, cfg.vocab_size), tgt[:, :4].reshape(-1)
    )
    torch.testing.assert_close(loss_masked, expected)


def test_model_is_causal_over_tokens():
    cfg = tiny_cfg()
    model = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    idx2 = idx.clone()
    idx2[:, 7] = (idx2[:, 7] + 1) % cfg.vocab_size
    l1, _ = model(idx)
    l2, _ = model(idx2)
    torch.testing.assert_close(l1[:, :7], l2[:, :7])
    assert (l1[:, 7:] - l2[:, 7:]).abs().max() > 1e-5


def _expected_counts(cfg: ModelConfig) -> tuple[int, int]:
    D, F, V, Dh = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.head_dim
    attn = D * cfg.n_heads * Dh + 2 * D * cfg.n_kv_heads * Dh + cfg.n_heads * Dh * D
    per_layer = attn + 3 * D * F + 2 * D
    non_emb = cfg.n_layers * per_layer + D
    total = non_emb + V * D + (0 if cfg.tie_embeddings else V * D)
    return non_emb, total


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("n_kv_heads", [4, 1])
def test_param_counts(tie, n_kv_heads):
    cfg = tiny_cfg(tie_embeddings=tie, n_kv_heads=n_kv_heads)
    model = Transformer(cfg)
    non_emb, total = _expected_counts(cfg)
    assert model.num_params(non_embedding=True) == non_emb
    assert model.num_params(non_embedding=False) == total


def test_tied_embeddings_share_the_same_parameter():
    model = Transformer(tiny_cfg(tie_embeddings=True))
    assert model.lm_head.weight is model.tok_emb.weight


def test_rope_buffers_are_not_in_state_dict():
    # Every state_dict entry must be a parameter: RoPE tables need persistent=False.
    model = Transformer(tiny_cfg())
    param_names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    extra = set(model.state_dict().keys()) - param_names
    assert not extra, f"non-parameter entries in state_dict: {extra}"


def test_initialization_scales():
    cfg = ModelConfig(vocab_size=512, d_model=256, n_layers=8, n_heads=4, d_ff=512, max_seq_len=16)
    model = Transformer(cfg)
    resid_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
    for blk in model.blocks:
        assert abs(blk.attn.wq.weight.std().item() - cfg.init_std) < 0.1 * cfg.init_std
        assert abs(blk.mlp.w1.weight.std().item() - cfg.init_std) < 0.1 * cfg.init_std
        assert abs(blk.attn.wo.weight.std().item() - resid_std) < 0.1 * resid_std
        assert abs(blk.mlp.w2.weight.std().item() - resid_std) < 0.1 * resid_std
        torch.testing.assert_close(blk.attn_norm.weight, torch.ones(cfg.d_model))
    assert abs(model.tok_emb.weight.std().item() - cfg.init_std) < 0.1 * cfg.init_std


def test_bf16_autocast_forward_backward():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(idx, idx)
    assert logits.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_can_overfit_a_single_batch():
    # End-to-end sanity: gradients flow everywhere and the model can learn.
    cfg = tiny_cfg()
    model = Transformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, f"final loss {loss.item():.3f}; something is not learning"
