"""Llama-style decoder-only Transformer.

TASK 1 — implement everything marked `raise NotImplementedError`.
Run `uv run pytest tests/test_model.py -v` until all tests pass.

Conventions (the tests rely on these):
  * Shapes: B = batch, T = sequence length, D = d_model, H = n_heads,
    KV = n_kv_heads, Dh = head_dim = D // H, F = d_ff, V = vocab_size.
  * No biases in any Linear layer.
  * Pre-norm residual blocks: x = x + attn(norm(x)); x = x + mlp(norm(x)).
  * Attention is causal. Grouped-query attention: each K/V head is shared by
    H // KV query heads.
  * Keep torch 2.2 compatibility (local Intel Mac): do not use torch.nn.RMSNorm
    or the `enable_gqa` argument of scaled_dot_product_attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (you will probably need it)


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # None -> same as n_heads (standard MHA)
    d_ff: int = 2048  # SwiGLU hidden size
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with a learnable per-channel gain.

    Attributes:
        weight: nn.Parameter of shape (dim,), initialized to ones.

    forward(x): x has shape (..., dim); returns the same shape AND the same
    dtype as x. The normalization statistics must be computed in float32
    even when x is bf16/fp16.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = self.weight * x / rms
        x = x.to(input_dtype)
        return x


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE rotation tables.

    Frequency for pair i (i = 0 .. head_dim/2 - 1) is theta ** (-2i / head_dim).

    Returns:
        (cos, sin), each float32 of shape (max_seq_len, head_dim // 2), where
        entry [t, i] is cos/sin of (t * freq_i).
    """
    pair_idx = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    freqs = theta ** (-2 * pair_idx / head_dim)
    position = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(position, freqs)
    angle_sin = angles.sin()
    angle_cos = angles.cos()
    return (angle_cos, angle_sin)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query/key vectors by their position.

    Args:
        x:   (B, n_heads_any, T, head_dim)
        cos: (T, head_dim // 2)  — already sliced to the first T positions
        sin: (T, head_dim // 2)

    Pairing convention ("rotate-half", as in Llama/GPT-NeoX): dimension i is
    paired with dimension i + head_dim // 2, and the pair (a, b) at position t
    is rotated by angle t * freq_i.

    Returns a tensor with the same shape and dtype as x.
    """
    x_dtype = x.dtype
    h = x.shape[-1]//2
    a = x[..., :h]
    b = x[..., h:]
    a, b = a.to(torch.float32), b.to(torch.float32)
    a_rotate = a * cos - b * sin
    b_rotate = a * sin + b * cos
    x_rotate = torch.cat([a_rotate, b_rotate], dim=-1)
    return x_rotate.to(x_dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Causal multi-head / grouped-query self-attention with RoPE.

    Attributes (names are relied on by tests and, later, by the Muon param
    grouping):
        wq: Linear(D -> H * Dh)
        wk: Linear(D -> KV * Dh)
        wv: Linear(D -> KV * Dh)
        wo: Linear(H * Dh -> D)

    Implement the attention math yourself first (scores, mask, softmax).
    Later you may switch to F.scaled_dot_product_attention and check that
    both paths agree.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.n_kv_heads = cfg.n_kv_heads
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D). cos/sin are the first T rows of the RoPE tables."""
        B, T, D = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # B, self.n_heads, T, self.head_dim
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # B, self.n_kv_heads, T, self.head_dim
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # B, self.n_kv_heads, T, self.head_dim

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        group = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn_score = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        attn = attn_score @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(attn)



# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: w2( silu(w1 x) * (w3 x) ).

    Attributes:
        w1: Linear(D -> F)   (gate)
        w3: Linear(D -> F)   (up)
        w2: Linear(F -> D)   (down)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Block and model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    """Pre-norm Transformer block.

    Attributes: attn_norm (RMSNorm), attn (Attention), mlp_norm (RMSNorm), mlp (SwiGLU).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Transformer(nn.Module):
    """Llama-style language model.

    Attributes:
        tok_emb: nn.Embedding(V, D)
        blocks:  nn.ModuleList of Block, length n_layers
        norm:    final RMSNorm
        lm_head: Linear(D -> V). If cfg.tie_embeddings, lm_head.weight must BE
                 tok_emb.weight (same Parameter object).
        RoPE tables registered as NON-persistent buffers (they should not end
        up in state_dict / checkpoints).

    Initialization (implement in __init__, e.g. via a helper):
        * every Linear and the Embedding: normal(0, init_std)
        * residual output projections (attn.wo and mlp.w2 in every block):
          normal(0, init_std / sqrt(2 * n_layers))
        * RMSNorm weights: ones
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) int64 token ids, T <= max_seq_len
            targets: (B, T) int64 next-token ids, or None. Positions with
                     value -100 are ignored in the loss.
        Returns:
            logits: (B, T, V) — computed in float32 for a stable loss
            loss:   scalar mean cross-entropy, or None if targets is None
        """
        raise NotImplementedError

    def num_params(self, non_embedding: bool = True) -> int:
        """Number of unique parameters.

        non_embedding=True excludes the token embedding and the lm_head
        (the "N" used for Chinchilla token budgets in this project). Tied
        weights must not be double-counted when non_embedding=False.
        """
        raise NotImplementedError
