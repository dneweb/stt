"""
Attention — Multi-head attention mechanisms for the STT model.

"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBlock(nn.Module):
    """Multi-Head Attention block for self-attention and cross-attention.

    Implements scaled dot-product attention with multiple parallel heads.

    Args:
        hidden_dim: Dimension of the model (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_q, hidden_dim).
            key: Key tensor of shape (batch, seq_k, hidden_dim).
            value: Value tensor of shape (batch, seq_k, hidden_dim).
            mask: Optional attention mask of shape (batch, seq_q, seq_k) or (batch, 1, seq_q, seq_k).
                  Values can be boolean (True = mask out) or additive (-inf).

        Returns:
            Tuple of:
                - Attention output of shape (batch, seq_q, hidden_dim).
                - Attention weights of shape (batch, num_heads, seq_q, seq_k).
        """
        batch_size, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        # Linear projections & reshape to (batch, num_heads, seq_len, head_dim)
        Q = self.query_proj(query).view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: (batch, num_heads, seq_q, seq_k)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                # Expand (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k)
                mask = mask.unsqueeze(1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores + mask

        # Softmax over key sequence length
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (batch, num_heads, seq_q, head_dim)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, seq_q, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, self.hidden_dim)

        output = self.out_proj(context)
        return output, attn_weights


if __name__ == "__main__":
    print("=" * 60)
    print("MultiHeadAttentionBlock — Self Verification Test")
    print("=" * 60)

    mha = MultiHeadAttentionBlock(hidden_dim=256, num_heads=4)
    q = torch.randn(2, 50, 256)
    k = torch.randn(2, 100, 256)
    v = torch.randn(2, 100, 256)

    out, weights = mha(q, k, v)
    print(f"  Input Query shape  : {q.shape}")
    print(f"  Input Key shape    : {k.shape}")
    print(f"  Output shape       : {out.shape} (Expected: (2, 50, 256))")
    print(f"  Weights shape      : {weights.shape} (Expected: (2, 4, 50, 100))")
    assert out.shape == (2, 50, 256), f"Shape mismatch: {out.shape}"
    print("  ✓ MultiHeadAttentionBlock verified successfully!")
    print("=" * 60)
