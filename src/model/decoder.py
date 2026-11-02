"""
Decoder — Autoregressive Transformer decoder & CTC projection head for STT.

Provides two complementary decoding strategies:
    1. CTCHead: Fast, non-autoregressive single-pass decoding (ideal for real-time CPU inference).
    2. Decoder: Autoregressive cross-attention Transformer decoder.

"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PositionalEncoding


class CTCHead(nn.Module):
    """Connectionist Temporal Classification (CTC) projection head.

    Linearly maps encoder hidden representations to vocabulary logits.
    Enables single-pass, real-time CPU speech recognition without autoregressive latency.

    Args:
        hidden_dim: Input dimension from encoder.
        vocab_size: Size of target vocabulary (including <BLANK> at index 0).
    """

    def __init__(self, hidden_dim: int = 256, vocab_size: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Project encoder output to vocabulary logits.

        Args:
            encoder_out: Tensor of shape (batch, time, hidden_dim).

        Returns:
            Logits of shape (batch, time, vocab_size).
        """
        return self.fc(encoder_out)


class Decoder(nn.Module):
    """Autoregressive Transformer Decoder for the STT model.

    Attends to encoder outputs and generates token predictions sequentially.

    Architecture:
        1. Token embedding layer.
        2. Sinusoidal positional encoding.
        3. Stack of Transformer decoder layers with self-attention and cross-attention.
        4. Linear projection to vocabulary logits.

    Args:
        vocab_size: Size of the output vocabulary.
        embedding_dim: Dimension of token embeddings (default: 256).
        hidden_dim: Hidden dimension size (must match encoder hidden_dim).
        num_layers: Number of Transformer decoder layers (default: 2).
        num_heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        vocab_size: int = 512,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Embedding + Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_scale = math.sqrt(embedding_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim=hidden_dim, dropout=dropout)

        # 2. Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # 3. Output Projection Head
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # 4. CTC Head for fast non-autoregressive inference
        self.ctc_head = CTCHead(hidden_dim=hidden_dim, vocab_size=vocab_size)

    @staticmethod
    def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal attention mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        targets: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training with teacher forcing.

        Args:
            targets: Target token IDs of shape (batch, target_len).
            encoder_out: Encoder outputs of shape (batch, enc_time, hidden_dim).
            encoder_lengths: Optional lengths of encoder sequences (batch,).
            target_lengths: Optional lengths of target sequences (batch,).

        Returns:
            Logits over vocabulary of shape (batch, target_len, vocab_size).
        """
        batch_size, tgt_len = targets.shape
        device = targets.device

        # Embed target tokens
        tgt_embed = self.embedding(targets) * self.embed_scale
        tgt_embed = self.pos_encoding(tgt_embed)

        # Causal mask for autoregressive generation
        causal_mask = self.generate_causal_mask(tgt_len, device)

        # Target key padding mask (mask out padded target tokens)
        tgt_key_padding_mask = None
        if target_lengths is not None:
            tgt_key_padding_mask = (
                torch.arange(tgt_len, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)
            )

        # Memory key padding mask (mask out padded encoder frames)
        memory_key_padding_mask = None
        if encoder_lengths is not None:
            enc_max_len = encoder_out.shape[1]
            memory_key_padding_mask = (
                torch.arange(enc_max_len, device=device).unsqueeze(0) >= encoder_lengths.unsqueeze(1)
            )

        # Transformer decoder forward pass
        dec_out = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.fc_out(dec_out)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("Decoder — Self Verification Test")
    print("=" * 60)

    vocab_size = 207  # Our project vocabulary size
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
    )

    batch_size = 2
    enc_time = 100
    tgt_len = 25

    dummy_encoder_out = torch.randn(batch_size, enc_time, 256)
    dummy_encoder_lens = torch.tensor([100, 80], dtype=torch.long)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, tgt_len))
    dummy_target_lens = torch.tensor([25, 18], dtype=torch.long)

    # Test Autoregressive Decoder
    logits = decoder(
        targets=dummy_targets,
        encoder_out=dummy_encoder_out,
        encoder_lengths=dummy_encoder_lens,
        target_lengths=dummy_target_lens,
    )
    print(f"  Autoregressive Logits shape : {logits.shape} (Expected: ({batch_size}, {tgt_len}, {vocab_size}))")
    assert logits.shape == (batch_size, tgt_len, vocab_size)

    # Test CTC Head
    ctc_logits = decoder.ctc_head(dummy_encoder_out)
    print(f"  CTC Logits shape            : {ctc_logits.shape} (Expected: ({batch_size}, {enc_time}, {vocab_size}))")
    assert ctc_logits.shape == (batch_size, enc_time, vocab_size)

    print("  ✓ Decoder & CTCHead verified successfully!")
    print("=" * 60)
