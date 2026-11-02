"""
Encoder — Converts Mel spectrograms into high-level acoustic representations.

Architecture:
    Conv2D Subsampling (4x temporal compression)
    → Linear Projection to hidden_dim
    → Positional Encoding
    → Stack of Transformer Encoder Layers

"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, hidden_dim: int = 256, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, hidden_dim)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x of shape (batch, seq_len, hidden_dim)."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling module.

    Downsamples the input Mel spectrogram by a factor of 4 in time using two
    stride-2 2D convolution layers with ReLU activations.

    Input shape: (batch, time, n_mels)
    Output shape: (batch, time // 4, hidden_dim)
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, conv_channels: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # After two stride-2 convs with input_dim=80, the frequency dimension becomes:
        # Layer 1: floor((80 + 2*1 - 3)/2 + 1) = 40
        # Layer 2: floor((40 + 2*1 - 3)/2 + 1) = 20
        freq_dim_out = ((input_dim + 1) // 2 + 1) // 2
        # Verify exact calculation dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16, input_dim)
            dummy_out = self.conv(dummy)
            freq_dim_out = dummy_out.shape[1] * dummy_out.shape[3]

        self.out_proj = nn.Linear(freq_dim_out, hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply convolutional subsampling.

        Args:
            x: Input Mel spectrogram tensor of shape (batch, time, input_dim).
            lengths: Feature lengths tensor of shape (batch,).

        Returns:
            Tuple of:
                - Subsampled tensor of shape (batch, time // 4, hidden_dim)
                - Subsampled lengths tensor of shape (batch,)
        """
        # (batch, time, input_dim) -> (batch, 1, time, input_dim)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (batch, conv_channels, time // 4, freq_dim)

        batch, channels, time_steps, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, time_steps, channels * freq)
        x = self.out_proj(x)  # (batch, time_steps, hidden_dim)

        # Subsample lengths: each stride-2 conv roughly halves length
        # L_out = ((L_in + 2*p - k) // s) + 1 with p=1, k=3, s=2 -> (L_in + 1) // 2
        subsampled_lengths = ((lengths + 1) // 2 + 1) // 2

        return x, subsampled_lengths


class Encoder(nn.Module):
    """Acoustic encoder for the STT model.

    Takes a Mel spectrogram and produces a sequence of hidden representations
    that capture the acoustic content of the speech signal.

    Architecture:
        1. Conv2D subsampling front-end for local acoustic modeling & 4x speedup.
        2. Sinusoidal positional encoding.
        3. Stack of Transformer encoder layers for global sequence context.

    Args:
        input_dim: Number of input features (n_mels = 80).
        hidden_dim: Hidden dimension size for Transformer layers (default: 256).
        num_layers: Number of Transformer encoder layers (default: 4).
        num_heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).
        conv_channels: Number of channels in convolutional front-end (default: 256).
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. 4x Convolutional Subsampling Front-End
        self.subsampling = Conv2dSubsampling(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            conv_channels=conv_channels,
        )

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(hidden_dim=hidden_dim, dropout=dropout)

        # 3. Stack of Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio features.

        Args:
            features: Mel spectrogram batch of shape (batch, time, input_dim)
                      or (batch, input_dim, time).
            feature_lengths: 1D tensor of original feature lengths (batch,).

        Returns:
            Tuple of:
                - Encoded acoustic representations of shape (batch, time // 4, hidden_dim).
                - Subsampled feature lengths of shape (batch,).
        """
        # If passed as (batch, n_mels, time), transpose to (batch, time, n_mels)
        if features.dim() == 3 and features.shape[1] == self.input_dim and features.shape[2] != self.input_dim:
            features = features.transpose(1, 2)

        # 1. Conv subsampling: downsample time dimension by 4x
        x, out_lengths = self.subsampling(features, feature_lengths)

        # 2. Add Positional Encoding
        x = self.pos_encoding(x)

        # 3. Create key padding mask (True for padded positions)
        batch_size, max_len, _ = x.shape
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        # 4. Transformer Encoder forward pass
        encoder_out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        return encoder_out, out_lengths


if __name__ == "__main__":
    print("=" * 60)
    print("Encoder — Self Verification Test")
    print("=" * 60)

    encoder = Encoder(
        input_dim=80,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    )

    batch_size = 2
    time_frames = 400  # 4 seconds of audio (100 frames/sec)
    dummy_features = torch.randn(batch_size, time_frames, 80)
    dummy_lengths = torch.tensor([400, 320], dtype=torch.long)

    out, out_lens = encoder(dummy_features, dummy_lengths)
    print(f"  Input Features shape : {dummy_features.shape}")
    print(f"  Input Lengths        : {dummy_lengths.tolist()}")
    print(f"  Encoder Output shape : {out.shape} (Time downsampled by 4x: {out.shape[1]})")
    print(f"  Subsampled Lengths   : {out_lens.tolist()}")
    assert out.shape == (2, time_frames // 4, 256), f"Unexpected shape: {out.shape}"
    print("  ✓ Encoder verified successfully!")
    print("=" * 60)
