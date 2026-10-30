"""
Conformer-CTC Speech-to-Text Model Architecture.
Combines Convolutional 2D Subsampling, Multi-Head Self-Attention,
and Depthwise Separable Convolutions into a lightweight, non-CUDA,
highly efficient speech recognition model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: [batch_size, time_steps, d_model]"""
        return x + self.pe[:, :x.size(1)]


class Conv2dSubsampling4(nn.Module):
    """
    Convolutional 2D Subsampling module (4x reduction in time).
    Reduces 80 Mel bins and T time steps -> [batch, T // 4, d_model].
    """

    def __init__(self, in_channels: int = 80, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # 80 mel channels with two stride-2 convs -> 20 channels
        self.out_proj = nn.Linear(d_model * 20, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, time_steps, 80]
            lengths: [batch] original time lengths
        Returns:
            subsampled: [batch, time_steps // 4, d_model]
            subsampled_lengths: [batch]
        """
        # [batch, 1, time_steps, 80]
        x = x.unsqueeze(1)
        # [batch, d_model, time_steps // 4, 20]
        x = self.conv(x)
        b, c, t, f = x.size()
        # [batch, time_steps // 4, c * f]
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        # [batch, time_steps // 4, d_model]
        out = self.out_proj(x)

        # Compute lengths after 2 stride-2 convolutions
        subsampled_lengths = torch.clamp((lengths + 1) // 2, min=1)
        subsampled_lengths = torch.clamp((subsampled_lengths + 1) // 2, min=1)

        return out, subsampled_lengths


class FeedForwardModule(nn.Module):
    """Macaron-style Feed-Forward Network with Swish activation."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module:
    LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(self, d_model: int = 256, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for same-length padding"
        padding = (kernel_size - 1) // 2

        self.ln = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1, padding=padding, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: [batch, time, d_model]"""
        res = x
        x = self.ln(x)
        # Conv1d operates on [batch, channels, time]
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # splits 2*d_model -> d_model
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return res + x


class ConformerBlock(nn.Module):
    """
    Standard Conformer Encoder Block:
    1. Feed-Forward (half step)
    2. Multi-Head Self-Attention
    3. Convolution Module
    4. Feed-Forward (half step)
    5. LayerNorm
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, time, d_model]
            key_padding_mask: [batch, time] (True where padded)
        """
        # Half-step FFN 1
        x = x + 0.5 * self.ffn1(x)

        # Multi-Head Self Attention
        attn_in = self.attn_ln(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask)
        x = x + self.attn_dropout(attn_out)

        # Convolution Module
        x = self.conv_module(x)

        # Half-step FFN 2
        x = x + 0.5 * self.ffn2(x)

        return self.final_ln(x)


class ConformerCTC(nn.Module):
    """
    End-to-End Conformer-CTC Speech Recognition Model.
    Architecture:
      1. Conv2d Subsampling (4x temporal reduction)
      2. Linear Projection + Sinusoidal Positional Encoding
      3. N x Conformer Encoder Blocks
      4. Final Linear Classification Head -> [batch, time, vocab_size]
    """

    def __init__(
        self,
        in_channels: int = 80,
        vocab_size: int = 250,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 6,
        kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 4x Temporal Subsampling
        self.subsampling = Conv2dSubsampling4(in_channels=in_channels, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # Conformer Encoder Blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final CTC Projection Head
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, time_steps, 80] Log-Mel spectrograms
            lengths: [batch] original feature lengths
        Returns:
            logits: [batch, time_subsampled, vocab_size]
            subsampled_lengths: [batch] lengths after 4x temporal subsampling
        """
        # Subsampling: [batch, time // 4, d_model]
        x, subsampled_lengths = self.subsampling(features, lengths)

        # Positional Encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Create padding mask for attention: True where padded
        batch_size, time_len, _ = x.size()
        device = x.device
        indices = torch.arange(time_len, device=device).unsqueeze(0).expand(batch_size, -1)
        padding_mask = indices >= subsampled_lengths.unsqueeze(1)

        # Forward through Conformer Blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        # Final CTC Logits: [batch, time_subsampled, vocab_size]
        logits = self.classifier(x)

        return logits, subsampled_lengths

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
