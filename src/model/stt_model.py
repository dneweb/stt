"""
STT Model — Full end-to-end Speech-to-Text model.

Combines Conv2D Subsampling Encoder, Multi-Head Attention, and Dual Decoder
(Fast CTC Head + Autoregressive Transformer Decoder) into a unified architecture.

Optimized for real-time CPU / macOS inference without CUDA requirements.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class STTModel(nn.Module):
    """End-to-end Multilingual Speech-to-Text model.

    Features:
        - Encoder: Conv2D Subsampling (4x) + Stack of Transformer Encoders
        - CTC Head: Instant single-pass inference (blazing fast on Mac/CPU)
        - Attention Decoder: Autoregressive seq2seq decoding with cross-attention
        - Joint CTC/Attention Training: Best of both worlds for stability and accuracy

    Args:
        encoder_config: Dictionary of Encoder constructor arguments.
        decoder_config: Dictionary of Decoder constructor arguments.
        ctc_weight: Weight for CTC loss during joint training (default: 0.5).
    """

    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        ctc_weight: float = 0.7,
    ) -> None:
        super().__init__()

        # Default configurations if not provided
        enc_cfg = {
            "input_dim": 80,
            "hidden_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.2,
            "conv_channels": 128,
        }
        if encoder_config:
            enc_cfg.update(encoder_config)

        dec_cfg = {
            "vocab_size": 207,
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.2,
        }
        if decoder_config:
            dec_cfg.update(decoder_config)

        self.encoder = Encoder(**enc_cfg)
        self.decoder = Decoder(**dec_cfg)
        self.ctc_weight = ctc_weight
        self.vocab_size = dec_cfg["vocab_size"]

        # CTC Loss Criterion
        self.ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        # Cross Entropy Criterion for Autoregressive Decoder (ignore padding token 1)
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the STT model.

        Args:
            features: Mel spectrogram batch of shape (batch, time, n_mels).
            feature_lengths: Sequence lengths of features (batch,).
            targets: Target token IDs of shape (batch, target_len).
            target_lengths: Sequence lengths of targets (batch,).

        Returns:
            Dictionary with:
                - "ctc_logits": (batch, enc_time, vocab_size)
                - "encoder_out": (batch, enc_time, hidden_dim)
                - "encoder_lengths": (batch,)
                - "decoder_logits": (batch, target_len, vocab_size) if targets provided
        """
        # 1. Encode acoustic features
        encoder_out, enc_lengths = self.encoder(features, feature_lengths)

        # 2. CTC projection
        ctc_logits = self.decoder.ctc_head(encoder_out)

        output: Dict[str, torch.Tensor] = {
            "ctc_logits": ctc_logits,
            "encoder_out": encoder_out,
            "encoder_lengths": enc_lengths,
        }

        # 3. Autoregressive Decoder (if targets provided)
        if targets is not None:
            dec_logits = self.decoder(
                targets=targets,
                encoder_out=encoder_out,
                encoder_lengths=enc_lengths,
                target_lengths=target_lengths,
            )
            output["decoder_logits"] = dec_logits

        return output

    def compute_loss(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute joint CTC and Attention loss.

        Args:
            features: (batch, time, n_mels)
            feature_lengths: (batch,)
            targets: (batch, target_len)
            target_lengths: (batch,)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        outputs = self.forward(features, feature_lengths, targets, target_lengths)

        # 1. CTC Loss
        # CTC expects log_probs of shape (T, B, C)
        ctc_logits = outputs["ctc_logits"]
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, C)
        enc_lengths = outputs["encoder_lengths"]

        if log_probs.device.type == "mps":
            ctc_loss = self.ctc_criterion(
                log_probs.cpu(), targets.cpu(), enc_lengths.cpu(), target_lengths.cpu()
            ).to(log_probs.device)
        else:
            ctc_loss = self.ctc_criterion(log_probs, targets, enc_lengths, target_lengths)

        # 2. Attention Cross-Entropy Loss
        # Shift targets for teacher forcing: inputs are targets[:, :-1], targets are targets[:, 1:]
        dec_logits = outputs["decoder_logits"]
        # Reshape for CrossEntropy: (B * T, C) and (B * T)
        ce_loss = self.ce_criterion(
            dec_logits[:, :-1, :].reshape(-1, self.vocab_size),
            targets[:, 1:].reshape(-1),
        )

        # 3. Joint weighted loss
        total_loss = self.ctc_weight * ctc_loss + (1.0 - self.ctc_weight) * ce_loss

        metrics = {
            "loss": total_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "ce_loss": ce_loss.item(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def transcribe(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
    ) -> List[str]:
        """Perform fast single-pass CTC inference (ideal for real-time CPU execution).

        Args:
            features: Mel spectrogram batch of shape (batch, time, n_mels) or (time, n_mels).
            feature_lengths: Optional feature lengths.
            tokenizer: CharTokenizer instance with decode_ctc method.

        Returns:
            List of decoded transcription strings (one per sample in batch).
        """
        self.eval()

        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, time, n_mels)

        if feature_lengths is None:
            feature_lengths = torch.tensor([features.shape[1]], device=features.device)

        # Single forward pass through Encoder + CTC Head
        encoder_out, _ = self.encoder(features, feature_lengths)
        ctc_logits = self.decoder.ctc_head(encoder_out)

        # Greedy CTC decoding: argmax over vocabulary
        pred_ids = torch.argmax(ctc_logits, dim=-1)  # (batch, time)

        transcriptions = []
        for i in range(pred_ids.shape[0]):
            token_ids = pred_ids[i].tolist()
            if tokenizer is not None:
                text = tokenizer.decode_ctc(token_ids)
            else:
                text = str(token_ids)
            transcriptions.append(text)

        return transcriptions

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_params,
            "decoder": decoder_params,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("STTModel — Full Architecture Verification Test")
    print("=" * 60)

    model = STTModel()
    param_info = model.count_parameters()

    print(f"Total Parameters   : {param_info['total']:,}")
    print(f"Trainable Params   : {param_info['trainable']:,}")
    print(f"Encoder Params     : {param_info['encoder']:,}")
    print(f"Decoder Params     : {param_info['decoder']:,}")

    # Test forward pass with realistic batch from STTDataset
    batch_size = 2
    time_frames = 500  # 5 seconds
    n_mels = 80
    target_len = 30

    dummy_features = torch.randn(batch_size, time_frames, n_mels)
    dummy_feat_lens = torch.tensor([500, 420], dtype=torch.long)
    dummy_targets = torch.randint(2, 207, (batch_size, target_len))
    dummy_tgt_lens = torch.tensor([30, 22], dtype=torch.long)

    # Test Loss computation
    loss, metrics = model.compute_loss(
        features=dummy_features,
        feature_lengths=dummy_feat_lens,
        targets=dummy_targets,
        target_lengths=dummy_tgt_lens,
    )

    print("\n--- Forward & Loss Computation ---")
    print(f"  Total Loss : {metrics['loss']:.4f}")
    print(f"  CTC Loss   : {metrics['ctc_loss']:.4f}")
    print(f"  CE Loss    : {metrics['ce_loss']:.4f}")

    # Test Real-Time CTC Inference
    from tokenizer.tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    tokenizer.load_vocab("tokenizer/vocab.json")

    results = model.transcribe(dummy_features, dummy_feat_lens, tokenizer=tokenizer)
    print(f"\n--- Real-Time CTC Inference (CPU) ---")
    print(f"  Batch size : {len(results)}")
    print(f"  Sample 1   : '{results[0]}'")
    print(f"  ✓ STTModel full architecture verified successfully!")
    print("=" * 60)
