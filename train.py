#!/usr/bin/env python3
"""
Production-grade Training Script for Multilingual Conformer-CTC STT.
Optimized for Apple Silicon Metal (MPS), Linux (CUDA/CPU), and Windows.
Includes Warmup + Cosine Scheduler, Validation Evaluation, CER Tracking, and Checkpointing.
"""

import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from src.tokenizer import CharTokenizer
from src.features import AudioFeatureExtractor
from src.dataset import STTDataset, SpeechCollate
from src.model import ConformerCTC
from src.metrics import calculate_cer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing learning rate schedule."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear Warmup
            factor = float(self.current_step) / float(self.warmup_steps)
        else:
            # Cosine Decay
            progress = float(self.current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def evaluate(model, val_loader, ctc_loss_fn, tokenizer, device, num_print_samples: int = 3):
    """Evaluates model on validation dataset, returns avg loss and avg CER."""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_clips = 0
    sample_outputs = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]
            texts = batch["texts"]

            # Forward pass: [B, T_sub, vocab_size]
            logits, out_lengths = model(features, feature_lengths)
            log_probs = F.log_softmax(logits, dim=-1)

            # CTCLoss expects [T, B, C]
            log_probs_t = log_probs.transpose(0, 1)

            # Loss computation on CPU for universal cross-platform stability
            loss = ctc_loss_fn(
                log_probs_t.to(torch.float32).cpu(),
                targets.cpu(),
                out_lengths.cpu(),
                target_lengths.cpu()
            )

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_clips += batch_size

            # Greedy Decode predictions
            preds = torch.argmax(log_probs, dim=-1)  # [B, T_sub]

            for i in range(batch_size):
                pred_ids = preds[i, :out_lengths[i]].cpu().tolist()
                pred_text = tokenizer.ctc_decode(pred_ids)
                ref_text = texts[i]

                cer = calculate_cer(ref_text, pred_text)
                total_cer += cer

                if len(sample_outputs) < num_print_samples:
                    sample_outputs.append((ref_text, pred_text, cer, batch["languages"][i]))

    avg_loss = total_loss / max(1, total_clips)
    avg_cer = total_cer / max(1, total_clips)

    return avg_loss, avg_cer, sample_outputs


def train(args):
    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "complete_stt_dataset_23hrs"
    vocab_path = dataset_dir / "tokenizer" / "vocab.json"
    train_csv = dataset_dir / "data" / "metadata" / "combined_train.csv"
    val_csv = dataset_dir / "data" / "metadata" / "combined_val.csv"
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device Resolution
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 65)
    print("  MULTILINGUAL CONFORMER-CTC STT TRAINING PIPELINE")
    print("=" * 65)
    print(f"• Hardware Device:       {device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'Standard'})")
    print(f"• Batch Size:            {args.batch_size}")
    print(f"• Epochs:                {args.epochs}")
    print(f"• Initial Learning Rate: {args.lr}")
    print(f"• Checkpoint Directory:  {checkpoint_dir.resolve()}")

    # 2. Tokenizer & Feature Extractor
    tokenizer = CharTokenizer(vocab_path)
    feature_extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=80)

    # 3. Datasets & Loaders
    print("\nLoading datasets...")
    train_ds = STTDataset(train_csv, dataset_dir, tokenizer, feature_extractor, is_train=True)
    val_ds = STTDataset(val_csv, dataset_dir, tokenizer, feature_extractor, is_train=False)
    collate_fn = SpeechCollate(pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    print(f"• Training Samples:      {len(train_ds):,} clips ({len(train_loader)} batches/epoch)")
    print(f"• Validation Samples:    {len(val_ds):,} clips ({len(val_loader)} batches)")
    print(f"• Vocabulary Size:       {tokenizer.vocab_size} tokens")

    # 4. Instantiate Model
    model = ConformerCTC(
        in_channels=80,
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    total_params = model.count_parameters()
    print(f"• Model Parameter Count: {total_params:,} ({total_params / 1e6:.2f}M parameters)")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    ctc_loss_fn = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    # 6. Training Loop
    best_val_cer = float("inf")
    start_train_time = time.time()

    print("\nStarting Training...\n" + "-" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_clips = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]

            optimizer.zero_grad()

            # Forward pass: [B, T_sub, vocab_size]
            logits, out_lengths = model(features, feature_lengths)
            log_probs = F.log_softmax(logits, dim=-1)

            # Transpose to [T, B, C] for CTCLoss
            log_probs_t = log_probs.transpose(0, 1)

            # Compute CTC loss with CPU bridging for universal stability
            loss = ctc_loss_fn(
                log_probs_t.to(torch.float32).cpu(),
                targets.cpu(),
                out_lengths.cpu(),
                target_lengths.cpu()
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            batch_size = features.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_clips += batch_size

            # Periodic batch progress logging
            if step % args.log_interval == 0 or step == len(train_loader):
                current_lr = scheduler.get_lr()
                elapsed = time.time() - epoch_start
                samples_per_sec = epoch_clips / max(1e-5, elapsed)
                avg_train_loss = epoch_loss / epoch_clips
                print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                    f"Step [{step:03d}/{len(train_loader):03d}] | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {samples_per_sec:.1f} clips/s"
                )

        train_loss = epoch_loss / max(1, epoch_clips)
        epoch_time = time.time() - epoch_start

        # 7. Validation Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\nEvaluating on Validation Set (Epoch {epoch})...")
            val_loss, val_cer, sample_preds = evaluate(
                model, val_loader, ctc_loss_fn, tokenizer, device
            )

            print(f"--> Epoch {epoch:02d} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val CER: {val_cer * 100:.2f}% | Time: {epoch_time:.1f}s")
            print("Sample Predictions:")
            for ref, pred, cer, lang in sample_preds:
                print(f"  [{lang.upper()}] Ref:  '{ref}'")
                print(f"         Pred: '{pred}' (CER: {cer * 100:.1f}%)")

            # Checkpoint: Best Model
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "args": vars(args),
                    "vocab_size": tokenizer.vocab_size
                }, best_path)
                print(f"  ★ New best model saved to: {best_path} (Val CER: {val_cer * 100:.2f}%)")

        # Save Latest Checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args)
        }, latest_path)
        print("-" * 65)

    total_time = (time.time() - start_train_time) / 60.0
    print(f"\nTraining Complete in {total_time:.2f} minutes!")
    print(f"Best Validation CER: {best_val_cer * 100:.2f}%")
    print(f"Best checkpoint saved at: {checkpoint_dir / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train Conformer-CTC Multilingual STT Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=800, help="Learning rate warmup steps")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Max gradient norm")
    parser.add_argument("--d-model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--d-ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Conformer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output directory for saved checkpoints")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--log-interval", type=int, default=50, help="Log progress every N batches")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
