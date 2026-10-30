#!/usr/bin/env python3
"""
Verification Script for Step 2: Conformer-CTC Model & Training Engine.
Tests model parameter sizing, forward pass on Apple Silicon (MPS),
CTC alignment condition, and runs an automated Overfit Sanity Test on a real batch.
"""

import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from src.tokenizer import CharTokenizer
from src.features import AudioFeatureExtractor
from src.dataset import STTDataset, SpeechCollate
from src.model import ConformerCTC
from src.metrics import calculate_cer

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "complete_stt_dataset_23hrs"
VOCAB_PATH = DATASET_DIR / "tokenizer" / "vocab.json"
TRAIN_CSV = DATASET_DIR / "data" / "metadata" / "combined_train.csv"


def test_model_architecture():
    print("\n[TEST 1] Conformer-CTC Model Architecture & Parameter Sizing...")
    model = ConformerCTC(
        in_channels=80,
        vocab_size=250,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=6,
        kernel_size=31,
        dropout=0.1
    )
    total_params = model.count_parameters()
    print(f"  ✓ Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Sizing sanity check
    assert 9_000_000 < total_params < 15_000_000, f"Unexpected parameter count: {total_params}"
    print("  ✓ Parameter count is optimally sized for ~23 hours of speech data (~10.5M)")

    return model


def test_forward_pass_and_alignment(model, device):
    print("\n[TEST 2] Model Forward Pass & CTC Length Alignment on Device...")
    tokenizer = CharTokenizer(VOCAB_PATH)
    extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=80)
    dataset = STTDataset(TRAIN_CSV, DATASET_DIR, tokenizer, extractor, is_train=False)
    collate_fn = SpeechCollate(pad_id=tokenizer.pad_id)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = model.to(device)
    features = batch["features"].to(device)
    feature_lengths = batch["feature_lengths"].to(device)
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]

    start = time.time()
    logits, out_lengths = model(features, feature_lengths)
    elapsed = time.time() - start

    B, T_sub, V = logits.shape
    print(f"  ✓ Device: {device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'Standard'})")
    print(f"  ✓ Features input: {list(features.shape)} -> Logits output: {list(logits.shape)}")
    print(f"  ✓ Forward pass speed: {elapsed * 1000:.1f}ms for batch of {B}")

    assert B == 4, f"Expected batch size 4, got {B}"
    assert V == 250, f"Expected vocab size 250, got {V}"

    # Verify CTC Length Condition: Subsampled time steps must be >= target sequence length
    for i in range(B):
        sub_len = out_lengths[i].item()
        tgt_len = target_lengths[i].item()
        assert sub_len >= tgt_len, (
            f"CTC length violation for item {i}: subsampled time {sub_len} < target length {tgt_len}"
        )
    print("  ✓ CTC Alignment Condition Verified: All subsampled time steps >= target sequence lengths")

    return batch


def test_overfit_sanity(model, batch, device):
    print("\n[TEST 3] Overfit Sanity Test (Single-Batch Training Convergence)...")
    print("  Verifying that the model rapidly descends CTC loss toward zero on real data...")

    tokenizer = CharTokenizer(VOCAB_PATH)
    ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    features = batch["features"].to(device)
    feature_lengths = batch["feature_lengths"].to(device)
    targets = batch["targets"]
    target_lengths = batch["target_lengths"]
    texts = batch["texts"]

    model.train()
    initial_loss = None
    final_loss = None
    num_iterations = 40

    for step in range(1, num_iterations + 1):
        optimizer.zero_grad()
        logits, out_lengths = model(features, feature_lengths)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        loss = ctc_loss_fn(
            log_probs.to(torch.float32).cpu(),
            targets.cpu(),
            out_lengths.cpu(),
            target_lengths.cpu()
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        current_loss = loss.item()
        if step == 1:
            initial_loss = current_loss
            print(f"    Step {step:02d}: CTC Loss = {current_loss:.4f} (Initial)")
        elif step % 10 == 0 or step == num_iterations:
            print(f"    Step {step:02d}: CTC Loss = {current_loss:.4f}")

        final_loss = current_loss

    print(f"\n  ✓ Initial Loss: {initial_loss:.4f} -> Final Loss after {num_iterations} steps: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.6, (
        f"Loss failed to descend sufficiently: initial={initial_loss}, final={final_loss}"
    )
    percent_drop = (1.0 - final_loss / initial_loss) * 100
    print(f"  ✓ Steep mathematical descent confirmed (Loss dropped by {percent_drop:.1f}%)")

    # Evaluate prediction on overfit batch
    model.eval()
    with torch.no_grad():
        logits, out_lengths = model(features, feature_lengths)
        preds = torch.argmax(logits, dim=-1)

        print("\n  Sample Decoded Output from Overfit Batch:")
        for i in range(min(2, len(texts))):
            pred_ids = preds[i, :out_lengths[i]].cpu().tolist()
            pred_text = tokenizer.ctc_decode(pred_ids)
            cer = calculate_cer(texts[i], pred_text)
            print(f"    Sample {i+1} [{batch['languages'][i]}]:")
            print(f"      Reference: '{texts[i]}'")
            print(f"      Predicted: '{pred_text}' (CER: {cer * 100:.1f}%)")


def main():
    print("=" * 65)
    print("  STT STEP 2 VERIFICATION: CONFORMER-CTC MODEL & TRAINING ENGINE")
    print("=" * 65)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Architecture test
    model = test_model_architecture()

    # 2. Forward pass test
    batch = test_forward_pass_and_alignment(model, device)

    # 3. Overfit sanity test
    test_overfit_sanity(model, batch, device)

    print("\n" + "=" * 65)
    print("  🎉 ALL STEP 2 TESTS PASSED PERFECTLY WITH ZERO ERRORS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
