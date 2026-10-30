#!/usr/bin/env python3
"""
Verification Script for Step 1: Data Loading & Audio Feature Pipeline.
Tests Tokenizer, Audio Feature Extraction, STTDataset, SpeechCollate,
and Apple Silicon Metal (MPS) device execution end-to-end.
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

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "complete_stt_dataset_23hrs"
VOCAB_PATH = DATASET_DIR / "tokenizer" / "vocab.json"
TRAIN_CSV = DATASET_DIR / "data" / "metadata" / "combined_train.csv"
VAL_CSV = DATASET_DIR / "data" / "metadata" / "combined_val.csv"


def test_tokenizer(vocab_path: Path):
    print("\n[TEST 1] Multilingual Character Tokenizer...")
    tokenizer = CharTokenizer(vocab_path)
    print(f"  ✓ Vocab loaded successfully. Size: {tokenizer.vocab_size} tokens")

    assert tokenizer.vocab_size == 250, f"Expected 250 tokens, got {tokenizer.vocab_size}"
    assert tokenizer.blank_id == 0, f"Expected blank_id=0, got {tokenizer.blank_id}"
    assert tokenizer.pad_id == 1, f"Expected pad_id=1, got {tokenizer.pad_id}"

    # Test Trilingual text encoding & decoding
    test_cases = [
        ("English", "The quick brown fox jumps over the lazy dog 123."),
        ("Hindi", "शाम सात बजे का reminder लगाओ।"),
        ("Gujarati", "ટ્રેન સાંજે સાત ને પાંત્રીસે ઊપડશે.")
    ]

    for lang, sample_text in test_cases:
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        print(f"  ✓ [{lang}] '{sample_text}'")
        print(f"       -> Encoded {len(encoded)} token IDs (min={min(encoded)}, max={max(encoded)})")
        print(f"       -> Decoded: '{decoded}'")
        assert len(encoded) > 0, f"Encoding failed for {lang}"

    # Test CTC Greedy Decoding
    # Simulate repeated tokens separated by blanks: e.g. 'c' 'c' <blank> 'a' 'a' 't'
    c_id = tokenizer.char_to_id["c"]
    a_id = tokenizer.char_to_id["a"]
    t_id = tokenizer.char_to_id["t"]
    blank = tokenizer.blank_id
    mock_ctc_seq = [c_id, c_id, blank, a_id, a_id, a_id, t_id, t_id]
    ctc_res = tokenizer.ctc_decode(mock_ctc_seq)
    print(f"  ✓ CTC Collapse Test: {mock_ctc_seq} -> '{ctc_res}' (Expected: 'cat')")
    assert ctc_res == "cat", f"CTC decode expected 'cat', got '{ctc_res}'"

    return tokenizer


def test_feature_extractor():
    print("\n[TEST 2] Audio Feature Extractor (80-channel Log-Mel Spectrogram)...")
    extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=80)

    # 1. Test on 1.0 second synthetic audio (16,000 samples)
    dummy_wav = torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 1, 16000))
    feat = extractor(dummy_wav, augment=False)
    print(f"  ✓ Synthetic audio [16,000 samples] -> Features shape: {list(feat.shape)}")

    assert feat.ndim == 2, f"Expected 2D tensor, got {feat.ndim}D"
    assert feat.shape[1] == 80, f"Expected 80 mel channels, got {feat.shape[1]}"
    assert not torch.isnan(feat).any(), "Features contain NaN!"
    assert not torch.isinf(feat).any(), "Features contain Inf!"

    # 2. Test SpecAugment (frequency and time masking)
    feat_aug = extractor(dummy_wav, augment=True)
    print(f"  ✓ SpecAugment features shape: {list(feat_aug.shape)}")
    assert feat_aug.shape == feat.shape, "Augmentation changed feature shape!"

    return extractor


def test_dataset_and_loader(tokenizer, extractor):
    print("\n[TEST 3] STTDataset and SpeechCollate Dynamic Batching...")

    train_ds = STTDataset(
        csv_path=TRAIN_CSV,
        base_dir=DATASET_DIR,
        tokenizer=tokenizer,
        feature_extractor=extractor,
        is_train=True
    )
    val_ds = STTDataset(
        csv_path=VAL_CSV,
        base_dir=DATASET_DIR,
        tokenizer=tokenizer,
        feature_extractor=extractor,
        is_train=False
    )

    print(f"  ✓ Train dataset loaded: {len(train_ds):,} clips")
    print(f"  ✓ Val dataset loaded:   {len(val_ds):,} clips")

    assert len(train_ds) > 9000, f"Unexpected train count: {len(train_ds)}"
    assert len(val_ds) > 1200, f"Unexpected val count: {len(val_ds)}"

    collate_fn = SpeechCollate(pad_id=tokenizer.pad_id)
    batch_size = 16

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"  ✓ Testing DataLoader with batch_size={batch_size}...")
    start_time = time.time()
    num_test_batches = 5

    for i, batch in enumerate(train_loader):
        features = batch["features"]
        feature_lengths = batch["feature_lengths"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        print(f"    - Batch {i+1}:")
        print(f"        Features: {list(features.shape)} (max_time={features.shape[1]}, n_mels={features.shape[2]})")
        print(f"        Feature Lengths: min={feature_lengths.min().item()}, max={feature_lengths.max().item()}")
        print(f"        Targets:  {list(targets.shape)} (max_label_len={targets.shape[1]})")
        print(f"        Sample text: '{batch['texts'][0][:45]}...' [{batch['languages'][0]}]")

        assert features.shape[0] == batch_size
        assert features.shape[2] == 80
        assert targets.shape[0] == batch_size
        assert not torch.isnan(features).any(), f"Batch {i+1} features contain NaN"

        if i + 1 >= num_test_batches:
            break

    elapsed = time.time() - start_time
    samples_per_sec = (num_test_batches * batch_size) / elapsed
    print(f"  ✓ Loader throughput: {samples_per_sec:.1f} audio clips/sec ({elapsed:.2f}s for {num_test_batches * batch_size} samples)")

    return batch


def test_device_and_ctc_loss(sample_batch, vocab_size):
    print("\n[TEST 4] Hardware Acceleration & CTC Loss Verification...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  ✓ Target Device: {device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'CPU'})")

    features = sample_batch["features"].to(device)
    feature_lengths = sample_batch["feature_lengths"].to(device)
    targets = sample_batch["targets"].to(device)
    target_lengths = sample_batch["target_lengths"].to(device)

    B = features.shape[0]
    T = features.shape[1]

    # Conformer models typically subsample time by 4x via convolution subsampling
    T_subsampled = max(1, T // 4)
    input_lengths_subsampled = torch.clamp(feature_lengths // 4, min=1)

    # Simulate model logits output: [T_subsampled, B, vocab_size] with requires_grad=True
    mock_logits = torch.randn(T_subsampled, B, vocab_size, device=device, requires_grad=True)
    log_probs = F.log_softmax(mock_logits, dim=-1)

    # Note: On Apple Silicon (MPS), PyTorch executes all heavy Conformer attention and convolutions
    # on Metal (MPS), but aten::_ctc_loss is computed on CPU. We bridge log_probs to CPU for the loss
    # while autograd cleanly computes gradients back into the MPS graph.
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss_fn(
        log_probs.to(torch.float32).cpu(),
        targets.cpu(),
        input_lengths_subsampled.cpu(),
        target_lengths.cpu()
    )

    print(f"  ✓ CTC Loss forward pass: {loss.item():.4f}")
    assert not torch.isnan(loss), "CTC Loss returned NaN!"
    assert not torch.isinf(loss), "CTC Loss returned Inf!"
    assert loss.item() > 0.0, "CTC Loss should be positive!"
    print(f"  ✓ Gradient backward sanity check...")
    loss.backward()
    assert mock_logits.grad is not None, "Gradients failed to compute on MPS!"
    print(f"  ✓ Backward gradient norm on {device.type.upper()}: {mock_logits.grad.norm().item():.4f}")


def main():
    print("=" * 65)
    print("  STT STEP 1 VERIFICATION: DATA PIPELINE & AUDIO FEATURES")
    print("=" * 65)

    if not VOCAB_PATH.exists():
        print(f"Error: {VOCAB_PATH} not found!", file=sys.stderr)
        sys.exit(1)
    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found!", file=sys.stderr)
        sys.exit(1)

    # 1. Tokenizer
    tokenizer = test_tokenizer(VOCAB_PATH)

    # 2. Features
    extractor = test_feature_extractor()

    # 3. Dataset & DataLoader
    sample_batch = test_dataset_and_loader(tokenizer, extractor)

    # 4. Device & CTC Loss
    test_device_and_ctc_loss(sample_batch, tokenizer.vocab_size)

    print("\n" + "=" * 65)
    print("  🎉 ALL STEP 1 TESTS PASSED PERFECTLY WITH ZERO ERRORS!")
    print("=" * 65)


if __name__ == "__main__":
    main()
