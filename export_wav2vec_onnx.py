#!/usr/bin/env python3
"""
Export fine-tuned Wav2Vec2-CTC model to ONNX and perform INT8 Dynamic Quantization.
Optimized for ultra-fast, CUDA-independent CPU inference on macOS (Apple Silicon / Intel)
and any standard server without GPU.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.tokenizer import CharTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "content" / "drive" / "MyDrive" / "stt_w2v" / "best"
VOCAB_PATH = MODEL_DIR / "vocab.json"
ONNX_DIR = MODEL_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

FP32_ONNX_PATH = ONNX_DIR / "model.onnx"
INT8_ONNX_PATH = ONNX_DIR / "model_int8.onnx"


def export_to_onnx():
    print("=" * 65)
    print(" 1. Loading PyTorch Wav2Vec2 Model for Export")
    print("=" * 65)
    
    print(f"Loading checkpoint from: {MODEL_DIR}")
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR))
    model.eval()

    # Create dummy input: 1 audio sample, 2 seconds at 16kHz = 32,000 samples
    dummy_input = torch.randn(1, 32000, dtype=torch.float32)

    print(f"\nExporting PyTorch model to ONNX: {FP32_ONNX_PATH}...")
    start_time = time.time()
    
    # Export with dynamic axes for variable audio lengths and batch sizes
    torch.onnx.export(
        model,
        dummy_input,
        str(FP32_ONNX_PATH),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "time_frames"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    export_duration = time.time() - start_time
    fp32_size_mb = os.path.getsize(FP32_ONNX_PATH) / (1024 * 1024)
    print(f"✓ ONNX FP32 model exported in {export_duration:.2f}s! Size: {fp32_size_mb:.1f} MB")

    # Verify ONNX model integrity
    print("Verifying ONNX model graph structure...")
    onnx_model = onnx.load(str(FP32_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model check passed successfully!")


def quantize_to_int8():
    print("\n" + "=" * 65)
    print(" 2. Applying INT8 Dynamic Quantization")
    print("=" * 65)
    
    print(f"Quantizing: {FP32_ONNX_PATH} -> {INT8_ONNX_PATH}...")
    start_time = time.time()

    quantize_dynamic(
        model_input=str(FP32_ONNX_PATH),
        model_output=str(INT8_ONNX_PATH),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False
    )

    quant_duration = time.time() - start_time
    int8_size_mb = os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024)
    fp32_size_mb = os.path.getsize(FP32_ONNX_PATH) / (1024 * 1024)
    compression_ratio = (1.0 - int8_size_mb / fp32_size_mb) * 100

    print(f"✓ Quantization completed in {quant_duration:.2f}s!")
    print(f"  • FP32 Size: {fp32_size_mb:.1f} MB")
    print(f"  • INT8 Size: {int8_size_mb:.1f} MB ({compression_ratio:.1f}% reduction!)")


def benchmark_and_verify():
    print("\n" + "=" * 65)
    print(" 3. Benchmarking & Verifying Output Consistency")
    print("=" * 65)

    sample_audio_path = BASE_DIR / "complete_stt_dataset_23hrs/data/processed/all_speakers/priyanshi_000475.wav"
    if not sample_audio_path.exists():
        print(f"Sample audio not found at: {sample_audio_path}")
        return

    tokenizer = CharTokenizer(VOCAB_PATH)
    wav, sr = torchaudio.load(str(sample_audio_path))
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    
    # Audio normalization
    norm_audio = ((wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)).numpy()

    # 1. PyTorch CPU Inference
    print("\n[A] PyTorch CPU (Original):")
    pytorch_model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR))
    pytorch_model.eval()
    t0 = time.time()
    with torch.no_grad():
        pt_logits = pytorch_model(torch.from_numpy(norm_audio)).logits
    pt_time = time.time() - t0
    pt_pred_ids = torch.argmax(pt_logits, dim=-1)[0].tolist()
    pt_text = tokenizer.ctc_decode(pt_pred_ids)
    print(f"  • Latency:     {pt_time * 1000:.1f} ms")
    print(f"  • Prediction:  {pt_text}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)

    # 2. ONNX FP32 Inference
    print("\n[B] ONNX FP32 (Unquantized CPU):")
    session_fp32 = ort.InferenceSession(str(FP32_ONNX_PATH), sess_options, providers=["CPUExecutionProvider"])
    t0 = time.time()
    onnx_fp32_logits = session_fp32.run(["logits"], {"input_values": norm_audio})[0]
    fp32_time = time.time() - t0
    fp32_pred_ids = onnx_fp32_logits.argmax(axis=-1)[0].tolist()
    fp32_text = tokenizer.ctc_decode(fp32_pred_ids)
    print(f"  • Latency:     {fp32_time * 1000:.1f} ms ({pt_time / fp32_time:.2f}x speedup)")
    print(f"  • Prediction:  {fp32_text}")

    # 3. ONNX INT8 Inference
    print("\n[C] ONNX INT8 (Quantized CPU):")
    session_int8 = ort.InferenceSession(str(INT8_ONNX_PATH), sess_options, providers=["CPUExecutionProvider"])
    t0 = time.time()
    onnx_int8_logits = session_int8.run(["logits"], {"input_values": norm_audio})[0]
    int8_time = time.time() - t0
    int8_pred_ids = onnx_int8_logits.argmax(axis=-1)[0].tolist()
    int8_text = tokenizer.ctc_decode(int8_pred_ids)
    print(f"  • Latency:     {int8_time * 1000:.1f} ms ({pt_time / int8_time:.2f}x speedup)")
    print(f"  • Prediction:  {int8_text}")

    print("\n" + "=" * 65)
    print("✓ Model is 100% CUDA-independent!")
    print(f"  FP32 ONNX: {FP32_ONNX_PATH} ({os.path.getsize(FP32_ONNX_PATH) / (1024*1024):.1f} MB)")
    print(f"  INT8 ONNX: {INT8_ONNX_PATH} ({os.path.getsize(INT8_ONNX_PATH) / (1024*1024):.1f} MB)")
    print("=" * 65 + "\n")


def main():
    export_to_onnx()
    quantize_to_int8()
    benchmark_and_verify()


if __name__ == "__main__":
    main()
