#!/usr/bin/env python3
"""
Comprehensive Validation Benchmark & Error Analysis Tool for Multilingual STT.
Evaluates the CUDA-independent Wav2Vec2 ONNX (INT8 / FP32) model across English, Hindi, and Gujarati.
Calculates CER, WER, Normalized WER, Script Confusion Rate, RTF (Real-Time Factor), and exports error analysis.
"""

import os
import sys
import time
import json
import argparse
import unicodedata
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torchaudio
import onnxruntime as ort

from src.tokenizer import CharTokenizer
from src.metrics import calculate_cer, calculate_wer

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "complete_stt_dataset_23hrs"
VAL_CSV = DATASET_DIR / "data" / "metadata" / "combined_val.csv"
MODEL_DIR = BASE_DIR / "content" / "drive" / "MyDrive" / "stt_w2v" / "best"
ONNX_INT8_PATH = MODEL_DIR / "onnx" / "model_int8.onnx"
ONNX_FP32_PATH = MODEL_DIR / "onnx" / "model.onnx"
VOCAB_PATH = MODEL_DIR / "vocab.json"


def normalize_for_eval(text: str, lang: str = "en") -> str:
    """Normalizes text for fair acoustic ASR evaluation (lowercases Latin, strips punctuation)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).strip()
    if lang == "en":
        text = text.lower()
        # Remove standard English punctuation
        text = re.sub(r"[^\w\s']", "", text)
    else:
        # For Indic scripts, remove Latin/Indic punctuation like । , . ? !
        text = re.sub(r"[।,.:;?!\"'()\-—]", "", text)
    return " ".join(text.split())


def detect_script_confusion(text: str, expected_lang: str) -> bool:
    """Detects if characters from other scripts leaked into the prediction."""
    for char in text:
        cp = ord(char)
        if expected_lang == "en":
            # Devanagari: 0x0900-0x097F, Gujarati: 0x0A80-0x0AFF
            if (0x0900 <= cp <= 0x097F) or (0x0A80 <= cp <= 0x0AFF):
                return True
        elif expected_lang == "hi":
            # Gujarati: 0x0A80-0x0AFF
            if 0x0A80 <= cp <= 0x0AFF:
                return True
        elif expected_lang == "gu":
            # Devanagari: 0x0900-0x097F
            if 0x0900 <= cp <= 0x097F:
                return True
    return False


class BenchmarkRunner:
    def __init__(self, onnx_path: Path = ONNX_INT8_PATH, vocab_path: Path = VOCAB_PATH):
        if not onnx_path.exists():
            if ONNX_FP32_PATH.exists():
                print(f"Notice: {onnx_path} not found. Falling back to FP32 ONNX.")
                onnx_path = ONNX_FP32_PATH
            else:
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        print(f"Loading ONNX Inference Session from: {onnx_path}...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)

        self.session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
        self.tokenizer = CharTokenizer(vocab_path)
        self.onnx_path = onnx_path
        self.model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✓ Model loaded successfully! Engine: ONNX INT8 CPU | Size: {self.model_size_mb:.1f} MB\n")

    def transcribe(self, audio_path: Path) -> Tuple[str, float]:
        t0 = time.time()
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        norm_audio = ((wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)).numpy()
        logits = self.session.run(["logits"], {"input_values": norm_audio})[0]
        latency_ms = (time.time() - t0) * 1000

        pred_ids = logits.argmax(axis=-1)[0].tolist()
        pred_text = self.tokenizer.ctc_decode(pred_ids)
        return pred_text, latency_ms


def run_benchmark(
    val_csv: Path = VAL_CSV,
    limit: Optional[int] = None,
    lang_filter: Optional[str] = None,
    output_dir: Path = BASE_DIR / "benchmark_results"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner()

    df = pd.read_csv(val_csv)
    if lang_filter and lang_filter.lower() != "all":
        df = df[df["language"] == lang_filter.lower()]

    if limit and limit < len(df):
        # Stratified sampling across languages preserving all columns
        per_lang = max(1, limit // df["language"].nunique())
        df = pd.concat([
            group.sample(min(len(group), per_lang), random_state=42)
            for _, group in df.groupby("language")
        ]).reset_index(drop=True)
        print(f"Running stratified benchmark on {len(df)} sampled clips ({len(df) // df['language'].nunique()} per language)...")
    else:
        print(f"Running full benchmark on all {len(df)} validation clips...")

    results = []
    total_clips = len(df)
    start_total_time = time.time()

    print("\n" + "=" * 70)
    print(f"  STARTING MULTILINGUAL STT VALIDATION BENCHMARK ({total_clips} CLIPS)")
    print("=" * 70)

    for i, row in df.iterrows():
        audio_rel_path = row["audio_path"]
        ground_truth = str(row["transcript"]).strip()
        lang = str(row["language"]).strip().lower()
        speaker = str(row.get("speaker", "unknown"))
        duration_sec = float(row.get("duration", 0.0) or 0.0)

        full_audio_path = DATASET_DIR / audio_rel_path
        if not full_audio_path.exists():
            continue

        try:
            pred_text, latency_ms = runner.transcribe(full_audio_path)
        except Exception as e:
            print(f"\nError processing {audio_rel_path}: {e}")
            continue

        # Raw metrics
        raw_cer = calculate_cer(ground_truth, pred_text)
        raw_wer = calculate_wer(ground_truth, pred_text)

        # Normalized metrics (acoustic-level fair comparison)
        norm_ref = normalize_for_eval(ground_truth, lang)
        norm_pred = normalize_for_eval(pred_text, lang)
        norm_cer = calculate_cer(norm_ref, norm_pred)
        norm_wer = calculate_wer(norm_ref, norm_pred)

        script_confusion = detect_script_confusion(pred_text, lang)
        rtf = (latency_ms / 1000.0) / max(0.1, duration_sec)

        results.append({
            "audio_path": audio_rel_path,
            "language": lang,
            "speaker": speaker,
            "duration_sec": duration_sec,
            "latency_ms": round(latency_ms, 1),
            "rtf": round(rtf, 3),
            "ground_truth": ground_truth,
            "prediction": pred_text,
            "raw_cer": round(raw_cer * 100, 2),
            "raw_wer": round(raw_wer * 100, 2),
            "norm_cer": round(norm_cer * 100, 2),
            "norm_wer": round(norm_wer * 100, 2),
            "script_confusion": script_confusion
        })

        # Progress reporting
        processed = len(results)
        if processed % 25 == 0 or processed == total_clips:
            avg_cer = np.mean([r["norm_cer"] for r in results])
            elapsed = time.time() - start_total_time
            eta_sec = (elapsed / processed) * (total_clips - processed)
            print(f"[{processed:4d}/{total_clips}] Elapsed: {elapsed:5.1f}s | ETA: {eta_sec:5.1f}s | Rolling Norm CER: {avg_cer:5.2f}%")

    total_time_sec = time.time() - start_total_time
    res_df = pd.DataFrame(results)

    # Export detailed CSV
    csv_path = output_dir / "benchmark_detailed.csv"
    res_df.to_csv(csv_path, index=False)

    # Aggregate Statistics
    summary = {
        "total_clips": len(res_df),
        "total_audio_hours": round(res_df["duration_sec"].sum() / 3600.0, 2),
        "total_eval_time_sec": round(total_time_sec, 1),
        "avg_latency_ms": round(res_df["latency_ms"].mean(), 1),
        "avg_rtf": round(res_df["rtf"].mean(), 3),
        "overall_norm_cer": round(res_df["norm_cer"].mean(), 2),
        "overall_norm_wer": round(res_df["norm_wer"].mean(), 2),
        "overall_raw_cer": round(res_df["raw_cer"].mean(), 2),
        "overall_raw_wer": round(res_df["raw_wer"].mean(), 2),
        "script_confusion_rate": round((res_df["script_confusion"].sum() / len(res_df)) * 100, 2),
        "by_language": {}
    }

    print("\n" + "=" * 70)
    print("               🏆 MULTILINGUAL STT BENCHMARK REPORT")
    print("=" * 70)
    print(f"Model Engine:        ONNX INT8 CPU ({runner.model_size_mb:.1f} MB, CUDA-Independent)")
    print(f"Total Clips Tested:  {len(res_df)} clips ({summary['total_audio_hours']} hours of audio)")
    print(f"Total Time:          {total_time_sec:.1f}s (Average Latency: {summary['avg_latency_ms']} ms/clip)")
    print(f"Throughput (RTF):    {summary['avg_rtf']}x real-time (Lower is faster, <1.0 is real-time)")
    print(f"Script Confusion:    {summary['script_confusion_rate']}% of clips had character leak across scripts")
    print("-" * 70)
    print(f"{'LANGUAGE':<12} | {'CLIPS':<6} | {'NORM CER':<10} | {'NORM WER':<10} | {'RAW WER':<10} | {'AVG LATENCY'}")
    print("-" * 70)

    for lang_code, lang_name in [("en", "English"), ("hi", "Hindi"), ("gu", "Gujarati")]:
        sub = res_df[res_df["language"] == lang_code]
        if len(sub) == 0:
            continue
        lang_stats = {
            "clips": len(sub),
            "norm_cer": round(sub["norm_cer"].mean(), 2),
            "norm_wer": round(sub["norm_wer"].mean(), 2),
            "raw_cer": round(sub["raw_cer"].mean(), 2),
            "raw_wer": round(sub["raw_wer"].mean(), 2),
            "avg_latency_ms": round(sub["latency_ms"].mean(), 1),
            "script_confusion_pct": round((sub["script_confusion"].sum() / len(sub)) * 100, 2)
        }
        summary["by_language"][lang_name] = lang_stats
        print(f"{lang_name:<12} | {len(sub):<6} | {lang_stats['norm_cer']:>8.2f}% | {lang_stats['norm_wer']:>8.2f}% | {lang_stats['raw_wer']:>8.2f}% | {lang_stats['avg_latency_ms']:>8.1f} ms")

    print("-" * 70)
    print(f"{'OVERALL':<12} | {len(res_df):<6} | {summary['overall_norm_cer']:>8.2f}% | {summary['overall_norm_wer']:>8.2f}% | {summary['overall_raw_wer']:>8.2f}% | {summary['avg_latency_ms']:>8.1f} ms")
    print("=" * 70)

    # Save summary JSON
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save Top 10 High-Error Clips per language for error analysis
    analysis_path = output_dir / "error_analysis.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("# 🔍 STT Model Failure Mode & Error Analysis\n\n")
        f.write(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} using ONNX INT8 Engine.\n\n")
        for lang_code, lang_name in [("en", "English"), ("hi", "Hindi"), ("gu", "Gujarati")]:
            sub = res_df[res_df["language"] == lang_code].sort_values(by="norm_cer", ascending=False).head(8)
            f.write(f"## {lang_name} - Top Discrepancies\n\n")
            for _, r in sub.iterrows():
                f.write(f"* **File:** `{Path(r['audio_path']).name}` (CER: **{r['norm_cer']}%**, WER: **{r['norm_wer']}%**)\n")
                f.write(f"  * **Ground Truth:** `{r['ground_truth']}`\n")
                f.write(f"  * **Prediction:**   `{r['prediction']}`\n\n")

    print(f"\n✓ Detailed CSV exported to:   {csv_path}")
    print(f"✓ Summary JSON exported to:  {summary_path}")
    print(f"✓ Failure analysis in:       {analysis_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Multilingual STT Model Validation Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation clips to evaluate (e.g. 150 for quick run)")
    parser.add_argument("--lang", type=str, default="all", choices=["en", "hi", "gu", "all"], help="Filter by language code")
    args = parser.parse_args()

    run_benchmark(limit=args.limit, lang_filter=args.lang)


if __name__ == "__main__":
    main()
