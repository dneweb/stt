#!/usr/bin/env python3
"""
Enterprise Trilingual Speech-to-Text Transcriber (Gujarati, Hindi, English).
High-accuracy acoustic CTC + KenLM beam search language model.
Supports word-level millisecond timestamps and SRT subtitle exports.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "stt_production_v3"

os.environ["STT_CHECKPOINT_DIR"] = str(DEFAULT_MODEL_DIR)
os.environ["STT_LM_PATH"] = str(DEFAULT_MODEL_DIR / "multilingual_3gram.arpa")
os.environ["STT_UNIGRAMS_PATH"] = str(DEFAULT_MODEL_DIR / "unigrams.txt")

from src.inference.inference import STTInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Proprietary Trilingual STT (Gujarati, Hindi, English) Inference CLI."
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        required=True,
        help="Path to audio file (WAV, MP3, FLAC, OGG, etc.)"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="auto",
        choices=["gu", "hi", "en", "auto"],
        help="Target language code: gu (Gujarati), hi (Hindi), en (English), auto (Auto-Detect)"
    )
    parser.add_argument(
        "--model-dir", "-m",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device for inference"
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Save transcription as .srt subtitle file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found at: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"============================================================")
    print(f"🎙️  Enterprise Multilingual Speech-to-Text Transcriber")
    print(f"    Languages: Gujarati (gu) | Hindi (hi) | English (en)")
    print(f"============================================================")
    print(f"📦 Loading STT Engine from: {args.model_dir} ...")

    engine = STTInference(model_path=args.model_dir, device=args.device)
    print(f"✓ Engine loaded successfully!")
    print(f"  • Acoustic Model : MMS-300M Fine-Tuned (is_w2v={engine.is_w2v})")
    print(f"  • Language Model : KenLM 3-Gram Beam Search ({'Active' if engine.lm_decoder is not None else 'Raw CTC'})")
    print(f"  • Post-Processor : Smart Multilingual Lexicon ({'Active' if engine.post_processor is not None else 'Off'})")
    print(f"------------------------------------------------------------")

    print(f"🎵 Transcribing: {audio_path.name} (Lang: {args.lang.upper()}) ...")
    start_time = time.time()
    result = engine.transcribe_file(str(audio_path), language=args.lang, source="file")
    elapsed = time.time() - start_time

    text = result.get("text", "")
    ctc_text = result.get("ctc_text", "")
    lm_text = result.get("lm_text", "")
    audio_duration = result.get("duration", 0.0)
    rtf = elapsed / max(0.01, audio_duration)

    print(f"\n====================== TRANSCRIPTION ======================")
    print(f"{text}")
    print(f"============================================================")

    if ctc_text and lm_text and ctc_text != lm_text:
        print(f"🔍 CTC Raw Output : {ctc_text}")
        print(f"✨ KenLM Output   : {lm_text}")

    print(f"\n📊 Diagnostics:")
    print(f"  • Audio Duration  : {audio_duration:.2f}s")
    print(f"  • Inference Time  : {elapsed:.3f}s")
    print(f"  • Real-Time Factor: {rtf:.2f}x (Blazing Fast CPU Inference)")

    word_timestamps = result.get("word_timestamps", [])
    if word_timestamps:
        print(f"\n⏱️ Word-Level Millisecond Timestamps:")
        for w in word_timestamps[:10]:
            print(f"  [{w.get('start', 0.0):.2f}s -> {w.get('end', 0.0):.2f}s] {w.get('word', '')}")
        if len(word_timestamps) > 10:
            print(f"  ... ({len(word_timestamps) - 10} more words)")

    if args.export_srt:
        srt_content = result.get("srt_subtitles", "")
        if srt_content:
            srt_path = audio_path.with_suffix(".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"\n💾 Subtitles exported: {srt_path}")


if __name__ == "__main__":
    main()
