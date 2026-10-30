#!/usr/bin/env python3
"""
Convert STT recorded WebM/Opus dataset into standard 16 kHz Mono 16-bit PCM WAV.
Validates audio integrity and generates updated metadata_wav.csv.
"""

import os
import csv
import sys
import wave
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
INPUT_AUDIO_DIR = DATASET_DIR / "audio"
OUTPUT_AUDIO_DIR = DATASET_DIR / "audio_wav"
INPUT_METADATA = DATASET_DIR / "metadata.csv"
OUTPUT_METADATA = DATASET_DIR / "metadata_wav.csv"

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"


def convert_file(webm_path: Path, wav_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(webm_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", TARGET_CODEC,
        str(wav_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {
            "success": False,
            "file": webm_path.name,
            "error": res.stderr.strip()
        }

    # Verify WAV integrity using standard wave module
    try:
        with wave.open(str(wav_path), "rb") as w:
            n_channels = w.getnchannels()
            samp_width = w.getsampwidth()
            frame_rate = w.getframerate()
            n_frames = w.getnframes()
            duration = round(n_frames / frame_rate, 3)

        if n_channels != TARGET_CHANNELS:
            raise ValueError(f"Expected {TARGET_CHANNELS} channel, got {n_channels}")
        if samp_width != 2:
            raise ValueError(f"Expected 16-bit (2 bytes), got {samp_width} bytes")
        if frame_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {frame_rate} Hz")

        return {
            "success": True,
            "file": webm_path.name,
            "stem": webm_path.stem,
            "wav_path": str(wav_path.relative_to(BASE_DIR)),
            "duration": duration,
            "frames": n_frames
        }
    except Exception as e:
        return {
            "success": False,
            "file": webm_path.name,
            "error": f"WAV validation error: {str(e)}"
        }


def main():
    if not INPUT_AUDIO_DIR.exists():
        print(f"Error: Input directory {INPUT_AUDIO_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    webm_files = sorted(list(INPUT_AUDIO_DIR.glob("*.webm")))
    print(f"Found {len(webm_files)} WebM audio files to convert.")

    if not webm_files:
        print("No .webm files found.")
        sys.exit(0)

    workers = min(os.cpu_count() or 4, 16)
    print(f"Starting parallel conversion to 16 kHz Mono WAV using {workers} worker threads...")

    conversion_results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(convert_file, webm, OUTPUT_AUDIO_DIR / f"{webm.stem}.wav"): webm
            for webm in webm_files
        }

        completed = 0
        for future in as_completed(future_to_file):
            res = future.result()
            completed += 1
            if res["success"]:
                conversion_results[res["stem"]] = res
            else:
                errors.append(res)
            if completed % 100 == 0 or completed == len(webm_files):
                print(f"  Progress: {completed}/{len(webm_files)} files processed...")

    if errors:
        print(f"\n[WARNING] {len(errors)} files failed conversion:")
        for err in errors:
            print(f"  - {err['file']}: {err['error']}")
        sys.exit(1)

    print(f"\nAll {len(conversion_results)} audio files successfully converted and verified!")

    # Synchronize with metadata
    if INPUT_METADATA.exists():
        print("\nUpdating metadata to reference converted WAV files...")
        with open(INPUT_METADATA, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            input_rows = list(reader)

        output_rows = []
        stats_by_lang = {}

        for row in input_rows:
            # Extract id from audio_path or text
            audio_path = row.get("audio_path", "")
            item_id = Path(audio_path).stem if audio_path else ""
            lang = row.get("language", "Unknown")

            if lang not in stats_by_lang:
                stats_by_lang[lang] = {"count": 0, "total_duration": 0.0}

            if item_id in conversion_results:
                info = conversion_results[item_id]
                wav_rel_path = f"dataset/audio_wav/{item_id}.wav"
                dur = info["duration"]
                status = "recorded"
                stats_by_lang[lang]["count"] += 1
                stats_by_lang[lang]["total_duration"] += dur
            else:
                wav_rel_path = ""
                dur = float(row.get("duration", 0.0) or 0.0)
                status = row.get("status", "pending")

            output_rows.append({
                "id": item_id,
                "audio_path": wav_rel_path,
                "language": lang,
                "text": row.get("text", ""),
                "duration": dur,
                "sample_rate": TARGET_SAMPLE_RATE if wav_rel_path else "",
                "channels": TARGET_CHANNELS if wav_rel_path else "",
                "status": status,
                "original_webm_path": audio_path
            })

        fieldnames = [
            "id", "audio_path", "language", "text", "duration",
            "sample_rate", "channels", "status", "original_webm_path"
        ]
        with open(OUTPUT_METADATA, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"Saved: {OUTPUT_METADATA}")

        print("\nDataset Summary by Language:")
        print("-" * 55)
        total_all_duration = 0.0
        total_all_count = 0
        for lang, s in stats_by_lang.items():
            mins = s["total_duration"] / 60.0
            print(f"  {lang:<10}: {s['count']:>4} files | {s['total_duration']:>7.2f} sec ({mins:>5.2f} min)")
            total_all_duration += s["total_duration"]
            total_all_count += s["count"]
        print("-" * 55)
        all_mins = total_all_duration / 60.0
        print(f"  TOTAL     : {total_all_count:>4} files | {total_all_duration:>7.2f} sec ({all_mins:>5.2f} min)")


if __name__ == "__main__":
    main()
