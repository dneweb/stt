#!/usr/bin/env python3
"""
Indian Accent English Ingestion Script
Downloads and processes VoiceArena Monsoon en-IN (conversational Indian-accented English),
resamples in-memory to 16 kHz 16-bit mono WAV, and merges into extened_recordings and combined_train.csv.
"""

import os
import sys
import time
import shutil
import urllib.request
import fastparquet
import io
import torch
import torchaudio
import soundfile as sf
import pandas as pd

WORKSPACE_DIR = "/Users/deep/Downloads/STT_Recording_Platform copy"
EXTENDED_DIR = "/Users/deep/Downloads/extened_recordings"
DATASET_DIR = os.path.join(WORKSPACE_DIR, "complete_stt_dataset_23hrs")
PROCESSED_EN_DIR = os.path.join(DATASET_DIR, "data", "processed", "indian_english")
EXTENDED_AUDIO_DIR = os.path.join(EXTENDED_DIR, "audio")
TEMP_DIR = "/tmp/monsoon_downloads"

HF_BASE_URL = "https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN/resolve/main/data"
SHARDS = [
    "test-00000-of-00004.parquet",
    "test-00001-of-00004.parquet",
    "test-00002-of-00004.parquet",
    "test-00003-of-00004.parquet"
]

def download_shard(shard_name, dest_path):
    # Check if pre-cached at /tmp/monsoon_part0.parquet
    if shard_name == "test-00000-of-00004.parquet" and os.path.exists("/tmp/monsoon_part0.parquet"):
        print(f"Using pre-cached /tmp/monsoon_part0.parquet")
        return "/tmp/monsoon_part0.parquet"
    
    url = f"{HF_BASE_URL}/{shard_name}"
    print(f"Downloading {shard_name} from Hugging Face...")
    start = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)
    elapsed = time.time() - start
    mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"Downloaded {shard_name}: {mb:.1f} MB in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s)")
    return dest_path

def process_shard(parquet_path, resampler):
    pf = fastparquet.ParquetFile(parquet_path)
    df = pf.to_pandas(columns=["id", "text", "speaker_id", "audio.bytes", "audio_length_s"])
    
    records = []
    print(f"Processing {len(df)} clips from {os.path.basename(parquet_path)}...")
    start_time = time.time()
    
    for idx, row in df.iterrows():
        clip_id = str(row["id"]).strip().replace("-", "_").replace(".", "_")
        text = str(row["text"]).strip()
        speaker = f"in_en_{row['speaker_id']}"
        raw_bytes = row["audio.bytes"]
        
        wav_filename = f"in_en_{clip_id}.wav"
        loc_wav_path = os.path.join(PROCESSED_EN_DIR, wav_filename)
        ext_wav_path = os.path.join(EXTENDED_AUDIO_DIR, wav_filename)
        
        try:
            # Decode audio in memory
            audio_io = io.BytesIO(raw_bytes)
            waveform, orig_sr = torchaudio.load(audio_io)
            
            # Resample to 16 kHz mono
            if orig_sr != 16000:
                waveform_16k = resampler(waveform)
            else:
                waveform_16k = waveform
                
            if waveform_16k.shape[0] > 1:
                waveform_16k = torch.mean(waveform_16k, dim=0, keepdim=True)
                
            # Save to processed folder
            torchaudio.save(loc_wav_path, waveform_16k, 16000, encoding="PCM_S", bits_per_sample=16)
            
            # Copy to extended recordings
            shutil.copy2(loc_wav_path, ext_wav_path)
            
            # Duration
            dur = round(waveform_16k.shape[1] / 16000, 3)
            
            records.append({
                "wav_filename": wav_filename,
                "transcript": text,
                "speaker": speaker,
                "duration": dur
            })
        except Exception as e:
            print(f"Error processing clip {clip_id}: {e}")
            continue
            
        if (len(records)) % 200 == 0:
            print(f"  Processed {len(records)}/{len(df)} clips...")

    elapsed = time.time() - start_time
    print(f"Shard processed: {len(records)} clips in {elapsed:.1f}s")
    return records

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_EN_DIR, exist_ok=True)
    os.makedirs(EXTENDED_AUDIO_DIR, exist_ok=True)
    
    resampler = torchaudio.transforms.Resample(44100, 16000)
    all_records = []
    
    for i, shard_name in enumerate(SHARDS):
        print(f"\n==========================================")
        print(f"PROCESSING SHARD {i+1}/{len(SHARDS)}: {shard_name}")
        print(f"==========================================")
        shard_path = os.path.join(TEMP_DIR, shard_name)
        downloaded_path = download_shard(shard_name, shard_path)
        shard_records = process_shard(downloaded_path, resampler)
        all_records.extend(shard_records)
        
        # Remove downloaded parquet to free space
        if os.path.exists(shard_path):
            os.remove(shard_path)
            print(f"Cleaned up {shard_path}")

    # Build manifest rows
    ext_rows = []
    train_rows = []
    for r in all_records:
        ext_rows.append({
            "audio_path": f"audio/{r['wav_filename']}",
            "transcript": r["transcript"],
            "language": "en",
            "speaker": r["speaker"],
            "duration": r["duration"]
        })
        train_rows.append({
            "audio_path": f"data/processed/indian_english/{r['wav_filename']}",
            "transcript": r["transcript"],
            "language": "en",
            "speaker": r["speaker"],
            "duration": r["duration"]
        })
        
    # Save split manifest
    split_manifest = os.path.join(DATASET_DIR, "data", "metadata", "indian_english_monsoon.csv")
    pd.DataFrame(train_rows).to_csv(split_manifest, index=False)
    print(f"\nSaved split manifest: {split_manifest} ({len(train_rows)} rows)")
    
    # Append to extened_recordings/metadata.csv
    ext_meta_path = os.path.join(EXTENDED_DIR, "metadata.csv")
    pd.DataFrame(ext_rows).to_csv(ext_meta_path, mode="a", header=False, index=False)
    print(f"Appended {len(ext_rows)} rows to {ext_meta_path}")
    
    # Append to combined_train.csv
    comb_train_path = os.path.join(DATASET_DIR, "data", "metadata", "combined_train.csv")
    pd.DataFrame(train_rows).to_csv(comb_train_path, mode="a", header=False, index=False)
    print(f"Appended {len(train_rows)} rows to {comb_train_path}")

    total_hours = sum(r["duration"] for r in all_records) / 3600
    print(f"\n==========================================")
    print(f"INDIAN ENGLISH INGESTION COMPLETE")
    print(f"==========================================")
    print(f"Total Indian English clips added: {len(all_records)}")
    print(f"Total Indian English hours added: {total_hours:.2f} hrs")

if __name__ == "__main__":
    main()

