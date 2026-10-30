#!/usr/bin/env python3
"""
AI4Bharat Kathbath Hindi Ingestion Script
Downloads high-quality Hindi speech data from IIT Madras Kathbath (IndicSUPERB),
converts .m4a to 16 kHz mono .wav, normalizes transcripts to match the CTC vocab,
and integrates the clips into both the training dataset and the user's extened_recordings folder.
"""

import os
import sys
import time
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import soundfile as sf
import pandas as pd

# Paths
WORKSPACE_DIR = "/Users/deep/Downloads/STT_Recording_Platform copy"
EXTENDED_DIR = "/Users/deep/Downloads/extened_recordings"
DATASET_DIR = os.path.join(WORKSPACE_DIR, "complete_stt_dataset_23hrs")
PROCESSED_HINDI_DIR = os.path.join(DATASET_DIR, "data", "processed", "kathbath_hindi")
EXTENDED_AUDIO_DIR = os.path.join(EXTENDED_DIR, "audio")
TEMP_DIR = "/tmp/kathbath_pipeline"
TRANSCRIPTS_DIR = "/tmp/kathbath_transcripts/hindi"

REPLACEMENTS = {
    'ॅ': 'े',
    'ऍ': 'ए',
    'ङ': 'ं',
    'ळ': 'ल',
    'ॆ': 'े',
    'ऽ': ' ',
    '॓': 'े',
    '\u200b': '',  # zero-width space
    '\u200c': '',  # ZWNJ
    '\ufeff': '',  # BOM
}

def normalize_text(text):
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    return " ".join(text.split())

def download_file_multithreaded(url, dest_path, num_threads=16):
    print(f"Checking remote file size: {url}")
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get('content-length', 0))
    
    print(f"Total size: {total_size / (1024*1024):.2f} MB ({total_size} bytes)")
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size:
        print(f"File already completely downloaded at {dest_path}")
        return

    part_size = total_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * part_size
        end = total_size - 1 if i == num_threads - 1 else (i + 1) * part_size - 1
        ranges.append((start, end, f"{dest_path}.part{i}"))

    print(f"Downloading using {num_threads} parallel threads...")
    start_time = time.time()
    
    def download_part(start, end, part_file):
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req) as resp:
            with open(part_file, 'wb') as f:
                shutil.copyfileobj(resp, f)
        return part_file

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(download_part, s, e, pf) for s, e, pf in ranges]
        completed = 0
        for f in as_completed(futures):
            f.result()
            completed += 1
            print(f"  Part {completed}/{num_threads} downloaded ({completed*100//num_threads}%)")

    print("Stitching parts together into final tarball...")
    with open(dest_path, 'wb') as outfile:
        for _, _, pf in ranges:
            with open(pf, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
            os.remove(pf)
    
    elapsed = time.time() - start_time
    speed = (total_size / (1024*1024)) / elapsed
    print(f"Download complete: {dest_path} ({total_size / (1024*1024):.2f} MB in {elapsed:.1f}s, {speed:.2f} MB/s)")

def extract_hindi_from_tar(tar_path, extract_dir):
    print(f"Extracting Hindi audio from {tar_path} into {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        # Use fast system tar
        cmd = ["tar", "-xf", tar_path, "-C", extract_dir, "kb_data_clean_m4a/hindi"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print("System tar extraction succeeded.")
            return
        print(f"System tar warning ({res.stderr}), falling back to python tarfile...")
    except Exception as e:
        print(f"System tar failed ({e}), falling back to python tarfile...")

    with tarfile.open(tar_path, 'r') as tf:
        members = [m for m in tf.getmembers() if 'hindi' in m.name and m.name.endswith('.m4a')]
        print(f"Found {len(members)} Hindi .m4a files in archive.")
        for m in members:
            tf.extract(m, path=extract_dir)
    print("Extraction complete.")

def convert_m4a_to_wav(m4a_path, wav_path):
    # Convert to 16 kHz, 16-bit, mono WAV
    cmd = [
        "/usr/local/bin/ffmpeg",
        "-y",
        "-i", m4a_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-hide_banner",
        "-loglevel", "error",
        wav_path
    ]
    res = subprocess.run(cmd)
    return res.returncode == 0

def process_split(split_name, tar_url):
    print(f"\n==========================================")
    print(f"PROCESSING KATHBATH SPLIT: {split_name}")
    print(f"==========================================")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PROCESSED_HINDI_DIR, exist_ok=True)
    os.makedirs(EXTENDED_AUDIO_DIR, exist_ok=True)
    
    tar_dest = os.path.join(TEMP_DIR, f"{split_name}_audio.tar")
    extract_dest = os.path.join(TEMP_DIR, f"extracted_{split_name}")
    
    # 1. Download
    download_file_multithreaded(tar_url, tar_dest, num_threads=16)
    
    # 2. Extract
    extract_hindi_from_tar(tar_dest, extract_dest)
    
    # Remove tar to free space
    if os.path.exists(tar_dest):
        os.remove(tar_dest)
        print(f"Removed temporary archive {tar_dest}")

    # 3. Load transcripts
    transcripts_file = os.path.join(TRANSCRIPTS_DIR, split_name, "transcription_n2w.txt")
    if not os.path.exists(transcripts_file):
        raise FileNotFoundError(f"Transcript file missing: {transcripts_file}")
    
    transcript_map = {}
    with open(transcripts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                m4a_id = parts[0].strip()
                text = normalize_text(parts[1].strip())
                transcript_map[m4a_id] = text

    print(f"Loaded {len(transcript_map)} transcripts for {split_name}.")

    # 4. Find all extracted m4a files
    m4a_files = []
    for root, _, files in os.walk(extract_dest):
        for file in files:
            if file.endswith('.m4a') and file in transcript_map:
                m4a_files.append((os.path.join(root, file), file))

    print(f"Matching m4a files found: {len(m4a_files)}")

    # 5. Parallel Conversion to WAV
    print("Converting audio files to 16 kHz mono WAV using ffmpeg...")
    conversion_tasks = []
    for m4a_path, filename in m4a_files:
        stem = filename[:-4]
        parts = stem.split('-')
        speaker = f"kb_{parts[1]}_{parts[2]}" if len(parts) >= 3 else "kb_hindi"
        wav_filename = f"kb_hi_{stem.replace('-', '_')}.wav"
        
        local_wav_path = os.path.join(PROCESSED_HINDI_DIR, wav_filename)
        ext_wav_path = os.path.join(EXTENDED_AUDIO_DIR, wav_filename)
        
        conversion_tasks.append((m4a_path, local_wav_path, ext_wav_path, filename, wav_filename, speaker))

    def convert_and_copy(task):
        m4a_p, loc_w, ext_w, orig_fn, wav_fn, spk = task
        if not os.path.exists(loc_w) or os.path.getsize(loc_w) == 0:
            ok = convert_m4a_to_wav(m4a_p, loc_w)
            if not ok:
                return None
        # Copy to extened_recordings/audio
        if not os.path.exists(ext_w) or os.path.getsize(ext_w) == 0:
            shutil.copy2(loc_w, ext_w)
        # Measure duration
        try:
            info = sf.info(loc_w)
            dur = round(info.duration, 3)
            return (wav_fn, orig_fn, spk, dur)
        except Exception:
            return None

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(convert_and_copy, t) for t in conversion_tasks]
        count = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
            count += 1
            if count % 500 == 0 or count == len(conversion_tasks):
                print(f"  Processed {count}/{len(conversion_tasks)} clips...")

    elapsed = time.time() - start_time
    print(f"Audio conversion completed: {len(results)} clips in {elapsed:.1f}s")

    # Clean up extracted temp directory
    shutil.rmtree(extract_dest, ignore_errors=True)

    # 6. Create manifest rows
    new_ext_rows = []
    new_train_rows = []
    
    for wav_fn, orig_fn, spk, dur in results:
        transcript = transcript_map[orig_fn]
        # For extened_recordings/metadata.csv: audio/<filename>,transcript,language,speaker,duration
        new_ext_rows.append({
            "audio_path": f"audio/{wav_fn}",
            "transcript": transcript,
            "language": "hi",
            "speaker": spk,
            "duration": dur
        })
        # For complete_stt_dataset_23hrs/data/metadata/combined_train.csv
        new_train_rows.append({
            "audio_path": f"data/processed/kathbath_hindi/{wav_fn}",
            "transcript": transcript,
            "language": "hi",
            "speaker": spk,
            "duration": dur
        })

    # Save split manifest
    split_manifest_path = os.path.join(DATASET_DIR, "data", "metadata", f"kathbath_hindi_{split_name}.csv")
    pd.DataFrame(new_train_rows).to_csv(split_manifest_path, index=False)
    print(f"Saved split manifest: {split_manifest_path} ({len(new_train_rows)} rows)")

    # 7. Append to extened_recordings/metadata.csv
    ext_meta_path = os.path.join(EXTENDED_DIR, "metadata.csv")
    ext_df = pd.DataFrame(new_ext_rows)
    if os.path.exists(ext_meta_path):
        ext_df.to_csv(ext_meta_path, mode='a', header=False, index=False)
    else:
        ext_df.to_csv(ext_meta_path, index=False)
    print(f"Appended {len(new_ext_rows)} rows to {ext_meta_path}")

    # 8. Append to combined_train.csv
    comb_train_path = os.path.join(DATASET_DIR, "data", "metadata", "combined_train.csv")
    train_df = pd.DataFrame(new_train_rows)
    train_df.to_csv(comb_train_path, mode='a', header=False, index=False)
    print(f"Appended {len(new_train_rows)} rows to {comb_train_path}")

    return len(new_train_rows), sum(r["duration"] for r in new_train_rows) / 3600

def main():
    splits = [
        ("test_known", "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/testkn_audio.tar"),
    ]
    
    total_clips = 0
    total_hours = 0.0
    for split_name, tar_url in splits:
        n_clips, n_hrs = process_split(split_name, tar_url)
        total_clips += n_clips
        total_hours += n_hrs

    print(f"\n==========================================")
    print(f"INGESTION COMPLETE SUMMARY")
    print(f"==========================================")
    print(f"New Hindi clips added: {total_clips}")
    print(f"New Hindi hours added: {total_hours:.2f} hrs")

if __name__ == "__main__":
    main()
