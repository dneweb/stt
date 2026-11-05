"""
Helper script to unpack and activate downloaded Kaggle best model (10.30% CER).
Extracts weights into models/stt_production_v3 and verifies model integrity.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = Path.home() / "Downloads"
TARGET_DIR = BASE_DIR / "models" / "stt_production_v3"
BACKUP_DIR = BASE_DIR / "models" / "stt_backup_11.4"


def find_downloaded_zip():
    candidates = [
        DOWNLOADS_DIR / "stt_master_best_model.zip",
        DOWNLOADS_DIR / "stt_master_best_model (1).zip",
        DOWNLOADS_DIR / "stt_master_final.zip",
        BASE_DIR / "stt_master_best_model.zip"
    ]
    for c in candidates:
        if c.exists():
            return c
    
    # Check any zip with 'stt' and 'model' in Downloads
    if DOWNLOADS_DIR.exists():
        for f in DOWNLOADS_DIR.iterdir():
            if f.suffix.lower() == ".zip" and ("stt" in f.name.lower() or "master" in f.name.lower() or "10" in f.name.lower()):
                return f
    return None


def install_model():
    zip_path = find_downloaded_zip()
    if not zip_path:
        print(f"❌ Zip file not found yet in {DOWNLOADS_DIR}.")
        print("👉 Please download 'stt_master_best_model.zip' from Kaggle into your Downloads folder first.")
        return False

    print(f"📦 Found downloaded model archive: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")

    # 1. Backup existing model
    if TARGET_DIR.exists() and not BACKUP_DIR.exists():
        print(f"💾 Backing up previous checkpoint to {BACKUP_DIR} ...")
        shutil.copytree(TARGET_DIR, BACKUP_DIR)

    # 2. Extract new model
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Unpacking new 10.30% model into {TARGET_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(TARGET_DIR)

    # 3. Verify extracted files
    required_files = ["model.safetensors", "config.json", "vocab.json"]
    all_ok = True
    for req in required_files:
        p = TARGET_DIR / req
        if p.exists():
            print(f"  ✓ Verified: {req} ({p.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  ❌ Missing: {req}")
            all_ok = False

    if all_ok:
        print("\n🎉 SUCCESS! New 10.30% CER Model is now installed in production.")
        print("🔄 The UI Studio will automatically use this model for all live microphone and recording testing.")
        return True
    else:
        print("\n⚠️ Extraction finished with warnings. Please verify files.")
        return False


if __name__ == "__main__":
    install_model()

