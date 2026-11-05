"""
Upload full STT Studio (10.30% CER Model + Custom UI + Inference Engine)
to Hugging Face Model Repository: deeeeep04/stt-gujarati-hindi-10cer
This allows Google Colab to 1-click clone and run the full server with GPU.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "deeeeep04/stt-gujarati-hindi-10cer"

BASE_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = BASE_DIR / "hf_model_staging"

def prepare_staging():
    print(f"📦 Staging files in {STAGING_DIR} ...")
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model weights
    models_dir = STAGING_DIR / "models" / "stt_production_v3"
    models_dir.mkdir(parents=True, exist_ok=True)
    src_models = BASE_DIR / "models" / "stt_production_v3"

    for fname in ["model.safetensors", "config.json", "vocab.json", "multilingual_3gram.arpa", "unigrams.txt"]:
        src_f = src_models / fname
        if src_f.exists():
            shutil.copy2(src_f, models_dir / fname)
            print(f"  ✓ Copied model file: {fname} ({src_f.stat().st_size / (1024*1024):.1f} MB)")
        else:
            raise FileNotFoundError(f"Missing required model file: {src_f}")

    # 2. App & Source code
    shutil.copy2(BASE_DIR / "app.py", STAGING_DIR / "app.py")
    print("  ✓ Copied app.py")

    shutil.copytree(BASE_DIR / "src", STAGING_DIR / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("  ✓ Copied src/ modules")

    shutil.copytree(BASE_DIR / "ui", STAGING_DIR / "ui")
    print("  ✓ Copied ui/ custom frontend")

    # 3. README.md
    readme = """---
language:
- gu
- hi
- en
license: mit
tags:
- automatic-speech-recognition
- wav2vec2
- mms
- speech
- kenlm
- trilingual
metrics:
- cer
- wer
model-index:
- name: trilingual-stt-10cer
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    metrics:
    - type: cer
      value: 10.30
      name: Character Error Rate
    - type: wer
      value: 34.88
      name: Word Error Rate
---

# 🎙️ Proprietary Trilingual STT Model (Gujarati • Hindi • English)
### Record Checkpoint: 10.30% Raw CER • Wav2Vec2/MMS-300M + KenLM

Fine-tuned on 41.77+ hours of trilingual speech audio with KenLM 3-Gram Language Model.
"""
    with open(STAGING_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print("  ✓ Created README.md")

    # 4. requirements.txt
    reqs = """torch
torchaudio
transformers
safetensors
soundfile
uvicorn
fastapi
python-multipart
pydantic
numpy<2.0.0
scipy
pyctcdecode
editdistance
"""
    with open(STAGING_DIR / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(reqs)
    print("  ✓ Created requirements.txt")
    print("✓ Staging complete!\n")


def upload_to_hf():
    api = HfApi(token=TOKEN)
    print(f"🚀 Uploading STT Studio to https://huggingface.co/{REPO_ID} ...")
    print("   (Uploading 1.2 GB model weights + KenLM + UI via Git LFS...)")

    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload 10.30% CER STT Production Model, KenLM, and Custom Studio UI"
    )

    print("\n" + "=" * 70)
    print("🎉 MODEL & STUDIO REPOSITORY UPLOAD COMPLETE!")
    print(f"🔗 Repository: https://huggingface.co/{REPO_ID}")
    print("=" * 70)


if __name__ == "__main__":
    prepare_staging()
    upload_to_hf()

