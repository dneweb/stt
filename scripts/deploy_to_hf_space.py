"""
Automated Deployment Script for Hugging Face Spaces (Docker SDK).
Deploys the Proprietary Trilingual STT (10.30% CER Model + Custom Studio UI)
directly to Hugging Face Spaces with 24/7 Free Cloud Hosting.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
USERNAME = "deeeeep04"
SPACE_NAME = "trilingual-stt-studio"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

BASE_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = BASE_DIR / "hf_space_build"

def prepare_build_directory():
    print(f"📦 Preparing build directory at {BUILD_DIR} ...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create Space README.md with YAML metadata
    readme_content = f"""---
title: Trilingual STT Studio (Gujarati • Hindi • English)
emoji: 🎙️
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🎙️ Proprietary Trilingual Speech Recognition Studio
### Flagship STT Engine (Gujarati, Hindi, English) • 10.30% Character Error Rate (CER)

This is the production deployment of our trilingual speech-to-text platform, fine-tuned on Tesla GPUs over 41.77+ hours of multilingual speech data.

- **Model:** Wav2Vec2 / MMS-300M Fine-Tuned (24-Layer Transformer Encoder + CTC)
- **Language Model:** Adaptive KenLM Trilingual 3-Gram Decoder
- **Validation Accuracy:** **10.30% Raw CER** / **34.88% Raw WER**
- **Exact Accuracy Benchmarking:** Real-time mathematical Levenshtein metric verification.
"""
    with open(BUILD_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  ✓ Created README.md with Docker Space metadata")

    # 2. Create requirements.txt
    requirements_content = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.0+cpu
torchaudio==2.3.0+cpu
transformers==4.40.2
safetensors>=0.4.2
soundfile>=0.12.1
uvicorn>=0.29.0
fastapi>=0.110.0
python-multipart>=0.0.9
pydantic>=2.7.0
numpy<2.0.0
scipy>=1.13.0
pyctcdecode>=0.5.0
editdistance>=0.8.1
"""
    with open(BUILD_DIR / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("  ✓ Created requirements.txt (Optimized for Fast CPU Container)")

    # 3. Create Dockerfile
    dockerfile_content = """FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    PORT=7860 \\
    STT_CHECKPOINT_DIR=/app/models/stt_production_v3 \\
    STT_LM_PATH=/app/models/stt_production_v3/multilingual_3gram.arpa \\
    STT_UNIGRAMS_PATH=/app/models/stt_production_v3/unigrams.txt

WORKDIR /app

# Install system audio and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ffmpeg \\
    libsndfile1 \\
    build-essential \\
    cmake \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir https://github.com/kpu/kenlm/archive/master.zip

# Copy application source & model weights
COPY app.py .
COPY src/ ./src/
COPY ui/ ./ui/
COPY models/stt_production_v3/ ./models/stt_production_v3/

# Create non-root user for Hugging Face Spaces security
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""
    with open(BUILD_DIR / "Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    print("  ✓ Created Dockerfile for Port 7860")

    # 4. Copy app.py
    shutil.copy2(BASE_DIR / "app.py", BUILD_DIR / "app.py")
    print("  ✓ Copied app.py")

    # 5. Copy src/ (ignoring __pycache__)
    shutil.copytree(BASE_DIR / "src", BUILD_DIR / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("  ✓ Copied src/ modules")

    # 6. Copy ui/
    shutil.copytree(BASE_DIR / "ui", BUILD_DIR / "ui")
    print("  ✓ Copied ui/ frontend assets")

    # 7. Copy models/stt_production_v3/ (10.30% model weights)
    target_model_dir = BUILD_DIR / "models" / "stt_production_v3"
    target_model_dir.mkdir(parents=True, exist_ok=True)
    src_model_dir = BASE_DIR / "models" / "stt_production_v3"

    for f in ["model.safetensors", "config.json", "vocab.json", "multilingual_3gram.arpa", "unigrams.txt"]:
        src_f = src_model_dir / f
        if src_f.exists():
            shutil.copy2(src_f, target_model_dir / f)
            print(f"  ✓ Copied model asset: {f} ({src_f.stat().st_size / (1024*1024):.1f} MB)")
        else:
            print(f"  ⚠️ Warning: {f} missing from {src_model_dir}")

    print("✓ Build directory preparation complete!\n")


def deploy_to_huggingface():
    api = HfApi(token=TOKEN)

    print("🚀 Connecting to Hugging Face...")
    user = api.whoami()
    print(f"✓ Authenticated as: {user['name']} ({user.get('fullname', '')})")

    # 1. Create Space repo
    print(f"📁 Ensuring Space repository '{REPO_ID}' exists ...")
    space_url = api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
    print(f"✓ Space repository ready: {space_url}")

    # 2. Upload build folder
    print(f"⬆️ Uploading package (~1.3 GB) to Hugging Face Space '{REPO_ID}' ...")
    print("   (This will upload model weights via LFS and application files...)")

    api.upload_folder(
        folder_path=str(BUILD_DIR),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Production Trilingual STT Studio (10.30% CER Model + Custom UI)"
    )

    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT UPLOAD COMPLETE!")
    print(f"🌐 Your 24/7 Live Public URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"⚡ Direct Fullscreen App URL : https://{USERNAME}-{SPACE_NAME}.hf.space")
    print("=" * 70)


if __name__ == "__main__":
    prepare_build_directory()
    deploy_to_huggingface()

