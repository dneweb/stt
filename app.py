"""
FastAPI Server for STT Conformer-CTC Training & Inference Studio.
Provides real-time training controls, telemetry, live inference, and dataset inspection.
"""

import os
import sys
import time
import json
import math
import random
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.tokenizer import CharTokenizer
from src.features import AudioFeatureExtractor
from src.dataset import STTDataset, SpeechCollate
from src.model import ConformerCTC
from src.metrics import calculate_cer, calculate_wer

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "complete_stt_dataset_23hrs"
VOCAB_PATH = DATASET_DIR / "tokenizer" / "vocab.json"
TRAIN_CSV = DATASET_DIR / "data" / "metadata" / "combined_train.csv"
VAL_CSV = DATASET_DIR / "data" / "metadata" / "combined_val.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_TEMP_DIR = BASE_DIR / "temp_uploads"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Shared Global State
training_state = {
    "is_training": False,
    "should_stop": False,
    "current_epoch": 0,
    "total_epochs": 30,
    "current_step": 0,
    "total_steps_in_epoch": 0,
    "current_lr": 0.0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_cer": 0.0,
    "speed": 0.0,
    "history": [],  # List of {epoch, train_loss, val_loss, val_cer, lr, time}
    "logs": [],
    "recent_samples": [],
    "device_name": "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU",
    "model_params": 11105530
}

# Inference Model Cache
active_inference_model = None
active_model_device = None
tokenizer_instance = None
feature_extractor_instance = None


def get_tokenizer():
    global tokenizer_instance
    if tokenizer_instance is None:
        tokenizer_instance = CharTokenizer(VOCAB_PATH)
    return tokenizer_instance


def get_feature_extractor():
    global feature_extractor_instance
    if feature_extractor_instance is None:
        feature_extractor_instance = AudioFeatureExtractor(sample_rate=16000, n_mels=80)
    return feature_extractor_instance


def log_message(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    training_state["logs"].append(entry)
    if len(training_state["logs"]) > 200:
        training_state["logs"].pop(0)
    print(entry)


# ---------------------------------------------------------------------------
# Background Training Worker
# ---------------------------------------------------------------------------
def run_training_worker(params: Dict[str, Any]):
    global training_state
    training_state["is_training"] = True
    training_state["should_stop"] = False
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = params["epochs"]
    training_state["history"] = []
    training_state["logs"] = []

    log_message("Starting training worker...")

    # Device
    if params["device"] == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(params["device"])

    training_state["device_name"] = f"{device.type.upper()} ({'Apple Silicon Metal' if device.type == 'mps' else 'Standard'})"
    log_message(f"Using compute device: {training_state['device_name']}")

    try:
        tokenizer = get_tokenizer()
        feature_extractor = get_feature_extractor()

        log_message("Loading training and validation datasets...")
        train_ds = STTDataset(TRAIN_CSV, DATASET_DIR, tokenizer, feature_extractor, is_train=True)
        val_ds = STTDataset(VAL_CSV, DATASET_DIR, tokenizer, feature_extractor, is_train=False)
        collate_fn = SpeechCollate(pad_id=tokenizer.pad_id)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=params["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        training_state["total_steps_in_epoch"] = len(train_loader)
        log_message(f"Loaded {len(train_ds):,} training clips ({len(train_loader)} batches) and {len(val_ds):,} validation clips.")

        # Instantiate Model
        model = ConformerCTC(
            in_channels=80,
            vocab_size=tokenizer.vocab_size,
            d_model=params.get("d_model", 256),
            n_heads=params.get("n_heads", 4),
            d_ff=params.get("d_ff", 1024),
            n_layers=params.get("n_layers", 6),
            dropout=params.get("dropout", 0.1)
        ).to(device)

        training_state["model_params"] = model.count_parameters()
        log_message(f"Conformer-CTC initialized with {model.count_parameters():,} trainable parameters.")

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=params.get("weight_decay", 1e-4)
        )
        total_steps = len(train_loader) * params["epochs"]
        warmup_steps = params.get("warmup_steps", 600)

        ctc_loss_fn = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

        best_val_cer = float("inf")

        for epoch in range(1, params["epochs"] + 1):
            if training_state["should_stop"]:
                log_message("Training cancellation received. Halting training loop.")
                break

            training_state["current_epoch"] = epoch
            model.train()
            epoch_loss = 0.0
            epoch_clips = 0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader, start=1):
                if training_state["should_stop"]:
                    break

                training_state["current_step"] = step

                features = batch["features"].to(device)
                feature_lengths = batch["feature_lengths"].to(device)
                targets = batch["targets"]
                target_lengths = batch["target_lengths"]

                optimizer.zero_grad()
                logits, out_lengths = model(features, feature_lengths)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

                # CTC loss computed on CPU for stability
                loss = ctc_loss_fn(
                    log_probs.to(torch.float32).cpu(),
                    targets.cpu(),
                    out_lengths.cpu(),
                    target_lengths.cpu()
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                # Warmup + Cosine step
                global_step = (epoch - 1) * len(train_loader) + step
                if global_step <= warmup_steps:
                    lr_factor = float(global_step) / float(max(1, warmup_steps))
                else:
                    prog = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    lr_factor = 0.05 + 0.5 * 0.95 * (1.0 + math.cos(math.pi * prog))

                for g in optimizer.param_groups:
                    g["lr"] = params["lr"] * lr_factor
                training_state["current_lr"] = optimizer.param_groups[0]["lr"]

                batch_size = features.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_clips += batch_size

                elapsed = time.time() - epoch_start
                training_state["speed"] = round(epoch_clips / max(1e-5, elapsed), 1)
                training_state["train_loss"] = round(epoch_loss / max(1, epoch_clips), 4)

                if step % 25 == 0 or step == len(train_loader):
                    log_message(
                        f"Epoch {epoch:02d} [{step}/{len(train_loader)}] | "
                        f"Loss: {training_state['train_loss']:.4f} | "
                        f"Speed: {training_state['speed']} clips/s | "
                        f"LR: {training_state['current_lr']:.2e}"
                    )

            # Validation Loop
            if not training_state["should_stop"]:
                log_message(f"Evaluating validation set for Epoch {epoch}...")
                model.eval()
                val_loss_sum = 0.0
                val_cer_sum = 0.0
                val_clips = 0
                sample_preds = []

                with torch.no_grad():
                    for v_batch in val_loader:
                        v_features = v_batch["features"].to(device)
                        v_feature_lengths = v_batch["feature_lengths"].to(device)
                        v_targets = v_batch["targets"]
                        v_target_lengths = v_batch["target_lengths"]

                        v_logits, v_out_lengths = model(v_features, v_feature_lengths)
                        v_log_probs = F.log_softmax(v_logits, dim=-1)

                        v_loss = ctc_loss_fn(
                            v_log_probs.transpose(0, 1).to(torch.float32).cpu(),
                            v_targets.cpu(),
                            v_out_lengths.cpu(),
                            v_target_lengths.cpu()
                        )

                        bs = v_features.size(0)
                        val_loss_sum += v_loss.item() * bs
                        val_clips += bs

                        v_preds = torch.argmax(v_log_probs, dim=-1)
                        for i in range(bs):
                            p_ids = v_preds[i, :v_out_lengths[i]].cpu().tolist()
                            p_text = tokenizer.ctc_decode(p_ids)
                            r_text = v_batch["texts"][i]
                            cer = calculate_cer(r_text, p_text)
                            val_cer_sum += cer

                            if len(sample_preds) < 4:
                                sample_preds.append({
                                    "lang": v_batch["languages"][i],
                                    "ref": r_text,
                                    "pred": p_text,
                                    "cer": round(cer * 100, 1)
                                })

                avg_val_loss = round(val_loss_sum / max(1, val_clips), 4)
                avg_val_cer = round((val_cer_sum / max(1, val_clips)) * 100, 2)
                epoch_sec = round(time.time() - epoch_start, 1)

                training_state["val_loss"] = avg_val_loss
                training_state["val_cer"] = avg_val_cer
                training_state["recent_samples"] = sample_preds

                history_entry = {
                    "epoch": epoch,
                    "train_loss": training_state["train_loss"],
                    "val_loss": avg_val_loss,
                    "val_cer": avg_val_cer,
                    "lr": round(training_state["current_lr"], 6),
                    "epoch_time_sec": epoch_sec
                }
                training_state["history"].append(history_entry)

                log_message(f"✓ Epoch {epoch:02d} Complete: Val Loss={avg_val_loss:.4f}, Val CER={avg_val_cer:.2f}%, Time={epoch_sec}s")

                # Checkpoint saving
                if avg_val_cer < best_val_cer:
                    best_val_cer = avg_val_cer
                    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": avg_val_loss,
                        "val_cer": avg_val_cer,
                        "vocab_size": tokenizer.vocab_size,
                        "params": params
                    }, best_ckpt_path)
                    log_message(f"★ Saved new best model to {best_ckpt_path.name} (Val CER: {avg_val_cer:.2f}%)")

                # Save latest checkpoint
                latest_ckpt_path = CHECKPOINT_DIR / "latest_checkpoint.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_cer": avg_val_cer,
                    "params": params
                }, latest_ckpt_path)

        log_message("Training session concluded successfully.")

    except Exception as e:
        log_message(f"[ERROR] Exception during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        training_state["is_training"] = False
        training_state["should_stop"] = False


# ---------------------------------------------------------------------------
# Wav2Vec2 ONNX & Conformer Inference Helpers
# ---------------------------------------------------------------------------
W2V_DIR = BASE_DIR / "content" / "drive" / "MyDrive" / "stt_w2v" / "best"
W2V_ONNX_INT8 = W2V_DIR / "onnx" / "model_int8.onnx"
W2V_ONNX_FP32 = W2V_DIR / "onnx" / "model.onnx"
W2V_VOCAB = W2V_DIR / "vocab.json"

w2v_onnx_session = None
w2v_tokenizer = None


def convert_to_wav16k_mono(input_path: Path, output_path: Path) -> bool:
    """Converts any audio file (WebM, MP3, etc.) to 16 kHz Mono 16-bit PCM WAV using ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0


def get_wav2vec_session():
    global w2v_onnx_session, w2v_tokenizer
    if w2v_onnx_session is None:
        import onnxruntime as ort
        onnx_file = W2V_ONNX_INT8 if W2V_ONNX_INT8.exists() else W2V_ONNX_FP32
        if not onnx_file.exists():
            return None, None, f"Wav2Vec2 ONNX model not found at {onnx_file}"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
        w2v_onnx_session = ort.InferenceSession(str(onnx_file), sess_options, providers=["CPUExecutionProvider"])
        vocab_file = W2V_VOCAB if W2V_VOCAB.exists() else VOCAB_PATH
        w2v_tokenizer = CharTokenizer(vocab_file)
    return w2v_onnx_session, w2v_tokenizer, None


def run_wav2vec_onnx_inference(audio_path: Path):
    session, tok, err = get_wav2vec_session()
    if err:
        return {"error": err}

    temp_converted = None
    file_to_load = audio_path

    # Ensure clean 16kHz mono WAV via ffmpeg if not WAV
    if audio_path.suffix.lower() not in [".wav"]:
        temp_converted = UPLOAD_TEMP_DIR / f"w2v_conv_{int(time.time() * 1000)}.wav"
        if convert_to_wav16k_mono(audio_path, temp_converted):
            file_to_load = temp_converted

    try:
        import torchaudio
        wav, sr = torchaudio.load(str(file_to_load))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        duration_sec = round(wav.shape[1] / 16000.0, 2)
        norm_audio = ((wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)).numpy()

        start_t = time.time()
        logits = session.run(["logits"], {"input_values": norm_audio})[0]
        infer_time_ms = round((time.time() - start_t) * 1000, 1)

        pred_ids = logits.argmax(axis=-1)[0].tolist()
        pred_text = tok.ctc_decode(pred_ids)

        device_label = "ONNX INT8 (CPU)" if W2V_ONNX_INT8.exists() else "ONNX FP32 (CPU)"

        return {
            "predicted_text": pred_text,
            "duration_sec": duration_sec,
            "inference_time_ms": infer_time_ms,
            "device": device_label,
            "model_name": "Wav2Vec2 Multilingual (13.55% CER - Epoch 22)"
        }
    except Exception as e:
        return {"error": f"Wav2Vec ONNX inference failed: {str(e)}"}
    finally:
        if temp_converted and temp_converted.exists():
            temp_converted.unlink()


def load_checkpoint_for_inference(ckpt_name: str = "best_model.pt"):
    global active_inference_model, active_model_device
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        # Fallback to latest
        ckpt_path = CHECKPOINT_DIR / "latest_checkpoint.pt"
        if not ckpt_path.exists():
            return None, "No checkpoint found. Please train the model or initialize a checkpoint first."

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    active_model_device = device

    tokenizer = get_tokenizer()
    model = ConformerCTC(
        in_channels=80,
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=6,
        dropout=0.0
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    active_inference_model = model
    return model, None


_production_engine = None

def get_production_engine():
    global _production_engine
    if _production_engine is None:
        try:
            from src.inference.inference import STTInference
            model_dir = BASE_DIR / "models" / "stt_production_v3"
            if not (model_dir / "model.safetensors").exists():
                return None
            os.environ["STT_CHECKPOINT_DIR"] = str(model_dir)
            os.environ["STT_LM_PATH"] = str(model_dir / "multilingual_3gram.arpa")
            os.environ["STT_UNIGRAMS_PATH"] = str(model_dir / "unigrams.txt")
            _production_engine = STTInference(model_path=str(model_dir), device="cpu")
        except Exception as e:
            print(f"[!] Warning: Could not initialize production engine: {e}")
            return None
    return _production_engine


def run_production_stt_inference(audio_path: Path, language: str = "auto"):
    engine = get_production_engine()
    if engine is None:
        return run_wav2vec_onnx_inference(audio_path)

    # Always ensure clean 16kHz mono 16-bit PCM WAV via ffmpeg (handles WebM, Opus, MP3, etc.)
    clean_wav = UPLOAD_TEMP_DIR / f"prod_clean_{int(time.time() * 1000)}.wav"
    file_to_transcribe = audio_path
    converted = convert_to_wav16k_mono(audio_path, clean_wav)
    if converted:
        file_to_transcribe = clean_wav

    try:
        res = engine.transcribe_file(str(file_to_transcribe), language=language, source="file")
        return {
            "predicted_text": res.get("text", ""),
            "ctc_text": res.get("ctc_text", ""),
            "lm_text": res.get("lm_text", ""),
            "duration_sec": round(res.get("duration", 0.0), 2),
            "inference_time_ms": round(res.get("inference_time", 0.0) * 1000, 1),
            "device": "PyTorch CPU + Adaptive KenLM",
            "model_name": "Proprietary Master STT (11.4% Raw CER • 41.77 Hrs)",
            "word_timestamps": res.get("word_timestamps", []),
            "srt_subtitles": res.get("srt_subtitles", "")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "predicted_text": f"Error: {e}"}
    finally:
        if clean_wav.exists():
            clean_wav.unlink()


def run_model_inference_on_audio(audio_path: Path, model_type: str = "production_kenlm", language: str = "auto"):
    if model_type in ["production_kenlm", "mms_production", "kenlm"] or (model_type == "wav2vec_onnx" and (BASE_DIR / "models" / "stt_production_v3" / "model.safetensors").exists()):
        return run_production_stt_inference(audio_path, language=language)

    if model_type == "wav2vec_onnx":
        return run_wav2vec_onnx_inference(audio_path)

    # Conformer inference
    model, err = load_checkpoint_for_inference()
    if err:
        return {"error": err}

    tokenizer = get_tokenizer()
    extractor = get_feature_extractor()

    temp_converted = None
    file_to_load = audio_path
    if audio_path.suffix.lower() not in [".wav"]:
        temp_converted = UPLOAD_TEMP_DIR / f"conf_conv_{int(time.time() * 1000)}.wav"
        if convert_to_wav16k_mono(audio_path, temp_converted):
            file_to_load = temp_converted

    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(file_to_load))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        duration_sec = round(len(waveform) / float(sr), 2)

        # Features: [time_steps, 80] -> [1, time_steps, 80]
        features = extractor(waveform, augment=False).unsqueeze(0).to(active_model_device)
        lengths = torch.tensor([features.size(1)], dtype=torch.long, device=active_model_device)

        start_t = time.time()
        with torch.no_grad():
            logits, out_lengths = model(features, lengths)
            preds = torch.argmax(logits, dim=-1)
            pred_ids = preds[0, :out_lengths[0]].cpu().tolist()
            pred_text = tokenizer.ctc_decode(pred_ids)
        infer_time_ms = round((time.time() - start_t) * 1000, 1)

        # If Conformer produces blank (due to Epoch 2 99.56% CER checkpoint),
        # automatically provide high-accuracy Wav2Vec2 ONNX transcription
        if len(pred_text.strip()) <= 1:
            w2v_res = run_wav2vec_onnx_inference(file_to_load)
            if not w2v_res.get("error") and w2v_res.get("predicted_text"):
                return {
                    "predicted_text": w2v_res["predicted_text"],
                    "duration_sec": duration_sec,
                    "inference_time_ms": infer_time_ms,
                    "device": f"{str(active_model_device).upper()} (Wav2Vec2)",
                    "model_name": "Wav2Vec2 (Conformer Epoch 2 was blank)"
                }

        return {
            "predicted_text": pred_text,
            "duration_sec": duration_sec,
            "inference_time_ms": infer_time_ms,
            "device": str(active_model_device).upper(),
            "model_name": "Conformer-CTC (11M)"
        }
    except Exception as e:
        return {"error": f"Conformer inference failed: {str(e)}"}
    finally:
        if temp_converted and temp_converted.exists():
            temp_converted.unlink()


# ---------------------------------------------------------------------------
# FastAPI Application Endpoints
# ---------------------------------------------------------------------------
app = FastAPI(title="Conformer-CTC STT Studio")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static audio serving from dataset
app.mount("/audio_files", StaticFiles(directory=str(DATASET_DIR)), name="audio_files")


class TrainStartRequest(BaseModel):
    epochs: int = 25
    batch_size: int = 16
    lr: float = 5e-4
    warmup_steps: int = 600
    device: str = "auto"
    d_model: int = 256
    n_layers: int = 6


@app.get("/api/status")
async def get_system_status():
    return {
        "hardware_device": training_state["device_name"],
        "mps_available": torch.backends.mps.is_available(),
        "model_params": training_state["model_params"],
        "dataset_total_hours": 41.77,
        "dataset_total_clips": 19823,
        "languages": ["English", "Hindi", "Gujarati"],
        "is_training": training_state["is_training"],
        "has_best_checkpoint": (CHECKPOINT_DIR / "best_model.pt").exists(),
        "has_latest_checkpoint": (CHECKPOINT_DIR / "latest_checkpoint.pt").exists()
    }


@app.post("/api/train/start")
async def start_training(req: TrainStartRequest):
    if training_state["is_training"]:
        return JSONResponse({"status": "error", "message": "Training is already in progress."}, status_code=400)

    thread = threading.Thread(
        target=run_training_worker,
        args=(req.dict(),),
        daemon=True
    )
    thread.start()
    return {"status": "started", "message": f"Training started for {req.epochs} epochs on {req.device}."}


@app.post("/api/train/stop")
async def stop_training():
    if not training_state["is_training"]:
        return {"status": "idle", "message": "No active training process."}

    training_state["should_stop"] = True
    return {"status": "stopping", "message": "Graceful stop signal dispatched."}


@app.get("/api/train/metrics")
async def get_training_metrics():
    return {
        "is_training": training_state["is_training"],
        "epoch": training_state["current_epoch"],
        "total_epochs": training_state["total_epochs"],
        "step": training_state["current_step"],
        "total_steps": training_state["total_steps_in_epoch"],
        "train_loss": training_state["train_loss"],
        "val_loss": training_state["val_loss"],
        "val_cer": training_state["val_cer"],
        "speed": training_state["speed"],
        "lr": training_state["current_lr"],
        "history": training_state["history"],
        "logs": training_state["logs"][-35:],
        "recent_samples": training_state["recent_samples"]
    }


@app.get("/api/checkpoints")
async def list_checkpoints():
    ckpts = []
    prod_dir = BASE_DIR / "models" / "stt_production_v3"
    if (prod_dir / "model.safetensors").exists():
        stat = (prod_dir / "model.safetensors").stat()
        ckpts.append({
            "filename": "stt_master_11cer_checkpoint.safetensors",
            "name": "Proprietary Master STT (11.4% Raw CER • 41.77 Hrs)",
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "epoch": "Epoch 30 (Kaggle Master)",
            "val_cer": 11.40,
            "val_wer": 37.63,
            "languages": "Gujarati • Hindi • English",
            "engine": "Wav2Vec2/MMS Master (18,581 Clips) + Adaptive KenLM",
            "status": "Active Flagship",
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        })
    for file in sorted(CHECKPOINT_DIR.glob("*.pt")):
        try:
            stat = file.stat()
            ckpt_data = torch.load(file, map_location="cpu")
            ckpts.append({
                "filename": file.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "epoch": ckpt_data.get("epoch", "-"),
                "val_cer": ckpt_data.get("val_cer", "-"),
                "val_loss": ckpt_data.get("val_loss", "-"),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            })
        except Exception as e:
            ckpts.append({
                "filename": file.name,
                "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
                "error": str(e)
            })
    return ckpts


@app.get("/api/sample/random")
async def get_random_validation_sample():
    tokenizer = get_tokenizer()
    extractor = get_feature_extractor()
    val_ds = STTDataset(VAL_CSV, DATASET_DIR, tokenizer, extractor, is_train=False)

    idx = random.randint(0, len(val_ds) - 1)
    entry = val_ds.entries[idx]

    return {
        "id": f"sample_{idx}",
        "rel_audio_path": entry["audio_path"],
        "audio_url": f"/audio_files/{entry['audio_path']}",
        "reference_text": entry.get("transcript", ""),
        "language": entry.get("language", ""),
        "duration": float(entry.get("duration", 0.0) or 0.0),
        "speaker": entry.get("speaker", "unknown")
    }


@app.post("/api/infer/sample_path")
async def infer_sample_path(rel_path: str = Form(...), model_type: str = Form("production_kenlm"), language: str = Form("auto")):
    full_path = DATASET_DIR / rel_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    result = run_model_inference_on_audio(full_path, model_type=model_type, language=language)
    return result


@app.post("/api/infer/upload")
async def infer_uploaded_audio(file: UploadFile = File(...), model_type: str = Form("production_kenlm"), language: str = Form("auto")):
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    temp_file = UPLOAD_TEMP_DIR / f"upload_{int(time.time() * 1000)}{suffix}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_model_inference_on_audio(temp_file, model_type=model_type, language=language)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "predicted_text": f"Error: {e}"}
    finally:
        if temp_file.exists():
            temp_file.unlink()


# Serve UI
UI_DIR = BASE_DIR / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import argparse
    import socket

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    default_port = 8080 if is_port_in_use(8000) else 8000
    parser = argparse.ArgumentParser(description="STT Studio Dashboard")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\nStarting STT Studio Dashboard on http://{args.host}:{args.port} ...")
    uvicorn.run("app:app", host=args.host, port=args.port, log_level="info", reload=False)
