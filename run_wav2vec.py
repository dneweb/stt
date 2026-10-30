#!/usr/bin/env python3
"""
Inference & Evaluation Runner for fine-tuned Wav2Vec2-CTC Multilingual STT Model.
Supports English, Hindi, and Gujarati.
Supports both PyTorch and high-performance, CUDA-independent ONNX (INT8 / FP32) runtime.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Union
import torch
import torchaudio

from src.tokenizer import CharTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "content" / "drive" / "MyDrive" / "stt_w2v" / "best"
DEFAULT_VOCAB_PATH = DEFAULT_MODEL_DIR / "vocab.json"
DEFAULT_ONNX_INT8 = DEFAULT_MODEL_DIR / "onnx" / "model_int8.onnx"
DEFAULT_ONNX_FP32 = DEFAULT_MODEL_DIR / "onnx" / "model.onnx"


class Wav2VecTranscriber:
    """
    Unified Transcriber supporting:
      1. 'onnx_int8': Ultra-fast, 70% compressed (~340MB), 100% CUDA-free CPU inference (Recommended).
      2. 'onnx_fp32': Unquantized ONNX engine.
      3. 'pytorch': Original PyTorch model.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        vocab_path: Union[str, Path] = DEFAULT_VOCAB_PATH,
        engine: str = "onnx_int8",
        device: Optional[str] = None
    ):
        self.model_dir = Path(model_dir)
        self.vocab_path = Path(vocab_path)
        self.engine = engine.lower()

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found at: {self.vocab_path}")
        self.tokenizer = CharTokenizer(self.vocab_path)

        # Fallback to pytorch if requested onnx file does not exist
        if self.engine == "onnx_int8" and not DEFAULT_ONNX_INT8.exists():
            print(f"Notice: {DEFAULT_ONNX_INT8} not found. Falling back to PyTorch engine.")
            self.engine = "pytorch"
        elif self.engine == "onnx_fp32" and not DEFAULT_ONNX_FP32.exists():
            print(f"Notice: {DEFAULT_ONNX_FP32} not found. Falling back to PyTorch engine.")
            self.engine = "pytorch"

        if self.engine in ("onnx_int8", "onnx_fp32"):
            import onnxruntime as ort
            onnx_path = DEFAULT_ONNX_INT8 if self.engine == "onnx_int8" else DEFAULT_ONNX_FP32
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)

            engine_label = "INT8 Quantized (340 MB)" if self.engine == "onnx_int8" else "FP32 (1.2 GB)"
            print(f"Loading Wav2Vec2 ONNX [{engine_label}] on CPU...")
            self.session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])
            print(f"✓ ONNX model ready! Vocab size: {self.tokenizer.vocab_size} (100% CUDA-Independent)")
        else:
            from transformers import Wav2Vec2ForCTC
            if device is not None:
                self.device = torch.device(device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            print(f"Loading Wav2Vec2 PyTorch model on device: {self.device}...")
            self.model = Wav2Vec2ForCTC.from_pretrained(str(self.model_dir))
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ PyTorch model ready! Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav, sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)

        # Standard Wav2Vec2 zero-mean unit-variance audio normalization
        norm_audio = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        if self.engine in ("onnx_int8", "onnx_fp32"):
            audio_np = norm_audio.numpy()
            onnx_logits = self.session.run(["logits"], {"input_values": audio_np})[0]
            predicted_ids = onnx_logits.argmax(axis=-1)[0].tolist()
        else:
            input_values = norm_audio.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        return self.tokenizer.ctc_decode(predicted_ids)


def run_sample_demo(transcriber: Wav2VecTranscriber):
    base_dir = Path(__file__).resolve().parent
    samples = [
        {
            "lang": "English",
            "path": base_dir / "complete_stt_dataset_23hrs/data/processed/all_speakers/priyanshi_000202.wav",
            "ground_truth": "Full, fool, pull. Don't be a fool, pull the door."
        },
        {
            "lang": "Hindi",
            "path": base_dir / "complete_stt_dataset_23hrs/data/processed/all_speakers/priyanshi_000475.wav",
            "ground_truth": "शाम सात बजे का reminder लगाओ।"
        },
        {
            "lang": "Gujarati",
            "path": base_dir / "complete_stt_dataset_23hrs/data/processed/all_speakers/divya_000343.wav",
            "ground_truth": "ઠંડો પવન ફૂંકાતાં પાંદડાં ખરી પડ્યાં."
        }
    ]

    print("\n" + "=" * 65)
    print(f"   Wav2Vec2 Inference Demo [{transcriber.engine.upper()} Engine]")
    print("=" * 65)

    for s in samples:
        if not s["path"].exists():
            print(f"Sample audio not found: {s['path']}")
            continue
        t0 = time.time()
        prediction = transcriber.transcribe(s["path"])
        latency = (time.time() - t0) * 1000

        print(f"\nLanguage:     {s['lang']}")
        print(f"File:         {s['path'].name}")
        print(f"Ground Truth: {s['ground_truth']}")
        print(f"Prediction:   {prediction}")
        print(f"Latency:      {latency:.1f} ms")
    print("\n" + "=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 STT inference (CUDA-independent ONNX / PyTorch)")
    parser.add_argument(
        "--engine",
        type=str,
        default="onnx_int8",
        choices=["onnx_int8", "onnx_fp32", "pytorch"],
        help="Inference engine: 'onnx_int8' (fastest & lightweight), 'onnx_fp32', or 'pytorch' (default: onnx_int8)"
    )
    parser.add_argument("--device", type=str, default=None, help="Device for PyTorch engine: 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--audio", type=str, help="Path to an audio file (.wav, .webm, .mp3, etc.)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration on sample English, Hindi, and Gujarati clips")
    args = parser.parse_args()

    transcriber = Wav2VecTranscriber(engine=args.engine, device=args.device)

    if args.audio:
        t0 = time.time()
        text = transcriber.transcribe(args.audio)
        latency = (time.time() - t0) * 1000
        print("\nTranscription Result:")
        print(f"  {text}")
        print(f"  (Processed in {latency:.1f} ms via {args.engine})\n")
    else:
        run_sample_demo(transcriber)


if __name__ == "__main__":
    main()
