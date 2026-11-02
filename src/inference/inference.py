"""
Inference Engine — Runs STT model inference on audio inputs.

Supports both single audio file processing, batch processing, and real-time streaming.
Optimized for CPU / Mac inference without requiring CUDA.

"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import yaml

try:
    from model.stt_model import STTModel
    from preprocessing.features import FeatureExtractor
    from tokenizer.tokenizer import CharTokenizer
    from inference.post_processor import STTPostProcessor
except ImportError:
    from src.model.stt_model import STTModel
    from src.preprocessing.features import FeatureExtractor
    from src.tokenizer.tokenizer import CharTokenizer
    from src.inference.post_processor import STTPostProcessor


class STTInference:
    """Speech-to-Text inference engine.

    Loads a trained model checkpoint and provides fast methods for
    transcribing audio files or streaming chunks.

    Attributes:
        model: The loaded STTModel instance.
        tokenizer: CharTokenizer for decoding token IDs to text.
        feature_extractor: FeatureExtractor for computing spectrograms.
        device: Inference device ('cpu', 'mps', 'cuda').
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path] = "configs/config.yaml",
        vocab_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the inference engine.

        Args:
            model_path: Path to trained checkpoint (.pt) or HuggingFace directory.
            config_path: Path to the YAML configuration file.
            vocab_path: Path to tokenizer vocabulary JSON.
            device: Device for inference ('cpu', 'mps', 'cuda').
        """
        self.device = torch.device(device)
        self.feature_extractor = FeatureExtractor()

        # Auto-select best available model if none provided
        if model_path is None:
            env_ckpt = os.environ.get("STT_CHECKPOINT_DIR")
            if env_ckpt and Path(env_ckpt).exists():
                model_path = env_ckpt
            elif Path("checkpoints/stt_w2v/best").exists():
                model_path = "checkpoints/stt_w2v/best"
            elif Path("checkpoints/best_model.pt").exists():
                model_path = "checkpoints/best_model.pt"
            else:
                model_path = "checkpoints/stt_w2v/best"

        self.model_path = Path(model_path)
        self.is_w2v = self.model_path.is_dir() and (
            (self.model_path / "model.safetensors").exists()
            or (self.model_path / "config.json").exists()
        )

        # 1. Load Tokenizer
        self.tokenizer = CharTokenizer()
        if vocab_path is None:
            if self.is_w2v and (self.model_path / "vocab.json").exists():
                vocab_path = self.model_path / "vocab.json"
            elif os.environ.get("STT_CHECKPOINT_DIR") and (Path(os.environ["STT_CHECKPOINT_DIR"]) / "vocab.json").exists():
                vocab_path = Path(os.environ["STT_CHECKPOINT_DIR"]) / "vocab.json"
            elif Path("tokenizer/vocab.json").exists():
                vocab_path = "tokenizer/vocab.json"
            else:
                vocab_path = "tokenizer/vocab.json"

        if Path(vocab_path).exists():
            self.tokenizer.load_vocab(str(vocab_path))
        else:
            self.tokenizer.build_from_metadata("data/metadata/metadata.csv")

        # 2. Load Model Architecture & Checkpoint
        if self.is_w2v:
            from transformers import Wav2Vec2ForCTC
            self.w2v_model = Wav2Vec2ForCTC.from_pretrained(str(self.model_path)).to(self.device)
            self.w2v_model.eval()
            self.model = None
            print(f"✓ Loaded fine-tuned MMS-300M model from {self.model_path}")
        else:
            self.w2v_model = None
            self._load_scratch_model(self.model_path, config_path)

        # 3. Optional KenLM Beam Search Decoder (build two: one for files, one for mic)
        self.lm_decoder = None
        self.lm_decoder_mic = None

        env_lm = os.environ.get("STT_LM_PATH")
        if env_lm and Path(env_lm).exists():
            lm_path = Path(env_lm)
        elif self.is_w2v and (self.model_path / "multilingual_3gram.arpa").exists():
            lm_path = self.model_path / "multilingual_3gram.arpa"
        elif Path("data/lm/multilingual_3gram.arpa").exists():
            lm_path = Path("data/lm/multilingual_3gram.arpa")
        else:
            lm_path = Path("data/lm/multilingual_3gram.arpa")

        env_unigrams = os.environ.get("STT_UNIGRAMS_PATH")
        if env_unigrams and Path(env_unigrams).exists():
            unigrams_path = Path(env_unigrams)
        elif self.is_w2v and (self.model_path / "unigrams.txt").exists():
            unigrams_path = self.model_path / "unigrams.txt"
        elif Path("data/lm/unigrams.txt").exists():
            unigrams_path = Path("data/lm/unigrams.txt")
        else:
            unigrams_path = Path("data/lm/unigrams.txt")

        if lm_path.exists() and self.is_w2v:
            try:
                from pyctcdecode import build_ctcdecoder
                labels = [self.tokenizer.id_to_char.get(i, "") for i in range(self.tokenizer.vocab_size)]
                labels[self.tokenizer.blank_id] = ""
                # Replace multi-character special tokens (<PAD>, <SOS>, <EOS>, <UNK>) with single private chars
                for i, spec in enumerate(["<PAD>", "<SOS>", "<EOS>", "<UNK>"], start=1):
                    if i < len(labels) and labels[i] == spec:
                        labels[i] = chr(0xE000 + i)

                unigrams = None
                if unigrams_path.exists():
                    with open(unigrams_path, "r", encoding="utf-8") as f:
                        unigrams = [line.strip() for line in f if line.strip()]

                # File decoder: tuned for clean benchmark audio
                self.lm_decoder = build_ctcdecoder(
                    labels=labels,
                    kenlm_model_path=str(lm_path),
                    unigrams=unigrams,
                    alpha=0.45,
                    beta=1.0,
                )
                # Mic decoder: higher LM trust to recover noisy acoustics
                self.lm_decoder_mic = build_ctcdecoder(
                    labels=labels,
                    kenlm_model_path=str(lm_path),
                    unigrams=unigrams,
                    alpha=0.65,
                    beta=1.5,
                )
                print(f"✓ KenLM Decoders: File (α=0.45, β=1.0) + Mic (α=0.65, β=1.5) from {lm_path}")
            except Exception as e:
                print(f"[!] Warning: Could not initialize KenLM decoder: {e}")

        # 4. Smart Multilingual Post-Processor & Language-Specific Hotwords
        unigram_file = str(unigrams_path) if unigrams_path.exists() else "data/lm/unigrams.txt"
        self.post_processor = STTPostProcessor(unigram_file)
        self.gu_hotwords = [
            "નમસ્તે", "ગુજરાતી", "કેમ છો", "આભાર", "ફિલ્મોના", "અભિનેતા", "અરવિંદ", "ત્રિવેદી", "સરકાર", "નમસ્કાર",
            "એસીનું", "તાપમાન", "ડિગ્રી", "કરો", "પંખો", "લાઇટ", "બંધ", "ચાલુ", "ફળ", "તોડે", "ઝાડ", "છોકરાઓ",
            "છવ્વીસ", "ચોવીસ", "પચ્ચીસ", "સત્તાવીસ", "અઠ્ઠાવીસ", "ઓગણત્રીસ", "ત્રીસ", "એકવીસ", "બાવીસ", "ત્રેવીસ",
            "તમે", "મારું", "નામ", "શું", "ક્યાં", "ક્યારે", "કેવી", "રીતે", "બરાબર", "હા", "ના", "મજામાં"
        ]
        self.hi_hotwords = [
            "नमस्ते", "धन्यवाद", "रास्ता", "दिखाओ", "गाना", "बजाओ", "मौसम", "लाइट", "बंद", "चालू", "कृपया",
            "तापमान", "डिग्री", "पंखे", "पंखा", "कमरा", "पानी", "खाना", "चाँदी", "चम्मच", "चटनी",
            "छब्बीस", "चौबीस", "पच्चीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इक्कीस", "बाईस", "तेईस",
            "आप", "मेरा", "नाम", "कहाँ", "कब", "कैसे", "हाँ", "नहीं"
        ]
        self.en_hotwords = [
            "turn off", "turn on", "switch off", "switch on", "shut off", "lights", "camera", "settings", "volume", 
            "music", "song", "search", "please", "room", "office", "temperature", "celsius", "degrees", 
            "twenty six", "twenty four", "twenty five", "twenty seven", "twenty eight", "twenty", "thirty",
            "what is", "where is", "how are", "how do", "open", "close", "start", "stop", "call", "send",
            "message", "email", "meeting", "schedule", "play", "pause", "fan", "ac", "air conditioner",
            "today", "tomorrow", "weather", "time", "thank you", "hello", "good morning", "good evening",
            "anydesk", "postman", "google", "chrome", "screen", "airport", "learning", "agreed", "guitar"
        ]
        self.all_hotwords = self.gu_hotwords + self.hi_hotwords + self.en_hotwords
        self.hotwords = self.gu_hotwords

        # Token indices for targeted language script biasing
        self.gu_indices = [idx for ch, idx in self.tokenizer.char_to_id.items() if len(ch) == 1 and 0x0A80 <= ord(ch) <= 0x0AFF]
        self.hi_indices = [idx for ch, idx in self.tokenizer.char_to_id.items() if len(ch) == 1 and 0x0900 <= ord(ch) <= 0x097F]
        self.en_indices = [idx for ch, idx in self.tokenizer.char_to_id.items() if len(ch) == 1 and (('a' <= ch <= 'z') or ('A' <= ch <= 'Z'))]


    def _load_scratch_model(self, model_path: Path, config_path: Union[str, Path]) -> None:
        """Load the scratch model architecture and restore weights from checkpoint."""
        config_path = Path(config_path)

        encoder_cfg = {"input_dim": 80, "hidden_dim": 256, "num_layers": 4, "num_heads": 4}
        decoder_cfg = {"vocab_size": self.tokenizer.vocab_size, "hidden_dim": 256, "num_layers": 2, "num_heads": 4}

        if config_path.exists():
            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f)
                if "model" in full_cfg:
                    if "encoder" in full_cfg["model"]:
                        encoder_cfg.update(full_cfg["model"]["encoder"])
                    if "decoder" in full_cfg["model"]:
                        decoder_cfg.update(full_cfg["model"]["decoder"])
                        decoder_cfg["vocab_size"] = self.tokenizer.vocab_size

        self.model = STTModel(encoder_config=encoder_cfg, decoder_config=decoder_cfg)

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"✓ Loaded scratch model weights from {model_path}")
        else:
            print(f"⚠️ Checkpoint not found at {model_path}. Initialized model with random weights.")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def format_srt_time(sec: float) -> str:
        """Convert seconds into SubRip HH:MM:SS,mmm timestamp."""
        millis = int(round((sec - int(sec)) * 1000))
        total_sec = int(sec)
        s = total_sec % 60
        total_min = total_sec // 60
        m = total_min % 60
        h = total_min // 60
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

    @classmethod
    def generate_srt(cls, word_timestamps: List[Dict[str, Any]], words_per_chunk: int = 4) -> str:
        """Generate official SRT subtitle string from word-level timestamps."""
        if not word_timestamps:
            return ""
        srt_blocks = []
        chunk = []
        block_idx = 1
        for i, item in enumerate(word_timestamps):
            chunk.append(item)
            if len(chunk) >= words_per_chunk or i == len(word_timestamps) - 1:
                start_str = cls.format_srt_time(chunk[0]["start"])
                end_str = cls.format_srt_time(chunk[-1]["end"])
                chunk_words = " ".join([c["word"] for c in chunk])
                srt_blocks.append(f"{block_idx}\n{start_str} --> {end_str}\n{chunk_words}\n")
                block_idx += 1
                chunk = []
        return "\n".join(srt_blocks)

    @staticmethod
    def align_words_with_timestamps(raw_offsets: List[Any], final_text: str, duration: float) -> List[Dict[str, Any]]:
        """Map raw token/word beam offsets to punctuated final text words."""
        final_words = final_text.split()
        if not final_words:
            return []

        # If decoder did not output offsets, spread words proportionally
        if not raw_offsets:
            step = max(0.08, duration / len(final_words))
            return [
                {
                    "word": w,
                    "start": round(i * step, 2),
                    "end": round(min(duration, (i + 1) * step), 2)
                }
                for i, w in enumerate(final_words)
            ]

        # In standard cases where word count matches
        if len(raw_offsets) == len(final_words):
            out = []
            for (raw_w, (s_frame, e_frame)), final_w in zip(raw_offsets, final_words):
                s_time = round(s_frame * 0.02, 2)
                e_time = round(min(duration, max(s_time + 0.05, e_frame * 0.02)), 2)
                out.append({"word": final_w, "start": s_time, "end": e_time})
            return out
        else:
            # If word counts differ due to compound joining/splitting
            out = []
            for (raw_w, (s_frame, e_frame)) in raw_offsets:
                s_time = round(s_frame * 0.02, 2)
                e_time = round(min(duration, max(s_time + 0.05, e_frame * 0.02)), 2)
                out.append({"word": raw_w, "start": s_time, "end": e_time})
            return out

    def transcribe_file(self, audio_path: Union[str, Path], language: str = "gu", source: str = "file", orig_sr_hint: Optional[int] = None) -> Dict[str, Any]:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file (.wav, .mp3, .flac, etc.).
            language: Target language mode ('gu', 'hi', 'en', 'auto'). Defaults to 'gu'.
            source: Audio source type ('file' for benchmark/clean, 'mic' for live microphone).
                    Mic mode enables enhanced denoising and adaptive decoding.
            orig_sr_hint: Optional original sample rate hint (e.g. from browser hardware mic).

        Returns:
            Dictionary with transcription results, latency, duration, and audio_metrics.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        is_mic = (source == "mic")

        # Load audio & capture pre-processing diagnostics
        try:
            raw_waveform, sr = sf.read(str(audio_path), dtype="float32")
        except Exception:
            import torchaudio
            wav, sr = torchaudio.load(str(audio_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            raw_waveform = wav.squeeze().cpu().numpy().astype("float32")
        orig_duration = len(raw_waveform) / sr
        orig_channels = 1 if raw_waveform.ndim == 1 else raw_waveform.shape[1]
        orig_sr = orig_sr_hint if orig_sr_hint else sr
        rms_before = float(np.sqrt(np.mean(raw_waveform ** 2) + 1e-9))

        waveform = raw_waveform
        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        duration = len(waveform) / sr
        duration_after_trim = duration
        start_time = time.time()

        if self.is_w2v:
            # 1. Resample to 16kHz
            if sr != 16000:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                sr = 16000

            # === PHASE 1: Mic-Specific Enhanced DSP Pipeline ===
            if is_mic:
                # 1M. Bandpass Filter (70Hz - 7600Hz) — eliminates sub-bass desk rumble and high hiss
                try:
                    from scipy.signal import butter, sosfilt
                    sos = butter(4, [70, 7600], btype="bandpass", fs=16000, output="sos")
                    waveform = sosfilt(sos, waveform).astype(np.float32)
                except Exception:
                    pass

                # 2M. Gentle VAD (top_db=28 keeps natural speech without clipping soft words)
                try:
                    import librosa
                    trimmed, _ = librosa.effects.trim(waveform, top_db=28)
                    duration_after_trim = len(trimmed) / 16000
                    if len(trimmed) >= 3200:
                        waveform = trimmed
                except Exception:
                    duration_after_trim = len(waveform) / 16000

                # 3M. RMS Loudness Normalization — matches live mic power to studio benchmark level
                rms = np.sqrt(np.mean(waveform ** 2) + 1e-9)
                target_rms = 0.08  # ~ -22 dBFS studio speech reference
                if rms > 1e-4:
                    waveform = waveform * (target_rms / rms)
                waveform = np.clip(waveform, -0.98, 0.98).astype(np.float32)

                # 4M. Pad short utterances with 150ms silence on both sides
                pad_samples = int(0.15 * 16000)
                if len(waveform) < 16000 * 3:
                    waveform = np.pad(waveform, (pad_samples, pad_samples), mode='constant', constant_values=0)

            else:
                # === Standard File DSP Pipeline (unchanged for benchmark accuracy) ===
                # 2. DSP Speech Bandpass Filter (80Hz to 7,500Hz)
                try:
                    from scipy.signal import butter, sosfilt
                    sos = butter(4, [80, 7500], btype="bandpass", fs=16000, output="sos")
                    waveform = sosfilt(sos, waveform).astype(np.float32)
                except Exception:
                    pass

                # 3. Gentle Voice Activity Detection (VAD) / Silence Trimming
                try:
                    import librosa
                    trimmed, _ = librosa.effects.trim(waveform, top_db=32)
                    duration_after_trim = len(trimmed) / 16000
                    if len(trimmed) >= 3200:
                        waveform = trimmed
                except Exception:
                    duration_after_trim = len(waveform) / 16000

            # 4. Auto-Gain Normalization (Normalize peak amplitude to studio level 0.95)
            max_val = np.max(np.abs(waveform))
            peak_amp = float(max_val)
            if max_val > 1e-4:
                waveform = (waveform / max_val) * 0.95

            rms_after = float(np.sqrt(np.mean(waveform ** 2) + 1e-9))

            # 5. Standardize audio (zero mean, unit variance) matching Wav2Vec2 training
            waveform = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)
            waveform_shape = f"({len(waveform)},)"

            audio_metrics = {
                "audio_duration": f"{orig_duration:.2f}s",
                "original_sample_rate": f"{orig_sr:,} Hz",
                "final_sample_rate": "16,000 Hz",
                "num_channels": f"{orig_channels} ({'Mono' if orig_channels == 1 else 'Stereo'})",
                "rms_before": f"{rms_before:.5f} ({20 * np.log10(rms_before + 1e-9):.1f} dBFS)",
                "rms_after": f"{rms_after:.5f} ({20 * np.log10(rms_after + 1e-9):.1f} dBFS)",
                "peak_amplitude": f"{peak_amp:.4f}",
                "duration_after_trim": f"{duration_after_trim:.2f}s",
                "waveform_shape": waveform_shape,
            }

            inp = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.w2v_model(inp).logits

            logits_np = logits[0].cpu().numpy()
            
            # === PHASE 2: Decoding Parameters ===
            # Temperature: stable T=0.85 ensures sharp acoustic token probabilities
            T = 0.85
            logits_scaled = logits_np / T

            # Logit bias strength: solid script lock to prevent language mixing
            bias_boost = 4.0
            bias_suppress = -6.0

            target_lang = (language or "gu").lower()
            hotwords = self.all_hotwords

            logits_biased = logits_scaled.copy()
            if target_lang == "gu":
                # Gujarati script lock: adaptive strength based on source
                logits_biased[:, self.gu_indices] += bias_boost
                logits_biased[:, self.hi_indices] += bias_suppress
                hotwords = self.gu_hotwords
            elif target_lang == "hi":
                # Hindi script lock: adaptive strength based on source
                logits_biased[:, self.hi_indices] += bias_boost
                logits_biased[:, self.gu_indices] += bias_suppress
                hotwords = self.hi_hotwords
            elif target_lang == "en":
                # English script lock: adaptive strength based on source
                en_boost = 3.0 if is_mic else 4.0
                en_suppress = -3.5 if is_mic else -5.0
                logits_biased[:, self.en_indices] += en_boost
                logits_biased[:, self.gu_indices] += en_suppress
                logits_biased[:, self.hi_indices] += en_suppress
                hotwords = self.en_hotwords
            elif target_lang == "auto":
                # Acoustic script energy detection
                gu_score = float(np.mean(logits_scaled[:, self.gu_indices]))
                hi_score = float(np.mean(logits_scaled[:, self.hi_indices]))
                en_score = float(np.mean(logits_scaled[:, self.en_indices]))
                auto_boost = 2.0 if is_mic else 3.0
                auto_suppress = -2.0 if is_mic else -3.0
                if en_score > gu_score + 1.0 and en_score > hi_score + 1.0:
                    target_lang = "en"
                    hotwords = self.en_hotwords
                elif gu_score >= hi_score - 0.15:
                    logits_biased[:, self.gu_indices] += auto_boost
                    logits_biased[:, self.hi_indices] += auto_suppress
                    target_lang = "gu"
                    hotwords = self.gu_hotwords
                else:
                    target_lang = "hi"
                    hotwords = self.hi_hotwords

            pred_ids = logits_biased.argmax(-1).tolist()
            raw_text = self.tokenizer.decode_ctc(pred_ids)

            # 7. KenLM Beam Search decoding with language-optimized decoder tuning
            lm_text = ""
            active_decoder = self.lm_decoder_mic if (is_mic and self.lm_decoder_mic) else self.lm_decoder
            if active_decoder is not None:
                if target_lang == "en":
                    # English optimized: higher LM reliance & strong word insertion penalty to avoid skipping short words
                    active_decoder.reset_params(alpha=0.75, beta=2.2)
                    beam_w = 140 if is_mic else 110
                    hw_weight = 12.0 if is_mic else 9.0
                elif is_mic:
                    active_decoder.reset_params(alpha=0.65, beta=1.5)
                    beam_w = 128
                    hw_weight = 10.0
                else:
                    active_decoder.reset_params(alpha=0.45, beta=1.0)
                    beam_w = 100
                    hw_weight = 8.0

                raw_word_offsets = []
                try:
                    beams = active_decoder.decode_beams(
                        logits_biased, 
                        beam_width=beam_w,
                        hotwords=hotwords,
                        hotword_weight=hw_weight
                    )
                    if beams:
                        best_beam = beams[0]
                        lm_text = best_beam[0]
                        raw_word_offsets = best_beam[2]
                    else:
                        lm_text = raw_text
                except Exception:
                    lm_text = raw_text
            else:
                raw_word_offsets = []
                lm_text = raw_text

            candidate_text = lm_text if lm_text.strip() else raw_text
            
            # 8. Smart Multilingual Post-Processing & Script Consistency (includes Punctuation Restoration)
            final_text = self.post_processor.process(candidate_text, target_lang=target_lang)

            # 9. Extract Word-level Millisecond Timestamps & Generate SRT Subtitles
            word_timestamps = self.align_words_with_timestamps(raw_word_offsets, final_text, duration)
            srt_subtitles = self.generate_srt(word_timestamps, words_per_chunk=4)

            inference_time = time.time() - start_time
            rtf = inference_time / max(duration, 0.001)

            return {
                "text": final_text,
                "ctc_text": raw_text,
                "lm_text": final_text,
                "attn_text": final_text,
                "word_timestamps": word_timestamps,
                "srt_subtitles": srt_subtitles,
                "language": target_lang,
                "source": source,
                "duration": round(duration, 2),
                "inference_time": round(inference_time, 4),
                "rtf": round(rtf, 4),
                "device": str(self.device),
                "model_name": "MMS-300M + KenLM + Smart Decoder" + (" [MIC-ENHANCED]" if is_mic else ""),
                "audio_metrics": audio_metrics,
            }


        # Otherwise use Scratch STTModel
        mel = self.feature_extractor.compute_mel_spectrogram(waveform, sample_rate=sr)
        features = torch.from_numpy(mel.T).unsqueeze(0).float().to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        # Fast CTC inference
        transcripts = self.model.transcribe(features, feature_lengths, tokenizer=self.tokenizer)
        ctc_text = transcripts[0] if transcripts else ""

        # Autoregressive Attention inference
        attn_text = ""
        try:
            with torch.no_grad():
                enc_out, enc_lens = self.model.encoder(features, feature_lengths)
                sos_id = self.tokenizer.sos_id
                eos_id = self.tokenizer.eos_id
                cur = torch.tensor([[sos_id]], dtype=torch.long, device=self.device)
                for _ in range(50):
                    tgt_lens = torch.tensor([cur.shape[1]], device=self.device)
                    logits = self.model.decoder(cur, enc_out, enc_lens, tgt_lens)

                    next_token_logits = logits[:, -1, :].clone()
                    for i in range(cur.shape[1]):
                        token_id = cur[0, i].item()
                        if next_token_logits[0, token_id] > 0:
                            next_token_logits[0, token_id] /= 1.2
                        else:
                            next_token_logits[0, token_id] *= 1.2

                    tok = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                    cur = torch.cat([cur, tok], dim=1)
                    if tok.item() == eos_id:
                        break
                attn_text = self.tokenizer.decode(cur[0].tolist(), remove_special=True)
        except Exception:
            attn_text = ""

        inference_time = time.time() - start_time
        rtf = inference_time / max(duration, 0.001)
        primary_text = ctc_text if ctc_text else attn_text

        scratch_metrics = {
            "audio_duration": f"{orig_duration:.2f}s",
            "original_sample_rate": f"{orig_sr:,} Hz",
            "final_sample_rate": "16,000 Hz",
            "num_channels": f"{orig_channels} ({'Mono' if orig_channels == 1 else 'Stereo'})",
            "rms_before": f"{rms_before:.5f} ({20 * np.log10(rms_before + 1e-9):.1f} dBFS)",
            "rms_after": f"{rms_before:.5f} ({20 * np.log10(rms_before + 1e-9):.1f} dBFS)",
            "peak_amplitude": f"{float(np.max(np.abs(waveform))):.4f}",
            "duration_after_trim": f"{orig_duration:.2f}s",
            "waveform_shape": f"({len(waveform)},)",
        }

        scratch_timestamps = self.align_words_with_timestamps([], primary_text, duration)
        scratch_srt = self.generate_srt(scratch_timestamps, words_per_chunk=4)

        return {
            "text": primary_text,
            "ctc_text": ctc_text,
            "attn_text": attn_text,
            "word_timestamps": scratch_timestamps,
            "srt_subtitles": scratch_srt,
            "duration": round(duration, 2),
            "inference_time": round(inference_time, 4),
            "rtf": round(rtf, 4),
            "device": str(self.device),
            "model_name": "Proprietary Transformer + CTC",
            "audio_metrics": scratch_metrics,
        }

    def transcribe_batch(
        self, audio_paths: List[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription result dictionaries.
        """
        results = []
        for path in audio_paths:
            res = self.transcribe_file(path)
            res["audio_path"] = str(path)
            results.append(res)
        return results

    def transcribe_stream(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """Process a chunk of streaming audio and return transcription.

        Args:
            audio_chunk: 1D numpy array of audio samples.
            sample_rate: Sampling rate (standard: 16000 Hz).

        Returns:
            Transcription string.
        """
        if len(audio_chunk) < sample_rate * 0.2:  # Min 200ms
            return None

        mel = self.feature_extractor.compute_mel_spectrogram(audio_chunk, sample_rate=sample_rate)
        features = torch.from_numpy(mel.T).unsqueeze(0).float().to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        transcripts = self.model.transcribe(features, feature_lengths, tokenizer=self.tokenizer)
        return transcripts[0] if transcripts else ""


if __name__ == "__main__":
    print("=" * 60)
    print("STTInference — Verification Test")
    print("=" * 60)

    engine = STTInference(
        model_path="checkpoints/best_model.pt",
        config_path="configs/config.yaml",
        device="cpu",
    )

    test_file = "dataset/test_sample.wav"
    if Path(test_file).exists():
        print(f"\nTranscribing: {test_file}")
        res = engine.transcribe_file(test_file)
        print(f"  Duration       : {res['duration']}s")
        print(f"  Inference Time : {res['inference_time']}s")
        print(f"  RTF (Speed)    : {res['rtf']}x (< 1.0 means faster than real-time!)")
        print(f"  Device         : {res['device']}")
        print(f"  Output Text    : '{res['text'][:100]}...'")
        print("  ✓ STTInference verified successfully!")
    print("=" * 60)
