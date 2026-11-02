# Preprocessing Module

This module contains the audio preprocessing pipeline that converts raw voice recordings into clean, standardized WAV files suitable for STT model training.

---

## Pipeline Steps

### 1. Why 16 kHz Sample Rate?

Human speech contains meaningful frequency content up to ~8 kHz. By the **Nyquist theorem**, a 16 kHz sample rate captures all frequencies up to 8 kHz — more than enough for accurate speech recognition. Using 16 kHz instead of the original 44.1 kHz or 48 kHz:

- **Reduces file size** by ~3× without losing speech information
- **Speeds up feature extraction** (fewer samples to process per second)
- **Matches industry standard** — most STT models (Whisper, DeepSpeech, Wav2Vec) train at 16 kHz

### 2. Why Mono Conversion?

Most microphones and phones record in stereo (2 channels) or even surround. For speech recognition:

- The **same voice signal** appears on both channels — the second channel is redundant
- Mono conversion **halves the data size** without any speech quality loss
- All STT model architectures expect **single-channel input**

We average the channels: `mono = (left + right) / 2`

### 3. Why Silence Trimming?

Raw recordings often have **dead air** at the start and end (before speaking and after stopping). Trimming this:

- **Removes unnecessary padding** that wastes computation
- **Improves model attention** — the model doesn't learn to predict blank segments
- **Standardizes clip boundaries** across different recording setups

We use a threshold of 20 dB below peak — anything quieter is considered silence.

### 4. Why Normalization?

Different recordings have wildly different volumes depending on the microphone, distance, and environment. Peak normalization scales every waveform so the loudest sample hits exactly ±1.0:

- **Consistent amplitude** across all training samples
- **Prevents gradient issues** — very quiet or very loud inputs can cause training instability
- **No clipping** — we scale, not clip, so no information is lost

### 5. Raw vs Processed Data

| Aspect | `data/raw/` | `data/processed/` |
|---|---|---|
| **Content** | Original recordings, untouched | Cleaned, standardized WAVs |
| **Format** | .m4a, .wav, .mp3, .flac | .wav (16-bit PCM) only |
| **Sample Rate** | Varies (44.1k, 48k, etc.) | Always 16,000 Hz |
| **Channels** | Mono or Stereo | Always Mono |
| **Silence** | May have leading/trailing | Trimmed |
| **Volume** | Varies per recording | Peak-normalized to ±1.0 |
| **Rule** | ⚠️ NEVER modify or delete | ✅ Can be regenerated anytime |

---

## Usage

Run from the **project root** directory:

```bash
python -m preprocessing.audio
```

This processes all files in `dataset/raw/` and saves cleaned WAVs to `dataset/processed/`, preserving the folder structure.

---

## Output Structure

```
dataset/processed/
├── englishvoice/
│   ├── sample.wav
│   ├── sample.wav
│   └── ...
├── gujarativoice/
│   ├── sample.wav
│   └── ...
└── hindivoice/
    ├── sample.wav
    └── ...
```
