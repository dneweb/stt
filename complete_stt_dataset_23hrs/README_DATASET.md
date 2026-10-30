# 🎙️ Multi-Language Speech-to-Text (STT) Complete Dataset (23.24 Hours)

This dataset contains transcribed speech across **Gujarati, Hindi, and English** designed for training end-to-end proprietary ASR/STT models.

---

## 📊 Dataset Overview

* **Total Audio Duration:** **23.24 Hours (1,395 Minutes)**
* **Total Audio Clips:** **10,319 clips**
* **Audio Format:** 16,000 Hz, 16-bit PCM Mono WAV
* **Languages:**
  * **Gujarati (GU):** ~9.01 Hours (3,850 clips)
  * **English (EN):** ~7.93 Hours (3,652 clips)
  * **Hindi (HI):** ~6.31 Hours (2,817 clips)
* **Dataset Splits:**
  * **Train Set:** `data/metadata/combined_train.csv` (9,077 clips, 20.49 hours)
  * **Validation Set:** `data/metadata/combined_val.csv` (1,242 clips, 2.75 hours)

---

## 📁 Directory Layout

```
├── data/
│   ├── metadata/
│   │   ├── combined_train.csv      # Main training manifest (9,077 rows)
│   │   ├── combined_val.csv        # Main validation manifest (1,242 rows)
│   │   ├── train.csv               # Team-only training split
│   │   ├── val.csv                 # Team-only validation split
│   │   ├── test.csv                # Team-only test split
│   │   ├── fleurs_train.csv        # Public FLEURS training split
│   │   └── fleurs_validation.csv   # Public FLEURS validation split
│   │
│   └── processed/
│       ├── all_speakers/           # Custom recordings from team members (2,605 WAVs)
│       └── external_fleurs/        # High-quality studio speech clips (7,975 WAVs)
│           ├── gu/ (train / validation)
│           ├── hi/ (train / validation)
│           └── en/ (train / validation)
│
└── tokenizer/
    └── vocab.json                  # Complete 250-character vocabulary (Devanagari, Gujarati, Latin)
```

---

## 📋 Manifest Schema

All CSV files follow the standard schema:
```csv
audio_path,transcript,language,speaker,duration
```

* `audio_path`: Relative path to WAV file (e.g. `data/processed/external_fleurs/gu/train/fleurs_gu_train_00000.wav`)
* `transcript`: Normalized ground truth transcription in native script
* `language`: Language code (`gu`, `hi`, `en`)
* `speaker`: Speaker identifier
* `duration`: Audio length in seconds
