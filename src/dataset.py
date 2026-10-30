"""
PyTorch Dataset and Dynamic Collation for Multilingual STT.
Loads audio with soundfile, extracts Log-Mel features, tokenizes transcripts,
and dynamically pads batches to minimize computational overhead.
"""

import csv
import torch
import soundfile as sf
import torchaudio.functional as F
from pathlib import Path
from typing import List, Dict, Any, Union
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer
from .features import AudioFeatureExtractor


class STTDataset(Dataset):
    """
    Dataset loader for Speech-to-Text manifests (CSV format).
    Reads 16 kHz WAV audio files and corresponding multilingual transcripts.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        base_dir: Union[str, Path],
        tokenizer: CharTokenizer,
        feature_extractor: AudioFeatureExtractor,
        is_train: bool = True,
        max_duration: float = 25.0,
        min_duration: float = 0.4
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.base_dir = Path(base_dir)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.is_train = is_train
        self.max_duration = max_duration
        self.min_duration = min_duration

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")

        # Parse CSV manifest
        self.entries = []
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dur = float(row.get("duration", 0.0) or 0.0)
                # Filter out outliers (too short or too long)
                if dur > 0.0 and (dur < self.min_duration or dur > self.max_duration):
                    continue
                self.entries.append(row)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.entries[idx]
        rel_audio_path = row["audio_path"]
        transcript = row.get("transcript", "")
        lang = row.get("language", "unknown")

        full_audio_path = self.base_dir / rel_audio_path

        # 1. Load Audio Waveform
        try:
            data, sr = sf.read(str(full_audio_path), dtype="float32")
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {full_audio_path}: {e}")

        waveform = torch.from_numpy(data)

        # Handle stereo -> mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        # Resample to 16 kHz if necessary
        if sr != self.feature_extractor.sample_rate:
            waveform = F.resample(waveform, sr, self.feature_extractor.sample_rate)

        # 2. Extract 80-channel Log-Mel Spectrogram features: [time_steps, 80]
        features = self.feature_extractor(waveform, augment=self.is_train)

        # 3. Tokenize Transcript
        token_ids = self.tokenizer.encode(transcript)
        targets = torch.tensor(token_ids, dtype=torch.long)

        return {
            "features": features,
            "targets": targets,
            "text": transcript,
            "language": lang,
            "audio_path": str(full_audio_path)
        }


class SpeechCollate:
    """
    Collate function for DataLoader.
    Dynamically pads audio features and target token IDs to the maximum
    sequence length present in the current batch.
    """

    def __init__(self, pad_id: int = 1):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Filter out empty items if any
        batch = [b for b in batch if b["features"].shape[0] > 0 and b["targets"].shape[0] > 0]
        if not batch:
            raise ValueError("All batch items are empty!")

        batch_size = len(batch)

        # Measure feature lengths and target lengths
        feature_lengths = torch.tensor([b["features"].shape[0] for b in batch], dtype=torch.long)
        target_lengths = torch.tensor([b["targets"].shape[0] for b in batch], dtype=torch.long)

        max_feat_len = feature_lengths.max().item()
        max_target_len = target_lengths.max().item()
        n_mels = batch[0]["features"].shape[1]

        # Allocate padded tensors
        # Audio padded with 0.0 (normalized silent frame)
        padded_features = torch.zeros(batch_size, max_feat_len, n_mels, dtype=torch.float32)
        # Targets padded with pad_id (1)
        padded_targets = torch.full((batch_size, max_target_len), fill_value=self.pad_id, dtype=torch.long)

        texts = []
        languages = []
        audio_paths = []

        for i, item in enumerate(batch):
            f_len = feature_lengths[i].item()
            t_len = target_lengths[i].item()

            padded_features[i, :f_len, :] = item["features"]
            padded_targets[i, :t_len] = item["targets"]

            texts.append(item["text"])
            languages.append(item["language"])
            audio_paths.append(item["audio_path"])

        return {
            "features": padded_features,          # [batch_size, max_feat_len, 80]
            "feature_lengths": feature_lengths,    # [batch_size]
            "targets": padded_targets,            # [batch_size, max_target_len]
            "target_lengths": target_lengths,      # [batch_size]
            "texts": texts,
            "languages": languages,
            "audio_paths": audio_paths
        }
