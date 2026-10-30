"""
Audio Feature Extraction Pipeline for STT.
Extracts 80-channel Log-Mel Spectrograms with optional SpecAugment (Time & Frequency Masking).
"""

import torch
import torch.nn as nn
import torchaudio


class AudioFeatureExtractor(nn.Module):
    """
    Extracts normalized 80-channel Log-Mel Spectrograms from 16 kHz audio.
    Applies per-utterance cepstral mean & variance normalization (CMVN).
    Supports SpecAugment data augmentation during training.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,          # 25 ms window
        hop_length: int = 160,     # 10 ms hop size
        win_length: int = 400,     # 25 ms
        f_min: float = 0.0,
        f_max: float = 8000.0,
        freq_mask_param: int = 15,
        time_mask_param: int = 35
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def forward(self, waveform: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        Args:
            waveform: FloatTensor of shape [num_samples] or [1, num_samples] or [batch, num_samples].
            augment: If True, applies SpecAugment (frequency and time masking).
        Returns:
            features: FloatTensor of shape [time_steps, n_mels] (if single audio)
                      or [batch, time_steps, n_mels] (if batched audio).
        """
        is_1d = (waveform.ndim == 1)
        if is_1d:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]

        # Extract Mel Spectrogram: [batch, n_mels, time_steps]
        mel = self.mel_spectrogram(waveform)

        # Convert to Log-Mel scale: log(clamp(mel, min=1e-5))
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # Apply SpecAugment if in training mode
        if augment:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)

        # Per-utterance Normalization (mean=0, std=1)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-5
        norm_mel = (log_mel - mean) / std

        # Transpose to [batch, time_steps, n_mels] (or [time_steps, n_mels])
        out = norm_mel.transpose(-2, -1)
        if is_1d:
            out = out.squeeze(0)

        return out
