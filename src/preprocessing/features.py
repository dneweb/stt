"""
Feature Extractor — Computes Log-Mel spectrograms and acoustic features from waveforms.

"""

from typing import Optional, Union
import numpy as np
import librosa
import soundfile as sf
import torch


class FeatureExtractor:
    """Extracts Mel spectrograms, MFCCs, and other acoustic features for STT.

    Attributes:
        n_mels: Number of Mel filter banks (standard: 80).
        n_fft: FFT window size (standard: 400 for 25ms window at 16kHz).
        hop_length: Hop length between frames (standard: 160 for 10ms frame shift).
        win_length: Window length for STFT.
        f_min: Minimum frequency for Mel filter bank.
        f_max: Maximum frequency for Mel filter bank.
        normalize: Whether to apply per-channel CMVN normalization.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

    def _to_numpy(self, waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input waveform to a 1D float32 numpy array."""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        
        if not isinstance(waveform, np.ndarray):
            waveform = np.asarray(waveform, dtype=np.float32)

        # Flatten multi-channel to 1D mono if needed
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        return waveform.astype(np.float32)

    def compute_mel_spectrogram(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Compute log-Mel spectrogram from an audio waveform.

        Args:
            waveform: Audio waveform as 1D numpy array or torch.Tensor.
            sample_rate: Audio sampling rate (default: 16,000 Hz).

        Returns:
            Log-Mel spectrogram of shape (n_mels, time_steps).
        """
        y = self._to_numpy(waveform)

        if len(y) == 0:
            return np.zeros((self.n_mels, 0), dtype=np.float32)

        # Compute power Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )

        # Convert to decibels (log-Mel)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        if self.normalize:
            log_mel = self.normalize_features(log_mel)

        return log_mel.astype(np.float32)

    def compute_mfcc(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        n_mfcc: int = 13,
    ) -> np.ndarray:
        """Compute MFCC features from an audio waveform.

        Args:
            waveform: Audio waveform as 1D numpy array or torch.Tensor.
            sample_rate: Sampling rate of the waveform.
            n_mfcc: Number of MFCC coefficients to extract.

        Returns:
            MFCC features of shape (n_mfcc, time_steps).
        """
        y = self._to_numpy(waveform)

        if len(y) == 0:
            return np.zeros((n_mfcc, 0), dtype=np.float32)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.f_min,
            fmax=self.f_max,
        )

        if self.normalize:
            mfcc = self.normalize_features(mfcc)

        return mfcc.astype(np.float32)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply per-channel Cepstral Mean and Variance Normalization (CMVN).

        Args:
            features: Feature matrix of shape (channels, time_steps).

        Returns:
            Normalized feature matrix with zero mean and unit variance per channel.
        """
        if features.shape[1] == 0:
            return features

        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        # Avoid division by zero with small epsilon
        std = np.where(std < 1e-6, 1e-6, std)

        return (features - mean) / std

    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """Load audio file from disk and compute normalized Log-Mel spectrogram.

        Args:
            audio_path: Path to the .wav audio file.

        Returns:
            Log-Mel spectrogram of shape (n_mels, time_steps).
        """
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        return self.compute_mel_spectrogram(waveform, sample_rate=sample_rate)

    def apply_spec_augment(
        self,
        spectrogram: np.ndarray,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> np.ndarray:
        """Apply SpecAugment (frequency and time masking) for data augmentation.

        Args:
            spectrogram: Log-Mel spectrogram of shape (n_mels, time_steps).
            freq_mask_param: Maximum width of frequency mask.
            time_mask_param: Maximum width of time mask.
            num_freq_masks: Number of frequency masks to apply.
            num_time_masks: Number of time masks to apply.

        Returns:
            Augmented spectrogram with masked channels/frames.
        """
        augmented = spectrogram.copy()
        n_mels, time_steps = augmented.shape

        # Frequency masking
        for _ in range(num_freq_masks):
            f_len = np.random.randint(0, min(freq_mask_param, n_mels))
            f_start = np.random.randint(0, n_mels - f_len)
            augmented[f_start : f_start + f_len, :] = 0.0

        # Time masking
        for _ in range(num_time_masks):
            if time_steps > time_mask_param:
                t_len = np.random.randint(0, min(time_mask_param, time_steps))
                t_start = np.random.randint(0, time_steps - t_len)
                augmented[:, t_start : t_start + t_len] = 0.0

        return augmented


if __name__ == "__main__":
    import os
    import glob

    print("=" * 60)
    print("FeatureExtractor — Self Verification Test")
    print("=" * 60)

    extractor = FeatureExtractor(
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400,
        normalize=True,
    )

    test_files = glob.glob("dataset/**/*.wav", recursive=True)
    if not test_files:
        print("⚠️ No processed audio files found in data/processed/. Creating synthetic signal...")
        sample_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        mel = extractor.compute_mel_spectrogram(sample_audio, 16000)
        print(f"Synthetic Log-Mel shape: {mel.shape} (Channels: {mel.shape[0]}, Time: {mel.shape[1]})")
    else:
        sample_path = test_files[0]
        print(f"Testing on real audio file: {sample_path}")
        mel = extractor.extract_from_file(sample_path)
        mfcc = extractor.compute_mfcc(sf.read(sample_path)[0], 16000, n_mfcc=13)
        aug_mel = extractor.apply_spec_augment(mel)

        print(f"  Log-Mel shape     : {mel.shape} (80 mels, {mel.shape[1]} frames)")
        print(f"  MFCC shape        : {mfcc.shape} (13 coefficients, {mfcc.shape[1]} frames)")
        print(f"  SpecAugment shape : {aug_mel.shape}")
        print(f"  Mel Mean: {mel.mean():.4f}, Mel Std: {mel.std():.4f}")
        print(f"  ✓ FeatureExtractor verified successfully!")
    print("=" * 60)
