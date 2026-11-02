"""
Audio Processor — Loads, preprocesses, and saves audio files for STT training.

Pipeline per file:
    1. Load audio (supports .m4a, .wav, .mp3, .flac via librosa + ffmpeg)
    2. Convert stereo / multi-channel → mono
    3. Resample to 16 000 Hz
    4. Trim leading and trailing silence
    5. Normalize amplitude to [-1.0, 1.0]
    6. Save as 16-bit PCM WAV

"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Supported input formats (librosa delegates .m4a decoding to ffmpeg)
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".m4a", ".wav", ".mp3", ".flac"}


class AudioProcessor:
    """Loads, resamples, normalizes, and validates raw audio files.

    Attributes:
        sample_rate: Target sample rate for all audio (default 16 000 Hz).
        mono: Whether to convert all audio to mono.
        max_duration: Maximum allowed duration in seconds.
        min_duration: Minimum allowed duration in seconds.
        top_db: Silence-trimming threshold in dB below peak.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        mono: bool = True,
        max_duration: float = 300.0,
        min_duration: float = 0.1,
        top_db: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.top_db = top_db

    # ------------------------------------------------------------------
    # Step 1 — Load audio from disk
    # ------------------------------------------------------------------
    def load(self, file_path: str | Path) -> Tuple[np.ndarray, int]:
        """Load an audio file and return (waveform, original_sample_rate).

        librosa.load handles format decoding via audioread / soundfile.
        For .m4a files it automatically uses ffmpeg as a backend.

        Args:
            file_path: Path to the audio file (.wav, .mp3, .flac, .m4a).

        Returns:
            Tuple of (waveform as 1-D float32 numpy array, original sample rate).

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load at the file's NATIVE sample rate (sr=None) so we can
        # report original SR before resampling.  mono=False lets us
        # inspect channel count before forcing mono.
        waveform, orig_sr = librosa.load(
            str(file_path), sr=None, mono=False
        )
        return waveform, orig_sr

    # ------------------------------------------------------------------
    # Step 2 — Convert to mono
    # ------------------------------------------------------------------
    def to_mono(self, waveform: np.ndarray) -> np.ndarray:
        """Convert a multi-channel waveform to mono by averaging channels.

        If the input is already 1-D (mono), it is returned unchanged.

        Args:
            waveform: Audio array. Shape (samples,) or (channels, samples).

        Returns:
            Mono waveform of shape (samples,).
        """
        if waveform.ndim == 1:
            # Already mono
            return waveform
        # Average across the channel axis → (samples,)
        return librosa.to_mono(waveform)

    # ------------------------------------------------------------------
    # Step 3 — Resample to target sample rate
    # ------------------------------------------------------------------
    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample waveform to self.sample_rate using high-quality resampling.

        If the original SR already matches the target, no work is done.

        Args:
            waveform: Mono waveform (1-D float array).
            orig_sr:  Original sample rate of the waveform.

        Returns:
            Resampled waveform at self.sample_rate.
        """
        if orig_sr == self.sample_rate:
            return waveform
        return librosa.resample(
            waveform, orig_sr=orig_sr, target_sr=self.sample_rate
        )

    # ------------------------------------------------------------------
    # Step 4 — Trim leading / trailing silence
    # ------------------------------------------------------------------
    def trim_silence(
        self, waveform: np.ndarray, top_db: float | None = None
    ) -> np.ndarray:
        """Trim leading and trailing silence from the waveform.

        Uses librosa.effects.trim which computes a dB envelope and
        removes segments quieter than `top_db` below peak.

        Args:
            waveform: Mono waveform (1-D float array).
            top_db:   Threshold in dB below peak to consider as silence.
                      Falls back to self.top_db if not provided.

        Returns:
            Trimmed waveform.
        """
        if top_db is None:
            top_db = self.top_db
        trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
        return trimmed

    # ------------------------------------------------------------------
    # Step 5 — Normalize amplitude
    # ------------------------------------------------------------------
    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Peak-normalize waveform amplitude to [-1.0, 1.0].

        Divides by the maximum absolute value.  If the waveform is
        all-zero (silence), it is returned as-is to avoid division by zero.

        Args:
            waveform: Mono waveform (1-D float array).

        Returns:
            Normalized waveform with values in [-1.0, 1.0].
        """
        peak = np.max(np.abs(waveform))
        if peak < 1e-8:
            # Effectively silent — nothing to normalize
            return waveform
        return waveform / peak

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------
    def validate(self, waveform: np.ndarray, sample_rate: int) -> bool:
        """Check if audio meets duration constraints.

        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate of the waveform.

        Returns:
            True if the audio is valid, False otherwise.
        """
        duration = len(waveform) / sample_rate
        return self.min_duration <= duration <= self.max_duration

    # ------------------------------------------------------------------
    # Full single-file pipeline
    # ------------------------------------------------------------------
    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> dict:
        """Run the full preprocessing pipeline on a single audio file.

        Steps:
            1. Load audio at native sample rate
            2. Convert to mono
            3. Resample to 16 kHz
            4. Trim silence
            5. Normalize amplitude
            6. Save as 16-bit PCM WAV

        Args:
            input_path:  Path to the raw audio file.
            output_path: Path where the processed WAV will be saved.

        Returns:
            dict with keys: input, output, orig_sr, channels, duration_s
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # — Load —
        waveform, orig_sr = self.load(input_path)

        # Determine original channel count for logging
        orig_channels = 1 if waveform.ndim == 1 else waveform.shape[0]

        # — Mono conversion —
        waveform = self.to_mono(waveform)

        # — Resample —
        waveform = self.resample(waveform, orig_sr)

        # — Trim silence —
        waveform = self.trim_silence(waveform)

        # — Normalize —
        waveform = self.normalize(waveform)

        # — Duration after processing —
        duration_s = round(len(waveform) / self.sample_rate, 3)

        # — Save processed WAV —
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, self.sample_rate, subtype="PCM_16")

        return {
            "input": str(input_path),
            "output": str(output_path),
            "orig_sr": orig_sr,
            "orig_channels": orig_channels,
            "target_sr": self.sample_rate,
            "duration_s": duration_s,
        }

    # ------------------------------------------------------------------
    # Batch pipeline — process an entire directory tree
    # ------------------------------------------------------------------
    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> dict:
        """Recursively process all supported audio files in input_dir.

        The original folder structure is mirrored under output_dir.
        All output files are saved as .wav regardless of original format.

        Raw files in input_dir are NEVER modified or deleted.

        Args:
            input_dir:  Root of the raw audio tree (e.g. dataset/raw).
            output_dir: Root of the processed tree  (e.g. dataset/processed).

        Returns:
            Summary dict with total, success, failed counts and total duration.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Discover all audio files with supported extensions
        audio_files = sorted(
            f
            for f in input_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        total = len(audio_files)
        success = 0
        failed = 0
        total_duration = 0.0
        failed_files: list[str] = []

        print("=" * 70)
        print("  AUDIO PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"  Input directory  : {input_dir}")
        print(f"  Output directory : {output_dir}")
        print(f"  Target SR        : {self.sample_rate} Hz")
        print(f"  Mono             : {self.mono}")
        print(f"  Silence trim dB  : {self.top_db}")
        print(f"  Files found      : {total}")
        print("=" * 70)
        print()

        start_time = time.time()

        for idx, file_path in enumerate(audio_files, start=1):
            # Mirror the relative path under the output directory,
            # but always save as .wav
            relative = file_path.relative_to(input_dir)
            out_path = output_dir / relative.with_suffix(".wav")

            print(f"[{idx}/{total}] Processing: {relative}")

            try:
                info = self.process_file(file_path, out_path)

                print(f"         Original SR : {info['orig_sr']} Hz")
                print(f"         Channels    : {info['orig_channels']} → 1 (mono)")
                print(f"         Target SR   : {info['target_sr']} Hz")
                print(f"         Duration    : {info['duration_s']} s")
                print(f"         Saved       : {info['output']}")
                print()

                success += 1
                total_duration += info["duration_s"]

            except Exception as exc:
                # Gracefully handle corrupted / unreadable files
                print(f"         ✖ ERROR: {exc}")
                print()
                failed += 1
                failed_files.append(str(file_path))

        elapsed = round(time.time() - start_time, 2)

        # ---- Summary ----
        print("=" * 70)
        print("  PREPROCESSING SUMMARY")
        print("=" * 70)
        print(f"  Total files found      : {total}")
        print(f"  Successfully processed : {success}")
        print(f"  Failed                 : {failed}")
        print(f"  Total duration         : {round(total_duration, 2)} s "
              f"({round(total_duration / 60, 2)} min)")
        print(f"  Elapsed time           : {elapsed} s")

        if failed_files:
            print()
            print("  Failed files:")
            for fp in failed_files:
                print(f"    - {fp}")

        print("=" * 70)

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "total_duration_s": round(total_duration, 2),
            "elapsed_s": elapsed,
            "failed_files": failed_files,
        }


# ======================================================================
# CLI entry point — run as:  python -m preprocessing.audio
# ======================================================================
def main() -> None:
    """Entry point for running the preprocessing pipeline from the command line."""
    # Default paths (relative to project root)
    input_dir = Path("dataset/raw")
    output_dir = Path("dataset/processed")

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    processor = AudioProcessor(
        sample_rate=16_000,
        mono=True,
        top_db=20.0,
    )

    summary = processor.process_directory(input_dir, output_dir)

    # Exit with non-zero code if everything failed
    if summary["success"] == 0 and summary["total"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
