"""MachineEcho — Audio Capture & Preprocessing.

Handles real-time microphone streaming, windowing, and
mel-spectrogram computation compatible with YAMNet input format.
"""
from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa

from config import (
    SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES, CHANNELS,
    MEL_BANDS, MEL_HOP_LENGTH, MEL_WIN_LENGTH, MEL_FMIN, MEL_FMAX,
    WINDOW_DURATION,
)


class AudioProcessor:
    """Captures audio from the microphone and yields analysis-ready windows."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
        device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.device = device

        self._buffer = np.zeros(0, dtype=np.float32)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    # ── microphone streaming ──────────────────────────────────

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            print(f"[AudioProcessor] status: {status}")
        # indata shape: (frames, channels) — take channel 0
        self._audio_queue.put(indata[:, 0].copy())

    def start_stream(self) -> None:
        """Open the microphone stream."""
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.hop_samples,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

    def stop_stream(self) -> None:
        """Close the microphone stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_windows(self) -> Generator[np.ndarray, None, None]:
        """Yield complete audio windows from the live stream.

        Each window is a 1-D float32 array of length `window_samples`.
        """
        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._buffer = np.concatenate([self._buffer, chunk])

            while len(self._buffer) >= self.window_samples:
                window = self._buffer[: self.window_samples]
                self._buffer = self._buffer[self.hop_samples :]
                yield window

    # ── WAV file loading ──────────────────────────────────────

    @staticmethod
    def load_wav(path: str | Path) -> np.ndarray:
        """Load a WAV file and return mono float32 audio at SAMPLE_RATE."""
        audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)

    @staticmethod
    def windows_from_audio(
        audio: np.ndarray,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
    ) -> list[np.ndarray]:
        """Split a long audio array into overlapping windows."""
        windows: list[np.ndarray] = []
        start = 0
        while start + window_samples <= len(audio):
            windows.append(audio[start : start + window_samples])
            start += hop_samples
        return windows

    # ── mel spectrogram ───────────────────────────────────────

    @staticmethod
    def compute_mel_spectrogram(audio_window: np.ndarray) -> np.ndarray:
        """Compute a log-mel spectrogram matching YAMNet's input format.

        Parameters
        ----------
        audio_window : np.ndarray
            1-D float32 array of `WINDOW_SAMPLES` samples at `SAMPLE_RATE`.

        Returns
        -------
        np.ndarray
            Shape (1, 96, 64) — batch of one spectrogram patch:
            96 time frames × 64 mel bands.
        """
        mel = librosa.feature.melspectrogram(
            y=audio_window,
            sr=SAMPLE_RATE,
            n_mels=MEL_BANDS,
            hop_length=MEL_HOP_LENGTH,
            win_length=MEL_WIN_LENGTH,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        # Transpose to (time, mel_bands) then add batch dim
        log_mel = log_mel.T  # (time_frames, 64)
        # Pad or trim to 96 frames
        target_frames = 96
        if log_mel.shape[0] < target_frames:
            pad = np.zeros((target_frames - log_mel.shape[0], MEL_BANDS), dtype=np.float32)
            log_mel = np.vstack([log_mel, pad])
        else:
            log_mel = log_mel[:target_frames]
        return log_mel[np.newaxis, ...].astype(np.float32)  # (1, 96, 64)


def record_to_file(
    path: str | Path,
    duration: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int] = None,
) -> None:
    """Record audio from the microphone and save as a WAV file."""
    print(f"Recording {duration:.1f}s of audio ...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()
    # Normalize to int16 range for WAV compatibility
    audio_int16 = np.int16(audio.flatten() * 32767)
    wavfile.write(str(path), sample_rate, audio_int16)
    print(f"Saved -> {path}")
