"""MachineEcho — Synthetic Audio Generator for Demo & Testing.

Generates synthetic normal and abnormal machine audio WAV files
so you can test the full pipeline without a physical machine.

Usage:
    python generate_demo_audio.py
"""
from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

from config import SAMPLE_RATE, NORMAL_DIR, ABNORMAL_DIR


def generate_fan_normal(duration: float = 30.0) -> np.ndarray:
    """Generate synthetic 'normal fan' audio.

    Simulates a steady rotating fan with:
    - Fundamental frequency at ~60 Hz (fan blade pass)
    - Harmonics at 120, 180, 240 Hz
    - Broadband white noise (air flow)
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    signal = np.zeros_like(t)

    # Blade-pass frequency and harmonics
    for harmonic, amplitude in [(60, 0.3), (120, 0.15), (180, 0.08), (240, 0.04)]:
        signal += amplitude * np.sin(2 * np.pi * harmonic * t)

    # Broadband noise (air flow)
    rng = np.random.RandomState(42)
    noise = rng.randn(len(t)).astype(np.float32) * 0.05
    signal += noise

    # Slight amplitude modulation (natural fan wobble)
    mod = 1.0 + 0.02 * np.sin(2 * np.pi * 1.5 * t)
    signal *= mod

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal


def generate_fan_abnormal(duration: float = 15.0) -> np.ndarray:
    """Generate synthetic 'abnormal fan' audio.

    Simulates a fan with:
    - Shifted fundamental (bearing wear)
    - Intermittent clicks/rattles
    - Increased noise floor
    - Amplitude instability
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    signal = np.zeros_like(t)

    # Slightly shifted frequencies (bearing wear)
    for harmonic, amplitude in [(63, 0.25), (126, 0.2), (189, 0.12), (252, 0.08)]:
        signal += amplitude * np.sin(2 * np.pi * harmonic * t)

    # Intermittent clicks (rattle)
    rng = np.random.RandomState(99)
    click_times = rng.uniform(0, duration, size=int(duration * 3))  # ~3 clicks/sec
    for ct in click_times:
        idx = int(ct * SAMPLE_RATE)
        if idx + 100 < len(signal):
            click = rng.randn(100).astype(np.float32) * 0.4
            click *= np.exp(-np.linspace(0, 5, 100))  # decay
            signal[idx : idx + 100] += click

    # Higher noise floor
    noise = rng.randn(len(t)).astype(np.float32) * 0.12
    signal += noise

    # Amplitude instability
    mod = 1.0 + 0.15 * np.sin(2 * np.pi * 0.5 * t)
    mod += 0.1 * np.sin(2 * np.pi * 3.7 * t)
    signal *= mod

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal


def main() -> None:
    print("\n" + "=" * 55)
    print("  MachineEcho -- Synthetic Audio Generator")
    print("=" * 55)

    # Ensure directories exist
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

    # Generate normal audio
    print("\n  Generating normal fan audio (30s) ...")
    normal = generate_fan_normal(duration=30.0)
    normal_path = NORMAL_DIR / "normal_fan_synthetic.wav"
    audio_int16 = np.int16(normal * 32767)
    wavfile.write(str(normal_path), SAMPLE_RATE, audio_int16)
    print(f"  [OK] Saved -> {normal_path}")

    # Generate abnormal audio samples
    for i, (dur, label) in enumerate(
        [(15.0, "rattle"), (10.0, "bearing_wear"), (10.0, "vibration")], 1
    ):
        print(f"\n  Generating abnormal audio #{i}: {label} ({dur}s) ...")
        abnormal = generate_fan_abnormal(duration=dur)
        # Add variation per sample
        rng = np.random.RandomState(i * 17)
        abnormal += rng.randn(len(abnormal)).astype(np.float32) * 0.03 * i
        abnormal = abnormal / np.max(np.abs(abnormal)) * 0.8
        abn_path = ABNORMAL_DIR / f"abnormal_fan_{label}_synthetic.wav"
        audio_int16 = np.int16(abnormal * 32767)
        wavfile.write(str(abn_path), SAMPLE_RATE, audio_int16)
        print(f"  [OK] Saved -> {abn_path}")

    print("\n" + "=" * 55)
    print("  Done! You can now train the anomaly detector:")
    print("    python -m anomaly.train_anomaly")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
