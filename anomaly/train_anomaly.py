"""MachineEcho — Train Anomaly Detector.

Usage:
    python -m anomaly.train_anomaly
    python -m anomaly.train_anomaly --data-dir data/normal --output anomaly_model/detector.pkl

Reads WAV files from the normal-operation directory, extracts
YAMNet embeddings, and trains an Isolation Forest anomaly detector.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NORMAL_DIR, ANOMALY_MODEL_PATH
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from anomaly.detector import AnomalyDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MachineEcho anomaly detector on normal audio."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(NORMAL_DIR),
        help="Directory containing normal-operation WAV files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ANOMALY_MODEL_PATH),
        help="Path to save the trained detector.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    # Find WAV files
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        print(f"\n[ERROR] No WAV files found in {data_dir}")
        print("\nTo get started:")
        print("  1. Run: python record_audio.py --mode normal --duration 30")
        print("  2. Then re-run this training script.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  MachineEcho -- Anomaly Detector Training")
    print(f"{'='*60}")
    print(f"  Data directory : {data_dir}")
    print(f"  WAV files      : {len(wav_files)}")
    print(f"  Output         : {output_path}")
    print()

    # Initialize components
    audio = AudioProcessor()
    extractor = FeatureExtractor()
    detector = AnomalyDetector()

    print(f"  Feature extractor: {extractor.active_provider}")
    print(f"  Using stub: {extractor._use_stub}")
    print()

    # Process all WAV files
    all_embeddings: list[np.ndarray] = []
    total_windows = 0

    for wav_path in wav_files:
        print(f"  Processing: {wav_path.name}")
        raw_audio = audio.load_wav(wav_path)
        windows = audio.windows_from_audio(raw_audio)
        print(f"    -> {len(windows)} windows ({len(raw_audio)/16000:.1f}s audio)")

        for win in windows:
            mel = audio.compute_mel_spectrogram(win)
            emb = extractor.extract(mel)
            all_embeddings.append(emb)

        total_windows += len(windows)

    if not all_embeddings:
        print("\n[ERROR] No audio windows extracted. Check your WAV files.")
        sys.exit(1)

    embeddings_matrix = np.stack(all_embeddings)  # (N, 1024)
    print(f"\n  Total embeddings: {embeddings_matrix.shape[0]}")
    print(f"  Embedding dim   : {embeddings_matrix.shape[1]}")

    # Train
    print("\n  Training Isolation Forest ...")
    t0 = time.perf_counter()
    stats = detector.fit(embeddings_matrix)
    t1 = time.perf_counter()

    print(f"  Training time   : {(t1 - t0)*1000:.1f} ms")
    print(f"  Train score mean: {stats['train_score_mean']:.4f}")
    print(f"  Train score std : {stats['train_score_std']:.4f}")

    # Save
    detector.save(output_path)

    print(f"\n{'='*60}")
    print("  [OK] Training complete!")
    print(f"  Model saved to: {output_path}")
    print(f"{'='*60}\n")

    # Quick sanity check: score the training data
    print("  Sanity check (scoring training samples):")
    scores = []
    for emb in all_embeddings[:5]:
        result = detector.score(emb)
        scores.append(result['anomaly_score'])
        print(f"    Score: {result['anomaly_score']:.4f} | Status: {result['status']}")
    print(f"    Mean training score: {np.mean(scores):.4f}")
    print()


if __name__ == "__main__":
    main()
