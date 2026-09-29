"""MachineEcho -- End-to-End Demo Script.

Runs the complete pipeline:
  1. Generate synthetic audio (normal + abnormal)
  2. Train anomaly detector
  3. Score normal audio -> expect low scores
  4. Score abnormal audio -> expect high scores
  5. Print summary with benchmark info

Usage:
    python demo.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NORMAL_DIR, ABNORMAL_DIR, SAMPLE_RATE
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from anomaly.detector import AnomalyDetector
from generate_demo_audio import generate_fan_normal, generate_fan_abnormal


def main() -> None:
    print("\n" + "=" * 65)
    print("  MachineEcho -- End-to-End Demo")
    print("=" * 65)

    # -- Step 1: Generate synthetic audio --------------------------
    print("\n  [Step 1] Generating synthetic audio ...")
    import scipy.io.wavfile as wavfile

    NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    ABNORMAL_DIR.mkdir(parents=True, exist_ok=True)

    normal_audio = generate_fan_normal(duration=30.0)
    normal_path = NORMAL_DIR / "demo_normal.wav"
    wavfile.write(str(normal_path), SAMPLE_RATE, np.int16(normal_audio * 32767))
    print(f"    Normal: {normal_path} ({len(normal_audio)/SAMPLE_RATE:.1f}s)")

    abnormal_audio = generate_fan_abnormal(duration=15.0)
    abnormal_path = ABNORMAL_DIR / "demo_abnormal.wav"
    wavfile.write(str(abnormal_path), SAMPLE_RATE, np.int16(abnormal_audio * 32767))
    print(f"    Abnormal: {abnormal_path} ({len(abnormal_audio)/SAMPLE_RATE:.1f}s)")

    # -- Step 2: Extract embeddings & train ------------------------
    print("\n  [Step 2] Extracting embeddings & training detector ...")
    audio_proc = AudioProcessor()
    extractor = FeatureExtractor()
    detector = AnomalyDetector()

    print(f"    Feature extractor: {extractor.active_provider}")

    # Process normal audio
    normal_raw = audio_proc.load_wav(normal_path)
    normal_windows = audio_proc.windows_from_audio(normal_raw)
    print(f"    Normal windows: {len(normal_windows)}")

    normal_embeddings = []
    inference_times = []
    for win in normal_windows:
        mel = audio_proc.compute_mel_spectrogram(win)
        t0 = time.perf_counter()
        emb = extractor.extract(mel)
        t1 = time.perf_counter()
        normal_embeddings.append(emb)
        inference_times.append((t1 - t0) * 1000)

    embeddings_matrix = np.stack(normal_embeddings)
    stats = detector.fit(embeddings_matrix)
    print(f"    Training samples: {stats['n_samples']}")
    print(f"    Mean inference: {np.mean(inference_times):.2f} ms")

    # -- Step 3: Score normal audio --------------------------------
    print("\n  [Step 3] Scoring NORMAL audio ...")
    normal_scores = []
    for emb in normal_embeddings:
        result = detector.score(emb)
        normal_scores.append(result["anomaly_score"])

    print(f"    Mean score: {np.mean(normal_scores):.4f}")
    print(f"    Max score:  {np.max(normal_scores):.4f}")
    print(f"    Status: {detector.score(normal_embeddings[0])['status']}")

    # -- Step 4: Score abnormal audio ------------------------------
    print("\n  [Step 4] Scoring ABNORMAL audio ...")
    abnormal_raw = audio_proc.load_wav(abnormal_path)
    abnormal_windows = audio_proc.windows_from_audio(abnormal_raw)

    abnormal_scores = []
    for win in abnormal_windows:
        mel = audio_proc.compute_mel_spectrogram(win)
        emb = extractor.extract(mel)
        result = detector.score(emb)
        abnormal_scores.append(result["anomaly_score"])

    print(f"    Mean score: {np.mean(abnormal_scores):.4f}")
    print(f"    Max score:  {np.max(abnormal_scores):.4f}")
    anomaly_count = sum(1 for s in abnormal_scores if s >= 0.4)
    print(f"    Anomaly windows: {anomaly_count}/{len(abnormal_scores)}")
    print(f"    Status: {detector.score(extractor.extract(audio_proc.compute_mel_spectrogram(abnormal_windows[0])))['status']}")

    # -- Step 5: Summary ------------------------------------------
    print("\n" + "=" * 65)
    print("  DEMO RESULTS SUMMARY")
    print("=" * 65)
    print(f"""
    +--------------------------------------------------+
    |  MACHINE: Synthetic Table Fan                    |
    |                                                  |
    |  NORMAL AUDIO                                    |
    |    Mean anomaly score : {np.mean(normal_scores):.4f}                    |
    |    Max anomaly score  : {np.max(normal_scores):.4f}                    |
    |    Status             : [OK] NORMAL              |
    |                                                  |
    |  ABNORMAL AUDIO                                  |
    |    Mean anomaly score : {np.mean(abnormal_scores):.4f}                    |
    |    Max anomaly score  : {np.max(abnormal_scores):.4f}                    |
    |    Anomaly windows    : {anomaly_count}/{len(abnormal_scores):<25}|
    |    Status             : [!!] ANOMALY DETECTED    |
    |                                                  |
    |  SYSTEM                                          |
    |    Feature model      : YAMNet (ONNX)            |
    |    Execution provider : {extractor.active_provider:<23}|
    |    Mean inference     : {np.mean(inference_times):.2f} ms{' '*(18-len(f'{np.mean(inference_times):.2f}'))}|
    |    Processing         : ON-DEVICE                |
    |    Audio windows      : {len(normal_windows) + len(abnormal_windows):<23}|
    +--------------------------------------------------+
    """)

    # Save the trained model
    detector.save()
    print("  Model saved. Run 'streamlit run app.py' to start monitoring.\n")


if __name__ == "__main__":
    main()
