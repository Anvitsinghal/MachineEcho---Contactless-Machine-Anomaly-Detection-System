"""MachineEcho — Audio Recording Utility.

Quick helper to record normal / abnormal audio samples from the
laptop microphone and save them as WAV files for training.

Usage:
    python record_audio.py --mode normal --duration 30
    python record_audio.py --mode abnormal --duration 10 --label rattle
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from config import NORMAL_DIR, ABNORMAL_DIR, SAMPLE_RATE
from audio_processor import record_to_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record audio samples for MachineEcho."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "abnormal"],
        default="normal",
        help="Type of audio to record.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label for abnormal recordings (e.g., 'rattle', 'grinding').",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (leave blank for default mic).",
    )
    args = parser.parse_args()

    # Choose output directory
    if args.mode == "normal":
        out_dir = NORMAL_DIR
    else:
        out_dir = ABNORMAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{args.label}" if args.label else ""
    filename = f"{args.mode}{label_part}_{ts}.wav"
    filepath = out_dir / filename

    print(f"\n{'='*50}")
    print(f"  MachineEcho -- Audio Recorder")
    print(f"{'='*50}")
    print(f"  Mode     : {args.mode}")
    print(f"  Duration : {args.duration:.1f}s")
    print(f"  Output   : {filepath}")
    print(f"  Device   : {args.device or 'default'}")
    print()
    input("  Press ENTER to start recording...")
    print()

    record_to_file(
        path=filepath,
        duration=args.duration,
        sample_rate=SAMPLE_RATE,
        device=args.device,
    )

    print(f"\n  [OK] Recording saved to {filepath}")
    print()


if __name__ == "__main__":
    main()
