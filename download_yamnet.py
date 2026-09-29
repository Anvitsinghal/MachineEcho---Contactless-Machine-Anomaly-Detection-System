"""MachineEcho -- YAMNet ONNX Downloader / Helper.

Downloads pre-converted YAMNet ONNX model or provides download instructions from Qualcomm AI Hub.

Usage:
    python download_yamnet.py
"""
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "yamnet.onnx"

# Public YAMNet ONNX model URLs or mirrors
YAMNET_ONNX_URLS = [
    "https://github.com/onnx/models/raw/main/validated/audio_event_classification/yamnet/model/yamnet-12.onnx",
    "https://storage.googleapis.com/antigravity-public-assets/yamnet.onnx"
]

def download_yamnet():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"[OK] YAMNet model already present at {MODEL_PATH}")
        return True

    print(f"Downloading YAMNet ONNX model to {MODEL_PATH} ...")
    for url in YAMNET_ONNX_URLS:
        try:
            print(f"Trying URL: {url}")
            urllib.request.urlretrieve(url, MODEL_PATH)
            if MODEL_PATH.stat().st_size > 1000:
                print(f"[OK] Downloaded YAMNet ONNX ({MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    print("\nNote: Official Qualcomm AI Hub YAMNet ONNX model profile can be exported directly from:")
    print("      https://aihub.qualcomm.com/compute/models/yamnet")
    print("Place the exported 'yamnet.onnx' in the models/ directory of this project.")
    return False

if __name__ == "__main__":
    download_yamnet()
