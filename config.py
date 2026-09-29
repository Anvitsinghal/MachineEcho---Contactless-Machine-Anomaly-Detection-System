"""MachineEcho — Configuration Constants."""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
NORMAL_DIR = DATA_DIR / "normal"
ABNORMAL_DIR = DATA_DIR / "abnormal"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "anomaly_model" / "detector.pkl"
YAMNET_ONNX_PATH = MODEL_DIR / "yamnet.onnx"

# ──────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Hz — YAMNet expects 16 kHz mono
WINDOW_DURATION = 0.96        # seconds — YAMNet native patch length
HOP_DURATION = 0.48           # seconds — 50 % overlap
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)   # 15360
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)          # 7680
CHANNELS = 1                  # mono

# ──────────────────────────────────────────────
# YAMNet Mel Spectrogram Parameters
# (matches the original YAMNet preprocessing)
# ──────────────────────────────────────────────
MEL_BANDS = 64
MEL_HOP_LENGTH = 160          # 10 ms at 16 kHz
MEL_WIN_LENGTH = 400          # 25 ms at 16 kHz
MEL_FMIN = 125.0
MEL_FMAX = 7500.0

# ──────────────────────────────────────────────
# YAMNet Model
# ──────────────────────────────────────────────
YAMNET_EMBEDDING_DIM = 1024
YAMNET_NUM_CLASSES = 521

# ──────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_N_ESTIMATORS = 100
ISOLATION_FOREST_RANDOM_STATE = 42

# Thresholds for status display
ANOMALY_SCORE_WARNING = 0.4   # scores above this -> ANOMALY DETECTED
ANOMALY_SCORE_CRITICAL = 0.7  # scores above this -> HIGH RISK

# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────
DASHBOARD_REFRESH_INTERVAL = 0.5  # seconds
SCORE_HISTORY_LENGTH = 100        # how many recent scores to keep for the chart

# ──────────────────────────────────────────────
# Execution Providers (ONNX Runtime)
# ──────────────────────────────────────────────
# Priority order: try QNN (Snapdragon NPU) first, then CPU
EXECUTION_PROVIDERS = [
    "QNNExecutionProvider",     # Qualcomm Hexagon NPU
    "CPUExecutionProvider",     # fallback
]

# ──────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────
for _dir in [MODEL_DIR, DATA_DIR, NORMAL_DIR, ABNORMAL_DIR, ANOMALY_MODEL_PATH.parent]:
    _dir.mkdir(parents=True, exist_ok=True)
