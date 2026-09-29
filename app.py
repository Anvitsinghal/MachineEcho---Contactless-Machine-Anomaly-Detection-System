"""MachineEcho — Real-Time Machine Anomaly Detection Dashboard.

Launch with:
    streamlit run app.py
"""
from __future__ import annotations

import time
import threading
import sys
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ANOMALY_SCORE_WARNING,
    ANOMALY_SCORE_CRITICAL,
    SCORE_HISTORY_LENGTH,
    SAMPLE_RATE,
    WINDOW_DURATION,
    ANOMALY_MODEL_PATH,
)
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from anomaly.detector import AnomalyDetector

# ────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MachineEcho",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-normal {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .status-warning {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .status-critical {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #333;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Session state initialization
# ────────────────────────────────────────────────────────────
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "score_history" not in st.session_state:
    st.session_state.score_history = deque(maxlen=SCORE_HISTORY_LENGTH)
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "inference_ms" not in st.session_state:
    st.session_state.inference_ms = 0.0
if "windows_processed" not in st.session_state:
    st.session_state.windows_processed = 0
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = None
if "extractor" not in st.session_state:
    st.session_state.extractor = None
if "detector" not in st.session_state:
    st.session_state.detector = None


# ────────────────────────────────────────────────────────────
# Initialize models (cached)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_extractor():
    """Load YAMNet feature extractor."""
    return FeatureExtractor()


@st.cache_resource
def load_detector():
    """Load trained anomaly detector."""
    det = AnomalyDetector()
    if ANOMALY_MODEL_PATH.exists():
        det.load()
    return det


# ────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🔊 MachineEcho</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Contactless Machine Anomaly Detection · Powered by Snapdragon NPU</p>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    machine_name = st.text_input("Machine Name", value="Table Fan", key="machine_name")

    st.divider()
    st.subheader("Model Info")

    extractor = load_extractor()
    detector = load_detector()

    ext_info = extractor.info()
    st.write(f"**Feature Model:** YAMNet (ONNX)")
    st.write(f"**Execution Provider:** `{ext_info['active_provider']}`")
    st.write(f"**Using Stub:** {ext_info['using_stub']}")
    st.write(f"**Detector Trained:** {detector.is_trained}")

    if detector.is_trained:
        det_info = detector.info()
        stats = det_info.get("training_stats", {})
        if stats:
            st.write(f"**Training Samples:** {stats.get('n_samples', 'N/A')}")

    st.divider()
    st.subheader("Thresholds")
    st.write(f"Warning: ≥ {ANOMALY_SCORE_WARNING}")
    st.write(f"Critical: ≥ {ANOMALY_SCORE_CRITICAL}")

    st.divider()
    st.subheader("Platform")
    st.write("**Target:** Snapdragon X Elite / X Plus")
    st.write("**NPU:** Qualcomm Hexagon")
    st.write("**Optimization:** Qualcomm AI Hub")

# ────────────────────────────────────────────────────────────
# Main layout
# ────────────────────────────────────────────────────────────
if not detector.is_trained:
    st.warning(
        "⚠️ **Anomaly detector not trained.** "
        "Record normal audio first, then train the model:\n\n"
        "```bash\n"
        "python record_audio.py --mode normal --duration 30\n"
        "python -m anomaly.train_anomaly\n"
        "```"
    )

# Control buttons
col_start, col_stop, col_spacer = st.columns([1, 1, 3])

with col_start:
    start_btn = st.button("▶️ Start Monitoring", type="primary", use_container_width=True)

with col_stop:
    stop_btn = st.button("⏹️ Stop", use_container_width=True)

if start_btn and detector.is_trained:
    st.session_state.monitoring = True
    st.session_state.score_history = deque(maxlen=SCORE_HISTORY_LENGTH)
    st.session_state.windows_processed = 0

if stop_btn:
    st.session_state.monitoring = False

# ────────────────────────────────────────────────────────────
# Status display
# ────────────────────────────────────────────────────────────
status_placeholder = st.empty()
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
detail_placeholder = st.empty()


def display_status(result: dict, inference_ms: float, windows: int) -> None:
    """Update the dashboard with current detection results."""
    score = result["anomaly_score"]
    status = result["status"]
    confidence = result["confidence"]

    # Status banner
    if status == "NORMAL":
        icon = "🟢"
        css_class = "status-normal"
    elif status == "ANOMALY DETECTED":
        icon = "🟠"
        css_class = "status-warning"
    else:
        icon = "🔴"
        css_class = "status-critical"

    status_placeholder.markdown(
        f'<div class="{css_class}">{icon} {status}</div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    with metrics_placeholder.container():
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Anomaly Score", f"{score:.2f}")
        with m2:
            st.metric("Confidence", f"{confidence:.1f}%")
        with m3:
            st.metric("Inference", f"{inference_ms:.1f} ms")
        with m4:
            st.metric("Windows", f"{windows}")
        with m5:
            npu_label = "ON" if "QNN" in extractor.active_provider else "OFF (CPU)"
            st.metric("NPU", npu_label)

    # Score history chart
    if len(st.session_state.score_history) > 1:
        scores = list(st.session_state.score_history)
        chart_data = {
            "Anomaly Score": scores,
            "Warning Threshold": [ANOMALY_SCORE_WARNING] * len(scores),
            "Critical Threshold": [ANOMALY_SCORE_CRITICAL] * len(scores),
        }
        chart_placeholder.line_chart(chart_data, height=250)

    # Detail panel
    with detail_placeholder.container():
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 Detection Details")
            st.write(f"**Machine:** {machine_name}")
            st.write(f"**Status:** {status}")
            st.write(f"**Raw Score:** {result['raw_score']}")
            st.write(f"**Prediction:** {'Normal' if result['prediction'] == 1 else 'Anomaly'}")
        with col_b:
            st.subheader("🖥️ System Info")
            st.write(f"**Provider:** {extractor.active_provider}")
            st.write(f"**Processing:** On-Device")
            st.write(f"**Sample Rate:** {SAMPLE_RATE} Hz")
            st.write(f"**Window:** {WINDOW_DURATION}s")


# ────────────────────────────────────────────────────────────
# Monitoring loop
# ────────────────────────────────────────────────────────────
if st.session_state.monitoring and detector.is_trained:
    audio_proc = AudioProcessor()

    try:
        audio_proc.start_stream()
        st.toast("🎤 Microphone active — listening...", icon="🔊")

        for window in audio_proc.get_windows():
            if not st.session_state.monitoring:
                break

            # Process window
            mel = audio_proc.compute_mel_spectrogram(window)
            embedding = extractor.extract(mel)
            result = detector.score(embedding)

            # Update state
            st.session_state.current_result = result
            st.session_state.inference_ms = extractor.last_inference_ms
            st.session_state.windows_processed += 1
            st.session_state.score_history.append(result["anomaly_score"])

            # Display
            display_status(
                result,
                extractor.last_inference_ms,
                st.session_state.windows_processed,
            )

            # Small sleep to prevent UI freeze
            time.sleep(0.05)

    except Exception as exc:
        st.error(f"Error during monitoring: {exc}")
    finally:
        audio_proc.stop_stream()

elif not st.session_state.monitoring:
    # Show idle state
    status_placeholder.markdown(
        '<div class="status-normal">⏸️ IDLE — Ready to Monitor</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.current_result:
        display_status(
            st.session_state.current_result,
            st.session_state.inference_ms,
            st.session_state.windows_processed,
        )

# ────────────────────────────────────────────────────────────
# Demo mode: file-based analysis
# ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Analyze Audio File")
st.write("Upload a WAV file to analyze it for anomalies (useful for demos and testing).")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None and detector.is_trained:
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        audio_proc = AudioProcessor()
        audio_data = audio_proc.load_wav(tmp_path)
        windows = audio_proc.windows_from_audio(audio_data)

        st.write(f"**Duration:** {len(audio_data)/SAMPLE_RATE:.1f}s | **Windows:** {len(windows)}")

        scores = []
        progress = st.progress(0)

        for i, win in enumerate(windows):
            mel = audio_proc.compute_mel_spectrogram(win)
            embedding = extractor.extract(mel)
            result = detector.score(embedding)
            scores.append(result["anomaly_score"])
            progress.progress((i + 1) / len(windows))

        progress.empty()

        # Results
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        anomaly_windows = sum(1 for s in scores if s >= ANOMALY_SCORE_WARNING)

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Avg Score", f"{avg_score:.3f}")
        rc2.metric("Max Score", f"{max_score:.3f}")
        rc3.metric("Anomaly Windows", f"{anomaly_windows}/{len(windows)}")
        rc4.metric("Inference", f"{extractor.last_inference_ms:.1f} ms")

        st.line_chart({"Anomaly Score": scores, "Threshold": [ANOMALY_SCORE_WARNING]*len(scores)}, height=200)

        if avg_score >= ANOMALY_SCORE_CRITICAL:
            st.error("🔴 HIGH-RISK ANOMALY — Immediate inspection recommended.")
        elif avg_score >= ANOMALY_SCORE_WARNING:
            st.warning("🟠 ANOMALY DETECTED — Acoustic deviation from normal baseline.")
        else:
            st.success("🟢 NORMAL — Audio matches learned baseline.")

    finally:
        os.unlink(tmp_path)

# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "MachineEcho v0.1.0 · Contactless Machine Anomaly Detection · "
    "Optimized for Snapdragon Hexagon NPU · On-Device · Private · Low Latency"
    "</p>",
    unsafe_allow_html=True,
)
