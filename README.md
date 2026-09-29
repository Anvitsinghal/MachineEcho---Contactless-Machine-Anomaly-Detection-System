<![CDATA[# 🔊 MachineEcho

**Contactless Machine Anomaly Detection on Snapdragon PCs**

MachineEcho turns an ordinary Snapdragon-powered HP laptop into a portable, contactless machine-health monitoring system. Using only the laptop's built-in microphone, it learns the normal acoustic signature of any machine and detects deviations in real time — entirely on-device, with inference accelerated by the Qualcomm Hexagon NPU.

---

## 🎯 Problem

Everyday machines — ceiling fans, water pumps, washing machines, compressors, small motors — often develop abnormal sounds and vibrations **before** they fail. Detecting these early usually requires specialized sensors, technicians, or periodic inspection, making predictive maintenance **inaccessible** for homes, small businesses, and resource-constrained environments.

## 💡 Solution

MachineEcho provides **contactless early anomaly detection** using multimodal on-device AI:

1. **Learn** — Record 30–60 seconds of a machine's normal operation
2. **Monitor** — Continuously analyze the acoustic behaviour in real time
3. **Alert** — Get immediate warnings when the machine deviates from its baseline

No specialized sensors. No cloud. No subscription. Just a laptop microphone and Snapdragon AI.

---

## 🏗️ Architecture

```
MACHINE (fan, pump, motor, …)
        │
        ▼
  ┌─────────────┐
  │  Microphone  │  Built-in laptop mic
  └──────┬──────┘
         │ 16 kHz mono audio stream
         ▼
  ┌─────────────────────┐
  │  Audio Preprocessor  │  Windowing + Log-Mel Spectrogram
  └──────────┬──────────┘
             │ (1, 96, 64) spectrogram patch
             ▼
  ┌─────────────────────┐
  │      YAMNet          │  Audio event classification model
  │   (ONNX Runtime)     │  from Qualcomm AI Hub
  │                      │
  │  Execution Provider: │
  │  QNN → Hexagon NPU  │
  └──────────┬──────────┘
             │ 1024-dim embedding
             ▼
  ┌─────────────────────┐
  │  Isolation Forest    │  One-class anomaly detector
  │  Anomaly Detector    │  trained on normal-operation
  └──────────┬──────────┘  embeddings
             │
             ▼
  ┌─────────────────────┐
  │   Status + Score     │
  │   🟢 NORMAL          │
  │   🟠 ANOMALY         │
  │   🔴 HIGH RISK       │
  └─────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YAMNet as feature extractor** (not classifier) | We don't classify machine types — we learn each machine's unique baseline |
| **Isolation Forest** for anomaly detection | Trains on normal data only; no need for labelled anomaly examples |
| **ONNX Runtime with QNN EP** | Direct Snapdragon NPU acceleration; falls back to CPU gracefully |
| **Streamlit dashboard** | Simple, effective, fast to build — focus is on the AI, not the UI |

---

## 🚀 Quick Start

### Prerequisites

- **Hardware:** Snapdragon X Elite / X Plus / X2 Elite laptop (e.g., HP OmniBook Ultra)
- **OS:** Windows 11 (ARM64)
- **Python:** 3.10+
- **ONNX Runtime:** With QNN Execution Provider for NPU acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MachineEcho.git
cd MachineEcho

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Record Normal Audio

Point your laptop microphone at the machine and record 30–60 seconds of normal operation:

```bash
python record_audio.py --mode normal --duration 30
```

### Step 2: Train the Anomaly Detector

```bash
python -m anomaly.train_anomaly
```

This extracts YAMNet embeddings from your normal audio and trains an Isolation Forest anomaly detector.

### Step 3: Start Monitoring

```bash
streamlit run app.py
```

The dashboard will show real-time anomaly detection with:
- **Status indicator** (Normal / Anomaly / High Risk)
- **Anomaly score** (0–1 scale)
- **Inference latency** (ms)
- **Score history chart**

### Step 4: Benchmark NPU Performance

```bash
python benchmark.py --runs 100
```

Compares inference latency across execution providers (QNN NPU, DirectML GPU, CPU).

---

## 📁 Project Structure

```
MachineEcho/
├── app.py                      # Streamlit real-time dashboard
├── audio_processor.py          # Microphone capture, windowing, mel spectrogram
├── feature_extractor.py        # YAMNet ONNX inference (QNN/CPU)
├── config.py                   # All configuration constants
├── record_audio.py             # CLI audio recording utility
├── benchmark.py                # NPU vs CPU latency benchmark
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── anomaly/
│   ├── __init__.py
│   ├── detector.py             # IsolationForest anomaly detector
│   └── train_anomaly.py        # Training script
│
├── anomaly_model/
│   └── detector.pkl            # Trained anomaly model (generated)
│
├── models/
│   └── yamnet.onnx             # YAMNet ONNX model (download from Qualcomm AI Hub)
│
├── data/
│   ├── normal/                 # Normal-operation WAV recordings
│   └── abnormal/               # Abnormal-operation WAV recordings (for testing)
│
└── docs/
    └── architecture.png        # Architecture diagram
```

---

## 🎯 Why Snapdragon

MachineEcho is **fundamentally designed** around Snapdragon's architecture:

| Capability | Snapdragon Advantage |
|------------|---------------------|
| **Sustained NPU inference** | Hexagon NPU provides dedicated AI compute without thermal throttling |
| **Low power consumption** | Continuous monitoring for hours without draining battery |
| **On-device privacy** | All audio processing stays on the laptop — nothing leaves the device |
| **Low latency** | Sub-millisecond YAMNet inference on NPU (est. ~266 μs on X Elite) |
| **Qualcomm AI Hub** | Pre-optimized YAMNet model with Snapdragon X-series NPU profiles |

### Execution Provider Priority

```
1. QNNExecutionProvider  →  Qualcomm Hexagon NPU (preferred)
2. CPUExecutionProvider  →  Fallback for non-Snapdragon hardware
```

---

## 📊 How It Works

### Training Phase (one-time, per machine)

1. Record 30–60 seconds of the machine running normally
2. Audio is split into 0.96-second overlapping windows
3. Each window → log-mel spectrogram → YAMNet → 1024-dim embedding
4. All normal embeddings train an Isolation Forest (one-class classifier)
5. The trained model is saved to `anomaly_model/detector.pkl`

### Monitoring Phase (continuous)

1. Microphone captures live audio in 0.96-second windows
2. Each window → spectrogram → YAMNet → embedding → anomaly score
3. Score is normalized to [0, 1]: higher = more anomalous
4. Dashboard displays status, score, chart, and evidence

### Scoring Thresholds

| Score Range | Status | Action |
|-------------|--------|--------|
| 0.0 – 0.4 | 🟢 **NORMAL** | Machine operating within baseline |
| 0.4 – 0.7 | 🟠 **ANOMALY DETECTED** | Acoustic deviation — schedule inspection |
| 0.7 – 1.0 | 🔴 **HIGH-RISK ANOMALY** | Significant deviation — immediate attention |

---

## 🔬 Technical Details

### Audio Processing

- **Sample rate:** 16,000 Hz (mono)
- **Window duration:** 0.96 seconds (15,360 samples)
- **Hop duration:** 0.48 seconds (50% overlap)
- **Mel spectrogram:** 64 bands, 10 ms hop, 25 ms window, 125–7,500 Hz

### YAMNet Model

- **Type:** Audio event classification (used as feature extractor)
- **Input:** Log-mel spectrogram patch (1, 96, 64)
- **Output:** 1024-dimensional embedding vector
- **Format:** ONNX (from Qualcomm AI Hub)
- **NPU support:** Snapdragon X Elite / X Plus / X2 Elite

### Anomaly Detection

- **Algorithm:** Isolation Forest (scikit-learn)
- **Training data:** Normal-operation embeddings only
- **Contamination:** 5% (expected false-positive rate)
- **Estimators:** 100 trees

---

## 🛣️ Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Acoustic anomaly detection | ✅ Implemented |
| 2 | Vision-assisted inspection (camera) | 🔜 Planned |
| 3 | Multi-machine simultaneous monitoring | 🔜 Planned |
| 4 | Failure-type classification | 🔜 Planned |
| 5 | Mobile companion app | 🔜 Planned |

---

## 📝 License

This project is developed for the Qualcomm & Unstop Innovation Challenge.

---

## 🙏 Acknowledgements

- **Qualcomm AI Hub** — YAMNet model and Snapdragon NPU profiles
- **Google Research** — Original YAMNet architecture
- **MIMII / ToyADMOS** — Machine-sound anomaly detection research
- **scikit-learn** — Isolation Forest implementation
- **Streamlit** — Dashboard framework

---

<p align="center">
  <b>MachineEcho</b> — Transforming ordinary laptops into intelligent machine-health monitors.<br>
  Built for Snapdragon. Powered by AI. Private by design.
</p>
]]>
