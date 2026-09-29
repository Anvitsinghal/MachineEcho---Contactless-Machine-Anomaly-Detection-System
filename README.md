# 🔊 MachineEcho

### Contactless Machine Anomaly Detection on Snapdragon PCs

**MachineEcho** turns a Snapdragon-powered Windows PC into a portable, contactless machine-health monitoring system.

Using only the computer's built-in microphone, MachineEcho learns the acoustic signature of a machine during normal operation and detects deviations from that baseline in real time.

The system combines **YAMNet audio embeddings**, **Isolation Forest anomaly detection**, and **Qualcomm Hexagon NPU acceleration through ONNX Runtime's QNN Execution Provider** to enable private, on-device acoustic monitoring.

> **No cloud audio streaming. No dedicated vibration sensors. No external hardware required.**

---

## 🎯 At a Glance

| Feature                    | Details                               |
| -------------------------- | ------------------------------------- |
| **Platform**               | Snapdragon X Series PCs               |
| **Target Hardware**        | Snapdragon X Elite / X Plus PCs       |
| **Audio Input**            | Built-in microphone                   |
| **Sampling Rate**          | 16 kHz, mono                          |
| **Audio Window**           | 0.96 seconds                          |
| **Feature Extractor**      | YAMNet                                |
| **Feature Representation** | 1024-dimensional audio embeddings     |
| **Anomaly Detector**       | Isolation Forest                      |
| **NPU Runtime**            | ONNX Runtime + QNN Execution Provider |
| **Acceleration Target**    | Qualcomm Hexagon NPU                  |
| **Interface**              | Streamlit                             |
| **Processing**             | On-device                             |
| **Cloud Dependency**       | None for inference                    |

---

# 💡 Problem

Machines such as:

* 🌀 Ceiling fans
* 💧 Water pumps
* 🧺 Washing machines
* ⚙️ Motors
* 🔧 Compressors
* 🏭 Small workshop equipment

often develop changes in their acoustic behavior before a noticeable failure.

Traditional predictive-maintenance systems commonly rely on dedicated sensors such as vibration, temperature, current, or industrial acoustic sensors. These solutions can add hardware cost and installation complexity.

For households and small workshops, a simpler approach is desirable.

### The question

**Can the microphone already present in a laptop be used as a contactless machine-health sensor?**

---

# 🚀 Our Solution

MachineEcho uses the laptop microphone to continuously listen to a machine and compare its current acoustic behavior with a learned normal-operation baseline.

### Pipeline

```text
Machine
   │
   ▼
Built-in Laptop Microphone
   │
   ▼
Audio Preprocessing
   │
   ├── 16 kHz Mono
   ├── 0.96 s Windows
   └── Log-Mel Features
   │
   ▼
YAMNet
   │
   ▼
1024-D Audio Embedding
   │
   ▼
Isolation Forest
   │
   ▼
Anomaly Score
   │
   ▼
NORMAL / ANOMALY
   │
   ▼
Streamlit Dashboard
```

---

# ⚙️ How It Works

## 1. Learn the Machine's Normal Sound

The user records approximately **30–60 seconds of normal machine operation**.

These recordings are converted into audio features and embeddings.

The resulting embeddings represent the machine's normal acoustic behavior.

---

## 2. Extract Acoustic Features

MachineEcho uses **YAMNet** as a general-purpose audio feature extractor.

Instead of using YAMNet's final sound-classification output, MachineEcho uses its learned representation as an embedding.

```text
Audio
  ↓
YAMNet
  ↓
1024-dimensional embedding
```

This allows the system to model machine-specific acoustic behavior without requiring a large labeled dataset of machine failures.

---

## 3. Build an Anomaly Model

The extracted embeddings from normal operation are used to train an **Isolation Forest**.

The model learns the distribution of normal acoustic observations.

During monitoring:

```text
Current Audio
     ↓
YAMNet Embedding
     ↓
Isolation Forest
     ↓
Anomaly Score
     ↓
Threshold
     ↓
NORMAL / ANOMALY
```

---

## 4. Real-Time Monitoring

During monitoring, incoming microphone audio is processed in short overlapping windows.

The dashboard displays:

* Current machine status
* Anomaly score
* Recent score history
* Detection threshold
* Processing information

Example:

```text
┌─────────────────────────────────────┐
│        MACHINEECHO MONITOR          │
├─────────────────────────────────────┤
│                                     │
│        🟢 NORMAL                    │
│                                     │
│        Score: 0.14                  │
│                                     │
│   ─────────────────────────────     │
│   Anomaly Score History             │
│                                     │
└─────────────────────────────────────┘
```

When the acoustic behavior moves sufficiently far from the learned baseline:

```text
🟠 ANOMALY DETECTED
```

---

# ⚡ Snapdragon NPU Acceleration

One of the main goals of MachineEcho is to demonstrate **on-device AI inference on Snapdragon PCs**.

The YAMNet model is exported to ONNX and executed using:

```text
ONNX Runtime
      │
      ▼
QNN Execution Provider
      │
      ▼
Qualcomm Hexagon NPU
```

A CPU fallback is also supported where the QNN execution provider is unavailable.

### Execution modes

```text
Snapdragon PC
     │
     ├── QNN / Hexagon NPU
     │
     └── CPU fallback
```

This makes the application easier to test across different environments while keeping NPU acceleration available on supported Snapdragon systems.

---

# 📊 Benchmark

MachineEcho was evaluated using synthetic and recorded acoustic samples.

Example evaluation results:

| Metric                     | Normal Operation |  Abnormal Operation |
| -------------------------- | ---------------: | ------------------: |
| Mean anomaly score         |       **0.1367** |          **0.6195** |
| Classification             |        🟢 NORMAL | 🟠 ANOMALY DETECTED |
| Detected anomalous windows |                — |         **28 / 30** |
| Detection rate             |                — |           **93.3%** |

### Inference

The evaluation showed a clear separation between the normal-operation baseline and the tested abnormal acoustic samples.

> **Important:** These results represent the current test setup and should not be interpreted as a general machine-failure prediction accuracy. Performance can vary depending on machine type, microphone placement, background noise, and the nature of the anomaly.

---

# 🏗️ System Architecture

```text
                 MACHINE
              Fan / Pump / Motor
                     │
                     ▼
          ┌─────────────────────┐
          │   Laptop Microphone │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Audio Preprocessor  │
          │                     │
          │ 16 kHz Mono         │
          │ 0.96s Windows       │
          │ Log-Mel Features    │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │      YAMNet         │
          │    ONNX Runtime     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ 1024-D Embedding    │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Isolation Forest   │
          │                     │
          │ Normal Baseline     │
          │       ↓             │
          │ Anomaly Detection   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Anomaly Score       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Streamlit Dashboard │
          │                     │
          │ NORMAL / ANOMALY    │
          │ Score / Graph       │
          └─────────────────────┘
```

---

# 🧠 Key Technical Decisions

## YAMNet as a Feature Extractor

Rather than training a neural network from scratch, MachineEcho uses YAMNet's pretrained acoustic representation.

This provides a compact representation of incoming audio that can subsequently be used by the anomaly detector.

### Why this approach?

* Reduces training requirements
* Works with limited machine-specific data
* Provides a reusable acoustic representation
* Separates feature extraction from anomaly detection

---

## Isolation Forest for Anomaly Detection

Machine failure data is difficult to collect because abnormal events are relatively rare.

Therefore, MachineEcho follows a **normal-only / one-class style approach**:

```text
Normal Machine Data
       ↓
Train Detector
       ↓
Learn Normal Distribution
       ↓
New Audio
       ↓
Detect Deviation
```

This avoids requiring a large labeled dataset containing every possible failure mode.

---

## On-Device Processing

Machine audio is processed locally on the computer.

The system does not require continuously uploading microphone recordings to a cloud server.

This provides advantages for:

* Privacy
* Offline operation
* Reduced network dependency
* Lower data transmission
* Local real-time processing

---

# 🔐 Privacy

MachineEcho is designed around an **on-device inference architecture**.

```text
Microphone
    │
    ▼
Local Processing
    │
    ▼
Local AI Inference
    │
    ▼
Local Dashboard
```

Raw microphone audio does not need to be transmitted to a remote server for anomaly detection.

---

# 🖥️ Dashboard

The Streamlit interface provides a simple monitoring experience.

### Dashboard capabilities

* Live machine status
* Current anomaly score
* Score history
* Detection threshold
* Audio monitoring controls
* Model/inference information

---

# 🚀 Quick Start

## 1. Clone the Repository

```powershell
git clone https://github.com/YOUR_USERNAME/MachineEcho.git
cd MachineEcho
```

---

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 3. Run the Demo

The repository includes a synthetic machine-audio demonstration.

```powershell
python demo.py
```

The demo demonstrates the complete pipeline:

```text
Synthetic Machine Audio
        ↓
Feature Extraction
        ↓
Baseline Training
        ↓
Anomaly Detection
        ↓
Result Summary
```

---

## 4. Launch the Dashboard

```powershell
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🎙️ Using a Real Machine

## Step 1 — Record Normal Operation

Record approximately 30 seconds of normal machine audio:

```powershell
python record_audio.py --mode normal --duration 30
```

For better results:

* Keep the microphone position consistent
* Record during stable machine operation
* Minimize background noise
* Avoid moving the laptop during recording

---

## Step 2 — Train the Anomaly Detector

```powershell
python -m anomaly.train_anomaly
```

---

## Step 3 — Start Monitoring

```powershell
streamlit run app.py
```

The system will use the trained baseline to evaluate incoming audio.

---

# 📁 Repository Structure

```text
MachineEcho/
│
├── app.py
├── audio_processor.py
├── feature_extractor.py
├── config.py
├── demo.py
├── generate_demo_audio.py
├── record_audio.py
├── benchmark.py
├── download_yamnet.py
├── requirements.txt
├── README.md
│
├── anomaly/
│   ├── detector.py
│   └── train_anomaly.py
│
└── docs/
    ├── MachineEcho_Pitch_Deck.pptx
    ├── MachineEcho_Brief_Description.pdf
    ├── pitch_deck.md
    └── brief_description.md
```

---

# 📦 Submission Materials

The `docs/` directory contains the supporting materials prepared for the innovation challenge:

### Pitch Deck

`docs/MachineEcho_Pitch_Deck.pptx`

Six-slide presentation covering:

* Problem
* Solution
* Technology
* Architecture
* Results
* Future scope

### Brief Description

`docs/MachineEcho_Brief_Description.pdf`

A concise project description covering the problem, solution, technology, and evaluation.

---

# 🔬 Current Limitations

MachineEcho is a prototype and has several important limitations.

### 1. Machine-specific calibration

Different machines produce very different acoustic signatures.

A baseline should therefore be collected for each machine/environment.

### 2. Environmental noise

Background sounds can affect the extracted acoustic features.

### 3. Limited anomaly coverage

The current evaluation does not represent every possible mechanical failure.

An anomaly score indicates **acoustic deviation from the learned baseline**, not a guaranteed prediction of physical failure.

### 4. Synthetic evaluation

Part of the current demonstration uses synthetically generated machine audio. Real-world validation across different machines and failure conditions is required before making reliability claims.

### 5. Hardware-dependent NPU acceleration

QNN/Hexagon acceleration depends on compatible Snapdragon hardware, drivers, ONNX Runtime configuration, and model support.

---

# 🔮 Future Scope

MachineEcho can be extended into a more comprehensive predictive-maintenance platform.

### Multi-machine profiles

```text
Machine A → Baseline A
Machine B → Baseline B
Machine C → Baseline C
```

### Edge AI model optimization

Further optimize the complete inference pipeline for Snapdragon NPU execution.

### More anomaly types

Build datasets containing:

* Bearing wear
* Fan imbalance
* Motor friction
* Loose components
* Pump cavitation
* Belt problems

### Sensor fusion

Combine acoustic monitoring with:

```text
Audio
  +
Vibration
  +
Temperature
  +
Current
```

to improve machine-health monitoring.

### Federated / privacy-preserving learning

Future versions could allow models to improve across multiple machines without requiring raw audio to leave the device.

---

# 🏆 Innovation

MachineEcho explores a simple idea:

> **The microphone already inside a laptop can potentially become a machine-health sensor.**

Instead of requiring dedicated industrial monitoring hardware, MachineEcho combines:

**Commodity microphone + pretrained audio representation + anomaly detection + Snapdragon NPU acceleration**

into a portable edge-AI monitoring system.

---

# 🛠️ Technology Stack

| Component          | Technology                         |
| ------------------ | ---------------------------------- |
| Language           | Python                             |
| Audio Processing   | NumPy / audio processing libraries |
| Feature Extraction | YAMNet                             |
| Model Format       | ONNX                               |
| Inference          | ONNX Runtime                       |
| NPU Acceleration   | Qualcomm QNN / Hexagon             |
| Anomaly Detection  | Scikit-learn Isolation Forest      |
| Dashboard          | Streamlit                          |
| Visualization      | Streamlit charts                   |
| Platform           | Snapdragon Windows PC              |

---

# 📜 License

This project was developed for the **Qualcomm & Unstop Innovation Challenge**.

### Acknowledgements

* **Qualcomm AI Hub** — model deployment resources and Snapdragon AI acceleration
* **Google Research** — YAMNet architecture and pretrained audio model
* **ONNX Runtime** — model inference framework
* **scikit-learn** — Isolation Forest implementation
* **Streamlit** — interactive application interface

---

# 👥 Project

**MachineEcho**

> Contactless acoustic anomaly detection for machines using Snapdragon-powered edge AI.

Built for the **Qualcomm & Unstop Innovation Challenge**.
