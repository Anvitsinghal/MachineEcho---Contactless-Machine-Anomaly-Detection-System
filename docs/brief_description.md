# MachineEcho — Brief Description

## Project Title
**MachineEcho: Contactless Machine Anomaly Detection on Snapdragon PCs**

## Problem Statement
Everyday machines — ceiling fans, water pumps, washing machines, compressors, and small motors — often develop abnormal sounds and vibrations before failure. Detecting these early requires specialized sensors, technicians, or periodic inspection, making predictive maintenance inaccessible for homes, small businesses, workshops, and resource-constrained environments.

## Solution
MachineEcho transforms a Snapdragon-powered HP laptop into a contactless machine-health monitoring system using only the built-in microphone. The system:

1. **Learns** a machine's normal acoustic signature from a short recording (30–60 seconds)
2. **Monitors** in real time using YAMNet audio embeddings accelerated by the Hexagon NPU
3. **Detects** deviations using Isolation Forest one-class anomaly detection
4. **Alerts** with interpretable status: Normal → Anomaly Detected → High-Risk Anomaly

## Technical Architecture
- **Audio Input:** Built-in laptop microphone → 16 kHz mono stream
- **Feature Extraction:** YAMNet (ONNX, from Qualcomm AI Hub) → 1024-dim embeddings
- **NPU Acceleration:** ONNX Runtime with QNN Execution Provider → Hexagon NPU
- **Anomaly Detection:** Isolation Forest trained on normal-operation embeddings only
- **Dashboard:** Streamlit real-time monitoring with score history

## Why Snapdragon
- **Dedicated NPU** for sustained, low-power inference — enabling continuous monitoring
- **On-device processing** — complete privacy; no audio data leaves the laptop
- **Qualcomm AI Hub** — pre-optimized YAMNet profiles for Snapdragon X Elite/X Plus
- **Sub-millisecond inference** — real-time detection without battery drain

## Key Innovation
MachineEcho does **not** classify known machine types. Instead, it learns any machine's unique baseline and detects deviations — making it applicable to machines the model has never seen before. This is contactless anomaly detection, not classification.

## Target Platform
- **Hardware:** HP OmniBook Ultra / Snapdragon X Elite / X Plus / X2 Elite
- **OS:** Windows 11 (ARM64)
- **NPU:** Qualcomm Hexagon via QNN Execution Provider

## Team
Anvit Singhal

---
*MachineEcho — Transforming ordinary laptops into intelligent machine-health monitors.*
