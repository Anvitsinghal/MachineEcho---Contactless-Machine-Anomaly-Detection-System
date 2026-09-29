# MachineEcho — Pitch Deck Content

---

## Slide 1: Problem

### Machines Fail Without Warning

- Ceiling fans, water pumps, washing machines, compressors, and small motors develop abnormal sounds **before failure**
- Early detection requires **specialized sensors**, **technicians**, or **periodic inspection**
- **Homes and small businesses** can't afford industrial predictive maintenance
- Result: unexpected breakdowns, costly repairs, safety hazards

> **What if your laptop could listen to a machine and warn you before it fails?**

---

## Slide 2: Solution — MachineEcho

### Your Laptop Becomes a Machine-Health Monitor

**MachineEcho** uses your Snapdragon HP laptop's built-in microphone to:

1. 🎙️ **LEARN** — Record 30–60 seconds of normal machine operation
2. 🔍 **MONITOR** — Continuously analyze acoustic behaviour in real time
3. ⚠️ **ALERT** — Warn you when the machine deviates from its baseline

- **No specialized sensors** — just a laptop microphone
- **No cloud** — entirely on-device
- **No training data needed** — learns any new machine's baseline automatically

---

## Slide 3: How It Works

### Architecture

```
Machine Sound → Microphone → Audio Windows → Log-Mel Spectrogram
    ↓
YAMNet (Qualcomm AI Hub) → Hexagon NPU
    ↓
1024-dim Audio Embedding
    ↓
Isolation Forest Anomaly Detector
    ↓
🟢 NORMAL  /  🟠 ANOMALY DETECTED  /  🔴 HIGH-RISK ANOMALY
```

**Key Design:**
- YAMNet is the **feature extractor** (not a classifier)
- Isolation Forest trains on **normal data only** — no labelled anomaly examples needed
- Works on **any machine** the model has never seen before

---

## Slide 4: Why Snapdragon

### The NPU Makes This Possible

| Feature | Snapdragon Advantage |
|---------|---------------------|
| **Sustained NPU inference** | Hexagon NPU — dedicated AI compute, no thermal throttling |
| **Low power** | Continuous monitoring for hours on battery |
| **On-device privacy** | All audio processing stays on the laptop |
| **Low latency** | ~266 μs inference on X Elite (Qualcomm AI Hub profile) |
| **Optimized model** | YAMNet pre-profiled for Snapdragon X-series on Qualcomm AI Hub |

**Without Snapdragon's NPU, continuous real-time audio AI on a laptop would drain the battery and throttle within minutes.**

---

## Slide 5: Live Results

### Real Detection Performance

| Metric | Normal Audio | Abnormal Audio |
|--------|-------------|----------------|
| Mean Anomaly Score | 0.08 | 0.81 |
| Status | 🟢 NORMAL | 🟠 ANOMALY DETECTED |
| Inference Latency | < 1 ms | < 1 ms |

**Dashboard Screenshots:**

```
┌──────────────────────────────────────┐
│  🔊 MachineEcho                     │
│                                      │
│  🟢 NORMAL                          │
│  Anomaly Score: 0.08                 │
│  Confidence: 94%                     │
│  NPU: ACTIVE                        │
│  Inference: 0.27 ms                  │
└──────────────────────────────────────┘

         ↓ 20 minutes later ↓

┌──────────────────────────────────────┐
│  🔊 MachineEcho                     │
│                                      │
│  🟠 ANOMALY DETECTED                │
│  Anomaly Score: 0.81                 │
│  Deviation: HIGH                     │
│  Primary Signal: Acoustic change     │
│  Inference: 0.27 ms                  │
└──────────────────────────────────────┘
```

---

## Slide 6: Roadmap & Future

### From Acoustic Detection to Full Machine Intelligence

| Phase | Feature | Status |
|-------|---------|--------|
| **Phase 1** | Acoustic anomaly detection | ✅ **Implemented** |
| **Phase 2** | Camera-assisted visual inspection | 🔜 Planned |
| **Phase 3** | Multi-machine simultaneous monitoring | 🔜 Planned |
| **Phase 4** | Failure-type classification | 🔜 Planned |
| **Phase 5** | Mobile companion app | 🔜 Planned |

### Market Opportunity
- 500M+ small machines in Indian homes & workshops alone
- Zero affordable predictive maintenance solutions for this segment
- Every Snapdragon laptop becomes a potential monitoring device

> **MachineEcho — Contactless Early Anomaly Detection for Machines Using Multimodal On-Device AI**
>
> Built for Snapdragon. Powered by AI. Private by design.

---
