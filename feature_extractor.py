"""MachineEcho — YAMNet Feature Extractor.

Loads a YAMNet ONNX model via ONNX Runtime and extracts
1024-dimensional audio embeddings. Supports QNN (Hexagon NPU)
and CPU execution providers.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from config import (
    YAMNET_ONNX_PATH,
    YAMNET_EMBEDDING_DIM,
    EXECUTION_PROVIDERS,
)


class FeatureExtractor:
    """YAMNet-based audio feature extractor using ONNX Runtime.

    Falls back gracefully:
      QNN EP (NPU) -> CPU EP -> pure-NumPy stub embeddings.
    """

    def __init__(
        self,
        model_path: str | Path = YAMNET_ONNX_PATH,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.providers = providers or list(EXECUTION_PROVIDERS)
        self.session: Optional[ort.InferenceSession] = None  # type: ignore
        self.active_provider: str = "none"
        self.last_inference_ms: float = 0.0
        self._use_stub = False

        self._load_model()

    # ── model loading ─────────────────────────────────────────

    def _load_model(self) -> None:
        """Try to load the ONNX model with the best available EP."""
        if ort is None:
            print("[FeatureExtractor] onnxruntime not installed -- using stub embeddings.")
            self._use_stub = True
            return

        if not self.model_path.exists():
            print(
                f"[FeatureExtractor] Model not found at {self.model_path}\n"
                "  -> Using stub embeddings. Download yamnet.onnx from Qualcomm AI Hub\n"
                "    and place it in the models/ directory."
            )
            self._use_stub = True
            return

        # Try each provider in priority order
        for provider in self.providers:
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_options,
                    providers=[provider],
                )
                self.active_provider = provider
                print(f"[FeatureExtractor] Loaded YAMNet with {provider}")
                return
            except Exception as exc:
                print(f"[FeatureExtractor] {provider} unavailable: {exc}")

        print("[FeatureExtractor] No ONNX provider worked -- using stub embeddings.")
        self._use_stub = True

    # ── inference ─────────────────────────────────────────────

    def extract(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Extract a 1024-dim embedding from a mel spectrogram.

        Parameters
        ----------
        mel_spectrogram : np.ndarray
            Shape (1, 96, 64) float32 log-mel spectrogram patch.

        Returns
        -------
        np.ndarray
            Shape (1024,) float32 embedding vector.
        """
        if self._use_stub:
            return self._stub_embedding(mel_spectrogram)

        input_name = self.session.get_inputs()[0].name
        input_data = mel_spectrogram.astype(np.float32)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        self.last_inference_ms = (t1 - t0) * 1000.0

        # YAMNet typically outputs [scores, embeddings, spectrogram]
        # The embedding is the second output with shape (N, 1024)
        # If the model only has one output, treat it as scores and
        # use it directly as the feature vector (truncated/padded).
        if len(outputs) >= 2:
            embedding = outputs[1]
        else:
            embedding = outputs[0]

        # Flatten and ensure correct dimension
        embedding = np.array(embedding, dtype=np.float32).flatten()
        if len(embedding) > YAMNET_EMBEDDING_DIM:
            embedding = embedding[:YAMNET_EMBEDDING_DIM]
        elif len(embedding) < YAMNET_EMBEDDING_DIM:
            embedding = np.pad(
                embedding,
                (0, YAMNET_EMBEDDING_DIM - len(embedding)),
                mode="constant",
            )
        return embedding

    def _stub_embedding(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Generate a discriminative pseudo-embedding from mel features.

        Used when the ONNX model is unavailable. Extracts per-band
        and per-frame statistics to create a rich 1024-dim vector
        that captures spectral shape, temporal dynamics, and energy
        distribution -- sufficient for anomaly detection in demos.
        """
        t0 = time.perf_counter()
        # spec shape: (1, 96, 64) -> (96, 64) = (time_frames, mel_bands)
        spec = mel_spectrogram.squeeze()
        if spec.ndim == 1:
            spec = spec.reshape(-1, 64)

        features = []

        # Per-band statistics (64 bands x 4 stats = 256 features)
        features.append(np.mean(spec, axis=0))    # 64
        features.append(np.std(spec, axis=0))      # 64
        features.append(np.max(spec, axis=0))      # 64
        features.append(np.min(spec, axis=0))      # 64

        # Per-frame statistics (96 frames x 4 stats = 384 features)
        features.append(np.mean(spec, axis=1))    # 96
        features.append(np.std(spec, axis=1))      # 96
        features.append(np.max(spec, axis=1))      # 96
        features.append(np.min(spec, axis=1))      # 96

        # Spectral shape features (64 features)
        total_energy = np.sum(spec, axis=0) + 1e-8
        spectral_centroid = np.sum(
            spec * np.arange(spec.shape[0])[:, None], axis=0
        ) / total_energy
        features.append(spectral_centroid)          # 64

        # Temporal dynamics: frame-to-frame differences (95 x 2 = 190)
        diff = np.diff(spec, axis=0)               # (95, 64)
        features.append(np.mean(diff, axis=1))     # 95
        features.append(np.std(diff, axis=1))       # 95

        # Flatten into a single vector
        embedding = np.concatenate([f.flatten() for f in features]).astype(np.float32)

        # Pad or trim to exactly 1024
        if len(embedding) > YAMNET_EMBEDDING_DIM:
            embedding = embedding[:YAMNET_EMBEDDING_DIM]
        elif len(embedding) < YAMNET_EMBEDDING_DIM:
            embedding = np.pad(
                embedding,
                (0, YAMNET_EMBEDDING_DIM - len(embedding)),
                mode="constant",
            )

        t1 = time.perf_counter()
        self.last_inference_ms = (t1 - t0) * 1000.0
        return embedding

    # ── info ──────────────────────────────────────────────────

    def info(self) -> dict:
        """Return a summary of the extractor state."""
        return {
            "model_path": str(self.model_path),
            "active_provider": self.active_provider,
            "using_stub": self._use_stub,
            "last_inference_ms": round(self.last_inference_ms, 3),
        }
