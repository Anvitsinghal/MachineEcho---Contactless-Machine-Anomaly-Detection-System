"""MachineEcho — Anomaly Detector.

Wraps scikit-learn's IsolationForest for one-class anomaly detection
on YAMNet embeddings. Supports training on normal-operation audio only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ANOMALY_MODEL_PATH,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    ANOMALY_SCORE_WARNING,
    ANOMALY_SCORE_CRITICAL,
)


class AnomalyDetector:
    """One-class anomaly detector trained on normal-operation embeddings."""

    def __init__(self) -> None:
        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self._training_stats: dict = {}

    # ── training ──────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> dict:
        """Train the anomaly detector on normal-operation embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (N, 1024) — one embedding per audio window.

        Returns
        -------
        dict
            Training statistics.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Standardize features
        X = self.scaler.fit_transform(embeddings)

        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=ISOLATION_FOREST_CONTAMINATION,
            n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
            random_state=ISOLATION_FOREST_RANDOM_STATE,
            warm_start=False,
        )
        self.model.fit(X)
        self.is_trained = True

        # Compute training baselines
        train_scores = self.model.decision_function(X)
        self._training_stats = {
            "n_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
            "train_score_min": float(np.min(train_scores)),
            "train_score_max": float(np.max(train_scores)),
        }
        return self._training_stats

    # ── scoring ───────────────────────────────────────────────

    def score(self, embedding: np.ndarray) -> dict:
        """Score a single embedding for anomaly.

        Parameters
        ----------
        embedding : np.ndarray
            Shape (1024,) — single audio-window embedding.

        Returns
        -------
        dict
            {
              'raw_score': float,       # IsolationForest decision_function
              'anomaly_score': float,    # normalized to [0, 1], higher = more anomalous
              'prediction': int,         # 1 = normal, -1 = anomaly
              'status': str,             # 'NORMAL', 'ANOMALY DETECTED', 'HIGH-RISK ANOMALY'
              'confidence': float,       # confidence percentage
            }
        """
        if not self.is_trained:
            return {
                "raw_score": 0.0,
                "anomaly_score": 0.0,
                "prediction": 1,
                "status": "NOT TRAINED",
                "confidence": 0.0,
            }

        X = embedding.reshape(1, -1)
        X = self.scaler.transform(X)

        raw_score = float(self.model.decision_function(X)[0])
        prediction = int(self.model.predict(X)[0])

        # Calibrated normalization using training baseline.
        # IsolationForest decision_function: positive = inlier, negative = outlier.
        # We compute how many training-stds below the training mean this score is,
        # then map to [0, 1] where 0 = perfectly normal, 1 = extreme anomaly.
        train_mean = self._training_stats.get("train_score_mean", 0.0)
        train_std = self._training_stats.get("train_score_std", 1.0)
        train_std = max(train_std, 1e-6)  # avoid division by zero

        # z-score: how many stds below the training mean
        z = (train_mean - raw_score) / train_std

        # Map z-score to [0, 1] with a sigmoid-like function.
        # z <= 0 (at or above training mean) -> score near 0
        # z = 2  (2 stds below mean)        -> score ~0.5
        # z = 4  (4 stds below mean)        -> score ~0.9
        anomaly_score = float(1.0 / (1.0 + np.exp(-1.5 * (z - 1.5))))
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

        # Determine status
        if anomaly_score >= ANOMALY_SCORE_CRITICAL:
            status = "HIGH-RISK ANOMALY"
            confidence = min(99.0, 70.0 + anomaly_score * 30)
        elif anomaly_score >= ANOMALY_SCORE_WARNING:
            status = "ANOMALY DETECTED"
            confidence = min(95.0, 60.0 + anomaly_score * 35)
        else:
            status = "NORMAL"
            confidence = max(50.0, (1.0 - anomaly_score) * 100)

        return {
            "raw_score": round(raw_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "prediction": prediction,
            "status": status,
            "confidence": round(confidence, 1),
        }

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str | Path = ANOMALY_MODEL_PATH) -> None:
        """Serialize the trained model + scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "training_stats": self._training_stats,
            },
            path,
        )
        print(f"[AnomalyDetector] Saved -> {path}")

    def load(self, path: str | Path = ANOMALY_MODEL_PATH) -> bool:
        """Load a previously trained model from disk."""
        path = Path(path)
        if not path.exists():
            print(f"[AnomalyDetector] No model at {path}")
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._training_stats = data.get("training_stats", {})
        self.is_trained = True
        print(f"[AnomalyDetector] Loaded <- {path}")
        return True

    # ── info ──────────────────────────────────────────────────

    def info(self) -> dict:
        """Return a summary of the detector state."""
        return {
            "is_trained": self.is_trained,
            "training_stats": self._training_stats,
        }
