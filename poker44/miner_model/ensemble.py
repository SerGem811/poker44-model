"""EnsembleClassifier and BlendedIsotonicCalibrator for Poker44 bot detection.

Kept in the poker44 package so joblib can unpickle EnsembleClassifier bundles
from any working directory without needing scripts/miner/train on sys.path.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class BlendedIsotonicCalibrator:
    """Blend isotonic-calibrated probabilities with raw scores.

    blend=0.5 de-saturates while preserving ranking monotonicity.
    """

    def __init__(self, blend: float = 0.5):
        self.blend = blend
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, proba_raw: np.ndarray, y: np.ndarray) -> "BlendedIsotonicCalibrator":
        self._iso.fit(proba_raw, y)
        return self

    def transform(self, proba_raw: np.ndarray) -> np.ndarray:
        cal = self._iso.predict(proba_raw)
        return self.blend * cal + (1.0 - self.blend) * proba_raw


class EnsembleClassifier:
    """5-way ensemble with embedded BlendedIsotonicCalibrator."""

    def __init__(self, base_models: list, calibrator: BlendedIsotonicCalibrator):
        self.base_models = base_models
        self.calibrator = calibrator
        self.feature_names_in_: list | None = None

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        proba_sum = None
        for _name, clf in self.base_models:
            p = clf.predict_proba(X)[:, 1]
            if proba_sum is None:
                proba_sum = p.copy()
            else:
                proba_sum += p
        return proba_sum / len(self.base_models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        avg = self._raw_proba(X)
        calibrated = np.clip(self.calibrator.transform(avg), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
