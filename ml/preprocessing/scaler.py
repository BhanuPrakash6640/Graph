"""Standardisation fitted on training time steps only.

A small, explicit implementation (instead of sklearn) so the scaler records
*which steps it was fitted on*; ``prepare_split`` refuses a scaler whose fit
steps are not a subset of the split's training steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StepScaler:
    mean: np.ndarray
    std: np.ndarray
    fit_steps: tuple[int, ...]
    feature_names: tuple[str, ...]

    @classmethod
    def fit(cls, X: np.ndarray, steps: np.ndarray, fit_steps, feature_names) -> "StepScaler":
        fit_steps = tuple(sorted(int(s) for s in fit_steps))
        mask = np.isin(steps, fit_steps)
        if not mask.any():
            raise ValueError("No rows in fit_steps")
        Xf = X[mask].astype(np.float64)
        mean = Xf.mean(axis=0)
        std = Xf.std(axis=0)
        std[std < 1e-12] = 1.0  # constant-in-train features pass through centred
        return cls(mean.astype(np.float32), std.astype(np.float32), fit_steps, tuple(feature_names))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} features, got {X.shape[1]}")
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std,
                 fit_steps=np.array(self.fit_steps, dtype=np.int16),
                 feature_names=np.array(self.feature_names))

    @classmethod
    def load(cls, path: str | Path) -> "StepScaler":
        z = np.load(path, allow_pickle=False)
        return cls(z["mean"], z["std"], tuple(int(s) for s in z["fit_steps"]),
                   tuple(str(n) for n in z["feature_names"]))
