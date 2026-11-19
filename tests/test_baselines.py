"""Tests for baseline model factories and tuning logic using synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from ml.baselines.models import (
    create_lr,
    create_rf,
    create_xgb,
)
from ml.baselines.train import tune_lr, tune_rf, tune_xgb


def test_model_factories():
    lr = create_lr(C=0.1, random_state=42)
    assert lr.C == 0.1
    assert lr.class_weight == "balanced"

    rf = create_rf(max_depth=10, max_features="sqrt", random_state=42, n_estimators=50)
    assert rf.max_depth == 10
    assert rf.n_estimators == 50

    xgb = create_xgb(max_depth=4, learning_rate=0.05, scale_pos_weight=2.5, random_state=42, n_estimators=100)
    assert xgb.max_depth == 4
    assert xgb.learning_rate == 0.05
    assert xgb.scale_pos_weight == 2.5


def test_tuning_functions_synthetic():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(60, 10)).astype(np.float32)
    y_train = rng.choice([0, 1], size=60, p=[0.7, 0.3]).astype(np.int8)

    X_val = rng.normal(size=(30, 10)).astype(np.float32)
    y_val = rng.choice([0, 1], size=30, p=[0.7, 0.3]).astype(np.int8)

    # LR tuning
    best_lr = tune_lr(X_train, y_train, X_val, y_val)
    assert "C" in best_lr

    # RF tuning
    best_rf = tune_rf(X_train, y_train, X_val, y_val, seed=0)
    assert "max_depth" in best_rf
    assert "max_features" in best_rf

    # XGB tuning
    best_xgb, best_iter = tune_xgb(X_train, y_train, X_val, y_val, scale_pos_weight=2.0, seed=0)
    assert "max_depth" in best_xgb
    assert "learning_rate" in best_xgb
    assert best_iter >= 0

