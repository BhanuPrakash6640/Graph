"""Model factories and hyperparameter grids for tabular baselines (LR, RF, XGBoost)."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


LR_HP_GRID: list[dict[str, Any]] = [
    {"C": 0.01},
    {"C": 0.1},
    {"C": 1.0},
    {"C": 10.0},
]

RF_HP_GRID: list[dict[str, Any]] = [
    {"max_depth": None, "max_features": "sqrt"},
    {"max_depth": None, "max_features": 0.3},
    {"max_depth": 20, "max_features": "sqrt"},
    {"max_depth": 20, "max_features": 0.3},
]

XGB_HP_GRID: list[dict[str, Any]] = [
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.1},
    {"max_depth": 8, "learning_rate": 0.05},
    {"max_depth": 8, "learning_rate": 0.1},
]


def create_lr(
    C: float = 1.0,
    random_state: int = 0,
    max_iter: int = 5000,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )


def create_rf(
    max_depth: int | None = None,
    max_features: str | float = "sqrt",
    random_state: int = 0,
    n_estimators: int = 500,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def create_xgb(
    max_depth: int = 6,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 0,
    n_estimators: int = 2000,
    early_stopping_rounds: int | None = 50,
    device: str | None = None,
) -> XGBClassifier:
    if device is None:
        device = "cuda" if HAS_CUDA else "cpu"

    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": random_state,
        "device": device,
    }
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds

    return XGBClassifier(**kwargs)

