"""Evaluation metrics, threshold selection, per-step metrics and bootstrap CIs.

Positive class is always illicit (1). Unknown (-1) nodes are strictly excluded.
Primary metric is PR-AUC (average precision).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.data.loader import UNKNOWN


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard classification metrics at the given threshold.

    y_true and scores must be 1D arrays of the same length without unknown (-1) labels.
    """
    y_arr = np.asarray(y_true, dtype=np.int8)
    s_arr = np.asarray(scores, dtype=np.float64)

    assert y_arr.ndim == 1, f"y_true must be 1D, got shape {y_arr.shape}"
    assert s_arr.ndim == 1, f"scores must be 1D, got shape {s_arr.shape}"
    assert len(y_arr) == len(s_arr), f"Length mismatch: {len(y_arr)} vs {len(s_arr)}"
    assert not np.isin(y_arr, UNKNOWN).any(), f"y_true contains unknown labels ({UNKNOWN})"
    assert 0.0 <= threshold <= 1.0, f"threshold must be in [0, 1], got {threshold}"

    n = int(len(y_arr))
    n_pos = int((y_arr == 1).sum())
    base_rate = float(n_pos / n) if n > 0 else 0.0

    y_pred = (s_arr >= threshold).astype(int)

    # Confusion matrix with explicit labels [0, 1] -> [[tn, fp], [fn, tp]]
    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1]).tolist()
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(precision_score(y_arr, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_arr, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, pos_label=1, zero_division=0))

    if 0 < n_pos < n:
        roc_auc = float(roc_auc_score(y_arr, s_arr))
        pr_auc = float(average_precision_score(y_arr, s_arr))
    elif n_pos == n or n_pos == 0:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "base_rate": base_rate,
        "n": n,
        "n_pos": n_pos,
        "threshold": float(threshold),
    }


def best_f1_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Find the decision threshold that maximizes F1 score on the validation set.

    Computed from the precision-recall curve on the validation set.
    """
    y_arr = np.asarray(y_val, dtype=np.int8)
    s_arr = np.asarray(s_val, dtype=np.float64)

    assert not np.isin(y_arr, UNKNOWN).any(), f"y_val contains unknown labels ({UNKNOWN})"
    if len(y_arr) == 0 or (y_arr == 1).sum() == 0:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_arr, s_arr, pos_label=1)
    # precision_recall_curve returns len(thresholds) + 1 for precision and recall
    p = precisions[:-1]
    r = recalls[:-1]

    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)

    if len(thresholds) == 0 or f1.max() <= 0:
        return 0.5

    best_idx = int(np.argmax(f1))
    best_t = float(thresholds[best_idx])
    return float(np.clip(best_t, 0.0, 1.0))


def per_step_metrics(
    y: np.ndarray,
    s: np.ndarray,
    steps: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute metrics broken down per time step."""
    y_arr = np.asarray(y, dtype=np.int8)
    s_arr = np.asarray(s, dtype=np.float64)
    step_arr = np.asarray(steps)

    assert not np.isin(y_arr, UNKNOWN).any(), f"y contains unknown labels ({UNKNOWN})"
    assert len(y_arr) == len(s_arr) == len(step_arr)

    unique_steps = np.sort(np.unique(step_arr))
    rows = []

    for step in unique_steps:
        mask = step_arr == step
        y_step = y_arr[mask]
        s_step = s_arr[mask]
        m = compute_metrics(y_step, s_step, threshold=threshold)
        rows.append({
            "step": int(step),
            "n": m["n"],
            "n_pos": m["n_pos"],
            "base_rate": m["base_rate"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "fpr": m["fpr"],
            "threshold": threshold,
        })

    return pd.DataFrame(rows)


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute empirical bootstrap 95% confidence interval (mean, lo, hi) for a metric."""
    y_arr = np.asarray(y, dtype=np.int8)
    s_arr = np.asarray(s, dtype=np.float64)
    assert not np.isin(y_arr, UNKNOWN).any()
    assert len(y_arr) == len(s_arr)

    rng = np.random.default_rng(seed)
    n_samples = len(y_arr)
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")

    estimates = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        estimates[i] = metric_fn(y_arr[idx], s_arr[idx])

    valid = estimates[~np.isnan(estimates)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(valid))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    return mean, lo, hi


def paired_bootstrap_diff(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute empirical bootstrap 95% confidence interval for metric(s_a) - metric(s_b)."""
    y_arr = np.asarray(y, dtype=np.int8)
    sa_arr = np.asarray(s_a, dtype=np.float64)
    sb_arr = np.asarray(s_b, dtype=np.float64)

    assert not np.isin(y_arr, UNKNOWN).any()
    assert len(y_arr) == len(sa_arr) == len(sb_arr)

    rng = np.random.default_rng(seed)
    n_samples = len(y_arr)
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")

    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        m_a = metric_fn(y_arr[idx], sa_arr[idx])
        m_b = metric_fn(y_arr[idx], sb_arr[idx])
        diffs[i] = m_a - m_b

    valid = diffs[~np.isnan(diffs)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(valid))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    return mean, lo, hi


def paired_block_bootstrap_diff(
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute empirical block-bootstrap 95% confidence interval for metric(s_a) - metric(s_b).

    Resamples whole groups (time steps) with replacement to account for intra-step
    temporal clustering and correlation. Skips resamples containing no positive instances.
    """
    y_arr = np.asarray(y, dtype=np.int8)
    sa_arr = np.asarray(s_a, dtype=np.float64)
    sb_arr = np.asarray(s_b, dtype=np.float64)
    grp_arr = np.asarray(groups)

    assert not np.isin(y_arr, UNKNOWN).any(), f"y contains unknown labels ({UNKNOWN})"
    assert len(y_arr) == len(sa_arr) == len(sb_arr) == len(grp_arr)

    unique_groups = np.unique(grp_arr)
    n_groups = len(unique_groups)
    if n_groups == 0 or len(y_arr) == 0:
        return float("nan"), float("nan"), float("nan")

    group_indices = {g: np.where(grp_arr == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    diffs: list[float] = []

    max_attempts = n * 5
    attempts = 0
    while len(diffs) < n and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_list = [group_indices[g] for g in sampled_groups]
        idx = np.concatenate(idx_list)

        y_boot = y_arr[idx]
        if (y_boot == 1).sum() == 0 or (y_boot == 0).sum() == 0:
            continue

        try:
            m_a = metric_fn(y_boot, sa_arr[idx])
            m_b = metric_fn(y_boot, sb_arr[idx])
            if not (np.isnan(m_a) or np.isnan(m_b)):
                diffs.append(float(m_a - m_b))
        except Exception:
            continue

    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")

    diffs_arr = np.array(diffs, dtype=np.float64)
    mean = float(np.mean(diffs_arr))
    lo = float(np.percentile(diffs_arr, 2.5))
    hi = float(np.percentile(diffs_arr, 97.5))
    return mean, lo, hi



