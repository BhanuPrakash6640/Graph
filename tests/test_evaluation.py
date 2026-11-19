"""Tests for evaluation metrics, threshold selection, latency, and run tracking."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from ml.evaluation.latency import measure_latency
from ml.evaluation.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    compute_metrics,
    paired_block_bootstrap_diff,
    paired_bootstrap_diff,
    per_step_metrics,
)
from ml.evaluation.tracking import RunRecorder, get_env_info


def test_compute_metrics_hand_crafted():
    # 6 samples: 3 licit (0), 3 illicit (1)
    # y_true = [0, 0, 0, 1, 1, 1]
    # scores = [0.1, 0.2, 0.8, 0.4, 0.7, 0.9]
    # At threshold = 0.5:
    # predictions: [0, 0, 1, 0, 1, 1]
    # TN = 2 (indices 0, 1), FP = 1 (index 2)
    # FN = 1 (index 3), TP = 2 (indices 4, 5)
    # precision = 2 / 3 = 0.6666667
    # recall = 2 / 3 = 0.6666667
    # f1 = 2/3 = 0.6666667
    # fpr = 1 / (1 + 2) = 1/3
    # base_rate = 3 / 6 = 0.5
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9], dtype=np.float64)

    m = compute_metrics(y_true, scores, threshold=0.5)

    assert m["n"] == 6
    assert m["n_pos"] == 3
    assert np.isclose(m["base_rate"], 0.5)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert np.isclose(m["precision"], 2.0 / 3.0)
    assert np.isclose(m["recall"], 2.0 / 3.0)
    assert np.isclose(m["f1"], 2.0 / 3.0)
    assert np.isclose(m["fpr"], 1.0 / 3.0)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_compute_metrics_rejects_unknown():
    y_true = np.array([0, 1, -1], dtype=np.int8)
    scores = np.array([0.2, 0.8, 0.5], dtype=np.float64)
    with pytest.raises(AssertionError, match="unknown labels"):
        compute_metrics(y_true, scores, threshold=0.5)


def test_threshold_selection_never_sees_test():
    """Verify that val threshold is tuned on val alone and not affected by test data."""
    # Val set has high scores for illicit: optimal threshold should be ~0.7
    y_val = np.array([0, 0, 1, 1], dtype=np.int8)
    s_val = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    # Test set has much lower scores: optimal threshold on test would be ~0.25
    y_test = np.array([0, 0, 1, 1], dtype=np.int8)
    s_test = np.array([0.05, 0.1, 0.3, 0.4], dtype=np.float64)

    thr_val = best_f1_threshold(y_val, s_val)
    thr_test = best_f1_threshold(y_test, s_test)

    # They must be different: val threshold must be >= 0.8, test threshold <= 0.3
    assert thr_val >= 0.7
    assert thr_test <= 0.4

    # When evaluating test with val threshold, it uses thr_val, not thr_test
    m_test_with_val_thr = compute_metrics(y_test, s_test, threshold=thr_val)
    assert m_test_with_val_thr["threshold"] == thr_val


def test_per_step_metrics():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    s = np.array([0.2, 0.8, 0.1, 0.9, 0.3, 0.7], dtype=np.float64)
    steps = np.array([1, 1, 2, 2, 3, 3], dtype=np.int16)

    df = per_step_metrics(y, s, steps, threshold=0.5)
    assert len(df) == 3
    assert list(df["step"]) == [1, 2, 3]
    assert list(df["n"]) == [2, 2, 2]
    assert list(df["n_pos"]) == [1, 1, 1]
    assert (df["f1"] == 1.0).all()


def test_bootstrap_ci_and_paired_diff():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    s_a = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.85], dtype=np.float64)
    s_b = np.array([0.4, 0.5, 0.6, 0.55, 0.45, 0.5, 0.4, 0.6], dtype=np.float64)

    from sklearn.metrics import average_precision_score
    mean, lo, hi = bootstrap_ci(y, s_a, average_precision_score, n=100, seed=42)
    assert lo <= mean <= hi

    diff_mean, diff_lo, diff_hi = paired_bootstrap_diff(
        y, s_a, s_b, average_precision_score, n=100, seed=42
    )
    assert diff_lo <= diff_mean <= diff_hi


def test_paired_block_bootstrap_diff():
    from sklearn.metrics import average_precision_score

    # 3 time steps, each with 10 nodes (2 positive, 8 negative)
    n_per_step = 10
    steps = np.repeat([35, 36, 37], n_per_step)
    y = np.tile([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 3)

    # Model A: inverted separation (poor PR-AUC)
    s_a = np.where(y == 1, 0.35, 0.65)
    # Model B: perfect separation (PR-AUC = 1.0)
    s_b = np.where(y == 1, 0.90, 0.10)

    mean, lo, hi = paired_block_bootstrap_diff(
        y=y,
        s_a=s_b,
        s_b=s_a,
        groups=steps,
        metric_fn=average_precision_score,
        n=200,
        seed=42,
    )

    # Diff is B - A > 0
    assert mean > 0
    assert lo > 0
    assert hi >= lo

    # Assert unknown rejected
    with pytest.raises(AssertionError):
        paired_block_bootstrap_diff(
            y=np.array([-1, 0, 1]),
            s_a=np.array([0.5, 0.5, 0.5]),
            s_b=np.array([0.5, 0.5, 0.5]),
            groups=np.array([1, 1, 1]),
            metric_fn=average_precision_score,
        )


def test_measure_latency():
    def dummy_predict(X):
        return np.ones(len(X))

    X = np.zeros((50, 10))
    lat = measure_latency(dummy_predict, X, repeats=10, warmup=2)
    assert "batch_p50_ms" in lat
    assert "batch_p95_ms" in lat
    assert "node_p50_ms" in lat
    assert "node_p95_ms" in lat
    assert lat["batch_size"] == 50
    assert lat["node_p50_ms"] <= lat["batch_p50_ms"]


def test_run_recorder_schema(tmp_path):
    config = {
        "model": "xgb",
        "split": "paper",
        "feature_set": "all",
        "seed": 0,
        "best_hps": {"max_depth": 4},
    }

    results_dir = tmp_path / "results"
    with RunRecorder("test_exp", config, base_dir=results_dir, timestamp="20261117-120000") as rec:
        rec.log_metrics({"test_at_opt": {"pr_auc": 0.85, "f1": 0.75}, "best_threshold": 0.45})
        rec.log_timing({"train_time_s": 1.25, "inference": {"batch_p50_ms": 5.0, "node_p50_ms": 0.05}})
        df_preds = pd.DataFrame({
            "node_idx": [0, 1],
            "txId": [1001, 1002],
            "time_step": [35, 36],
            "y": [1, 0],
            "score": [0.88, 0.12],
            "part": ["test", "test"],
        })
        rec.log_predictions(df_preds)

    run_dir = rec.output_dir
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "timing.json").exists()
    assert (run_dir / "predictions.parquet").exists()

    with open(run_dir / "env.json", "r", encoding="utf-8") as f:
        env = json.load(f)
        assert "git_hash" in env
        assert "python_version" in env
        assert "package_versions" in env
        assert "device" in env

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        met = json.load(f)
        assert "test_at_opt" in met
        assert met["best_threshold"] == 0.45

    with open(run_dir / "timing.json", "r", encoding="utf-8") as f:
        tim = json.load(f)
        assert "train_time_s" in tim
        assert "peak_memory_mb" in tim
        assert "rss_peak_mb" in tim

    preds_read = pd.read_parquet(run_dir / "predictions.parquet")
    expected_cols = ["node_idx", "txId", "time_step", "y", "score", "part"]
    assert list(preds_read.columns) == expected_cols

    # Verify no overwrite when creating a recorder with identical parameters
    with RunRecorder("test_exp", config, base_dir=results_dir, timestamp="20261117-120000") as rec2:
        pass
    assert rec2.output_dir != rec.output_dir
    assert rec2.output_dir.exists()


def test_compare_configurations_synthetic(tmp_path: Path):
    from ml.evaluation.compare import compare_configurations

    # Create synthetic run dirs for two models on a split
    results_dir = tmp_path / "exp_test"
    results_dir.mkdir(parents=True)

    n_nodes = 50
    y_test = (np.arange(n_nodes) % 5 == 0).astype(int)  # 10 positives, 40 negatives

    # Model A: overlapping scores (lower PR-AUC)
    scores_a = np.where(y_test == 1, 0.45, 0.55)
    # Model B: perfect separation (PR-AUC = 1.0)
    scores_b = np.where(y_test == 1, 0.9, 0.1)

    for model, scores in [("rf", scores_a), ("xgb", scores_b)]:
        run_d = results_dir / f"20261117-000000_{model}_paper_refit_all_s0"
        run_d.mkdir()
        with open(run_d / "config.yaml", "w") as f:
            yaml.safe_dump({"model": model, "split": "paper_refit", "feature_set": "all", "seed": 0}, f)
        df = pd.DataFrame({
            "node_idx": np.arange(n_nodes),
            "txId": [f"tx_{i}" for i in range(n_nodes)],
            "time_step": np.full(n_nodes, 35),
            "y": y_test,
            "score": scores,
            "part": "test",
        })
        df.to_parquet(run_d / "predictions.parquet")

    comparisons = [{
        "name": "RF vs XGB test",
        "split": "paper_refit",
        "cfg_a": {"model": "rf", "feature_set": "all"},
        "cfg_b": {"model": "xgb", "feature_set": "all"},
    }]

    res_df = compare_configurations(results_dir, comparisons, n_bootstraps=100, seed=42)
    assert len(res_df) == 1
    assert "Diff (B - A)" in res_df.columns
    assert "Block 95% CI (Primary)" in res_df.columns
    assert "Node 95% CI (Secondary)" in res_df.columns
    assert "B better on k/n steps" in res_df.columns
    assert float(res_df["PR-AUC (B)"].iloc[0]) > float(res_df["PR-AUC (A)"].iloc[0])


