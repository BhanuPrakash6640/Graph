"""CLI and training pipeline for tabular baselines (LR, RF, XGBoost).

Implements hyperparameter tuning on val PR-AUC, decision threshold tuning on val F1,
leak-free test evaluation, paper_refit retraining, and full run recording.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.baselines.models import (
    LR_HP_GRID,
    RF_HP_GRID,
    XGB_HP_GRID,
    create_lr,
    create_rf,
    create_xgb,
)
from ml.data.loader import UNKNOWN
from ml.evaluation.latency import measure_latency
from ml.evaluation.metrics import best_f1_threshold, compute_metrics, per_step_metrics
from ml.evaluation.tracking import RunRecorder
from ml.preprocessing.dataset import ProcessedData, SplitView, load_processed, prepare_split
from ml.utils.config import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.baselines.train")


def tune_lr(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    best_score = -1.0
    best_hp = LR_HP_GRID[0]
    for hp in LR_HP_GRID:
        clf = create_lr(C=hp["C"], random_state=0)
        clf.fit(X_train, y_train)
        s_val = clf.predict_proba(X_val)[:, 1]
        score = float(average_precision_score(y_val, s_val))
        if score > best_score:
            best_score = score
            best_hp = hp
    logger.info("Best LR HPs: %s (val PR-AUC: %.4f)", best_hp, best_score)
    return best_hp


def tune_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    best_score = -1.0
    best_hp = RF_HP_GRID[0]
    for hp in RF_HP_GRID:
        clf = create_rf(
            max_depth=hp["max_depth"],
            max_features=hp["max_features"],
            random_state=seed,
            n_estimators=100,  # Fast proxy for tuning
        )
        clf.fit(X_train, y_train)
        s_val = clf.predict_proba(X_val)[:, 1]
        score = float(average_precision_score(y_val, s_val))
        if score > best_score:
            best_score = score
            best_hp = hp
    logger.info("Best RF HPs: %s (val PR-AUC: %.4f)", best_hp, best_score)
    return best_hp


def tune_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    seed: int = 0,
) -> tuple[dict[str, Any], int]:
    best_score = -1.0
    best_hp = XGB_HP_GRID[0]
    best_iter = 100
    for hp in XGB_HP_GRID:
        clf = create_xgb(
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_estimators=2000,
            early_stopping_rounds=50,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        s_val = clf.predict_proba(X_val)[:, 1]
        score = float(average_precision_score(y_val, s_val))
        if score > best_score:
            best_score = score
            best_hp = hp
            best_iter = int(clf.best_iteration) if hasattr(clf, "best_iteration") else 200
    logger.info("Best XGB HPs: %s, best_iteration: %d (val PR-AUC: %.4f)", best_hp, best_iter, best_score)
    return best_hp, best_iter


def run_single_model(
    model_name: str,
    split_name: str,
    feature_set: str,
    seed: int,
    data: ProcessedData,
    experiment: str,
    best_hps: dict[str, Any],
    val_threshold: float | None = None,
    xgb_best_iter: int | None = None,
    results_dir: str | Path = "experiments/results",
    latency_repeats: int = 20,
) -> tuple[dict[str, Any], float, int | None]:
    """Train a single model configuration, evaluate, and record."""
    view: SplitView = prepare_split(data, split_name, feature_set)
    set_seed(seed)

    X_train = view.X[view.train_mask]
    y_train = view.y[view.train_mask]
    has_val = bool(view.val_mask.any())

    X_val = view.X[view.val_mask] if has_val else np.empty((0, view.X.shape[1]), dtype=np.float32)
    y_val = view.y[view.val_mask] if has_val else np.empty(0, dtype=np.int8)

    X_test = view.X[view.test_mask]
    y_test = view.y[view.test_mask]

    test_steps = view.split.test_steps
    unlabelled_test_mask = np.isin(view.time_step, list(test_steps)) & (view.y == UNKNOWN)
    X_test_unlab = view.X[unlabelled_test_mask]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # Build model instance
    best_iteration = xgb_best_iter
    if model_name == "lr":
        model = create_lr(C=best_hps.get("C", 1.0), random_state=seed)
    elif model_name == "rf":
        model = create_rf(
            max_depth=best_hps.get("max_depth"),
            max_features=best_hps.get("max_features", "sqrt"),
            random_state=seed,
            n_estimators=500,
        )
    elif model_name == "xgb":
        if has_val:
            model = create_xgb(
                max_depth=best_hps.get("max_depth", 6),
                learning_rate=best_hps.get("learning_rate", 0.1),
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_estimators=2000,
                early_stopping_rounds=50,
            )
        else:
            # For refit splits (no val set), fix n_estimators to best_iteration + 1
            n_est = (best_iteration + 1) if best_iteration is not None else 300
            model = create_xgb(
                max_depth=best_hps.get("max_depth", 6),
                learning_rate=best_hps.get("learning_rate", 0.1),
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                n_estimators=n_est,
                early_stopping_rounds=None,
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Train timing
    t0 = time.perf_counter()
    if model_name == "xgb" and has_val:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iteration = int(model.best_iteration) if hasattr(model, "best_iteration") else 200
    else:
        model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    # Validation predictions and threshold selection
    m_val_opt = {}
    m_val_05 = {}
    s_val = np.empty(0, dtype=np.float32)
    threshold = val_threshold
    if has_val:
        s_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
        if threshold is None:
            threshold = best_f1_threshold(y_val, s_val)
        m_val_opt = compute_metrics(y_val, s_val, threshold=threshold)
        m_val_05 = compute_metrics(y_val, s_val, threshold=0.5)
    elif threshold is None:
        threshold = 0.5

    # Test predictions & latency
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    latency = measure_latency(predict_fn, X_test, repeats=latency_repeats)

    s_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    m_test_opt = compute_metrics(y_test, s_test, threshold=threshold)
    m_test_05 = compute_metrics(y_test, s_test, threshold=0.5)

    test_steps_arr = view.time_step[view.test_mask]
    per_step_df = per_step_metrics(y_test, s_test, test_steps_arr, threshold=threshold)

    # Predictions for unlabelled test nodes
    s_test_unlab = np.empty(0, dtype=np.float32)
    if len(X_test_unlab) > 0:
        s_test_unlab = model.predict_proba(X_test_unlab)[:, 1].astype(np.float32)

    # Build predictions dataframe
    pred_parts = []
    if has_val and len(s_val) > 0:
        val_idx = np.where(view.val_mask)[0]
        pred_parts.append(pd.DataFrame({
            "node_idx": val_idx.astype(np.int64),
            "txId": data.tx_ids[val_idx],
            "time_step": view.time_step[val_idx],
            "y": view.y[val_idx],
            "score": s_val,
            "part": "val",
        }))

    test_idx = np.where(view.test_mask)[0]
    pred_parts.append(pd.DataFrame({
        "node_idx": test_idx.astype(np.int64),
        "txId": data.tx_ids[test_idx],
        "time_step": view.time_step[test_idx],
        "y": view.y[test_idx],
        "score": s_test,
        "part": "test",
    }))

    if len(s_test_unlab) > 0:
        unlab_idx = np.where(unlabelled_test_mask)[0]
        pred_parts.append(pd.DataFrame({
            "node_idx": unlab_idx.astype(np.int64),
            "txId": data.tx_ids[unlab_idx],
            "time_step": view.time_step[unlab_idx],
            "y": view.y[unlab_idx],
            "score": s_test_unlab,
            "part": "test_unlabelled",
        }))

    predictions_df = pd.concat(pred_parts, ignore_index=True)

    # Save via RunRecorder
    run_cfg = {
        "model": model_name,
        "split": split_name,
        "feature_set": feature_set,
        "seed": seed,
        "best_hps": best_hps,
        "threshold": float(threshold),
        "seed_invariant": bool(model_name == "lr"),
    }

    with RunRecorder(experiment=experiment, config=run_cfg, base_dir=results_dir) as recorder:
        recorder.log_metrics({
            "val_at_opt": m_val_opt,
            "val_at_05": m_val_05,
            "test_at_opt": m_test_opt,
            "test_at_05": m_test_05,
            "per_step_test": per_step_df.to_dict(orient="records"),
            "best_threshold": float(threshold),
            "best_hps": best_hps,
            "best_iteration": best_iteration,
            "seed_invariant": bool(model_name == "lr"),
        })
        recorder.log_timing({
            "train_time_s": float(train_time_s),
            "inference": latency,
        })
        recorder.log_predictions(predictions_df)

    logger.info(
        "Finished %s on %s (%s, s%d): test PR-AUC=%.4f, F1=%.4f (thr=%.3f)",
        model_name, split_name, feature_set, seed,
        m_test_opt["pr_auc"], m_test_opt["f1"], threshold,
    )
    return m_test_opt, float(threshold), best_iteration


def load_paper_seed_cache(
    results_dir: Path,
    experiment: str,
    model_name: str,
    featset: str,
    seed: int,
) -> dict[str, Any] | None:
    exp_dir = results_dir / experiment
    if not exp_dir.exists():
        return None
    target_suffix = f"_{model_name}_paper_{featset}_s{seed}"
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.endswith(target_suffix) or target_suffix in run_dir.name:
            if "paper_refit" in run_dir.name:
                continue
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    return {
                        "best_hps": metrics.get("best_hps", {}),
                        "threshold": metrics.get("best_threshold"),
                        "best_iteration": metrics.get("best_iteration"),
                    }
                except Exception:
                    pass
    return None


def run_baselines(
    config_path: str | Path,
    override_models: list[str] | None = None,
    override_splits: list[str] | None = None,
    override_refit_splits: list[str] | None = None,
    override_feat_sets: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    experiment = config.get("experiment", "exp1_baselines")
    proc_dir = Path(config["data"]["processed_dir"])
    data = load_processed(proc_dir)

    models = override_models or config.get("models", ["lr", "rf", "xgb"])
    feat_sets = override_feat_sets or config.get("feature_sets", ["local", "all"])
    splits = override_splits if override_splits is not None else config.get("splits", ["paper", "fc_24", "fc_29", "fc_34", "fc_39", "fc_44"])
    refit_splits = override_refit_splits if override_refit_splits is not None else config.get("refit_splits", ["paper_refit"])
    seeds_cfg = config.get("seeds", {"rf": [0, 1, 2, 3, 4], "xgb": [0, 1, 2, 3, 4], "lr": [0]})
    latency_repeats = config.get("latency_repeats", 20)
    results_dir = Path(config.get("results_dir", "experiments/results"))

    # Track chosen HPs, thresholds and best_iterations per seed from paper split for paper_refit
    paper_per_seed: dict[tuple[str, str, int], dict[str, Any]] = {}

    for model_name in models:
        seeds = seeds_cfg.get(model_name, [0])
        for featset in feat_sets:
            for split_name in splits:
                logger.info("=== Running %s | %s | %s ===", model_name, split_name, featset)
                # 1. HPO on val PR-AUC using split view
                hpo_view = prepare_split(data, split_name, featset)
                X_tr = hpo_view.X[hpo_view.train_mask]
                y_tr = hpo_view.y[hpo_view.train_mask]
                X_va = hpo_view.X[hpo_view.val_mask]
                y_va = hpo_view.y[hpo_view.val_mask]

                n_pos = int((y_tr == 1).sum())
                n_neg = int((y_tr == 0).sum())
                scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

                best_iter = None
                if model_name == "lr":
                    best_hps = tune_lr(X_tr, y_tr, X_va, y_va)
                elif model_name == "rf":
                    best_hps = tune_rf(X_tr, y_tr, X_va, y_va, seed=0)
                elif model_name == "xgb":
                    best_hps, best_iter = tune_xgb(X_tr, y_tr, X_va, y_va, scale_pos_weight, seed=0)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                # 2. Run across seeds
                last_best_iter = best_iter
                for seed in seeds:
                    m_test, thr, cur_iter = run_single_model(
                        model_name=model_name,
                        split_name=split_name,
                        feature_set=featset,
                        seed=seed,
                        data=data,
                        experiment=experiment,
                        best_hps=best_hps,
                        val_threshold=None,  # chosen from val per seed
                        xgb_best_iter=last_best_iter,
                        results_dir=results_dir,
                        latency_repeats=latency_repeats,
                    )
                    if cur_iter is not None:
                        last_best_iter = cur_iter

                    if split_name == "paper":
                        paper_per_seed[(model_name, featset, seed)] = {
                            "best_hps": best_hps,
                            "threshold": thr,
                            "best_iteration": cur_iter if cur_iter is not None else last_best_iter,
                        }

            # 3. paper_refit runs (retrain on 1-34 with seed-paired paper HPs, threshold and best_iteration)
            if "paper_refit" in refit_splits:
                for seed in seeds:
                    seed_data = paper_per_seed.get((model_name, featset, seed)) or load_paper_seed_cache(
                        results_dir, experiment, model_name, featset, seed
                    )
                    if seed_data is None:
                        logger.warning(
                            "No paper run found for %s | %s | s%d; cannot pair refit",
                            model_name, featset, seed,
                        )
                        continue
                    logger.info(
                        "=== Retraining paper_refit for %s | %s | s%d using paper HPs: %s, thr: %.4f, iter: %s ===",
                        model_name, featset, seed,
                        seed_data["best_hps"],
                        seed_data["threshold"],
                        str(seed_data["best_iteration"]),
                    )
                    run_single_model(
                        model_name=model_name,
                        split_name="paper_refit",
                        feature_set=featset,
                        seed=seed,
                        data=data,
                        experiment=experiment,
                        best_hps=seed_data["best_hps"],
                        val_threshold=seed_data["threshold"],
                        xgb_best_iter=seed_data["best_iteration"],
                        results_dir=results_dir,
                        latency_repeats=latency_repeats,
                    )


def main():
    parser = argparse.ArgumentParser(description="Train tabular baselines")
    parser.add_argument("--config", default="experiments/configs/baselines.yaml", help="Path to config file")
    parser.add_argument("--models", help="Comma-separated models to run (e.g. lr,rf,xgb)")
    parser.add_argument("--splits", help="Comma-separated splits to run (e.g. paper,fc_24)")
    parser.add_argument("--refit-splits", help="Comma-separated refit splits (or 'none')")
    parser.add_argument("--feature-sets", help="Comma-separated feature sets (e.g. local,all)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    splits = [] if args.splits == "none" else ([s.strip() for s in args.splits.split(",")] if args.splits else None)
    refit_splits = [] if args.refit_splits == "none" else ([s.strip() for s in args.refit_splits.split(",")] if args.refit_splits else None)
    feat_sets = [f.strip() for f in args.feature_sets.split(",")] if args.feature_sets else None

    run_baselines(
        args.config,
        override_models=models,
        override_splits=splits,
        override_refit_splits=refit_splits,
        override_feat_sets=feat_sets,
    )


if __name__ == "__main__":
    main()
