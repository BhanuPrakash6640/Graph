"""CLI and training pipeline for static GNN models (GCN, GraphSAGE, GAT, and MLP control).

Implements hyperparameter tuning on val PR-AUC, decision threshold tuning on val F1,
leak-free test evaluation, paper_refit retraining, checkpointing, curves logging,
and full run recording.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.data.loader import UNKNOWN
from ml.evaluation.latency import measure_latency
from ml.evaluation.metrics import best_f1_threshold, compute_metrics, per_step_metrics
from ml.evaluation.tracking import RunRecorder
from ml.gnn.graph import build_period_graph
from ml.gnn.models import create_gnn_model
from ml.preprocessing.dataset import ProcessedData, SplitView, load_processed, prepare_split
from ml.utils.config import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.gnn.train")


def tune_gnn(
    model_name: str,
    train_graph,
    val_graph,
    hpo_config: dict[str, Any],
    weight_decay: float = 5e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
    max_tune_epochs: int = 200,
    tune_patience: int = 40,
) -> tuple[dict[str, Any], int]:
    """Perform grid search on validation PR-AUC using seed 0."""
    hidden_options = hpo_config.get("hidden_channels", [64, 128])
    if model_name.lower() == "gat":
        # 4 heads concat yields 4x channels (256 dims); HPO on paper chose 64
        hidden_options = [64]
    dropout_options = hpo_config.get("dropout", [0.3, 0.5])
    lr_options = hpo_config.get("lr", [1e-3, 5e-3])
    gat_heads = hpo_config.get("gat_heads", 4)

    best_score = -1.0
    best_hp = {"hidden_channels": hidden_options[0], "dropout": dropout_options[0], "lr": lr_options[0]}
    best_epoch = 50

    val_y = val_graph.y[val_graph.val_mask].cpu().numpy()

    # Pre-calculate class weights on train_graph
    y_train = train_graph.y[train_graph.train_mask]
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    total = n_pos + n_neg
    w_pos = float(total / (2.0 * n_pos)) if n_pos > 0 else 1.0
    w_neg = float(total / (2.0 * n_neg)) if n_neg > 0 else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    in_channels = train_graph.x.shape[1]

    for hc in hidden_options:
        for dp in dropout_options:
            for lr in lr_options:
                set_seed(seed)
                model = create_gnn_model(
                    model_name=model_name,
                    in_channels=in_channels,
                    hidden_channels=hc,
                    dropout=dp,
                    heads=gat_heads,
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                cand_best_score = -1.0
                cand_best_epoch = 0
                patience_counter = 0

                for epoch in range(max_tune_epochs):
                    model.train()
                    optimizer.zero_grad()
                    out = model(train_graph.x, train_graph.edge_index)
                    loss = criterion(out[train_graph.train_mask], train_graph.y[train_graph.train_mask])
                    loss.backward()
                    optimizer.step()

                    # Eval on val
                    model.eval()
                    with torch.no_grad():
                        val_out = model(val_graph.x, val_graph.edge_index)
                        val_probs = F.softmax(val_out, dim=-1)[:, 1][val_graph.val_mask].cpu().numpy()
                        val_score = float(average_precision_score(val_y, val_probs))

                    if val_score > cand_best_score:
                        cand_best_score = val_score
                        cand_best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= tune_patience:
                            break

                cur_hp = {"hidden_channels": hc, "dropout": dp, "lr": lr}
                if model_name.lower() == "gat":
                    cur_hp["heads"] = gat_heads

                if cand_best_score > best_score:
                    best_score = cand_best_score
                    best_hp = cur_hp
                    best_epoch = cand_best_epoch

                # Free VRAM between HPO candidates
                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("Best %s HPs: %s (val PR-AUC: %.4f, best_epoch: %d)", model_name, best_hp, best_score, best_epoch)
    return best_hp, best_epoch


def run_single_gnn(
    model_name: str,
    split_name: str,
    feature_set: str,
    seed: int,
    data: ProcessedData,
    experiment: str,
    best_hps: dict[str, Any],
    val_threshold: float | None = None,
    target_epochs: int | None = None,
    device: str | torch.device = "cuda",
    results_dir: str | Path = "experiments/results",
    checkpoints_dir: str | Path = "checkpoints",
    latency_repeats: int = 20,
    max_epochs: int = 1000,
    patience: int = 100,
    weight_decay: float = 5e-4,
) -> tuple[dict[str, Any], float, int]:
    """Train, evaluate, and record a single GNN model run."""
    view: SplitView = prepare_split(data, split_name, feature_set)
    set_seed(seed)

    dev = torch.device(device)

    # 1. Build leak-free period subgraphs
    train_graph = build_period_graph(view, view.split.train_steps).to(dev)
    has_val = bool(view.val_mask.any())
    val_graph = build_period_graph(view, view.split.val_steps).to(dev) if has_val else None
    test_graph = build_period_graph(view, view.split.test_steps).to(dev)

    # 2. Setup loss and class weights
    y_train = train_graph.y[train_graph.train_mask]
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    total = n_pos + n_neg
    w_pos = float(total / (2.0 * n_pos)) if n_pos > 0 else 1.0
    w_neg = float(total / (2.0 * n_neg)) if n_neg > 0 else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model & optimizer
    in_channels = train_graph.x.shape[1]
    hc = best_hps.get("hidden_channels", 64)
    dp = best_hps.get("dropout", 0.3)
    lr = best_hps.get("lr", 1e-3)
    heads = best_hps.get("heads", 4)

    model = create_gnn_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=hc,
        dropout=dp,
        heads=heads,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training loop with timing
    curves: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0

    if has_val:
        assert val_graph is not None
        val_y = val_graph.y[val_graph.val_mask].cpu().numpy()
        best_val_score = -1.0
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_graph.x, train_graph.edge_index)
            loss = criterion(out[train_graph.train_mask], train_graph.y[train_graph.train_mask])
            loss.backward()
            optimizer.step()

            # Eval on val
            model.eval()
            with torch.no_grad():
                val_out = model(val_graph.x, val_graph.edge_index)
                val_probs = F.softmax(val_out, dim=-1)[:, 1][val_graph.val_mask].cpu().numpy()
                val_score = float(average_precision_score(val_y, val_probs))

            curves.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_pr_auc": val_score,
            })

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_weights)
    else:
        # Refit mode without validation set: train for target_epochs
        epochs_to_run = (target_epochs + 1) if target_epochs is not None else 150
        for epoch in range(epochs_to_run):
            model.train()
            optimizer.zero_grad()
            out = model(train_graph.x, train_graph.edge_index)
            loss = criterion(out[train_graph.train_mask], train_graph.y[train_graph.train_mask])
            loss.backward()
            optimizer.step()

            curves.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_pr_auc": None,
            })
        best_epoch = epochs_to_run - 1

    train_time_s = time.perf_counter() - t0

    # 5. Validation metrics and threshold selection
    model.eval()
    m_val_opt: dict[str, Any] = {}
    m_val_05: dict[str, Any] = {}
    s_val = np.empty(0, dtype=np.float32)
    threshold = val_threshold

    if has_val:
        assert val_graph is not None
        with torch.no_grad():
            val_out = model(val_graph.x, val_graph.edge_index)
            val_probs = F.softmax(val_out, dim=-1)[:, 1]
            s_val = val_probs[val_graph.val_mask].cpu().numpy().astype(np.float32)
            val_y = val_graph.y[val_graph.val_mask].cpu().numpy()
            if threshold is None:
                threshold = best_f1_threshold(val_y, s_val)
            m_val_opt = compute_metrics(val_y, s_val, threshold=threshold)
            m_val_05 = compute_metrics(val_y, s_val, threshold=0.5)
    elif threshold is None:
        threshold = 0.5

    # 6. Test evaluation & Latency profiling
    def predict_fn(x, _m=model, _edge_index=test_graph.edge_index):
        return _m(x, _edge_index)

    latency = measure_latency(predict_fn, test_graph.x, repeats=latency_repeats)

    att_weights = None
    with torch.no_grad():
        if model_name.lower() == "gat":
            test_out, att_weights = model(test_graph.x, test_graph.edge_index, return_attention_weights=True)
        else:
            test_out = model(test_graph.x, test_graph.edge_index)

        probs = F.softmax(test_out, dim=-1)[:, 1]
        s_test = probs[test_graph.test_mask].cpu().numpy().astype(np.float32)
        y_test = test_graph.y[test_graph.test_mask].cpu().numpy()

        m_test_opt = compute_metrics(y_test, s_test, threshold=threshold)
        m_test_05 = compute_metrics(y_test, s_test, threshold=0.5)

        test_steps_arr = test_graph.time_step[test_graph.test_mask].cpu().numpy()
        per_step_df = per_step_metrics(y_test, s_test, test_steps_arr, threshold=threshold)

        # Unlabelled test predictions
        unlab_mask = (test_graph.y == UNKNOWN)
        s_unlab = np.empty(0, dtype=np.float32)
        if unlab_mask.any():
            s_unlab = probs[unlab_mask].cpu().numpy().astype(np.float32)

    # 7. Construct predictions dataframe
    pred_parts = []
    if has_val and len(s_val) > 0:
        assert val_graph is not None
        val_orig_idx = val_graph.orig_idx[val_graph.val_mask].cpu().numpy()
        pred_parts.append(pd.DataFrame({
            "node_idx": val_orig_idx.astype(np.int64),
            "txId": data.tx_ids[val_orig_idx],
            "time_step": val_graph.time_step[val_graph.val_mask].cpu().numpy(),
            "y": val_graph.y[val_graph.val_mask].cpu().numpy(),
            "score": s_val,
            "part": "val",
        }))

    test_orig_idx = test_graph.orig_idx[test_graph.test_mask].cpu().numpy()
    pred_parts.append(pd.DataFrame({
        "node_idx": test_orig_idx.astype(np.int64),
        "txId": data.tx_ids[test_orig_idx],
        "time_step": test_graph.time_step[test_graph.test_mask].cpu().numpy(),
        "y": y_test,
        "score": s_test,
        "part": "test",
    }))

    if len(s_unlab) > 0:
        unlab_orig_idx = test_graph.orig_idx[unlab_mask].cpu().numpy()
        pred_parts.append(pd.DataFrame({
            "node_idx": unlab_orig_idx.astype(np.int64),
            "txId": data.tx_ids[unlab_orig_idx],
            "time_step": test_graph.time_step[unlab_mask].cpu().numpy(),
            "y": test_graph.y[unlab_mask].cpu().numpy(),
            "score": s_unlab,
            "part": "test_unlabelled",
        }))

    predictions_df = pd.concat(pred_parts, ignore_index=True)

    # 8. Record run with RunRecorder
    run_cfg = {
        "model": model_name,
        "split": split_name,
        "feature_set": feature_set,
        "seed": seed,
        "best_hps": best_hps,
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "device": str(device),
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
            "best_epoch": int(best_epoch),
        })
        recorder.log_timing({
            "train_time_s": float(train_time_s),
            "inference": latency,
        })
        recorder.log_predictions(predictions_df)

        # Log training curves
        curves_df = pd.DataFrame(curves)
        curves_df.to_csv(recorder.run_dir / "curves.csv", index=False)

        # Save GAT attention weights if available
        if att_weights is not None:
            torch.save(att_weights, recorder.run_dir / "attention.pt")

    # 9. Save checkpoint
    chk_dir = Path(checkpoints_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = chk_dir / f"{recorder.run_dir.name}.pt"
    torch.save({
        "model_class": model.__class__.__name__,
        "model_name": model_name,
        "model_kwargs": {
            "in_channels": in_channels,
            "hidden_channels": hc,
            "dropout": dp,
            "heads": heads,
        },
        "feature_set": feature_set,
        "state_dict": model.state_dict(),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "best_hps": best_hps,
    }, checkpoint_path)

    logger.info(
        "Finished %s on %s (%s, s%d): test PR-AUC=%.4f, F1=%.4f (thr=%.3f, epoch=%d)",
        model_name, split_name, feature_set, seed,
        m_test_opt["pr_auc"], m_test_opt["f1"], threshold, best_epoch,
    )

    del model, optimizer, train_graph, test_graph
    if val_graph is not None:
        del val_graph
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return m_test_opt, float(threshold), int(best_epoch)


def check_single_run_finished(
    results_dir: Path,
    experiment: str,
    model_name: str,
    split_name: str,
    featset: str,
    seed: int,
) -> bool:
    """Check if a specific seed run has already finished successfully."""
    exp_dir = results_dir / experiment
    if not exp_dir.exists():
        return False
    target_suffix = f"_{model_name}_{split_name}_{featset}_s{seed}"
    for d in exp_dir.iterdir():
        if d.is_dir() and (target_suffix in d.name or d.name.endswith(target_suffix)) and (d / "metrics.json").exists():
            return True
    return False


def check_runs_finished(
    results_dir: Path,
    experiment: str,
    model_name: str,
    split_name: str,
    featset: str,
    seeds: list[int],
) -> bool:
    """Check if all seeds for a given configuration have already finished successfully."""
    exp_dir = results_dir / experiment
    if not exp_dir.exists():
        return False
    for seed in seeds:
        target_suffix = f"_{model_name}_{split_name}_{featset}_s{seed}"
        found = False
        for d in exp_dir.iterdir():
            if d.is_dir() and (target_suffix in d.name or d.name.endswith(target_suffix)) and (d / "metrics.json").exists():
                found = True
                break
        if not found:
            return False
    return True


def load_gnn_paper_seed_cache(
    results_dir: Path,
    experiment: str,
    model_name: str,
    featset: str,
    seed: int,
) -> dict[str, Any] | None:
    """Load HPs, threshold, and best_epoch from a finished paper split run for seed-paired refit."""
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
                        "best_epoch": metrics.get("best_epoch"),
                    }
                except Exception:
                    pass
    return None


def run_gnn_experiments(
    config_path: str | Path,
    override_models: list[str] | None = None,
    override_splits: list[str] | None = None,
    override_refit_splits: list[str] | None = None,
    override_feature_sets: list[str] | None = None,
    override_device: str | None = None,
) -> None:
    """Run full GNN experiment suite defined in config."""
    cfg = load_config(config_path)
    experiment = cfg.get("experiment", "exp1_gnn")
    processed_dir = cfg["data"]["processed_dir"]
    results_dir = Path("experiments/results")
    checkpoints_dir = Path("checkpoints")

    models = override_models or cfg["models"]
    splits = override_splits if override_splits is not None else cfg.get("splits", [])
    refit_splits = override_refit_splits if override_refit_splits is not None else cfg.get("refit_splits", [])
    feat_sets = override_feature_sets or cfg.get("feature_sets", ["local", "all"])
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    hpo_cfg = cfg.get("hpo", {})
    training_cfg = cfg.get("training", {})

    device_str = override_device or training_cfg.get("device", "cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    max_epochs = training_cfg.get("max_epochs", 1000)
    patience = training_cfg.get("early_stopping_patience", 100)
    weight_decay = training_cfg.get("weight_decay", 5e-4)

    logger.info("Loading processed data from %s...", processed_dir)
    data = load_processed(processed_dir)
    logger.info("Data loaded. Total nodes: %d, edges: %d. Device: %s", len(data.tx_ids), data.edge_index.shape[1], device_str)

    paper_per_seed: dict[tuple[str, str, int], dict[str, Any]] = {}

    for model_name in models:
        for featset in feat_sets:
            # 1. Main splits (paper, fc_*)
            for split_name in splits:
                if check_runs_finished(results_dir, experiment, model_name, split_name, featset, seeds):
                    logger.info("Skipping already finished split %s | %s | %s", model_name, split_name, featset)
                    if split_name == "paper":
                        for seed in seeds:
                            cached = load_gnn_paper_seed_cache(results_dir, experiment, model_name, featset, seed)
                            if cached:
                                paper_per_seed[(model_name, featset, seed)] = {
                                    "best_hps": cached["best_hps"],
                                    "threshold": cached["threshold"],
                                    "best_epoch": cached["best_epoch"],
                                }
                    continue

                logger.info("=== Running %s | %s | %s on %s ===", model_name, split_name, featset, device_str)
                hpo_view = prepare_split(data, split_name, featset)
                train_graph = build_period_graph(hpo_view, hpo_view.split.train_steps).to(device)
                has_val = bool(hpo_view.val_mask.any())
                val_graph = build_period_graph(hpo_view, hpo_view.split.val_steps).to(device) if has_val else None

                best_epoch = 100
                if has_val:
                    best_hps, best_epoch = tune_gnn(
                        model_name=model_name,
                        train_graph=train_graph,
                        val_graph=val_graph,
                        hpo_config=hpo_cfg,
                        weight_decay=weight_decay,
                        device=device,
                        seed=0,
                    )
                else:
                    best_hps = {"hidden_channels": 64, "dropout": 0.3, "lr": 1e-3}

                # Run seeds 0-4
                for seed in seeds:
                    if check_single_run_finished(results_dir, experiment, model_name, split_name, featset, seed):
                        logger.info("Skipping already finished seed %s | %s | %s | s%d", model_name, split_name, featset, seed)
                        if split_name == "paper":
                            cached = load_gnn_paper_seed_cache(results_dir, experiment, model_name, featset, seed)
                            if cached:
                                paper_per_seed[(model_name, featset, seed)] = {
                                    "best_hps": cached["best_hps"],
                                    "threshold": cached["threshold"],
                                    "best_epoch": cached["best_epoch"],
                                }
                        continue

                    m_test, thr, epoch_ran = run_single_gnn(
                        model_name=model_name,
                        split_name=split_name,
                        feature_set=featset,
                        seed=seed,
                        data=data,
                        experiment=experiment,
                        best_hps=best_hps,
                        val_threshold=None,
                        device=device,
                        results_dir=results_dir,
                        checkpoints_dir=checkpoints_dir,
                        max_epochs=max_epochs,
                        patience=patience,
                        weight_decay=weight_decay,
                    )

                    if split_name == "paper":
                        paper_per_seed[(model_name, featset, seed)] = {
                            "best_hps": best_hps,
                            "threshold": thr,
                            "best_epoch": epoch_ran,
                        }

            # 2. paper_refit runs (seed-paired from paper runs)
            if "paper_refit" in refit_splits:
                for seed in seeds:
                    if check_single_run_finished(results_dir, experiment, model_name, "paper_refit", featset, seed):
                        logger.info("Skipping already finished refit %s | paper_refit | %s | s%d", model_name, featset, seed)
                        continue

                    seed_data = paper_per_seed.get((model_name, featset, seed)) or load_gnn_paper_seed_cache(
                        results_dir, experiment, model_name, featset, seed
                    )
                    if seed_data is None:
                        logger.warning(
                            "No paper run found for %s | %s | s%d; cannot pair refit",
                            model_name, featset, seed,
                        )
                        continue
                    logger.info(
                        "=== Retraining paper_refit for %s | %s | s%d using paper HPs: %s, thr: %.4f, epoch: %d ===",
                        model_name, featset, seed,
                        seed_data["best_hps"],
                        seed_data["threshold"],
                        seed_data["best_epoch"],
                    )
                    run_single_gnn(
                        model_name=model_name,
                        split_name="paper_refit",
                        feature_set=featset,
                        seed=seed,
                        data=data,
                        experiment=experiment,
                        best_hps=seed_data["best_hps"],
                        val_threshold=seed_data["threshold"],
                        target_epochs=seed_data["best_epoch"],
                        device=device,
                        results_dir=results_dir,
                        checkpoints_dir=checkpoints_dir,
                        max_epochs=max_epochs,
                        patience=patience,
                        weight_decay=weight_decay,
                    )


def main():
    parser = argparse.ArgumentParser(description="Train GNN models and MLP control")
    parser.add_argument("--config", default="experiments/configs/gnn.yaml", help="Path to config file")
    parser.add_argument("--models", help="Comma-separated models (e.g. mlp,gcn,sage_mean,sage_max,gat)")
    parser.add_argument("--splits", help="Comma-separated splits (e.g. paper,fc_24) or 'none'")
    parser.add_argument("--refit-splits", help="Comma-separated refit splits or 'none'")
    parser.add_argument("--feature-sets", help="Comma-separated feature sets (e.g. local,all)")
    parser.add_argument("--device", help="Torch device override (e.g. cuda, cuda:0, cpu)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    splits = [] if args.splits == "none" else ([s.strip() for s in args.splits.split(",")] if args.splits else None)
    refit_splits = [] if args.refit_splits == "none" else ([s.strip() for s in args.refit_splits.split(",")] if args.refit_splits else None)
    feat_sets = [f.strip() for f in args.feature_sets.split(",")] if args.feature_sets else None

    run_gnn_experiments(
        config_path=args.config,
        override_models=models,
        override_splits=splits,
        override_refit_splits=refit_splits,
        override_feature_sets=feat_sets,
        override_device=args.device,
    )


if __name__ == "__main__":
    main()

