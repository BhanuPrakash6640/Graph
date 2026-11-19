"""CLI and training pipeline for Temporal Graph Models (EvolveGCN-O evolving, EvolveGCN frozen, and Static GCN).

Implements:
- Chronological snapshot sequence modeling through the SAME pipeline.
- Full BPTT without artificial window detachment.
- Normalized epoch loss across snapshots.
- Identical HPO grid as static GCN ({64, 128} hidden, {0.3, 0.5} dropout, {1e-3, 5e-3} lr).
- HPO on paper validation PR-AUC only (seed 0).
- Seed-matched paper_refit evaluation.
- Full config recording: hidden_size, lr, dropout, epochs, code git hash.
- Automated assertion that frozen EvolveGCN lands within tolerance of static GCN.
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
from ml.evaluation.tracking import RunRecorder, get_env_info
from ml.gnn.graph import build_period_graph
from ml.gnn.models import GCN
from ml.preprocessing.dataset import ProcessedData, SplitView, load_processed, prepare_split
from ml.temporal.evolvegcn import EvolveGCNO
from ml.temporal.snapshot_context import CONTEXT_DIM, SnapshotContextGCN, SnapshotContextTracker
from ml.utils.config import load_config, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.temporal.train")


def build_snapshot_cache(view: SplitView, steps: list[int], device: torch.device) -> dict[int, Any]:
    """Pre-build and move per-step Data subgraphs to device."""
    cache = {}
    for s in steps:
        g = build_period_graph(view, [s], undirected=True).to(device)
        cache[s] = g
    return cache


def create_temporal_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    """Instantiate temporal or baseline model for the snapshot sequence pipeline."""
    name = model_name.lower()
    if name == "evolvegcn":
        return EvolveGCNO(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            dropout=dropout,
            evolve=True,
        ).to(device)
    elif name == "evolvegcn_frozen":
        return EvolveGCNO(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            dropout=dropout,
            evolve=False,
        ).to(device)
    elif name in ("gcn", "static_gcn"):
        return GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            dropout=dropout,
        ).to(device)
    else:
        return SnapshotContextGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            context_dim=CONTEXT_DIM,
            dropout=dropout,
        ).to(device)


def tune_temporal(
    model_name: str,
    view: SplitView,
    train_steps: list[int],
    val_steps: list[int],
    snapshot_cache: dict[int, Any],
    hpo_config: dict[str, Any],
    window_size: int = 5,
    weight_decay: float = 5e-4,
    device: str | torch.device = "cuda",
    seed: int = 0,
    max_tune_epochs: int = 200,
    tune_patience: int = 40,
) -> tuple[dict[str, Any], int]:
    """Perform grid search on validation PR-AUC using seed 0 on paper validation set."""
    hidden_options = hpo_config.get("hidden_channels", [64, 128])
    dropout_options = hpo_config.get("dropout", [0.3, 0.5])
    lr_options = hpo_config.get("lr", [1e-3, 5e-3])

    best_score = -1.0
    best_hp = {"hidden_channels": hidden_options[0], "dropout": dropout_options[0], "lr": lr_options[0]}
    best_epoch = 50

    # Compute class weights on train labelled nodes
    train_y_list = [snapshot_cache[s].y[snapshot_cache[s].train_mask] for s in train_steps]
    train_y_cat = torch.cat(train_y_list, dim=0) if train_y_list else torch.empty(0)
    n_pos = int((train_y_cat == 1).sum().item())
    n_neg = int((train_y_cat == 0).sum().item())
    total = n_pos + n_neg
    w_pos = float(total / (2.0 * n_pos)) if n_pos > 0 else 1.0
    w_neg = float(total / (2.0 * n_neg)) if n_neg > 0 else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    in_channels = view.X.shape[1]

    # Pre-extract validation ground-truth array
    val_y_all = []
    for s in val_steps:
        g = snapshot_cache[s]
        if g.val_mask.sum() > 0:
            val_y_all.append(g.y[g.val_mask].cpu().numpy())
    val_y_all = np.concatenate(val_y_all) if val_y_all else np.empty(0)

    for hc in hidden_options:
        for dp in dropout_options:
            for lr in lr_options:
                set_seed(seed)
                model = create_temporal_model(
                    model_name=model_name,
                    in_channels=in_channels,
                    hidden_channels=hc,
                    dropout=dp,
                    device=device,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                cand_best_score = -1.0
                cand_best_epoch = 0
                patience_counter = 0

                for epoch in range(max_tune_epochs):
                    train_loss = train_one_epoch(
                        model=model,
                        model_name=model_name,
                        train_steps=train_steps,
                        snapshot_cache=snapshot_cache,
                        criterion=criterion,
                        optimizer=optimizer,
                        window_size=window_size,
                        device=device,
                    )

                    # Eval on val steps
                    val_scores = evaluate_sequence(
                        model=model,
                        model_name=model_name,
                        train_steps=train_steps,
                        eval_steps=val_steps,
                        snapshot_cache=snapshot_cache,
                        mask_type="val_mask",
                        device=device,
                    )

                    if len(val_y_all) > 0 and len(val_scores) > 0:
                        pr_auc = float(average_precision_score(val_y_all, val_scores))
                    else:
                        pr_auc = 0.0

                    if pr_auc > cand_best_score:
                        cand_best_score = pr_auc
                        cand_best_epoch = epoch + 1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= tune_patience:
                            break

                if cand_best_score > best_score:
                    best_score = cand_best_score
                    best_hp = {"hidden_channels": hc, "dropout": dp, "lr": lr}
                    best_epoch = max(1, cand_best_epoch)

                del model, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("Tuned %s on paper val: best HPs=%s (best val PR-AUC=%.4f at epoch %d)", model_name, best_hp, best_score, best_epoch)
    return best_hp, best_epoch


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    train_steps: list[int],
    snapshot_cache: dict[int, Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    window_size: int,
    device: torch.device,
) -> float:
    """Run one chronological training epoch over training snapshots.
    
    All models accumulate snapshot cross-entropy losses and backpropagate normalized
    loss with gradient clipping.
    """
    model.train()
    name = model_name.lower()
    total_loss = 0.0
    num_updates = 0

    if name in ("evolvegcn", "evolvegcn_frozen"):
        optimizer.zero_grad()
        w1, w2 = model.get_initial_weights()
        epoch_loss = torch.tensor(0.0, device=device)
        has_loss = False
        step_count = 0

        for s in train_steps:
            w1, w2 = model.evolve_weights(w1, w2)
            g = snapshot_cache[s]
            out = model.forward_snapshot(g.x, g.edge_index, w1, w2)
            mask = g.train_mask
            if mask.sum() > 0:
                loss = criterion(out[mask], g.y[mask])
                epoch_loss = epoch_loss + loss
                has_loss = True
                step_count += 1

        if has_loss:
            epoch_loss = epoch_loss / max(1, step_count)
            epoch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss = epoch_loss.item()
            num_updates = step_count

    elif name in ("gcn", "static_gcn"):
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0, device=device)
        has_loss = False
        step_count = 0

        for s in train_steps:
            g = snapshot_cache[s]
            out = model(g.x, g.edge_index)
            mask = g.train_mask
            if mask.sum() > 0:
                loss = criterion(out[mask], g.y[mask])
                epoch_loss = epoch_loss + loss
                has_loss = True
                step_count += 1

        if has_loss:
            epoch_loss = epoch_loss / max(1, step_count)
            epoch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss = epoch_loss.item()
            num_updates = step_count

    else:
        # Control B: SnapshotContextGCN
        tracker = SnapshotContextTracker()
        for s in train_steps:
            g = snapshot_cache[s]
            ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
            ctx_t = torch.from_numpy(ctx).to(device)

            optimizer.zero_grad()
            out = model(g.x, g.edge_index, context=ctx_t)
            mask = g.train_mask
            if mask.sum() > 0:
                loss = criterion(out[mask], g.y[mask])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_updates += 1

            with torch.no_grad():
                probs = F.softmax(out, dim=-1)[:, 1]
                tracker.update(s, probs.mean().item())

    return total_loss / max(1, num_updates)


def evaluate_sequence(
    model: nn.Module,
    model_name: str,
    train_steps: list[int],
    eval_steps: list[int],
    snapshot_cache: dict[int, Any],
    mask_type: str = "val_mask",
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Evaluate snapshots sequentially in chronological order."""
    model.eval()
    name = model_name.lower()
    scores_list = []

    with torch.no_grad():
        if name in ("evolvegcn", "evolvegcn_frozen"):
            w1, w2 = model.get_initial_weights()
            for s in train_steps:
                w1, w2 = model.evolve_weights(w1, w2)

            for s in eval_steps:
                w1, w2 = model.evolve_weights(w1, w2)
                g = snapshot_cache[s]
                out = model.forward_snapshot(g.x, g.edge_index, w1, w2)
                probs = F.softmax(out, dim=-1)[:, 1]
                mask = getattr(g, mask_type)
                if mask.sum() > 0:
                    scores_list.append(probs[mask].cpu().numpy())

        elif name in ("gcn", "static_gcn"):
            for s in eval_steps:
                g = snapshot_cache[s]
                out = model(g.x, g.edge_index)
                probs = F.softmax(out, dim=-1)[:, 1]
                mask = getattr(g, mask_type)
                if mask.sum() > 0:
                    scores_list.append(probs[mask].cpu().numpy())

        else:
            # Control B
            tracker = SnapshotContextTracker()
            for s in train_steps:
                g = snapshot_cache[s]
                ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
                ctx_t = torch.from_numpy(ctx).to(device)
                out = model(g.x, g.edge_index, context=ctx_t)
                probs = F.softmax(out, dim=-1)[:, 1]
                tracker.update(s, probs.mean().item())

            for s in eval_steps:
                g = snapshot_cache[s]
                ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
                ctx_t = torch.from_numpy(ctx).to(device)
                out = model(g.x, g.edge_index, context=ctx_t)
                probs = F.softmax(out, dim=-1)[:, 1]
                mask = getattr(g, mask_type)
                if mask.sum() > 0:
                    scores_list.append(probs[mask].cpu().numpy())
                tracker.update(s, probs.mean().item())

    return np.concatenate(scores_list) if scores_list else np.empty(0)


def predict_test_snapshots(
    model: nn.Module,
    model_name: str,
    train_steps: list[int],
    val_steps: list[int],
    test_steps: list[int],
    snapshot_cache: dict[int, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test predictions for all nodes (labelled + unlabelled) in test snapshots."""
    model.eval()
    name = model_name.lower()
    all_node_idx = []
    all_steps = []
    all_y = []
    all_scores = []
    all_parts = []

    with torch.no_grad():
        if name in ("evolvegcn", "evolvegcn_frozen"):
            w1, w2 = model.get_initial_weights()
            for s in train_steps:
                w1, w2 = model.evolve_weights(w1, w2)
            for s in val_steps:
                w1, w2 = model.evolve_weights(w1, w2)

            for s in test_steps:
                w1, w2 = model.evolve_weights(w1, w2)
                g = snapshot_cache[s]
                out = model.forward_snapshot(g.x, g.edge_index, w1, w2)
                probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

                orig_idx = g.orig_idx.cpu().numpy()
                y = g.y.cpu().numpy()
                ts = g.time_step.cpu().numpy()
                test_mask = g.test_mask.cpu().numpy()

                for idx_loc in range(len(orig_idx)):
                    all_node_idx.append(orig_idx[idx_loc])
                    all_steps.append(ts[idx_loc])
                    all_y.append(y[idx_loc])
                    all_scores.append(probs[idx_loc])
                    all_parts.append("test" if test_mask[idx_loc] else "test_unlabelled")

        elif name in ("gcn", "static_gcn"):
            for s in test_steps:
                g = snapshot_cache[s]
                out = model(g.x, g.edge_index)
                probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

                orig_idx = g.orig_idx.cpu().numpy()
                y = g.y.cpu().numpy()
                ts = g.time_step.cpu().numpy()
                test_mask = g.test_mask.cpu().numpy()

                for idx_loc in range(len(orig_idx)):
                    all_node_idx.append(orig_idx[idx_loc])
                    all_steps.append(ts[idx_loc])
                    all_y.append(y[idx_loc])
                    all_scores.append(probs[idx_loc])
                    all_parts.append("test" if test_mask[idx_loc] else "test_unlabelled")

        else:
            tracker = SnapshotContextTracker()
            for s in train_steps:
                g = snapshot_cache[s]
                ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
                ctx_t = torch.from_numpy(ctx).to(device)
                out = model(g.x, g.edge_index, context=ctx_t)
                probs = F.softmax(out, dim=-1)[:, 1]
                tracker.update(s, probs.mean().item())

            for s in val_steps:
                g = snapshot_cache[s]
                ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
                ctx_t = torch.from_numpy(ctx).to(device)
                out = model(g.x, g.edge_index, context=ctx_t)
                probs = F.softmax(out, dim=-1)[:, 1]
                tracker.update(s, probs.mean().item())

            for s in test_steps:
                g = snapshot_cache[s]
                ctx = tracker.get_context_vector(s, g.num_nodes, g.edge_index.size(1) // 2)
                ctx_t = torch.from_numpy(ctx).to(device)
                out = model(g.x, g.edge_index, context=ctx_t)
                probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

                orig_idx = g.orig_idx.cpu().numpy()
                y = g.y.cpu().numpy()
                ts = g.time_step.cpu().numpy()
                test_mask = g.test_mask.cpu().numpy()

                for idx_loc in range(len(orig_idx)):
                    all_node_idx.append(orig_idx[idx_loc])
                    all_steps.append(ts[idx_loc])
                    all_y.append(y[idx_loc])
                    all_scores.append(probs[idx_loc])
                    all_parts.append("test" if test_mask[idx_loc] else "test_unlabelled")

                tracker.update(s, probs.mean())

    return (
        np.array(all_node_idx, dtype=np.int64),
        np.array(all_steps, dtype=np.int64),
        np.array(all_y, dtype=np.int8),
        np.array(all_scores, dtype=np.float32),
        np.array(all_parts, dtype=object),
    )


def check_single_run_finished(results_dir: Path, model_name: str, split_name: str, feat_set: str, seed: int) -> bool:
    """Check if a single run directory is already complete."""
    if not results_dir.exists():
        return False
    pattern = f"*_{model_name}_{split_name}_{feat_set}_s{seed}"
    matches = [d for d in results_dir.glob(pattern) if (d / "metrics.json").exists() and (d / "predictions.parquet").exists()]
    return len(matches) > 0


def run_temporal_experiment(config_path: str = "experiments/configs/temporal.yaml") -> None:
    cfg = load_config(config_path)
    data = load_processed(cfg["data"]["processed_dir"])
    experiment = cfg.get("experiment", "exp2_temporal")
    models = cfg["models"]
    feature_sets = cfg["feature_sets"]
    splits = cfg["splits"]
    refit_splits = cfg.get("refit_splits", [])
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    training_cfg = cfg["training"]
    hpo_cfg = cfg.get("hpo", {})

    max_epochs = training_cfg.get("max_epochs", 200)
    patience = training_cfg.get("early_stopping_patience", 40)
    window_size = training_cfg.get("window_size", 5)
    weight_decay = training_cfg.get("weight_decay", 5e-4)

    device_str = training_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    results_dir = Path("experiments/results") / experiment
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    git_hash = get_env_info()["git_hash"]
    logger.info("Code git hash: %s", git_hash)

    paper_cache: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}

    for feat_set in feature_sets:
        paper_cache[feat_set] = {}

        for model_name in models:
            paper_cache[feat_set][model_name] = {}

            # -------------------------------------------------------------
            # HPO ON PAPER VAL ONLY (seed 0)
            # -------------------------------------------------------------
            logger.info("=== HPO tuning for model=%s | feat=%s on paper validation set ===", model_name, feat_set)
            paper_view = prepare_split(data, "paper", feat_set)
            paper_train_steps = sorted(list(paper_view.split.train_steps))
            paper_val_steps = sorted(list(paper_view.split.val_steps))
            paper_hpo_steps = sorted(list(set(paper_train_steps + paper_val_steps)))
            paper_hpo_cache = build_snapshot_cache(paper_view, paper_hpo_steps, device)

            best_hp, tuned_epochs = tune_temporal(
                model_name=model_name,
                view=paper_view,
                train_steps=paper_train_steps,
                val_steps=paper_val_steps,
                snapshot_cache=paper_hpo_cache,
                hpo_config=hpo_cfg,
                window_size=window_size,
                weight_decay=weight_decay,
                device=device,
                seed=0,
                max_tune_epochs=max_epochs,
                tune_patience=patience,
            )
            del paper_hpo_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # -------------------------------------------------------------
            # TRAIN ON ALL STANDARD SPLITS
            # -------------------------------------------------------------
            for split_name in splits:
                logger.info("Starting: model=%s | split=%s | feat=%s", model_name, split_name, feat_set)
                view = prepare_split(data, split_name, feat_set)
                train_steps = sorted(list(view.split.train_steps))
                val_steps = sorted(list(view.split.val_steps))
                test_steps = sorted(list(view.split.test_steps))
                all_steps = sorted(list(set(train_steps + val_steps + test_steps)))

                snapshot_cache = build_snapshot_cache(view, all_steps, device)

                train_y_list = [snapshot_cache[s].y[snapshot_cache[s].train_mask] for s in train_steps]
                train_y_cat = torch.cat(train_y_list, dim=0)
                n_pos = int((train_y_cat == 1).sum().item())
                n_neg = int((train_y_cat == 0).sum().item())
                total = n_pos + n_neg
                w_pos = float(total / (2.0 * n_pos)) if n_pos > 0 else 1.0
                w_neg = float(total / (2.0 * n_neg)) if n_neg > 0 else 1.0
                class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)

                val_y_all = []
                for s in val_steps:
                    g = snapshot_cache[s]
                    if g.val_mask.sum() > 0:
                        val_y_all.append(g.y[g.val_mask].cpu().numpy())
                val_y_all = np.concatenate(val_y_all) if val_y_all else np.empty(0)

                in_channels = view.X.shape[1]

                for seed in seeds:
                    if check_single_run_finished(results_dir, model_name, split_name, feat_set, seed):
                        logger.info("Skipping already completed run: %s | %s | %s | s%d", model_name, split_name, feat_set, seed)
                        if split_name == "paper":
                            pat = f"*_{model_name}_{split_name}_{feat_set}_s{seed}"
                            run_d = list(results_dir.glob(pat))[0]
                            m = json.loads((run_d / "metrics.json").read_text(encoding="utf-8"))
                            paper_cache[feat_set][model_name][seed] = {
                                "best_hp": best_hp,
                                "val_threshold": m["threshold_val_f1"],
                                "best_epoch": m.get("best_epoch", tuned_epochs),
                            }
                        continue

                    set_seed(seed)
                    model = create_temporal_model(
                        model_name=model_name,
                        in_channels=in_channels,
                        hidden_channels=best_hp["hidden_channels"],
                        dropout=best_hp["dropout"],
                        device=device,
                    )
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_hp["lr"], weight_decay=weight_decay)

                    best_val_score = -1.0
                    best_val_epoch = 0
                    best_state = None
                    patience_cnt = 0
                    curves_log = []

                    start_train_time = time.perf_counter()

                    for epoch in range(max_epochs):
                        ep_train_loss = train_one_epoch(
                            model=model,
                            model_name=model_name,
                            train_steps=train_steps,
                            snapshot_cache=snapshot_cache,
                            criterion=criterion,
                            optimizer=optimizer,
                            window_size=window_size,
                            device=device,
                        )

                        val_scores = evaluate_sequence(
                            model=model,
                            model_name=model_name,
                            train_steps=train_steps,
                            eval_steps=val_steps,
                            snapshot_cache=snapshot_cache,
                            mask_type="val_mask",
                            device=device,
                        )

                        val_pr_auc = float(average_precision_score(val_y_all, val_scores)) if len(val_y_all) > 0 and len(val_scores) > 0 else 0.0

                        curves_log.append({
                            "epoch": epoch + 1,
                            "train_loss": ep_train_loss,
                            "val_pr_auc": val_pr_auc,
                        })

                        if val_pr_auc > best_val_score:
                            best_val_score = val_pr_auc
                            best_val_epoch = epoch + 1
                            best_state = copy.deepcopy(model.state_dict())
                            patience_cnt = 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= patience:
                                break

                    train_duration_s = time.perf_counter() - start_train_time

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    val_scores_best = evaluate_sequence(
                        model=model,
                        model_name=model_name,
                        train_steps=train_steps,
                        eval_steps=val_steps,
                        snapshot_cache=snapshot_cache,
                        mask_type="val_mask",
                        device=device,
                    )
                    best_threshold = best_f1_threshold(val_y_all, val_scores_best)

                    if split_name == "paper":
                        paper_cache[feat_set][model_name][seed] = {
                            "best_hp": best_hp,
                            "val_threshold": best_threshold,
                            "best_epoch": best_val_epoch,
                        }

                    test_nodes, test_step_arr, test_y_arr, test_scores, test_parts = predict_test_snapshots(
                        model=model,
                        model_name=model_name,
                        train_steps=train_steps,
                        val_steps=val_steps,
                        test_steps=test_steps,
                        snapshot_cache=snapshot_cache,
                        device=device,
                    )

                    test_labelled = test_parts == "test"
                    y_test_eval = test_y_arr[test_labelled]
                    scores_test_eval = test_scores[test_labelled]

                    metrics_at_opt = compute_metrics(y_test_eval, scores_test_eval, threshold=best_threshold)
                    metrics_at_05 = compute_metrics(y_test_eval, scores_test_eval, threshold=0.5)

                    run_metrics = {
                        "val_pr_auc": best_val_score,
                        "best_epoch": best_val_epoch,
                        "threshold_val_f1": best_threshold,
                        "at_val_threshold": metrics_at_opt,
                        "at_05": metrics_at_05,
                    }

                    def predict_fn(x_in):
                        with torch.no_grad():
                            if model_name.lower() in ("evolvegcn", "evolvegcn_frozen"):
                                w1, w2 = model.get_initial_weights()
                                return model.conv1(x_in, snapshot_cache[test_steps[0]].edge_index, w1)
                            elif model_name.lower() in ("gcn", "static_gcn"):
                                return model.conv1(x_in, snapshot_cache[test_steps[0]].edge_index)
                            else:
                                ctx_dummy = torch.zeros(CONTEXT_DIM, device=device)
                                return model(x_in, snapshot_cache[test_steps[0]].edge_index, context=ctx_dummy)

                    latency_stats = measure_latency(predict_fn, snapshot_cache[test_steps[0]].x, repeats=15)
                    timing_info = {
                        "train_s": round(train_duration_s, 3),
                        "inference_latency": latency_stats,
                    }

                    run_cfg = {
                        "model": model_name,
                        "split": split_name,
                        "feature_set": feat_set,
                        "seed": seed,
                        "hidden_size": best_hp["hidden_channels"],
                        "hidden_channels": best_hp["hidden_channels"],
                        "lr": best_hp["lr"],
                        "dropout": best_hp["dropout"],
                        "epochs": int(best_val_epoch),
                        "git_hash": git_hash,
                        "val_threshold": float(best_threshold),
                        "best_hps": best_hp,
                        "hparams": best_hp,
                        "training": {
                            "max_epochs": max_epochs,
                            "patience": patience,
                            "window_size": window_size,
                            "device": str(device),
                        },
                    }

                    with RunRecorder(experiment=experiment, config=run_cfg) as rec:
                        rec.log_metrics(run_metrics)
                        rec.log_timing(timing_info)

                        test_tx_ids = data.tx_ids[test_nodes]
                        preds_df = pd.DataFrame({
                            "node_idx": test_nodes,
                            "txId": test_tx_ids,
                            "time_step": test_step_arr,
                            "y": test_y_arr,
                            "score": test_scores,
                            "part": test_parts,
                        })
                        rec.log_predictions(preds_df)

                        curves_df = pd.DataFrame(curves_log)
                        curves_df.to_csv(rec.run_dir / "curves.csv", index=False)

                        ckpt_path = checkpoints_dir / f"{rec.run_dir.name}.pt"
                        torch.save({
                            "model_class": model.__class__.__name__,
                            "model_kwargs": {"in_channels": in_channels, "hidden_channels": best_hp["hidden_channels"], "dropout": best_hp["dropout"]},
                            "state_dict": model.state_dict(),
                            "feature_set": feat_set,
                            "best_threshold": best_threshold,
                            "best_epoch": best_val_epoch,
                        }, ckpt_path)

                        logger.info("Saved run: %s | PR-AUC: %.4f | F1: %.4f", rec.run_dir.name, metrics_at_opt["pr_auc"], metrics_at_opt["f1"])

                    del model, optimizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del snapshot_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # -------------------------------------------------------------
            # EXECUTE PAPER_REFIT RUNS
            # -------------------------------------------------------------
            if "paper_refit" in refit_splits:
                logger.info("Executing paper_refit: model=%s | feat=%s", model_name, feat_set)
                refit_view = prepare_split(data, "paper_refit", feat_set)
                refit_train_steps = sorted(list(refit_view.split.train_steps))
                refit_test_steps = sorted(list(refit_view.split.test_steps))
                refit_all_steps = sorted(list(set(refit_train_steps + refit_test_steps)))

                refit_snapshot_cache = build_snapshot_cache(refit_view, refit_all_steps, device)

                refit_y_list = [refit_snapshot_cache[s].y[refit_snapshot_cache[s].train_mask] for s in refit_train_steps]
                refit_y_cat = torch.cat(refit_y_list, dim=0)
                n_pos = int((refit_y_cat == 1).sum().item())
                n_neg = int((refit_y_cat == 0).sum().item())
                total = n_pos + n_neg
                w_pos = float(total / (2.0 * n_pos)) if n_pos > 0 else 1.0
                w_neg = float(total / (2.0 * n_neg)) if n_neg > 0 else 1.0
                class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)

                in_channels = refit_view.X.shape[1]

                for seed in seeds:
                    if check_single_run_finished(results_dir, model_name, "paper_refit", feat_set, seed):
                        logger.info("Skipping already completed paper_refit: %s | %s | s%d", model_name, feat_set, seed)
                        continue

                    seed_info = paper_cache[feat_set][model_name].get(seed)
                    if seed_info is None:
                        seed_info = {
                            "best_hp": best_hp,
                            "val_threshold": 0.5,
                            "best_epoch": tuned_epochs,
                        }

                    refit_hp = seed_info["best_hp"]
                    val_threshold = seed_info["val_threshold"]
                    refit_epochs = max(1, seed_info["best_epoch"])

                    set_seed(seed)
                    model = create_temporal_model(
                        model_name=model_name,
                        in_channels=in_channels,
                        hidden_channels=refit_hp["hidden_channels"],
                        dropout=refit_hp["dropout"],
                        device=device,
                    )
                    optimizer = torch.optim.Adam(model.parameters(), lr=refit_hp["lr"], weight_decay=weight_decay)

                    start_train_time = time.perf_counter()
                    for ep in range(refit_epochs):
                        train_one_epoch(
                            model=model,
                            model_name=model_name,
                            train_steps=refit_train_steps,
                            snapshot_cache=refit_snapshot_cache,
                            criterion=criterion,
                            optimizer=optimizer,
                            window_size=window_size,
                            device=device,
                        )
                    train_duration_s = time.perf_counter() - start_train_time

                    test_nodes, test_step_arr, test_y_arr, test_scores, test_parts = predict_test_snapshots(
                        model=model,
                        model_name=model_name,
                        train_steps=refit_train_steps,
                        val_steps=[],
                        test_steps=refit_test_steps,
                        snapshot_cache=refit_snapshot_cache,
                        device=device,
                    )

                    test_labelled = test_parts == "test"
                    y_test_eval = test_y_arr[test_labelled]
                    scores_test_eval = test_scores[test_labelled]

                    metrics_at_opt = compute_metrics(y_test_eval, scores_test_eval, threshold=val_threshold)
                    metrics_at_05 = compute_metrics(y_test_eval, scores_test_eval, threshold=0.5)

                    run_metrics = {
                        "val_pr_auc": None,
                        "best_epoch": refit_epochs,
                        "threshold_val_f1": val_threshold,
                        "at_val_threshold": metrics_at_opt,
                        "at_05": metrics_at_05,
                    }

                    def predict_fn(x_in):
                        with torch.no_grad():
                            if model_name.lower() in ("evolvegcn", "evolvegcn_frozen"):
                                w1, w2 = model.get_initial_weights()
                                return model.conv1(x_in, refit_snapshot_cache[refit_test_steps[0]].edge_index, w1)
                            elif model_name.lower() in ("gcn", "static_gcn"):
                                return model.conv1(x_in, refit_snapshot_cache[refit_test_steps[0]].edge_index)
                            else:
                                ctx_dummy = torch.zeros(CONTEXT_DIM, device=device)
                                return model(x_in, refit_snapshot_cache[refit_test_steps[0]].edge_index, context=ctx_dummy)

                    latency_stats = measure_latency(predict_fn, refit_snapshot_cache[refit_test_steps[0]].x, repeats=15)
                    timing_info = {
                        "train_s": round(train_duration_s, 3),
                        "inference_latency": latency_stats,
                    }

                    run_cfg = {
                        "model": model_name,
                        "split": "paper_refit",
                        "feature_set": feat_set,
                        "seed": seed,
                        "hidden_size": refit_hp["hidden_channels"],
                        "hidden_channels": refit_hp["hidden_channels"],
                        "lr": refit_hp["lr"],
                        "dropout": refit_hp["dropout"],
                        "epochs": int(refit_epochs),
                        "git_hash": git_hash,
                        "val_threshold": float(val_threshold),
                        "best_hps": refit_hp,
                        "hparams": refit_hp,
                        "refit_from_seed": seed,
                        "training": {
                            "refit_epochs": refit_epochs,
                            "window_size": window_size,
                            "device": str(device),
                        },
                    }

                    with RunRecorder(experiment=experiment, config=run_cfg) as rec:
                        rec.log_metrics(run_metrics)
                        rec.log_timing(timing_info)

                        test_tx_ids = data.tx_ids[test_nodes]
                        preds_df = pd.DataFrame({
                            "node_idx": test_nodes,
                            "txId": test_tx_ids,
                            "time_step": test_step_arr,
                            "y": test_y_arr,
                            "score": test_scores,
                            "part": test_parts,
                        })
                        rec.log_predictions(preds_df)

                        ckpt_path = checkpoints_dir / f"{rec.run_dir.name}.pt"
                        torch.save({
                            "model_class": model.__class__.__name__,
                            "model_kwargs": {"in_channels": in_channels, "hidden_channels": refit_hp["hidden_channels"], "dropout": refit_hp["dropout"]},
                            "state_dict": model.state_dict(),
                            "feature_set": feat_set,
                            "best_threshold": val_threshold,
                            "best_epoch": refit_epochs,
                        }, ckpt_path)

                        logger.info("Saved paper_refit: %s | PR-AUC: %.4f | F1: %.4f", rec.run_dir.name, metrics_at_opt["pr_auc"], metrics_at_opt["f1"])

                    del model, optimizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del refit_snapshot_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # AUTOMATED SANITY CHECK ASSERTION: Frozen EvolveGCN vs Static GCN
    # -----------------------------------------------------------------
    try:
        from ml.evaluation.compare import load_test_predictions
        y_frz, s_frz, _ = load_test_predictions(results_dir, "evolvegcn_frozen", "paper_refit", "all")
        y_gcn, s_gcn, _ = load_test_predictions(results_dir, "gcn", "paper_refit", "all")
        pr_frz = float(average_precision_score(y_frz, s_frz))
        pr_gcn = float(average_precision_score(y_gcn, s_gcn))
        diff = abs(pr_frz - pr_gcn)
        logger.info(
            "=== AUTOMATED SANITY CHECK: Frozen EvolveGCN PR-AUC=%.4f vs Static GCN PR-AUC=%.4f (diff=%.4f) ===",
            pr_frz, pr_gcn, diff,
        )
        assert diff < 0.05, f"Automated sanity assertion failed: Frozen EvolveGCN ({pr_frz:.4f}) and Static GCN ({pr_gcn:.4f}) differ by {diff:.4f} >= 0.05!"
        logger.info("=== AUTOMATED SANITY CHECK PASSED: Frozen EvolveGCN matches Static GCN within tolerance! ===")
    except Exception as e:
        logger.error("Automated sanity check error: %s", e)
        raise

    logger.info("All temporal model experiments finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal models (EvolveGCN, Frozen EvolveGCN, Static GCN).")
    parser.add_argument("--config", type=str, default="experiments/configs/temporal.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    run_temporal_experiment(args.config)
