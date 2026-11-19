"""Experiment 4: Class Imbalance Strategy Comparison on paper_refit.

Compares:
1. none (unweighted)
2. class_weights (inverse frequency)
3. focal_loss (gamma=1, gamma=2)
4. oversampling_random (RandomOverSampler on train rows, tabular only)
5. oversampling_smote (SMOTE on train rows, tabular only)
6. threshold tuning (0.5 vs val-optimal)

Demonstrates PR-AUC invariance under threshold tuning alongside large shifts in F1/precision/recall.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import time
from typing import Any

from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from ml.evaluation.latency import measure_latency
from ml.evaluation.metrics import best_f1_threshold, compute_metrics
from ml.evaluation.tracking import RunRecorder
from ml.gnn.graph import build_period_graph
from ml.gnn.models import GCN
from ml.preprocessing.dataset import SplitView, load_processed, prepare_split
from ml.utils.config import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.experiments.exp4")


class FocalLoss(nn.Module):
    """Focal Loss for binary classification: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: float = 0.1, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_t = probs[torch.arange(len(targets)), targets]
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        log_p_t = torch.log(p_t.clamp(min=1e-8))
        return (-focal_weight * log_p_t).mean()


def run_exp4_imbalance(
    data_dir: str = "data/processed",
    results_base: str = "experiments/results",
    tables_dir: str = "reports/tables",
    seeds: list[int] = [0, 1, 2, 3, 4],
    device_str: str = "cuda",
) -> pd.DataFrame:
    data = load_processed(data_dir)
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    out_path = Path(tables_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_name = "exp4_imbalance"

    view_paper = prepare_split(data, "paper", "all")
    view_refit = prepare_split(data, "paper_refit", "all")

    # Train / Val masks for paper
    train_idx_p = np.where(view_paper.train_mask)[0]
    val_idx_p = np.where(view_paper.val_mask)[0]
    test_idx_p = np.where(view_paper.test_mask)[0]

    X_train_p, y_train_p = view_paper.X[train_idx_p], view_paper.y[train_idx_p]
    X_val_p, y_val_p = view_paper.X[val_idx_p], view_paper.y[val_idx_p]
    X_test_p, y_test_p = view_paper.X[test_idx_p], view_paper.y[test_idx_p]

    # Train / Test for paper_refit (steps 1..34)
    train_idx_r = np.where(view_refit.train_mask)[0]
    test_idx_r = np.where(view_refit.test_mask)[0]
    X_train_r, y_train_r = view_refit.X[train_idx_r], view_refit.y[train_idx_r]
    X_test_r, y_test_r = view_refit.X[test_idx_r], view_refit.y[test_idx_r]

    n_pos_r = int((y_train_r == 1).sum())
    n_neg_r = int((y_train_r == 0).sum())
    alpha_train = n_pos_r / max(1, n_pos_r + n_neg_r)
    scale_pos = float(n_neg_r / max(1, n_pos_r))

    # Pre-build period graphs for GCN
    g_train_p = build_period_graph(view_paper, view_paper.split.train_steps).to(device)
    g_val_p = build_period_graph(view_paper, view_paper.split.val_steps).to(device)
    g_test_p = build_period_graph(view_paper, view_paper.split.test_steps).to(device)

    g_train_r = build_period_graph(view_refit, view_refit.split.train_steps).to(device)
    g_test_r = build_period_graph(view_refit, view_refit.split.test_steps).to(device)

    strategies_xgb = ["none", "class_weights", "oversampling_random", "oversampling_smote"]
    strategies_gcn = ["none", "class_weights", "focal_loss_gamma1", "focal_loss_gamma2"]

    results = []

    # 1. XGBoost Imbalance Evaluation
    for strat in strategies_xgb:
        for seed in seeds:
            set_seed(seed)
            # Paper phase: determine threshold and best iteration
            if strat == "none":
                spw = 1.0
                X_tr, y_tr = X_train_p, y_train_p
            elif strat == "class_weights":
                spw = scale_pos
                X_tr, y_tr = X_train_p, y_train_p
            elif strat == "oversampling_random":
                ros = RandomOverSampler(random_state=seed)
                X_tr, y_tr = ros.fit_resample(X_train_p, y_train_p)
                spw = 1.0
            elif strat == "oversampling_smote":
                smote = SMOTE(random_state=seed)
                X_tr, y_tr = smote.fit_resample(X_train_p, y_train_p)
                spw = 1.0

            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val_p, label=y_val_p)

            params = {
                "tree_method": "hist",
                "device": str(device),
                "eval_metric": "aucpr",
                "scale_pos_weight": spw,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
            }

            evals = [(dval, "val")]
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            best_iter = bst.best_iteration + 1
            val_scores = bst.predict(dval)
            val_thr = best_f1_threshold(y_val_p, val_scores)

            # Refit phase on 1..34
            if strat == "none":
                X_tr_r, y_tr_r = X_train_r, y_train_r
            elif strat == "class_weights":
                X_tr_r, y_tr_r = X_train_r, y_train_r
            elif strat == "oversampling_random":
                ros = RandomOverSampler(random_state=seed)
                X_tr_r, y_tr_r = ros.fit_resample(X_train_r, y_train_r)
            elif strat == "oversampling_smote":
                smote = SMOTE(random_state=seed)
                X_tr_r, y_tr_r = smote.fit_resample(X_train_r, y_train_r)

            dtrain_r = xgb.DMatrix(X_tr_r, label=y_tr_r)
            dtest_r = xgb.DMatrix(X_test_r, label=y_test_r)

            rec_config = {
                "experiment": exp_name,
                "model": f"xgb_{strat}",
                "split": "paper_refit",
                "feature_set": "all",
                "strategy": strat,
                "seed": seed,
                "val_threshold": float(val_thr),
            }
            with RunRecorder(experiment=exp_name, config=rec_config, base_dir=results_base) as recorder:
                start_tr = time.perf_counter()
                bst_r = xgb.train(params, dtrain_r, num_boost_round=best_iter, verbose_eval=False)
                tr_dur = time.perf_counter() - start_tr

                start_ev = time.perf_counter()
                test_scores = bst_r.predict(dtest_r)
                ev_dur_ms = (time.perf_counter() - start_ev) * 1000.0

                m_opt = compute_metrics(y_test_r, test_scores, threshold=val_thr)
                m_05 = compute_metrics(y_test_r, test_scores, threshold=0.5)

                recorder.log_metrics({
                    "val_threshold": float(val_thr),
                    "at_val_threshold": m_opt,
                    "at_05": m_05,
                    "pr_auc": float(m_opt["pr_auc"]),
                    "f1": float(m_opt["f1"]),
                })
                recorder.log_timing({
                    "train_wall_s": float(tr_dur),
                    "test_eval_ms": float(ev_dur_ms),
                })
                pred_df = pd.DataFrame({
                    "node_idx": test_idx_r,
                    "txId": data.tx_ids[test_idx_r],
                    "time_step": view_refit.time_step[test_idx_r],
                    "y": view_refit.y[test_idx_r],
                    "score": test_scores.astype(np.float32),
                    "part": "test",
                })
                recorder.log_predictions(pred_df)

            results.append({
                "model": "XGB",
                "strategy": strat,
                "seed": seed,
                "val_threshold": val_thr,
                "pr_auc": m_opt["pr_auc"],
                "f1_opt": m_opt["f1"],
                "f1_05": m_05["f1"],
                "prec_opt": m_opt["precision"],
                "prec_05": m_05["precision"],
                "rec_opt": m_opt["recall"],
                "rec_05": m_05["recall"],
            })

    # 2. GCN Imbalance Evaluation
    for strat in strategies_gcn:
        for seed in seeds:
            set_seed(seed)
            in_ch = g_train_p.x.size(1)
            model = GCN(in_channels=in_ch, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

            if strat == "none":
                crit = nn.CrossEntropyLoss()
            elif strat == "class_weights":
                crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, scale_pos], device=device))
            elif strat == "focal_loss_gamma1":
                crit = FocalLoss(alpha=alpha_train, gamma=1.0)
            elif strat == "focal_loss_gamma2":
                crit = FocalLoss(alpha=alpha_train, gamma=2.0)

            # Paper phase training
            best_val_pr = -1.0
            best_ep = 1
            best_state = None
            patience = 0

            for ep in range(300):
                model.train()
                optimizer.zero_grad()
                out = model(g_train_p.x, g_train_p.edge_index)
                mask = g_train_p.train_mask
                loss = crit(out[mask], g_train_p.y[mask])
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    out_val = model(g_val_p.x, g_val_p.edge_index)
                    s_val = F.softmax(out_val[g_val_p.val_mask], dim=-1)[:, 1].cpu().numpy()
                    y_val = g_val_p.y[g_val_p.val_mask].cpu().numpy()
                    pr = average_precision_score(y_val, s_val)
                    if pr > best_val_pr:
                        best_val_pr = pr
                        best_ep = ep + 1
                        best_state = copy.deepcopy(model.state_dict())
                        patience = 0
                    else:
                        patience += 1
                        if patience >= 40:
                            break

            model.load_state_dict(best_state)
            with torch.no_grad():
                out_val = model(g_val_p.x, g_val_p.edge_index)
                s_val = F.softmax(out_val[g_val_p.val_mask], dim=-1)[:, 1].cpu().numpy()
            val_thr = best_f1_threshold(y_val, s_val)

            # Refit phase on 1..34
            model_r = GCN(in_channels=in_ch, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
            optimizer_r = torch.optim.Adam(model_r.parameters(), lr=0.005, weight_decay=5e-4)

            rec_config = {
                "experiment": exp_name,
                "model": f"gcn_{strat}",
                "split": "paper_refit",
                "feature_set": "all",
                "strategy": strat,
                "seed": seed,
                "val_threshold": float(val_thr),
            }
            with RunRecorder(experiment=exp_name, config=rec_config, base_dir=results_base) as recorder:
                start_tr = time.perf_counter()
                for _ in range(best_ep):
                    model_r.train()
                    optimizer_r.zero_grad()
                    out_r = model_r(g_train_r.x, g_train_r.edge_index)
                    loss_r = crit(out_r[g_train_r.train_mask], g_train_r.y[g_train_r.train_mask])
                    loss_r.backward()
                    optimizer_r.step()
                tr_dur = time.perf_counter() - start_tr

                model_r.eval()
                start_ev = time.perf_counter()
                with torch.no_grad():
                    out_test = model_r(g_test_r.x, g_test_r.edge_index)
                    s_test = F.softmax(out_test[g_test_r.test_mask], dim=-1)[:, 1].cpu().numpy()
                    y_test = g_test_r.y[g_test_r.test_mask].cpu().numpy()
                ev_dur_ms = (time.perf_counter() - start_ev) * 1000.0

                m_opt = compute_metrics(y_test, s_test, threshold=val_thr)
                m_05 = compute_metrics(y_test, s_test, threshold=0.5)

                recorder.log_metrics({
                    "val_threshold": float(val_thr),
                    "at_val_threshold": m_opt,
                    "at_05": m_05,
                    "pr_auc": float(m_opt["pr_auc"]),
                    "f1": float(m_opt["f1"]),
                })
                recorder.log_timing({
                    "train_wall_s": float(tr_dur),
                    "test_eval_ms": float(ev_dur_ms),
                })
                pred_df = pd.DataFrame({
                    "node_idx": test_idx_r,
                    "txId": data.tx_ids[test_idx_r],
                    "time_step": view_refit.time_step[test_idx_r],
                    "y": view_refit.y[test_idx_r],
                    "score": s_test.astype(np.float32),
                    "part": "test",
                })
                recorder.log_predictions(pred_df)

            results.append({
                "model": "GCN",
                "strategy": strat,
                "seed": seed,
                "val_threshold": val_thr,
                "pr_auc": m_opt["pr_auc"],
                "f1_opt": m_opt["f1"],
                "f1_05": m_05["f1"],
                "prec_opt": m_opt["precision"],
                "prec_05": m_05["precision"],
                "rec_opt": m_opt["recall"],
                "rec_05": m_05["recall"],
            })

            del model, model_r
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(results)

    # Aggregate over seeds
    agg_rows = []
    for (model, strat), grp in df.groupby(["model", "strategy"], sort=False):
        n = len(grp)
        def _f(col):
            return f"{grp[col].mean():.4f} ± {grp[col].std():.4f}"

        agg_rows.append({
            "model": model,
            "strategy": strat,
            "n_seeds": n,
            "pr_auc": _f("pr_auc"),
            "f1 (val thr)": _f("f1_opt"),
            "f1 (0.5)": _f("f1_05"),
            "precision (val thr)": _f("prec_opt"),
            "precision (0.5)": _f("prec_05"),
            "recall (val thr)": _f("rec_opt"),
            "recall (0.5)": _f("rec_05"),
            "val_threshold": f"{grp['val_threshold'].mean():.3f} ± {grp['val_threshold'].std():.3f}",
        })

    sum_df = pd.DataFrame(agg_rows)
    sum_df.to_csv(out_path / "exp4_imbalance.csv", index=False)
    sum_df.to_markdown(out_path / "exp4_imbalance.md", index=False)
    logger.info("Saved Exp 4 imbalance table to %s", out_path)
    return sum_df


if __name__ == "__main__":
    run_exp4_imbalance()

