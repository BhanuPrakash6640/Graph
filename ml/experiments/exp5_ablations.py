"""Experiment 5: Ablation Suite on paper_refit.

Each experiment isolates and ablates exactly one component:
1. Feature Sets: local only (93) vs aggregated only (72) vs all (165).
2. Topology Only: GCN with constant input features (no node features).
3. Receptive Field Hops: 1 hop (1-layer GCN) vs 2 hops (2-layer GCN reference).
4. Edge Direction: Symmetrised undirected (reference) vs Raw directed payment flow.
5. Temporal Dynamics: EvolveGCN with weight evolution frozen (static) vs on (reference).

All runs are recorded to experiments/results/exp5_ablations/ using RunRecorder for complete traceability.
Full-model reference rows and paired block-bootstrap 95% confidence intervals are included in the final report tables.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import xgboost as xgb
import yaml

from ml.evaluation.compare import load_test_predictions
from ml.evaluation.metrics import best_f1_threshold, compute_metrics, paired_block_bootstrap_diff
from ml.evaluation.tracking import RunRecorder
from ml.gnn.graph import build_period_graph
from ml.gnn.models import GCN
from ml.preprocessing.dataset import SplitView, load_processed, prepare_split
from ml.temporal.evolvegcn import EvolveGCNO
from ml.utils.config import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.experiments.exp5")


class OneLayerGCN(nn.Module):
    """1-hop GCN directly mapping in_channels to output logits."""

    def __init__(self, in_channels: int, out_channels: int = 2) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


def get_ref_model_metrics(res_dir: Path, model: str, split: str, feat_set: str) -> tuple[float, float, float, float]:
    """Load PR-AUC and F1 mean ± std from reference runs."""
    pr_aucs = []
    f1s = []
    for cfg_file in sorted(res_dir.rglob("config.yaml")):
        if "_archive" in cfg_file.parts:
            continue
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg.get("model") == model and cfg.get("split") == split and cfg.get("feature_set") == feat_set:
                m_file = cfg_file.parent / "metrics.json"
                if m_file.exists():
                    m = json.loads(m_file.read_text(encoding="utf-8"))
                    if "at_val_threshold" in m:
                        sub = m["at_val_threshold"]
                    elif "test_at_opt" in m:
                        sub = m["test_at_opt"]
                    elif "test_at_05" in m:
                        sub = m["test_at_05"]
                    else:
                        sub = m
                    pr_aucs.append(sub.get("pr_auc", m.get("pr_auc", 0.0)))
                    f1s.append(sub.get("f1", m.get("f1", 0.0)))
        except Exception:
            continue
    if not pr_aucs:
        logger.warning("Could not find metrics for reference %s %s %s in %s", model, split, feat_set, res_dir)
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(pr_aucs)), float(np.std(pr_aucs)), float(np.mean(f1s)), float(np.std(f1s))


def generate_exp5_table(
    preds_by_variant: dict[str, list[pd.DataFrame]],
    df_abl: pd.DataFrame,
    results_base: str,
    out_path: Path,
) -> pd.DataFrame:
    var_xgb_agg = "XGB (aggregated only, 72d)"
    var_gcn_agg = "GCN (aggregated only, 72d)"
    var_gcn_topo = "GCN (constant 1s, no features)"
    var_gcn_1hop = "GCN 1-hop (1 layer)"
    var_gcn_dir = "GCN (directed edges)"
    var_evolve_frozen = "EvolveGCN (frozen weights)"

    logger.info("Loading reference model predictions for paired statistical comparisons...")
    res_p = Path(results_base)

    y_ref_xgb, s_ref_xgb, t_ref_xgb = load_test_predictions(res_p / "exp1_baselines", "xgb", "paper_refit", "all")
    xgb_pr_m, xgb_pr_s, xgb_f1_m, xgb_f1_s = get_ref_model_metrics(res_p / "exp1_baselines", "xgb", "paper_refit", "all")

    y_ref_gcn, s_ref_gcn, t_ref_gcn = load_test_predictions(res_p / "exp1_gnn", "gcn", "paper_refit", "all")
    gcn_pr_m, gcn_pr_s, gcn_f1_m, gcn_f1_s = get_ref_model_metrics(res_p / "exp1_gnn", "gcn", "paper_refit", "all")

    y_ref_ev, s_ref_ev, t_ref_ev = load_test_predictions(res_p / "exp2_temporal", "evolvegcn", "paper_refit", "all")
    ev_pr_m, ev_pr_s, ev_f1_m, ev_f1_s = get_ref_model_metrics(res_p / "exp2_temporal", "evolvegcn", "paper_refit", "all")

    ref_data = {
        "xgb_all": {
            "name": "XGB (all features, 165d)",
            "y": y_ref_xgb,
            "s": s_ref_xgb,
            "t": t_ref_xgb,
            "pr_auc": f"{xgb_pr_m:.4f} ± {xgb_pr_s:.4f}",
            "f1": f"{xgb_f1_m:.4f} ± {xgb_f1_s:.4f}",
        },
        "gcn_all": {
            "name": "GCN (all 2-layer undirected, 165d)",
            "y": y_ref_gcn,
            "s": s_ref_gcn,
            "t": t_ref_gcn,
            "pr_auc": f"{gcn_pr_m:.4f} ± {gcn_pr_s:.4f}",
            "f1": f"{gcn_f1_m:.4f} ± {gcn_f1_s:.4f}",
        },
        "evolvegcn_evolving": {
            "name": "EvolveGCN (evolving weights, 165d)",
            "y": y_ref_ev,
            "s": s_ref_ev,
            "t": t_ref_ev,
            "pr_auc": f"{ev_pr_m:.4f} ± {ev_pr_s:.4f}",
            "f1": f"{ev_f1_m:.4f} ± {ev_f1_s:.4f}",
        },
    }

    def get_mean_scores(variant_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dfs = preds_by_variant[variant_name]
        cat = pd.concat(dfs, ignore_index=True)
        grp = cat.groupby(["node_idx", "y", "time_step"], as_index=False)["score"].mean()
        grp = grp.sort_values("node_idx").reset_index(drop=True)
        return grp["y"].to_numpy(dtype=np.int8), grp["score"].to_numpy(dtype=np.float64), grp["time_step"].to_numpy()

    final_table_rows = []

    # Dimension 1: Features
    final_table_rows.append({
        "Ablation Dimension": "1. Features",
        "Variant / Condition": f"{ref_data['xgb_all']['name']} [Reference]",
        "PR-AUC (mean ± std)": ref_data["xgb_all"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["xgb_all"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_xgb_agg)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["xgb_all"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_xgb = df_abl[df_abl["variant"] == var_xgb_agg]
    final_table_rows.append({
        "Ablation Dimension": "1. Features",
        "Variant / Condition": var_xgb_agg,
        "PR-AUC (mean ± std)": f"{grp_xgb['pr_auc'].mean():.4f} ± {grp_xgb['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_xgb['f1'].mean():.4f} ± {grp_xgb['f1'].std():.4f}",
    })

    final_table_rows.append({
        "Ablation Dimension": "1. Features",
        "Variant / Condition": f"{ref_data['gcn_all']['name']} [Reference]",
        "PR-AUC (mean ± std)": ref_data["gcn_all"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["gcn_all"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_gcn_agg)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["gcn_all"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_gcn_agg = df_abl[df_abl["variant"] == var_gcn_agg]
    final_table_rows.append({
        "Ablation Dimension": "1. Features",
        "Variant / Condition": var_gcn_agg,
        "PR-AUC (mean ± std)": f"{grp_gcn_agg['pr_auc'].mean():.4f} ± {grp_gcn_agg['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_gcn_agg['f1'].mean():.4f} ± {grp_gcn_agg['f1'].std():.4f}",
    })

    # Dimension 2: Topology Only
    final_table_rows.append({
        "Ablation Dimension": "2. Topology Only",
        "Variant / Condition": f"{ref_data['gcn_all']['name']} [Reference]",
        "PR-AUC (mean ± std)": ref_data["gcn_all"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["gcn_all"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_gcn_topo)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["gcn_all"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_topo = df_abl[df_abl["variant"] == var_gcn_topo]
    final_table_rows.append({
        "Ablation Dimension": "2. Topology Only",
        "Variant / Condition": var_gcn_topo,
        "PR-AUC (mean ± std)": f"{grp_topo['pr_auc'].mean():.4f} ± {grp_topo['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_topo['f1'].mean():.4f} ± {grp_topo['f1'].std():.4f}",
    })

    # Dimension 3: Neighbour Hops
    final_table_rows.append({
        "Ablation Dimension": "3. Neighbour Hops",
        "Variant / Condition": "GCN 2-hop (2 layers, 165d) [Reference]",
        "PR-AUC (mean ± std)": ref_data["gcn_all"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["gcn_all"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_gcn_1hop)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["gcn_all"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_1hop = df_abl[df_abl["variant"] == var_gcn_1hop]
    final_table_rows.append({
        "Ablation Dimension": "3. Neighbour Hops",
        "Variant / Condition": var_gcn_1hop,
        "PR-AUC (mean ± std)": f"{grp_1hop['pr_auc'].mean():.4f} ± {grp_1hop['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_1hop['f1'].mean():.4f} ± {grp_1hop['f1'].std():.4f}",
    })

    # Dimension 4: Edge Direction
    final_table_rows.append({
        "Ablation Dimension": "4. Edge Direction",
        "Variant / Condition": "GCN undirected (symmetrised) [Reference]",
        "PR-AUC (mean ± std)": ref_data["gcn_all"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["gcn_all"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_gcn_dir)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["gcn_all"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_dir = df_abl[df_abl["variant"] == var_gcn_dir]
    final_table_rows.append({
        "Ablation Dimension": "4. Edge Direction",
        "Variant / Condition": var_gcn_dir,
        "PR-AUC (mean ± std)": f"{grp_dir['pr_auc'].mean():.4f} ± {grp_dir['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_dir['f1'].mean():.4f} ± {grp_dir['f1'].std():.4f}",
    })

    # Dimension 5: Temporal Dynamics
    final_table_rows.append({
        "Ablation Dimension": "5. Temporal Dynamics",
        "Variant / Condition": f"{ref_data['evolvegcn_evolving']['name']} [Reference]",
        "PR-AUC (mean ± std)": ref_data["evolvegcn_evolving"]["pr_auc"],
        "Δ PR-AUC vs Ref [95% CI]": "— [Reference]",
        "F1 (mean ± std)": ref_data["evolvegcn_evolving"]["f1"],
    })
    y_a, s_a, t_a = get_mean_scores(var_evolve_frozen)
    d_m, d_lo, d_hi = paired_block_bootstrap_diff(y_a, s_a, ref_data["evolvegcn_evolving"]["s"], groups=t_a, metric_fn=average_precision_score, n=2000, seed=42)
    grp_frz = df_abl[df_abl["variant"] == var_evolve_frozen]
    final_table_rows.append({
        "Ablation Dimension": "5. Temporal Dynamics",
        "Variant / Condition": var_evolve_frozen,
        "PR-AUC (mean ± std)": f"{grp_frz['pr_auc'].mean():.4f} ± {grp_frz['pr_auc'].std():.4f}",
        "Δ PR-AUC vs Ref [95% CI]": f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}]",
        "F1 (mean ± std)": f"{grp_frz['f1'].mean():.4f} ± {grp_frz['f1'].std():.4f}",
    })

    sum_df = pd.DataFrame(final_table_rows)
    sum_df.to_csv(out_path / "exp5_ablations.csv", index=False)
    sum_df.to_markdown(out_path / "exp5_ablations.md", index=False)
    logger.info("Saved Exp 5 ablations table to %s:\n%s", out_path, sum_df.to_markdown(index=False))
    return sum_df


def run_exp5_ablations(
    data_dir: str = "data/processed",
    results_base: str = "experiments/results",
    tables_dir: str = "reports/tables",
    seeds: list[int] = [0, 1, 2, 3, 4],
    device_str: str = "cuda",
    rerun: bool = False,
) -> pd.DataFrame:
    out_path = Path(tables_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_name = "exp5_ablations"

    var_xgb_agg = "XGB (aggregated only, 72d)"
    var_gcn_agg = "GCN (aggregated only, 72d)"
    var_gcn_topo = "GCN (constant 1s, no features)"
    var_gcn_1hop = "GCN 1-hop (1 layer)"
    var_gcn_dir = "GCN (directed edges)"
    var_evolve_frozen = "EvolveGCN (frozen weights)"

    preds_by_variant: dict[str, list[pd.DataFrame]] = {
        var_xgb_agg: [],
        var_gcn_agg: [],
        var_gcn_topo: [],
        var_gcn_1hop: [],
        var_gcn_dir: [],
        var_evolve_frozen: [],
    }

    exp_dir = Path(results_base) / exp_name
    exp2_dir = Path(results_base) / "exp2_temporal"
    existing_exp5_runs = list(exp_dir.glob("*/predictions.parquet")) if exp_dir.exists() else []
    existing_exp2_frozen = list(exp2_dir.glob("*_evolvegcn_frozen_paper_refit_all_s*/predictions.parquet")) if exp2_dir.exists() else []

    if len(existing_exp5_runs) >= 25 and len(existing_exp2_frozen) >= 5 and not rerun:
        logger.info("Found completed ablation runs (25 in %s, 5 in %s), loading from disk...", exp_dir, exp2_dir)
        rows = []
        model_to_meta = {
            "xgb_agg": ("1. Features", var_xgb_agg, "xgb_all"),
            "gcn_agg": ("1. Features", var_gcn_agg, "gcn_all"),
            "gcn_topo": ("2. Topology Only", var_gcn_topo, "gcn_all"),
            "gcn_1hop": ("3. Neighbour Hops", var_gcn_1hop, "gcn_all"),
            "gcn_directed": ("4. Edge Direction", var_gcn_dir, "gcn_all"),
        }
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            cfg_p = run_dir / "config.yaml"
            met_p = run_dir / "metrics.json"
            prd_p = run_dir / "predictions.parquet"
            if not (cfg_p.exists() and met_p.exists() and prd_p.exists()):
                continue
            cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
            met = json.loads(met_p.read_text(encoding="utf-8"))
            pred_df = pd.read_parquet(prd_p)
            model_key = cfg.get("model", "")
            if model_key in model_to_meta:
                abl_dim, var_name, ref_m = model_to_meta[model_key]
                preds_by_variant[var_name].append(pred_df)
                at_val = met.get("at_val_threshold", met)
                rows.append({
                    "ablation": abl_dim,
                    "variant": var_name,
                    "ref_model": ref_m,
                    "seed": cfg.get("seed", 0),
                    "pr_auc": at_val.get("pr_auc", met.get("pr_auc", 0.0)),
                    "f1": at_val.get("f1", met.get("f1", 0.0)),
                })

        # Load frozen EvolveGCN runs directly from exp2_temporal
        for seed in seeds:
            pat = f"*_evolvegcn_frozen_paper_refit_all_s{seed}"
            matches = sorted(list(exp2_dir.glob(pat)))
            if matches:
                run_d = matches[-1]
                met = json.loads((run_d / "metrics.json").read_text(encoding="utf-8"))
                pred_df = pd.read_parquet(run_d / "predictions.parquet")
                test_pred_df = pred_df[pred_df["part"] == "test"].copy()
                preds_by_variant[var_evolve_frozen].append(test_pred_df)
                at_val = met.get("at_val_threshold", met)
                rows.append({
                    "ablation": "5. Temporal Dynamics",
                    "variant": var_evolve_frozen,
                    "ref_model": "evolvegcn_evolving",
                    "seed": seed,
                    "pr_auc": at_val.get("pr_auc", met.get("pr_auc", 0.0)),
                    "f1": at_val.get("f1", met.get("f1", 0.0)),
                })

        df_abl = pd.DataFrame(rows)
        return generate_exp5_table(preds_by_variant, df_abl, results_base, out_path)

    data = load_processed(data_dir)
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    rows = []
    # ---------------------------------------------------------
    # 1. Feature Set Ablation: Aggregated only (72 dims)
    # ---------------------------------------------------------
    logger.info("Running Ablation 1: Aggregated Features Only...")
    view_agg_p = prepare_split(data, "paper", "aggregated")
    view_agg_r = prepare_split(data, "paper_refit", "aggregated")

    # XGBoost on aggregated
    train_idx_p = np.where(view_agg_p.train_mask)[0]
    val_idx_p = np.where(view_agg_p.val_mask)[0]
    train_idx_r = np.where(view_agg_r.train_mask)[0]
    test_idx_r = np.where(view_agg_r.test_mask)[0]

    X_tr_p, y_tr_p = view_agg_p.X[train_idx_p], view_agg_p.y[train_idx_p]
    X_va_p, y_va_p = view_agg_p.X[val_idx_p], view_agg_p.y[val_idx_p]
    X_tr_r, y_tr_r = view_agg_r.X[train_idx_r], view_agg_r.y[train_idx_r]
    X_te_r, y_te_r = view_agg_r.X[test_idx_r], view_agg_r.y[test_idx_r]

    scale_pos = float((y_tr_r == 0).sum() / max(1, (y_tr_r == 1).sum()))

    var_xgb_agg = "XGB (aggregated only, 72d)"
    preds_by_variant[var_xgb_agg] = []

    for seed in seeds:
        set_seed(seed)
        dtr = xgb.DMatrix(X_tr_p, label=y_tr_p)
        dva = xgb.DMatrix(X_va_p, label=y_va_p)
        params = {
            "tree_method": "hist",
            "device": str(device),
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
        }
        bst = xgb.train(params, dtr, num_boost_round=1000, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
        best_iter = bst.best_iteration + 1
        val_thr = best_f1_threshold(y_va_p, bst.predict(dva))

        rec_cfg = {
            "experiment": exp_name,
            "model": "xgb_agg",
            "split": "paper_refit",
            "feature_set": "aggregated",
            "ablation": "1. Features",
            "seed": seed,
            "val_threshold": float(val_thr),
        }
        with RunRecorder(experiment=exp_name, config=rec_cfg, base_dir=results_base) as recorder:
            st = time.perf_counter()
            bst_r = xgb.train(params, xgb.DMatrix(X_tr_r, label=y_tr_r), num_boost_round=best_iter, verbose_eval=False)
            tr_dur = time.perf_counter() - st

            st_ev = time.perf_counter()
            scores = bst_r.predict(xgb.DMatrix(X_te_r, label=y_te_r))
            ev_ms = (time.perf_counter() - st_ev) * 1000.0

            m = compute_metrics(y_te_r, scores, threshold=val_thr)
            recorder.log_metrics({"val_threshold": float(val_thr), "at_val_threshold": m, "pr_auc": float(m["pr_auc"]), "f1": float(m["f1"])})
            recorder.log_timing({"train_wall_s": float(tr_dur), "test_eval_ms": float(ev_ms)})

            pred_df = pd.DataFrame({
                "node_idx": test_idx_r,
                "txId": data.tx_ids[test_idx_r],
                "time_step": view_agg_r.time_step[test_idx_r],
                "y": view_agg_r.y[test_idx_r],
                "score": scores.astype(np.float32),
                "part": "test",
            })
            recorder.log_predictions(pred_df)
            preds_by_variant[var_xgb_agg].append(pred_df)

        rows.append({"ablation": "1. Features", "variant": var_xgb_agg, "ref_model": "xgb_all", "seed": seed, "pr_auc": m["pr_auc"], "f1": m["f1"]})

    # GCN on aggregated
    g_tr_p = build_period_graph(view_agg_p, view_agg_p.split.train_steps).to(device)
    g_va_p = build_period_graph(view_agg_p, view_agg_p.split.val_steps).to(device)
    g_tr_r = build_period_graph(view_agg_r, view_agg_r.split.train_steps).to(device)
    g_te_r = build_period_graph(view_agg_r, view_agg_r.split.test_steps).to(device)

    var_gcn_agg = "GCN (aggregated only, 72d)"
    preds_by_variant[var_gcn_agg] = []

    for seed in seeds:
        set_seed(seed)
        model = GCN(in_channels=72, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, scale_pos], device=device))
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_pr, best_ep, best_st, pat = -1.0, 1, None, 0
        for ep in range(300):
            model.train()
            opt.zero_grad()
            out = model(g_tr_p.x, g_tr_p.edge_index)
            loss = crit(out[g_tr_p.train_mask], g_tr_p.y[g_tr_p.train_mask])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_va = model(g_va_p.x, g_va_p.edge_index)
                s_va = F.softmax(out_va[g_va_p.val_mask], dim=-1)[:, 1].cpu().numpy()
                pr = average_precision_score(g_va_p.y[g_va_p.val_mask].cpu().numpy(), s_va)
                if pr > best_pr:
                    best_pr, best_ep, best_st, pat = pr, ep + 1, copy.deepcopy(model.state_dict()), 0
                else:
                    pat += 1
                    if pat >= 40:
                        break

        model.load_state_dict(best_st)
        with torch.no_grad():
            s_va = F.softmax(model(g_va_p.x, g_va_p.edge_index)[g_va_p.val_mask], dim=-1)[:, 1].cpu().numpy()
        val_thr = best_f1_threshold(g_va_p.y[g_va_p.val_mask].cpu().numpy(), s_va)

        rec_cfg = {
            "experiment": exp_name,
            "model": "gcn_agg",
            "split": "paper_refit",
            "feature_set": "aggregated",
            "ablation": "1. Features",
            "seed": seed,
            "val_threshold": float(val_thr),
        }
        with RunRecorder(experiment=exp_name, config=rec_cfg, base_dir=results_base) as recorder:
            st = time.perf_counter()
            model_r = GCN(in_channels=72, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
            opt_r = torch.optim.Adam(model_r.parameters(), lr=0.005, weight_decay=5e-4)
            for _ in range(best_ep):
                model_r.train()
                opt_r.zero_grad()
                out_r = model_r(g_tr_r.x, g_tr_r.edge_index)
                loss_r = crit(out_r[g_tr_r.train_mask], g_tr_r.y[g_tr_r.train_mask])
                loss_r.backward()
                opt_r.step()
            tr_dur = time.perf_counter() - st

            model_r.eval()
            st_ev = time.perf_counter()
            with torch.no_grad():
                out_te = model_r(g_te_r.x, g_te_r.edge_index)
                s_te = F.softmax(out_te[g_te_r.test_mask], dim=-1)[:, 1].cpu().numpy()
            ev_ms = (time.perf_counter() - st_ev) * 1000.0

            m = compute_metrics(g_te_r.y[g_te_r.test_mask].cpu().numpy(), s_te, threshold=val_thr)
            recorder.log_metrics({"val_threshold": float(val_thr), "at_val_threshold": m, "pr_auc": float(m["pr_auc"]), "f1": float(m["f1"])})
            recorder.log_timing({"train_wall_s": float(tr_dur), "test_eval_ms": float(ev_ms)})

            pred_df = pd.DataFrame({
                "node_idx": test_idx_r,
                "txId": data.tx_ids[test_idx_r],
                "time_step": view_agg_r.time_step[test_idx_r],
                "y": view_agg_r.y[test_idx_r],
                "score": s_te.astype(np.float32),
                "part": "test",
            })
            recorder.log_predictions(pred_df)
            preds_by_variant[var_gcn_agg].append(pred_df)

        rows.append({"ablation": "1. Features", "variant": var_gcn_agg, "ref_model": "gcn_all", "seed": seed, "pr_auc": m["pr_auc"], "f1": m["f1"]})

    # ---------------------------------------------------------
    # 2. Topology Only (Constant 1s as node features)
    # ---------------------------------------------------------
    logger.info("Running Ablation 2: Topology Only (Constant Features)...")
    view_all_p = prepare_split(data, "paper", "all")
    view_all_r = prepare_split(data, "paper_refit", "all")

    g_tr_p_const = build_period_graph(view_all_p, view_all_p.split.train_steps).to(device)
    g_va_p_const = build_period_graph(view_all_p, view_all_p.split.val_steps).to(device)
    g_tr_r_const = build_period_graph(view_all_r, view_all_r.split.train_steps).to(device)
    g_te_r_const = build_period_graph(view_all_r, view_all_r.split.test_steps).to(device)

    g_tr_p_const.x = torch.ones((g_tr_p_const.num_nodes, 1), device=device)
    g_va_p_const.x = torch.ones((g_va_p_const.num_nodes, 1), device=device)
    g_tr_r_const.x = torch.ones((g_tr_r_const.num_nodes, 1), device=device)
    g_te_r_const.x = torch.ones((g_te_r_const.num_nodes, 1), device=device)

    var_gcn_topo = "GCN (constant 1s, no features)"
    preds_by_variant[var_gcn_topo] = []

    for seed in seeds:
        set_seed(seed)
        model = GCN(in_channels=1, hidden_channels=32, out_channels=2, dropout=0.3).to(device)
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, scale_pos], device=device))
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_pr, best_ep, best_st, pat = -1.0, 1, None, 0
        for ep in range(300):
            model.train()
            opt.zero_grad()
            out = model(g_tr_p_const.x, g_tr_p_const.edge_index)
            loss = crit(out[g_tr_p_const.train_mask], g_tr_p_const.y[g_tr_p_const.train_mask])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_va = model(g_va_p_const.x, g_va_p_const.edge_index)
                s_va = F.softmax(out_va[g_va_p_const.val_mask], dim=-1)[:, 1].cpu().numpy()
                pr = average_precision_score(g_va_p_const.y[g_va_p_const.val_mask].cpu().numpy(), s_va)
                if pr > best_pr:
                    best_pr, best_ep, best_st, pat = pr, ep + 1, copy.deepcopy(model.state_dict()), 0
                else:
                    pat += 1
                    if pat >= 40:
                        break

        model.load_state_dict(best_st)
        with torch.no_grad():
            s_va = F.softmax(model(g_va_p_const.x, g_va_p_const.edge_index)[g_va_p_const.val_mask], dim=-1)[:, 1].cpu().numpy()
        val_thr = best_f1_threshold(g_va_p_const.y[g_va_p_const.val_mask].cpu().numpy(), s_va)

        rec_cfg = {
            "experiment": exp_name,
            "model": "gcn_topo",
            "split": "paper_refit",
            "feature_set": "constant_1s",
            "ablation": "2. Topology Only",
            "seed": seed,
            "val_threshold": float(val_thr),
        }
        with RunRecorder(experiment=exp_name, config=rec_cfg, base_dir=results_base) as recorder:
            st = time.perf_counter()
            model_r = GCN(in_channels=1, hidden_channels=32, out_channels=2, dropout=0.3).to(device)
            opt_r = torch.optim.Adam(model_r.parameters(), lr=0.005, weight_decay=5e-4)
            for _ in range(best_ep):
                model_r.train()
                opt_r.zero_grad()
                out_r = model_r(g_tr_r_const.x, g_tr_r_const.edge_index)
                loss_r = crit(out_r[g_tr_r_const.train_mask], g_tr_r_const.y[g_tr_r_const.train_mask])
                loss_r.backward()
                opt_r.step()
            tr_dur = time.perf_counter() - st

            model_r.eval()
            st_ev = time.perf_counter()
            with torch.no_grad():
                out_te = model_r(g_te_r_const.x, g_te_r_const.edge_index)
                s_te = F.softmax(out_te[g_te_r_const.test_mask], dim=-1)[:, 1].cpu().numpy()
            ev_ms = (time.perf_counter() - st_ev) * 1000.0

            m = compute_metrics(g_te_r_const.y[g_te_r_const.test_mask].cpu().numpy(), s_te, threshold=val_thr)
            recorder.log_metrics({"val_threshold": float(val_thr), "at_val_threshold": m, "pr_auc": float(m["pr_auc"]), "f1": float(m["f1"])})
            recorder.log_timing({"train_wall_s": float(tr_dur), "test_eval_ms": float(ev_ms)})

            pred_df = pd.DataFrame({
                "node_idx": test_idx_r,
                "txId": data.tx_ids[test_idx_r],
                "time_step": view_all_r.time_step[test_idx_r],
                "y": view_all_r.y[test_idx_r],
                "score": s_te.astype(np.float32),
                "part": "test",
            })
            recorder.log_predictions(pred_df)
            preds_by_variant[var_gcn_topo].append(pred_df)

        rows.append({"ablation": "2. Topology Only", "variant": var_gcn_topo, "ref_model": "gcn_all", "seed": seed, "pr_auc": m["pr_auc"], "f1": m["f1"]})

    # ---------------------------------------------------------
    # 3. Neighbour Hops: 1 Layer GCN (1 hop)
    # ---------------------------------------------------------
    logger.info("Running Ablation 3: 1-Hop GCN...")
    g_tr_p_all = build_period_graph(view_all_p, view_all_p.split.train_steps).to(device)
    g_va_p_all = build_period_graph(view_all_p, view_all_p.split.val_steps).to(device)
    g_tr_r_all = build_period_graph(view_all_r, view_all_r.split.train_steps).to(device)
    g_te_r_all = build_period_graph(view_all_r, view_all_r.split.test_steps).to(device)

    var_gcn_1hop = "GCN 1-hop (1 layer)"
    preds_by_variant[var_gcn_1hop] = []

    for seed in seeds:
        set_seed(seed)
        model = OneLayerGCN(in_channels=165, out_channels=2).to(device)
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, scale_pos], device=device))
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_pr, best_ep, best_st, pat = -1.0, 1, None, 0
        for ep in range(300):
            model.train()
            opt.zero_grad()
            out = model(g_tr_p_all.x, g_tr_p_all.edge_index)
            loss = crit(out[g_tr_p_all.train_mask], g_tr_p_all.y[g_tr_p_all.train_mask])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_va = model(g_va_p_all.x, g_va_p_all.edge_index)
                s_va = F.softmax(out_va[g_va_p_all.val_mask], dim=-1)[:, 1].cpu().numpy()
                pr = average_precision_score(g_va_p_all.y[g_va_p_all.val_mask].cpu().numpy(), s_va)
                if pr > best_pr:
                    best_pr, best_ep, best_st, pat = pr, ep + 1, copy.deepcopy(model.state_dict()), 0
                else:
                    pat += 1
                    if pat >= 40:
                        break

        model.load_state_dict(best_st)
        with torch.no_grad():
            s_va = F.softmax(model(g_va_p_all.x, g_va_p_all.edge_index)[g_va_p_all.val_mask], dim=-1)[:, 1].cpu().numpy()
        val_thr = best_f1_threshold(g_va_p_all.y[g_va_p_all.val_mask].cpu().numpy(), s_va)

        rec_cfg = {
            "experiment": exp_name,
            "model": "gcn_1hop",
            "split": "paper_refit",
            "feature_set": "all",
            "ablation": "3. Neighbour Hops",
            "seed": seed,
            "val_threshold": float(val_thr),
        }
        with RunRecorder(experiment=exp_name, config=rec_cfg, base_dir=results_base) as recorder:
            st = time.perf_counter()
            model_r = OneLayerGCN(in_channels=165, out_channels=2).to(device)
            opt_r = torch.optim.Adam(model_r.parameters(), lr=0.005, weight_decay=5e-4)
            for _ in range(best_ep):
                model_r.train()
                opt_r.zero_grad()
                out_r = model_r(g_tr_r_all.x, g_tr_r_all.edge_index)
                loss_r = crit(out_r[g_tr_r_all.train_mask], g_tr_r_all.y[g_tr_r_all.train_mask])
                loss_r.backward()
                opt_r.step()
            tr_dur = time.perf_counter() - st

            model_r.eval()
            st_ev = time.perf_counter()
            with torch.no_grad():
                out_te = model_r(g_te_r_all.x, g_te_r_all.edge_index)
                s_te = F.softmax(out_te[g_te_r_all.test_mask], dim=-1)[:, 1].cpu().numpy()
            ev_ms = (time.perf_counter() - st_ev) * 1000.0

            m = compute_metrics(g_te_r_all.y[g_te_r_all.test_mask].cpu().numpy(), s_te, threshold=val_thr)
            recorder.log_metrics({"val_threshold": float(val_thr), "at_val_threshold": m, "pr_auc": float(m["pr_auc"]), "f1": float(m["f1"])})
            recorder.log_timing({"train_wall_s": float(tr_dur), "test_eval_ms": float(ev_ms)})

            pred_df = pd.DataFrame({
                "node_idx": test_idx_r,
                "txId": data.tx_ids[test_idx_r],
                "time_step": view_all_r.time_step[test_idx_r],
                "y": view_all_r.y[test_idx_r],
                "score": s_te.astype(np.float32),
                "part": "test",
            })
            recorder.log_predictions(pred_df)
            preds_by_variant[var_gcn_1hop].append(pred_df)

        rows.append({"ablation": "3. Neighbour Hops", "variant": var_gcn_1hop, "ref_model": "gcn_all", "seed": seed, "pr_auc": m["pr_auc"], "f1": m["f1"]})

    # ---------------------------------------------------------
    # 4. Edge Direction: Directed GCN (no symmetrisation)
    # ---------------------------------------------------------
    logger.info("Running Ablation 4: Directed Graph GCN...")
    g_tr_p_dir = build_period_graph(view_all_p, view_all_p.split.train_steps, undirected=False).to(device)
    g_va_p_dir = build_period_graph(view_all_p, view_all_p.split.val_steps, undirected=False).to(device)
    g_tr_r_dir = build_period_graph(view_all_r, view_all_r.split.train_steps, undirected=False).to(device)
    g_te_r_dir = build_period_graph(view_all_r, view_all_r.split.test_steps, undirected=False).to(device)

    var_gcn_dir = "GCN (directed edges)"
    preds_by_variant[var_gcn_dir] = []

    for seed in seeds:
        set_seed(seed)
        model = GCN(in_channels=165, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, scale_pos], device=device))
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        best_pr, best_ep, best_st, pat = -1.0, 1, None, 0
        for ep in range(300):
            model.train()
            opt.zero_grad()
            out = model(g_tr_p_dir.x, g_tr_p_dir.edge_index)
            loss = crit(out[g_tr_p_dir.train_mask], g_tr_p_dir.y[g_tr_p_dir.train_mask])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_va = model(g_va_p_dir.x, g_va_p_dir.edge_index)
                s_va = F.softmax(out_va[g_va_p_dir.val_mask], dim=-1)[:, 1].cpu().numpy()
                pr = average_precision_score(g_va_p_dir.y[g_va_p_dir.val_mask].cpu().numpy(), s_va)
                if pr > best_pr:
                    best_pr, best_ep, best_st, pat = pr, ep + 1, copy.deepcopy(model.state_dict()), 0
                else:
                    pat += 1
                    if pat >= 40:
                        break

        model.load_state_dict(best_st)
        with torch.no_grad():
            s_va = F.softmax(model(g_va_p_dir.x, g_va_p_dir.edge_index)[g_va_p_dir.val_mask], dim=-1)[:, 1].cpu().numpy()
        val_thr = best_f1_threshold(g_va_p_dir.y[g_va_p_dir.val_mask].cpu().numpy(), s_va)

        rec_cfg = {
            "experiment": exp_name,
            "model": "gcn_directed",
            "split": "paper_refit",
            "feature_set": "all",
            "ablation": "4. Edge Direction",
            "seed": seed,
            "val_threshold": float(val_thr),
        }
        with RunRecorder(experiment=exp_name, config=rec_cfg, base_dir=results_base) as recorder:
            st = time.perf_counter()
            model_r = GCN(in_channels=165, hidden_channels=64, out_channels=2, dropout=0.3).to(device)
            opt_r = torch.optim.Adam(model_r.parameters(), lr=0.005, weight_decay=5e-4)
            for _ in range(best_ep):
                model_r.train()
                opt_r.zero_grad()
                out_r = model_r(g_tr_r_dir.x, g_tr_r_dir.edge_index)
                loss_r = crit(out_r[g_tr_r_dir.train_mask], g_tr_r_dir.y[g_tr_r_dir.train_mask])
                loss_r.backward()
                opt_r.step()
            tr_dur = time.perf_counter() - st

            model_r.eval()
            st_ev = time.perf_counter()
            with torch.no_grad():
                out_te = model_r(g_te_r_dir.x, g_te_r_dir.edge_index)
                s_te = F.softmax(out_te[g_te_r_dir.test_mask], dim=-1)[:, 1].cpu().numpy()
            ev_ms = (time.perf_counter() - st_ev) * 1000.0

            m = compute_metrics(g_te_r_dir.y[g_te_r_dir.test_mask].cpu().numpy(), s_te, threshold=val_thr)
            recorder.log_metrics({"val_threshold": float(val_thr), "at_val_threshold": m, "pr_auc": float(m["pr_auc"]), "f1": float(m["f1"])})
            recorder.log_timing({"train_wall_s": float(tr_dur), "test_eval_ms": float(ev_ms)})

            pred_df = pd.DataFrame({
                "node_idx": test_idx_r,
                "txId": data.tx_ids[test_idx_r],
                "time_step": view_all_r.time_step[test_idx_r],
                "y": view_all_r.y[test_idx_r],
                "score": s_te.astype(np.float32),
                "part": "test",
            })
            recorder.log_predictions(pred_df)
            preds_by_variant[var_gcn_dir].append(pred_df)

        rows.append({"ablation": "4. Edge Direction", "variant": var_gcn_dir, "ref_model": "gcn_all", "seed": seed, "pr_auc": m["pr_auc"], "f1": m["f1"]})

    # ---------------------------------------------------------
    # 5. Temporal Dynamics: EvolveGCN Frozen (evolve=False)
    # ---------------------------------------------------------
    logger.info("Loading Ablation 5: EvolveGCN Frozen Mode directly from exp2_temporal...")
    var_evolve_frozen = "EvolveGCN (frozen weights)"
    preds_by_variant[var_evolve_frozen] = []

    exp2_dir = Path(results_base) / "exp2_temporal"
    for seed in seeds:
        pat = f"*_evolvegcn_frozen_paper_refit_all_s{seed}"
        matches = sorted(list(exp2_dir.glob(pat)))
        if not matches:
            raise FileNotFoundError(f"Missing exp2 frozen run for seed {seed} matching {pat} in {exp2_dir}")
        run_d = matches[-1]
        met = json.loads((run_d / "metrics.json").read_text(encoding="utf-8"))
        pred_df = pd.read_parquet(run_d / "predictions.parquet")
        test_pred_df = pred_df[pred_df["part"] == "test"].copy()
        preds_by_variant[var_evolve_frozen].append(test_pred_df)
        at_val = met.get("at_val_threshold", met)
        rows.append({
            "ablation": "5. Temporal Dynamics",
            "variant": var_evolve_frozen,
            "ref_model": "evolvegcn_evolving",
            "seed": seed,
            "pr_auc": at_val.get("pr_auc", met.get("pr_auc", 0.0)),
            "f1": at_val.get("f1", met.get("f1", 0.0)),
        })

    df_abl = pd.DataFrame(rows)

    return generate_exp5_table(preds_by_variant, df_abl, results_base, out_path)


if __name__ == "__main__":
    run_exp5_ablations()
