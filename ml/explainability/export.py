"""Precompute and export explainability JSON artefacts.

Precomputes multi-faceted explanations (SHAP tabular attributions, PyG GNNExplainer
fidelity & edge attributions, neighbour occlusion deltas, and structural flags)
for high-risk test transactions, false positives, and baseline controls.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F

from ml.data.loader import UNKNOWN, feature_columns
from ml.explainability.gnn_explain import explain_node_gnn, load_gnn_checkpoint
from ml.explainability.neighbors import compute_neighbor_occlusion, compute_pattern_flags
from ml.explainability.shap_tabular import DISCLAIMER, compute_global_shap, explain_node_shap, get_or_train_xgb
from ml.gnn.graph import build_period_graph
from ml.gnn.models import GCN
from ml.preprocessing.dataset import ProcessedData, load_processed, prepare_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.explainability.export")


def export_explanations(
    processed_dir: str = "data/processed",
    results_dir: str = "experiments/results",
    checkpoints_dir: str = "checkpoints",
    out_dir: str = "data/processed/explanations",
    split_name: str = "paper_refit",
    feature_set: str = "all",
    top_n_illicit: int = 50,
    n_false_positives: int = 5,
    n_licit_controls: int = 15,
    n_unknown_controls: int = 10,
    gnn_epochs: int = 30,
) -> list[dict[str, Any]]:
    """Generate and save complete explanation JSON records.

    Returns:
        List of generated explanation dictionaries.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Processed Dataset
    logger.info("Loading processed dataset from %s...", processed_dir)
    data = load_processed(processed_dir)
    view = prepare_split(data, split_name=split_name, feature_set=feature_set)
    feat_names = feature_columns(feature_set)

    # 2. Get or Train XGBoost Model
    logger.info("Setting up XGBoost model and TreeExplainer...")
    xgb_model, xgb_thresh = get_or_train_xgb(
        data=data,
        split_name=split_name,
        feature_set=feature_set,
        seed=0,
        checkpoint_dir=checkpoints_dir,
    )
    shap_explainer = shap.TreeExplainer(xgb_model)

    # Generate Global SHAP Feature Importance Plot
    test_mask = view.test_mask
    X_test_all = view.X[test_mask]
    sample_size = min(1000, len(X_test_all))
    sample_indices = np.random.RandomState(42).choice(len(X_test_all), size=sample_size, replace=False)
    compute_global_shap(
        model=xgb_model,
        X_sample=X_test_all[sample_indices],
        feature_names=feat_names,
        out_figure="reports/figures/shap_summary.png",
    )

    # 3. Locate Best GCN Checkpoint
    ckpt_dir = Path(checkpoints_dir)
    gcn_ckpts = list(ckpt_dir.glob("*gcn_paper_refit_all_s0.pt"))
    if not gcn_ckpts:
        raise FileNotFoundError("Could not find GCN paper_refit_all_s0 checkpoint in checkpoints/")
    gcn_ckpt_path = gcn_ckpts[0]
    logger.info("Loading GCN checkpoint: %s", gcn_ckpt_path)

    gcn_model, gcn_meta = load_gnn_checkpoint(gcn_ckpt_path, device="cpu")
    gcn_thresh = gcn_meta["threshold"]
    logger.info("Loaded GCN threshold=%.4f", gcn_thresh)

    # 4. Build Period Graph for Test Period
    test_graph = build_period_graph(view=view, steps=view.split.test_steps, undirected=True)
    test_x_tensor = test_graph.x
    test_edge_tensor = test_graph.edge_index
    test_orig_idx = test_graph.orig_idx.numpy()
    global_to_local = {int(g): int(loc) for loc, g in enumerate(test_orig_idx)}
    local_tx_ids = data.tx_ids[test_orig_idx]

    # Precompute GCN scores on test graph nodes
    with torch.no_grad():
        gnn_logits = gcn_model(test_x_tensor, test_edge_tensor)
        gnn_scores_graph = F.softmax(gnn_logits, dim=-1)[:, 1].numpy()

    # Precompute XGBoost scores on test nodes
    test_node_indices = np.where(view.test_mask)[0]
    test_scores_xgb = xgb_model.predict_proba(X_test_all)[:, 1].astype(np.float32)
    test_y = view.y[test_node_indices]
    test_tx_ids = data.tx_ids[test_node_indices]
    test_steps = view.time_step[test_node_indices]

    # Map test nodes to GCN scores
    test_scores_gcn = np.array([
        gnn_scores_graph[global_to_local[idx]] if idx in global_to_local else 0.0
        for idx in test_node_indices
    ], dtype=np.float32)

    # Helper function to select candidate cohorts for a specific model
    def select_cohorts(scores: np.ndarray, threshold: float) -> pd.DataFrame:
        df = pd.DataFrame({
            "node_idx": test_node_indices,
            "txId": test_tx_ids,
            "time_step": test_steps,
            "y": test_y,
            "score": scores,
        })
        top_ill = df[df["score"] >= threshold].sort_values("score", ascending=False).head(top_n_illicit)
        fp = df[(df["score"] >= threshold) & (df["y"] == 0)].sort_values("score", ascending=False).head(n_false_positives)
        tn = df[(df["score"] < 0.2) & (df["y"] == 0)].sample(n=min(n_licit_controls, len(df)), random_state=42)
        
        unlab_mask = np.isin(view.time_step, list(view.split.test_steps)) & (view.y == UNKNOWN)
        unlab_idx = np.where(unlab_mask)[0]
        unlab_samp = np.random.RandomState(42).choice(unlab_idx, size=min(n_unknown_controls, len(unlab_idx)), replace=False)
        unlab_df = pd.DataFrame({
            "node_idx": unlab_samp,
            "txId": data.tx_ids[unlab_samp],
            "time_step": view.time_step[unlab_samp],
            "y": UNKNOWN,
            "score": [
                float(scores[np.where(test_node_indices == u)[0][0]]) if u in test_node_indices
                else float(gnn_scores_graph[global_to_local[u]]) if u in global_to_local
                else 0.0
                for u in unlab_samp
            ],
        })
        return pd.concat([top_ill, fp, tn, unlab_df]).drop_duplicates(subset=["node_idx"])

    # -------------------------------------------------------------------------
    # 5. Export XGBoost Explanations (model_run_id = "xgb_paper_refit_all")
    # -------------------------------------------------------------------------
    xgb_run_id = "xgb_paper_refit_all"
    xgb_run_dir = out_path / xgb_run_id
    xgb_run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating XGBoost explanations in %s...", xgb_run_dir)

    selected_xgb = select_cohorts(test_scores_xgb, xgb_thresh).to_dict(orient="records")
    index_records = []
    explanations_list = []

    for item in selected_xgb:
        global_idx = int(item["node_idx"])
        tx_id = str(item["txId"])
        step = int(item["time_step"])
        y_val = int(item["y"])
        score = float(item["score"])
        pred_class = int(score >= xgb_thresh)

        label_str = "illicit" if y_val == 1 else "licit" if y_val == 0 else "unknown"

        # Tabular SHAP attributions
        x_vec = view.X[global_idx]
        top_features = explain_node_shap(shap_explainer, x_vec, feat_names, top_k=10)

        # Structural flags evaluated using XGBoost scores on 1-hop neighbours
        local_idx = global_to_local.get(global_idx)
        if local_idx is not None and local_idx < test_x_tensor.size(0):
            # Compute XGBoost scores for all graph neighbours
            src_edges = test_edge_tensor[0].numpy()
            dst_edges = test_edge_tensor[1].numpy()
            nbrs_loc = dst_edges[src_edges == local_idx]
            nbrs_loc = np.unique(nbrs_loc[nbrs_loc != local_idx])
            
            if len(nbrs_loc) > 0:
                nbrs_global = test_orig_idx[nbrs_loc]
                nbrs_X = view.X[nbrs_global]
                nbrs_xgb_scores = xgb_model.predict_proba(nbrs_X)[:, 1].astype(np.float32)
                # Map back to local array
                nbr_scores_map = np.zeros(test_x_tensor.size(0), dtype=np.float32)
                nbr_scores_map[nbrs_loc] = nbrs_xgb_scores
                pattern_flags = compute_pattern_flags(
                    edge_index=test_edge_tensor,
                    target_node_idx=local_idx,
                    neighbor_scores=nbr_scores_map,
                    threshold=xgb_thresh,
                )
            else:
                pattern_flags = ["isolated_or_leaf"]
        else:
            pattern_flags = ["isolated_or_leaf"]

        record = {
            "txId": tx_id,
            "node_idx": global_idx,
            "time_step": step,
            "model_run_id": xgb_run_id,
            "score": round(score, 4),
            "predicted_class": pred_class,
            "threshold": round(float(xgb_thresh), 4),
            "known_label": label_str,
            "top_features": top_features,
            "neighbor_contributions": [],
            "explainer_edges": [],
            "fidelity": {"fidelity_plus": 0.0, "fidelity_minus": 0.0},
            "pattern_flags": pattern_flags,
            "disclaimer": DISCLAIMER,
        }

        with open(xgb_run_dir / f"{tx_id}.json", "w") as f:
            json.dump(record, f, indent=2)

        index_records.append({
            "txId": tx_id,
            "time_step": step,
            "model_run_id": xgb_run_id,
            "score": round(score, 4),
            "predicted_class": pred_class,
            "known_label": label_str,
            "pattern_flags": pattern_flags,
            "file": f"{xgb_run_id}/{tx_id}.json",
        })

    # -------------------------------------------------------------------------
    # 6. Export GCN Explanations (model_run_id = "gcn_paper_refit_all")
    # -------------------------------------------------------------------------
    gcn_run_id = "gcn_paper_refit_all"
    gcn_run_dir = out_path / gcn_run_id
    gcn_run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating GCN explanations in %s...", gcn_run_dir)

    selected_gcn = select_cohorts(test_scores_gcn, gcn_thresh).to_dict(orient="records")

    for item in selected_gcn:
        global_idx = int(item["node_idx"])
        tx_id = str(item["txId"])
        step = int(item["time_step"])
        y_val = int(item["y"])
        local_idx = global_to_local.get(global_idx)

        gcn_score = float(gnn_scores_graph[local_idx]) if local_idx is not None else float(item["score"])
        pred_class = int(gcn_score >= gcn_thresh)
        label_str = "illicit" if y_val == 1 else "licit" if y_val == 0 else "unknown"

        if local_idx is not None and local_idx < test_x_tensor.size(0):
            # PyG GNNExplainer with global node index resolution and hard thresholding
            gnn_exp = explain_node_gnn(
                model=gcn_model,
                x=test_x_tensor,
                edge_index=test_edge_tensor,
                target_node_idx=local_idx,
                feature_names=feat_names,
                node_tx_ids=local_tx_ids,
                orig_idx=test_orig_idx,
                epochs=gnn_epochs,
            )
            fidelity_dict = gnn_exp["fidelity"]
            explainer_edges = gnn_exp["explainer_edges"]
            top_node_features = gnn_exp["top_node_features"]

            # Neighbour Occlusion
            neighbor_contribs = compute_neighbor_occlusion(
                model=gcn_model,
                x=test_x_tensor,
                edge_index=test_edge_tensor,
                target_node_idx=local_idx,
                node_tx_ids=local_tx_ids,
            )

            # Structural Pattern Flags
            pattern_flags = compute_pattern_flags(
                edge_index=test_edge_tensor,
                target_node_idx=local_idx,
                neighbor_scores=gnn_scores_graph,
                threshold=gcn_thresh,
            )
        else:
            fidelity_dict = {"fidelity_plus": 0.0, "fidelity_minus": 0.0}
            explainer_edges = []
            neighbor_contribs = []
            top_node_features = []
            pattern_flags = ["isolated_or_leaf"]

        # Adapt top_node_features into top_features schema for UI compatibility
        top_features_gcn = [
            {"feature": f["feature"], "value": 0.0, "shap_value": f["weight"], "direction": "+" if f["weight"] > 0 else "-"}
            for f in top_node_features
        ]

        record = {
            "txId": tx_id,
            "node_idx": global_idx,
            "time_step": step,
            "model_run_id": gcn_run_id,
            "score": round(gcn_score, 4),
            "predicted_class": pred_class,
            "threshold": round(float(gcn_thresh), 4),
            "known_label": label_str,
            "top_features": top_features_gcn,
            "top_node_features": top_node_features,
            "neighbor_contributions": neighbor_contribs[:10],
            "explainer_edges": explainer_edges[:10],
            "fidelity": fidelity_dict,
            "pattern_flags": pattern_flags,
            "disclaimer": DISCLAIMER,
        }

        # Write to GCN directory
        with open(gcn_run_dir / f"{tx_id}.json", "w") as f:
            json.dump(record, f, indent=2)

        # Also write canonical root alias
        with open(out_path / f"{tx_id}.json", "w") as f:
            json.dump(record, f, indent=2)

        explanations_list.append(record)
        index_records.append({
            "txId": tx_id,
            "time_step": step,
            "model_run_id": gcn_run_id,
            "score": round(gcn_score, 4),
            "predicted_class": pred_class,
            "known_label": label_str,
            "pattern_flags": pattern_flags,
            "file": f"{gcn_run_id}/{tx_id}.json",
            "fidelity_plus": fidelity_dict.get("fidelity_plus", 0.0),
        })

    # Save index.json
    with open(out_path / "index.json", "w") as f:
        json.dump(index_records, f, indent=2)

    logger.info("Successfully exported %d explanations across %s and %s", len(index_records), xgb_run_id, gcn_run_id)
    return explanations_list


def main():
    parser = argparse.ArgumentParser(description="Export explanations for transaction nodes")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out-dir", default="data/processed/explanations")
    parser.add_argument("--top-n-illicit", type=int, default=50)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    args = parser.parse_args()

    export_explanations(
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        out_dir=args.out_dir,
        top_n_illicit=args.top_n_illicit,
        gnn_epochs=args.gnn_epochs,
    )


if __name__ == "__main__":
    main()
