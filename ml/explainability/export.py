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

    raw_ckpt = torch.load(gcn_ckpt_path, map_location="cpu", weights_only=False)
    gcn_kwargs = raw_ckpt.get("model_kwargs", {"in_channels": 165, "hidden_channels": 128, "out_channels": 2})
    gcn_model = GCN(
        in_channels=gcn_kwargs.get("in_channels", 165),
        hidden_channels=gcn_kwargs.get("hidden_channels", 128),
        out_channels=gcn_kwargs.get("out_channels", 2),
        dropout=0.0,
    )
    gcn_model.load_state_dict(raw_ckpt["state_dict"])
    gcn_model.eval()
    gcn_thresh = float(raw_ckpt.get("threshold", 0.5))

    # 4. Predict on Test Nodes using XGBoost
    test_node_indices = np.where(view.test_mask)[0]
    test_scores = xgb_model.predict_proba(X_test_all)[:, 1].astype(np.float32)
    test_y = view.y[test_node_indices]
    test_tx_ids = data.tx_ids[test_node_indices]
    test_steps = view.time_step[test_node_indices]

    df_candidates = pd.DataFrame({
        "node_idx": test_node_indices,
        "txId": test_tx_ids,
        "time_step": test_steps,
        "y": test_y,
        "score": test_scores,
    })

    # Select node cohorts
    # A. Top N illicit predictions (sorted by score descending)
    top_illicit_df = df_candidates[df_candidates["score"] >= xgb_thresh].sort_values(
        "score", ascending=False
    ).head(top_n_illicit)

    # B. Confirmed False Positives (high score >= threshold, but ground-truth y == 0 [licit])
    fp_df = df_candidates[
        (df_candidates["score"] >= xgb_thresh) & (df_candidates["y"] == 0)
    ].sort_values("score", ascending=False).head(n_false_positives)

    # C. Confirmed True Negatives (low score < threshold, ground-truth y == 0)
    tn_df = df_candidates[
        (df_candidates["score"] < 0.2) & (df_candidates["y"] == 0)
    ].sample(n=min(n_licit_controls, len(df_candidates)), random_state=42)

    # D. Unknown Test Nodes (y == UNKNOWN)
    unlab_mask = np.isin(view.time_step, list(view.split.test_steps)) & (view.y == UNKNOWN)
    unlab_indices = np.where(unlab_mask)[0]
    unlab_sample = np.random.RandomState(42).choice(
        unlab_indices, size=min(n_unknown_controls, len(unlab_indices)), replace=False
    )
    X_unlab = view.X[unlab_sample]
    s_unlab = xgb_model.predict_proba(X_unlab)[:, 1].astype(np.float32)
    unlab_df = pd.DataFrame({
        "node_idx": unlab_sample,
        "txId": data.tx_ids[unlab_sample],
        "time_step": view.time_step[unlab_sample],
        "y": UNKNOWN,
        "score": s_unlab,
    })

    selected_nodes = pd.concat([top_illicit_df, fp_df, tn_df, unlab_df]).drop_duplicates(
        subset=["node_idx"]
    ).to_dict(orient="records")

    logger.info("Selected %d candidate nodes for full multi-faceted explanations.", len(selected_nodes))

    # 5. Build Period Graph for Test Period
    test_graph = build_period_graph(view=view, steps=view.split.test_steps, undirected=True)
    test_x_tensor = test_graph.x
    test_edge_tensor = test_graph.edge_index
    global_to_local = {int(g): int(loc) for loc, g in enumerate(test_graph.orig_idx.numpy())}
    local_tx_ids = data.tx_ids[test_graph.orig_idx.numpy()]

    # Precompute GNN scores on test graph nodes for neighbour risk flags
    with torch.no_grad():
        gnn_logits = gcn_model(test_x_tensor, test_edge_tensor)
        gnn_scores = F.softmax(gnn_logits, dim=-1)[:, 1].numpy()

    # 6. Generate Multi-Faceted Explanations
    model_run_id = "xgb_gcn_paper_refit_all"
    run_dir = out_path / model_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    index_records = []
    explanations_list = []

    for item in selected_nodes:
        global_idx = int(item["node_idx"])
        tx_id = str(item["txId"])
        step = int(item["time_step"])
        y_val = int(item["y"])
        score = float(item["score"])
        pred_class = int(score >= xgb_thresh)

        if y_val == 1:
            label_str = "illicit"
        elif y_val == 0:
            label_str = "licit"
        else:
            label_str = "unknown"

        # Tabular SHAP
        x_vec = view.X[global_idx]
        top_features = explain_node_shap(shap_explainer, x_vec, feat_names, top_k=10)

        # Graph Local Index in Test Graph
        local_idx = global_to_local.get(global_idx)
        if local_idx is not None and local_idx < test_x_tensor.size(0):
            # PyG GNNExplainer
            gnn_exp = explain_node_gnn(
                model=gcn_model,
                x=test_x_tensor,
                edge_index=test_edge_tensor,
                target_node_idx=local_idx,
                feature_names=feat_names,
                node_tx_ids=local_tx_ids,
                epochs=gnn_epochs,
            )
            fidelity_dict = gnn_exp["fidelity"]
            explainer_edges = gnn_exp["explainer_edges"]

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
                neighbor_scores=gnn_scores,
                threshold=gcn_thresh,
            )
        else:
            fidelity_dict = {"fidelity_plus": 0.0, "fidelity_minus": 0.0}
            explainer_edges = []
            neighbor_contribs = []
            pattern_flags = ["isolated_or_leaf"]

        record = {
            "txId": tx_id,
            "node_idx": global_idx,
            "time_step": step,
            "model_run_id": model_run_id,
            "score": round(score, 4),
            "predicted_class": pred_class,
            "threshold": round(xgb_thresh, 4),
            "known_label": label_str,
            "top_features": top_features,
            "neighbor_contributions": neighbor_contribs[:10],
            "explainer_edges": explainer_edges[:10],
            "fidelity": fidelity_dict,
            "pattern_flags": pattern_flags,
            "disclaimer": DISCLAIMER,
        }

        # Write to individual json file
        tx_file = run_dir / f"{tx_id}.json"
        with open(tx_file, "w") as f:
            json.dump(record, f, indent=2)

        # Also write canonical root alias
        with open(out_path / f"{tx_id}.json", "w") as f:
            json.dump(record, f, indent=2)

        explanations_list.append(record)
        index_records.append({
            "txId": tx_id,
            "time_step": step,
            "score": round(score, 4),
            "predicted_class": pred_class,
            "known_label": label_str,
            "pattern_flags": pattern_flags,
            "file": f"{model_run_id}/{tx_id}.json",
        })

    # Save index.json
    with open(out_path / "index.json", "w") as f:
        json.dump(index_records, f, indent=2)

    logger.info("Successfully exported %d explanations to %s", len(explanations_list), out_dir)
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
