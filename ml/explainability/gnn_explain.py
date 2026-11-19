"""Graph Neural Network explainability using PyG GNNExplainer.

Extracts local 2-hop subgraphs within a transaction snapshot and computes
edge and feature attribution masks with PyG fidelity metrics (fidelity+ / fidelity-).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, fidelity
from torch_geometric.utils import k_hop_subgraph

from ml.gnn.models import GCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.explainability.gnn_explain")


def load_gnn_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a trained GNN checkpoint and its metadata."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    meta = {
        "model_class": ckpt.get("model_class", "GCN"),
        "in_channels": ckpt.get("in_channels", 165),
        "hidden_channels": ckpt.get("hidden_channels", 64),
        "out_channels": ckpt.get("out_channels", 2),
        "dropout": ckpt.get("dropout", 0.3),
        "feature_set": ckpt.get("feature_set", "all"),
        "best_epoch": ckpt.get("best_epoch", 0),
        "threshold": ckpt.get("threshold", 0.5),
    }

    model = GCN(
        in_channels=meta["in_channels"],
        hidden_channels=meta["hidden_channels"],
        out_channels=meta["out_channels"],
        dropout=0.0,  # Inference / explainability mode
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta


def explain_node_gnn(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    feature_names: list[str] | None = None,
    node_tx_ids: list[str] | np.ndarray | None = None,
    epochs: int = 50,
    top_k_edges: int = 10,
    top_k_features: int = 10,
) -> dict[str, Any]:
    """Compute GNNExplainer edge and feature attributions on the 2-hop subgraph.

    Returns:
        Dict with fidelity (fidelity_plus, fidelity_minus), explainer_edges, and top_node_features.
    """
    model.eval()
    device = x.device

    # 1. Extract 2-hop local subgraph around the target node
    subset, sub_edge_index, mapping, edge_mask_hop = k_hop_subgraph(
        node_idx=int(target_node_idx),
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=True,
    )

    sub_x = x[subset].detach()
    target_local_idx = int(mapping)

    model_config = ModelConfig(
        mode="multiclass_classification",
        task_level="node",
        return_type="raw",
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config,
    )

    # 2. Run explainer on the 2-hop subgraph
    explanation = explainer(sub_x, sub_edge_index, index=target_local_idx)

    # 3. Compute PyG fidelity metrics
    try:
        fid_plus, fid_minus = fidelity(explainer, explanation)
    except Exception as e:
        logger.warning("Fidelity computation encountered warning: %s. Defaulting to 0.0.", e)
        fid_plus, fid_minus = 0.0, 0.0

    # 4. Extract top attributed edges
    sub_edge_mask = explanation.edge_mask
    if sub_edge_mask is not None and sub_edge_mask.numel() > 0:
        edge_weights = sub_edge_mask.detach().cpu().numpy()
        order = np.argsort(-edge_weights)[:top_k_edges]

        top_edges = []
        for i in order:
            src_local = int(sub_edge_index[0, i])
            dst_local = int(sub_edge_index[1, i])
            src_global = int(subset[src_local])
            dst_global = int(subset[dst_local])

            src_tx = str(node_tx_ids[src_global]) if node_tx_ids is not None else str(src_global)
            dst_tx = str(node_tx_ids[dst_global]) if node_tx_ids is not None else str(dst_global)
            top_edges.append({
                "source": src_tx,
                "target": dst_tx,
                "source_idx": src_global,
                "target_idx": dst_global,
                "edge_weight": round(float(edge_weights[i]), 4),
            })
    else:
        top_edges = []

    # 5. Extract top attributed features for the target node
    top_node_features = []
    if explanation.node_mask is not None:
        target_feat_mask = explanation.node_mask[target_local_idx].detach().cpu().numpy()
        feat_order = np.argsort(-np.abs(target_feat_mask))[:top_k_features]
        for f_idx in feat_order:
            f_name = feature_names[f_idx] if feature_names is not None else f"feature_{f_idx}"
            top_node_features.append({
                "feature": str(f_name),
                "weight": round(float(target_feat_mask[f_idx]), 4),
            })

    return {
        "target_node_idx": int(target_node_idx),
        "fidelity": {
            "fidelity_plus": round(float(fid_plus), 4),
            "fidelity_minus": round(float(fid_minus), 4),
        },
        "explainer_edges": top_edges,
        "top_node_features": top_node_features,
        "subgraph_nodes_count": int(subset.numel()),
        "subgraph_edges_count": int(sub_edge_index.size(1)),
    }

