"""Neighbour occlusion and structural transaction pattern flags.

Calculates model-based neighbour attribution by leave-one-out edge ablation
and computes graph structural indicators (fan-out, fan-in, chaining/layering,
and neighbour risk concentration) without looking at target ground-truth labels.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.explainability.neighbors")


def compute_neighbor_occlusion(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    node_tx_ids: list[str] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute leave-one-out score difference for each 1-hop neighbour.

    Args:
        model: Trained GNN returning raw logits of shape (N, 2).
        x: Node feature tensor (N, D).
        edge_index: Graph edge tensor (2, E).
        target_node_idx: Index of target node.
        node_tx_ids: Optional list/array of string txIds.

    Returns:
        List of dicts sorted by absolute score delta descending:
        [{'neighbor_idx': v, 'neighbor_txId': str, 'score_delta': float, 'impact': '+' | '-'}, ...]
    """
    model.eval()
    device = x.device

    with torch.no_grad():
        # Baseline prediction
        base_logits = model(x, edge_index)
        base_score = float(F.softmax(base_logits, dim=-1)[target_node_idx, 1].item())

        src = edge_index[0]
        dst = edge_index[1]

        # Identify all 1-hop neighbours (both directions)
        out_neighbors = dst[src == target_node_idx].tolist()
        in_neighbors = src[dst == target_node_idx].tolist()
        unique_neighbors = sorted(list(set(out_neighbors + in_neighbors)))

        # Exclude self-loop if present
        if target_node_idx in unique_neighbors:
            unique_neighbors.remove(target_node_idx)

        contributions = []
        for nbr in unique_neighbors:
            # Mask out all edges between target and this neighbour
            mask = ~(
                ((src == target_node_idx) & (dst == nbr))
                | ((src == nbr) & (dst == target_node_idx))
            )
            ab_edge_index = edge_index[:, mask]

            # Forward pass without this neighbour
            ab_logits = model(x, ab_edge_index)
            ab_score = float(F.softmax(ab_logits, dim=-1)[target_node_idx, 1].item())
            score_delta = base_score - ab_score  # Positive: neighbour increased illicit score

            nbr_tx = str(node_tx_ids[nbr]) if node_tx_ids is not None else str(nbr)
            contributions.append({
                "neighbor_idx": int(nbr),
                "neighbor_txId": nbr_tx,
                "score_delta": round(float(score_delta), 4),
                "impact": "+" if score_delta > 0 else "-",
            })

    # Sort by magnitude of score change
    contributions.sort(key=lambda item: abs(item["score_delta"]), reverse=True)
    return contributions


def compute_pattern_flags(
    edge_index: torch.Tensor | np.ndarray,
    target_node_idx: int,
    neighbor_scores: np.ndarray | torch.Tensor | None = None,
    threshold: float = 0.5,
) -> list[str]:
    """Compute structural graph pattern flags for a transaction node.

    Evaluates topological patterns:
    - fan_out: out-degree >= 5 (dispersion / peeling chain split)
    - fan_in: in-degree >= 5 (aggregation / consolidation)
    - chain_node: in-degree in {1, 2} and out-degree in {1, 2} (pass-through / layering)
    - high_illicit_neighborhood: >= 30% of 1-hop neighbours predicted illicit
    - isolated_or_leaf: total degree <= 1

    Enforced constraint: Never evaluates the ground-truth label of target_node_idx.
    """
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
    else:
        src = edge_index[0]
        dst = edge_index[1]

    out_degree = int(np.sum(src == target_node_idx))
    in_degree = int(np.sum(dst == target_node_idx))
    total_degree = in_degree + out_degree

    flags = []

    if out_degree >= 5:
        flags.append("fan_out")
    if in_degree >= 5:
        flags.append("fan_in")
    if in_degree in (1, 2) and out_degree in (1, 2):
        flags.append("chain_node")
    if total_degree <= 1:
        flags.append("isolated_or_leaf")

    # Neighbour risk concentration
    if neighbor_scores is not None and total_degree > 0:
        out_nbrs = dst[src == target_node_idx]
        in_nbrs = src[dst == target_node_idx]
        all_nbrs = np.unique(np.concatenate([out_nbrs, in_nbrs]))
        all_nbrs = all_nbrs[all_nbrs != target_node_idx]

        if len(all_nbrs) > 0:
            if isinstance(neighbor_scores, torch.Tensor):
                scores_arr = neighbor_scores.detach().cpu().numpy()
            else:
                scores_arr = np.asarray(neighbor_scores)

            nbr_scores = scores_arr[all_nbrs]
            illicit_ratio = float(np.mean(nbr_scores >= threshold))
            if illicit_ratio >= 0.30:
                flags.append("high_illicit_neighborhood")

    return sorted(flags)

