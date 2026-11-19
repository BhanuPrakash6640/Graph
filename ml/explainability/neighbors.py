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
    edge_index: torch.Tensor | np.ndarray | tuple[np.ndarray, np.ndarray],
    target_node_idx: int,
    neighbor_scores: np.ndarray | torch.Tensor | None = None,
    threshold: float = 0.5,
    neighbor_labels: np.ndarray | None = None,
) -> list[str]:
    """Compute structural graph pattern flags for a transaction node.

    Each flag is defined precisely as follows:
    - fan_out:
        Node sends outputs to >= 5 distinct recipients (out_degree >= 5).
        Indicates fund dispersion, mixing split, or peeling chain branch.
    - fan_in:
        Node receives inputs from >= 5 distinct senders (in_degree >= 5).
        Indicates fund consolidation or multi-source aggregation.
    - chain_node:
        Pass-through layering node having in_degree in {1, 2} and out_degree in {1, 2},
        or total unique degree == 2. Represents intermediate chain step.
    - isolated_or_leaf:
        Node with total unique degree <= 1 (terminal endpoint or isolated transaction).
    - high_illicit_neighborhood:
        >= 30% of the node's unique 1-hop neighbours are classified as illicit.
        When neighbor_scores is provided, an illicit neighbour is defined strictly
        as one with score >= threshold (using the explicit threshold of the model being explained).
        When neighbor_labels is provided, an illicit neighbour is one with label == 1.
        Requires at least 1 neighbour classified as illicit; if all neighbours have
        score < threshold (e.g. licit or unknown controls), this flag is NEVER set.

    Enforced constraint: Never evaluates the ground-truth label or prediction of target_node_idx itself.
    """
    if isinstance(edge_index, tuple):
        src = np.asarray(edge_index[0])
        dst = np.asarray(edge_index[1])
    elif isinstance(edge_index, torch.Tensor):
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
    else:
        src = np.asarray(edge_index[0])
        dst = np.asarray(edge_index[1])

    out_degree = int(np.sum(src == target_node_idx))
    in_degree = int(np.sum(dst == target_node_idx))

    out_nbrs = dst[src == target_node_idx]
    in_nbrs = src[dst == target_node_idx]
    all_nbrs = np.unique(np.concatenate([out_nbrs, in_nbrs]))
    all_nbrs = all_nbrs[all_nbrs != target_node_idx]
    total_degree = len(all_nbrs)

    flags = []

    if out_degree >= 5:
        flags.append("fan_out")
    if in_degree >= 5:
        flags.append("fan_in")
    if (in_degree in (1, 2) and out_degree in (1, 2)) or total_degree == 2:
        flags.append("chain_node")
    if total_degree <= 1:
        flags.append("isolated_or_leaf")

    # Neighbour risk concentration
    if total_degree > 0:
        if neighbor_scores is not None:
            if isinstance(neighbor_scores, torch.Tensor):
                scores_arr = neighbor_scores.detach().cpu().numpy()
            else:
                scores_arr = np.asarray(neighbor_scores)

            nbr_scores = scores_arr[all_nbrs]
            num_illicit = int(np.sum(nbr_scores >= threshold))
            illicit_ratio = float(num_illicit / total_degree)
            if num_illicit >= 1 and illicit_ratio >= 0.30:
                flags.append("high_illicit_neighborhood")
        elif neighbor_labels is not None:
            nbr_labels = np.asarray(neighbor_labels)[all_nbrs]
            num_illicit = int(np.sum(nbr_labels == 1))
            illicit_ratio = float(num_illicit / total_degree)
            if num_illicit >= 1 and illicit_ratio >= 0.30:
                flags.append("high_illicit_neighborhood")

    return sorted(flags)

