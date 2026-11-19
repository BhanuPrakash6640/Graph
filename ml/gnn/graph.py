"""Graph construction for PyG static GNN models.

Builds leak-free period subgraphs (train, val, test) with strictly re-indexed nodes
and isolated intra-period edges.
"""
from __future__ import annotations

from typing import Collection

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from ml.preprocessing.dataset import SplitView, edges_for_steps


def build_period_graph(
    view: SplitView,
    steps: Collection[int] | set[int],
    undirected: bool = True,
) -> Data:
    """Build a leak-free PyG Data object containing only nodes in ``steps``.

    All node arrays and edge indices are re-indexed locally to [0, N_period - 1].
    Labels, features, and masks are preserved for the selected nodes.

    Args:
        view: SplitView containing preprocessed features and global masks.
        steps: Time steps to include in this period subgraph.
        undirected: Whether to symmetrise edges via ``to_undirected``.

    Returns:
        Data object with x, edge_index, y, train_mask, val_mask, test_mask,
        orig_idx, and time_step.
    """
    step_set = set(steps)
    if not step_set:
        raise ValueError("Cannot build period graph with empty steps")

    # 1. Identify nodes belonging to this period
    in_period = np.isin(view.time_step, list(step_set))
    orig_idx = np.where(in_period)[0]
    n_nodes = len(orig_idx)

    # 2. Extract features and labels
    x = torch.from_numpy(view.X[orig_idx]).to(torch.float32)
    y = torch.from_numpy(view.y[orig_idx]).to(torch.long)

    # 3. Extract masks restricted to this period's nodes
    train_mask = torch.from_numpy(view.train_mask[orig_idx]).to(torch.bool)
    val_mask = torch.from_numpy(view.val_mask[orig_idx]).to(torch.bool)
    test_mask = torch.from_numpy(view.test_mask[orig_idx]).to(torch.bool)
    time_step_tensor = torch.from_numpy(view.time_step[orig_idx]).to(torch.long)

    # 4. Extract and re-index edges
    # Filter edges whose both endpoints belong to steps
    edges = edges_for_steps(view.edge_index, view.time_step, step_set)

    if edges.shape[1] > 0:
        node_map = np.full(len(view.X), -1, dtype=np.int64)
        node_map[orig_idx] = np.arange(n_nodes, dtype=np.int64)

        src = node_map[edges[0]]
        dst = node_map[edges[1]]

        assert not (src == -1).any(), "Source node not found in period graph mapping"
        assert not (dst == -1).any(), "Destination node not found in period graph mapping"

        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).to(torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 5. Symmetrise if undirected
    if undirected and edge_index.shape[1] > 0:
        edge_index = to_undirected(edge_index, num_nodes=n_nodes)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        orig_idx=torch.from_numpy(orig_idx).to(torch.long),
        time_step=time_step_tensor,
    )
    return data

