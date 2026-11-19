"""Snapshot Context Control Model (Control B).

Implements static GCN augmented with leak-free per-snapshot temporal context features:
1. Unlabelled structural graph statistics of the current snapshot:
   - log(1 + N_t)
   - log(1 + E_t)
   - Edge density: 2 * E_t / (N_t * (N_t - 1) + 1e-8)
   - Mean degree: 2 * E_t / (N_t + 1e-8)
2. Autoregressive prediction signals from the model's own predictions on past snapshots:
   - Mean predicted illicit score of snapshot t - 1
   - Mean predicted illicit score of snapshot t - 2
   - Mean predicted illicit score of snapshot t - 3
   - Window average of available past mean predictions [t - 3 .. t - 1]

Ground-truth labels of current, val, or test snapshots are NEVER accessed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


CONTEXT_DIM = 8


def compute_snapshot_structural_stats(num_nodes: int, num_edges: int) -> np.ndarray:
    """Compute 4 unlabelled structural stats for a snapshot."""
    log_nodes = float(np.log1p(max(0, num_nodes)))
    log_edges = float(np.log1p(max(0, num_edges)))
    density = float((2.0 * num_edges) / (max(1, num_nodes * (num_nodes - 1)) + 1e-8))
    mean_deg = float((2.0 * num_edges) / (max(1, num_nodes) + 1e-8))
    return np.array([log_nodes, log_edges, density, mean_deg], dtype=np.float32)


class SnapshotContextTracker:
    """Tracks model's own past predictions to build context features without label leakage."""

    def __init__(self, default_prior: float = 0.05) -> None:
        self.default_prior = default_prior
        self.history: dict[int, float] = {}

    def reset(self) -> None:
        self.history.clear()

    def update(self, step: int, mean_pred: float) -> None:
        """Record model's own mean prediction on snapshot step."""
        self.history[int(step)] = float(mean_pred)

    def get_context_vector(self, step: int, num_nodes: int, num_edges: int) -> np.ndarray:
        """Build the 8-dim context vector for snapshot step."""
        # 4 structural stats
        struct_stats = compute_snapshot_structural_stats(num_nodes, num_edges)

        # 4 autoregressive prediction signals
        p1 = self.history.get(step - 1, self.default_prior)
        p2 = self.history.get(step - 2, self.default_prior)
        p3 = self.history.get(step - 3, self.default_prior)

        past_vals = [self.history[s] for s in (step - 1, step - 2, step - 3) if s in self.history]
        p_win = float(np.mean(past_vals)) if past_vals else self.default_prior

        pred_stats = np.array([p1, p2, p3, p_win], dtype=np.float32)
        return np.concatenate([struct_stats, pred_stats], axis=0)


class SnapshotContextGCN(nn.Module):
    """Static 2-layer GCN augmented with snapshot-level context vector."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        context_dim: int = CONTEXT_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        total_in = in_channels + context_dim
        self.conv1 = GCNConv(total_in, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, in_channels) or (N, in_channels + context_dim).
            edge_index: Graph edge index (2, E).
            context: Optional context tensor (context_dim,) or (N, context_dim).
        """
        if context is not None:
            if context.dim() == 1:
                context_expanded = context.unsqueeze(0).expand(x.size(0), -1)
            else:
                context_expanded = context
            x = torch.cat([x, context_expanded], dim=-1)

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, edge_index)
        return logits

