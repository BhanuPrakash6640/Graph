"""Unit tests for Phase 6 Experiments (imbalance strategies and ablations)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.gnn.graph import build_period_graph
from ml.gnn.models import GCN
from ml.preprocessing.dataset import SplitView


class FocalLoss(nn.Module):
    """Focal Loss for binary classification logits: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: float = 0.1, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, 2), targets: (N,) in {0, 1}
        probs = F.softmax(logits, dim=-1)
        p_t = probs[torch.arange(len(targets)), targets]
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        log_p_t = torch.log(p_t.clamp(min=1e-8))
        loss = -focal_weight * log_p_t
        return loss.mean()


def test_focal_loss_forward_and_backward():
    loss_fn = FocalLoss(alpha=0.1, gamma=2.0)
    logits = torch.randn(20, 2, requires_grad=True)
    targets = torch.randint(0, 2, (20,))

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (20, 2)


def test_directed_vs_undirected_period_graph(tmp_path):
    from ml.preprocessing.build import build
    from ml.preprocessing.dataset import load_processed, prepare_split
    from tests.conftest import write_synthetic
    from tests.test_preprocessing import CFG

    raw_dir = tmp_path / "raw"
    write_synthetic(raw_dir, n_steps=10, per_step=30)
    proc_dir = tmp_path / "proc"
    build(raw_dir, proc_dir, CFG, expected_steps=10)
    data = load_processed(proc_dir)
    view = prepare_split(data, "paper", "local")

    g_undirected = build_period_graph(view, {1, 2}, undirected=True)
    g_directed = build_period_graph(view, {1, 2}, undirected=False)

    # Undirected should have >= edges than directed (symmetrized)
    assert g_undirected.edge_index.size(1) >= g_directed.edge_index.size(1)
    assert g_directed.num_nodes == g_undirected.num_nodes


def test_one_layer_vs_two_layer_gnn():
    """Verify 1-hop vs 2-hop GNN architectures."""
    # 2-layer standard GCN
    gcn2 = GCN(in_channels=16, hidden_channels=32, out_channels=2)
    x = torch.randn(10, 16)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out2 = gcn2(x, ei)
    assert out2.shape == (10, 2)

    # 1-layer GCN (1 hop of message passing)
    from torch_geometric.nn import GCNConv
    conv1 = GCNConv(16, 2)
    out1 = conv1(x, ei)
    assert out1.shape == (10, 2)
