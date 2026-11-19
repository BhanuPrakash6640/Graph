"""Tests for EvolveGCN-O and Snapshot Context Temporal Models."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.gnn.models import GCN
from ml.temporal.evolvegcn import EvolveGCNConv, EvolveGCNO
from ml.temporal.snapshot_context import (
    CONTEXT_DIM,
    SnapshotContextGCN,
    SnapshotContextTracker,
    compute_snapshot_structural_stats,
)


def test_evolvegcn_forward_and_evolution():
    in_channels = 16
    hidden_channels = 32
    out_channels = 2
    model = EvolveGCNO(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.2,
        evolve=True,
    )

    w1, w2 = model.get_initial_weights()
    assert w1.shape == (in_channels, hidden_channels)
    assert w2.shape == (hidden_channels, out_channels)

    # Synthetic snapshot 1
    n_nodes_1 = 15
    x1 = torch.randn(n_nodes_1, in_channels)
    edge_index1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    # Step 1
    w1_t1, w2_t1 = model.evolve_weights(w1, w2)
    assert not torch.allclose(w1_t1, w1)
    out1 = model.forward_snapshot(x1, edge_index1, w1_t1, w2_t1)
    assert out1.shape == (n_nodes_1, out_channels)

    # Step 2
    n_nodes_2 = 25
    x2 = torch.randn(n_nodes_2, in_channels)
    edge_index2 = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    w1_t2, w2_t2 = model.evolve_weights(w1_t1, w2_t1)
    assert not torch.allclose(w1_t2, w1_t1)
    out2 = model.forward_snapshot(x2, edge_index2, w1_t2, w2_t2)
    assert out2.shape == (n_nodes_2, out_channels)

    # Optimization step: loss backward through sequential snapshots
    loss = out1[:5, 1].mean() + out2[:5, 1].mean()
    loss.backward()

    assert model.initial_w1.grad is not None
    assert model.recurrent_cell1.weight_hh.grad is not None
    assert model.bias1.grad is not None


def test_evolvegcn_frozen_ablation():
    """Verify that frozen mode (evolve=False) preserves static weights."""
    model = EvolveGCNO(in_channels=10, hidden_channels=20, evolve=False)
    w1, w2 = model.get_initial_weights()

    next_w1, next_w2 = model.evolve_weights(w1, w2)
    assert torch.equal(next_w1, w1)
    assert torch.equal(next_w2, w2)


def test_snapshot_context_tracker_leak_free():
    tracker = SnapshotContextTracker(default_prior=0.05)

    # Step 1 with no prior history
    ctx1 = tracker.get_context_vector(step=1, num_nodes=100, num_edges=200)
    assert ctx1.shape == (CONTEXT_DIM,)
    # Autoregressive signals should be equal to default prior
    assert np.allclose(ctx1[4:], 0.05)

    # Model predicts 0.12 on step 1
    tracker.update(step=1, mean_pred=0.12)

    # Step 2 context should incorporate step 1 prediction
    ctx2 = tracker.get_context_vector(step=2, num_nodes=80, num_edges=150)
    assert np.isclose(ctx2[4], 0.12)  # p1
    assert np.isclose(ctx2[5], 0.05)  # p2 (default)
    assert np.isclose(ctx2[7], 0.12)  # window mean

    # Model predicts 0.18 on step 2
    tracker.update(step=2, mean_pred=0.18)

    # Step 3 context
    ctx3 = tracker.get_context_vector(step=3, num_nodes=120, num_edges=300)
    assert np.isclose(ctx3[4], 0.18)  # p1
    assert np.isclose(ctx3[5], 0.12)  # p2
    assert np.isclose(ctx3[7], 0.15)  # mean of (0.18, 0.12)


def test_snapshot_context_gcn_forward():
    in_channels = 16
    hidden_channels = 32
    out_channels = 2
    model = SnapshotContextGCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        context_dim=CONTEXT_DIM,
    )

    n_nodes = 30
    x = torch.randn(n_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    context_vec = torch.randn(CONTEXT_DIM)

    # Forward with 1D context vector
    out = model(x, edge_index, context=context_vec)
    assert out.shape == (n_nodes, out_channels)

    # Loss and backward
    loss = out[:, 1].mean()
    loss.backward()
    assert model.conv1.lin.weight.grad is not None


def test_evolvegcn_gradient_flow_from_last_snapshot():
    """Verify that with full BPTT, loss on the final snapshot produces non-zero gradient on initial_w1/w2."""
    torch.manual_seed(42)
    in_channels = 8
    hidden_channels = 16
    out_channels = 2
    model = EvolveGCNO(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.0,
        evolve=True,
    )

    # Sequence of 10 snapshots (>= 2 windows)
    num_snapshots = 10
    w1, w2 = model.get_initial_weights()
    for _ in range(num_snapshots - 1):
        w1, w2 = model.evolve_weights(w1, w2)

    # Final snapshot
    w1, w2 = model.evolve_weights(w1, w2)
    x_last = torch.randn(15, in_channels)
    edge_index_last = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out_last = model.forward_snapshot(x_last, edge_index_last, w1, w2)

    # Loss computed ONLY on the last snapshot
    loss = out_last[:, 1].mean()
    loss.backward()

    assert model.initial_w1.grad is not None
    assert model.initial_w1.grad.abs().sum().item() > 0, "initial_w1 must receive non-zero gradient from last snapshot"
    assert model.initial_w2.grad is not None
    assert model.initial_w2.grad.abs().sum().item() > 0, "initial_w2 must receive non-zero gradient from last snapshot"


def test_frozen_evolvegcn_matches_static_gcn():
    """Verify that EvolveGCNO(evolve=False) and static 2-layer GCN with same init produce identical logits."""
    torch.manual_seed(42)
    in_channels = 16
    hidden_channels = 32
    out_channels = 2

    # Frozen EvolveGCNO
    evolve_model = EvolveGCNO(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.0,
        evolve=False,
    )
    evolve_model.eval()

    # Static GCN
    static_model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=0.0,
    )
    static_model.eval()

    # Copy weights from evolve_model to static_model
    # PyG GCNConv.lin.weight is (out_features, in_features)
    # EvolveGCNO.initial_w1 is (in_features, out_features)
    with torch.no_grad():
        static_model.conv1.lin.weight.copy_(evolve_model.initial_w1.t())
        static_model.conv1.bias.copy_(evolve_model.bias1)
        static_model.conv2.lin.weight.copy_(evolve_model.initial_w2.t())
        static_model.conv2.bias.copy_(evolve_model.bias2)

    # Test snapshot
    x = torch.randn(25, in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]], dtype=torch.long)

    out_static = static_model(x, edge_index)
    w1, w2 = evolve_model.get_initial_weights()
    w1, w2 = evolve_model.evolve_weights(w1, w2)
    out_evolve = evolve_model.forward_snapshot(x, edge_index, w1, w2)

    diff = (out_static - out_evolve).abs().max().item()
    assert torch.allclose(out_static, out_evolve, atol=1e-5), f"Logits mismatch, max diff: {diff}"
