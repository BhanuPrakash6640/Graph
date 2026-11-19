"""Tests for GNN graph construction, models, and training on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.data.loader import UNKNOWN
from ml.gnn.graph import build_period_graph
from ml.gnn.models import GAT, GCN, MLP, GraphSAGE, create_gnn_model
from ml.preprocessing.build import build
from ml.preprocessing.dataset import SplitView, load_processed, prepare_split
from tests.test_preprocessing import CFG


def test_build_period_graph_synthetic(tmp_path):
    from tests.conftest import write_synthetic

    raw_dir = tmp_path / "raw"
    write_synthetic(raw_dir, n_steps=10, per_step=40)
    proc_dir = tmp_path / "proc"
    build(raw_dir, proc_dir, CFG, expected_steps=10)
    data = load_processed(proc_dir)

    # Prepare split view
    view = prepare_split(data, "paper", "local")

    # Build train period graph (steps 1..34 in paper, but synthetic has 1..4)
    # Synthetic split: let's build for steps {1, 2}
    train_graph = build_period_graph(view, {1, 2}, undirected=True)

    # 1. Verify node count matches nodes in steps 1 and 2
    expected_nodes = np.isin(view.time_step, [1, 2])
    n_expected = int(expected_nodes.sum())
    assert train_graph.num_nodes == n_expected
    assert train_graph.x.shape[0] == n_expected
    assert train_graph.x.shape[1] == view.X.shape[1]

    # 2. Verify all time_steps are 1 or 2 (strictly no leak)
    ts = train_graph.time_step.numpy()
    assert np.all(np.isin(ts, [1, 2]))

    # 3. Verify edges are strictly in range [0, n_expected - 1]
    if train_graph.edge_index.shape[1] > 0:
        assert train_graph.edge_index.min() >= 0
        assert train_graph.edge_index.max() < n_expected

    # 4. Verify orig_idx correctly maps back to view
    orig = train_graph.orig_idx.numpy()
    np.testing.assert_array_equal(view.time_step[orig], ts)
    np.testing.assert_array_equal(view.y[orig], train_graph.y.numpy())


def test_gnn_models_forward():
    in_channels = 16
    n_nodes = 20
    x = torch.randn(n_nodes, in_channels)
    # Create simple ring edge_index
    src = torch.arange(n_nodes)
    dst = (src + 1) % n_nodes
    edge_index = torch.stack([src, dst], dim=0)

    models = [
        ("mlp", MLP(in_channels, hidden_channels=32, out_channels=2)),
        ("gcn", GCN(in_channels, hidden_channels=32, out_channels=2)),
        ("sage_mean", GraphSAGE(in_channels, hidden_channels=32, out_channels=2, aggr="mean")),
        ("sage_max", GraphSAGE(in_channels, hidden_channels=32, out_channels=2, aggr="max")),
        ("gat", GAT(in_channels, hidden_channels=16, out_channels=2, heads=2)),
    ]

    for name, model in models:
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (n_nodes, 2), f"{name} output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"{name} has NaN in output"

    # Test GAT attention weights output
    gat_model = GAT(in_channels, hidden_channels=16, out_channels=2, heads=2)
    gat_model.eval()
    out, att = gat_model(x, edge_index, return_attention_weights=True)
    assert out.shape == (n_nodes, 2)
    assert isinstance(att, tuple)
    att_edge_index, att_alpha = att
    assert att_edge_index.shape[0] == 2
    assert att_alpha.shape[0] == att_edge_index.shape[1]


def test_gnn_optimization_step():
    in_channels = 8
    n_nodes = 15
    x = torch.randn(n_nodes, in_channels)
    edge_index = torch.stack([torch.arange(n_nodes), (torch.arange(n_nodes) + 1) % n_nodes], dim=0)
    y = torch.randint(0, 2, (n_nodes,))

    model = GCN(in_channels, hidden_channels=16, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()

    # Check weights updated
    for param in model.parameters():
        assert param.grad is not None


def test_create_gnn_model_factory():
    for name in ["mlp", "gcn", "sage_mean", "sage_max", "gat"]:
        m = create_gnn_model(name, in_channels=10, hidden_channels=32, out_channels=2)
        assert isinstance(m, torch.nn.Module)

    with pytest.raises(ValueError):
        create_gnn_model("invalid_model", in_channels=10)
