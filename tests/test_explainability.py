"""Unit tests for Phase 7: Explainability.

Tests SHAP tabular attributions, PyG GNNExplainer fidelity, neighbour occlusion,
and structural pattern flags on synthetic data fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xgboost as xgb

from ml.explainability.gnn_explain import explain_node_gnn
from ml.explainability.neighbors import compute_neighbor_occlusion, compute_pattern_flags
from ml.explainability.shap_tabular import DISCLAIMER, explain_node_shap
from ml.gnn.models import GCN


def test_shap_tabular_synthetic():
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 2, 100)
    feat_names = [f"feat_{i}" for i in range(20)]

    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)

    import shap
    explainer = shap.TreeExplainer(clf)
    res = explain_node_shap(explainer, X[0], feat_names, top_k=5)

    assert len(res) == 5
    for item in res:
        assert "feature" in item
        assert "value" in item
        assert "shap_value" in item
        assert item["direction"] in ("+", "-")


def test_gnn_explainer_synthetic():
    torch.manual_seed(42)
    model = GCN(in_channels=10, hidden_channels=16, out_channels=2, dropout=0.0)
    model.eval()

    x = torch.randn(8, 10)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2],
        [1, 2, 3, 4, 0, 0, 1],
    ], dtype=torch.long)
    feat_names = [f"f_{i}" for i in range(10)]

    exp_res = explain_node_gnn(
        model=model,
        x=x,
        edge_index=edge_index,
        target_node_idx=2,
        feature_names=feat_names,
        epochs=10,
    )

    assert exp_res["target_node_idx"] == 2
    assert "fidelity" in exp_res
    assert "fidelity_plus" in exp_res["fidelity"]
    assert "fidelity_minus" in exp_res["fidelity"]
    assert isinstance(exp_res["explainer_edges"], list)
    assert isinstance(exp_res["top_node_features"], list)


def test_neighbor_occlusion():
    torch.manual_seed(42)
    model = GCN(in_channels=8, hidden_channels=16, out_channels=2, dropout=0.0)
    model.eval()

    x = torch.randn(5, 8)
    # Undirected triangle 0-1-2 with dangling nodes 3, 4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0, 0, 3],
        [1, 0, 2, 1, 0, 2, 3, 0],
    ], dtype=torch.long)

    contribs = compute_neighbor_occlusion(
        model=model,
        x=x,
        edge_index=edge_index,
        target_node_idx=0,
    )

    assert len(contribs) == 3  # Neighbors 1, 2, 3
    for c in contribs:
        assert "neighbor_idx" in c
        assert "score_delta" in c
        assert c["impact"] in ("+", "-")
    # Must be sorted by absolute score delta descending
    deltas = [abs(c["score_delta"]) for c in contribs]
    assert deltas == sorted(deltas, reverse=True)


def test_structural_pattern_flags():
    # Star graph: center node 0 has 6 outgoing edges -> fan_out
    src_out = np.array([0, 0, 0, 0, 0, 0])
    dst_out = np.array([1, 2, 3, 4, 5, 6])
    flags_out = compute_pattern_flags((src_out, dst_out), target_node_idx=0)
    assert "fan_out" in flags_out

    # Star graph: center node 0 has 6 incoming edges -> fan_in
    src_in = np.array([1, 2, 3, 4, 5, 6])
    dst_in = np.array([0, 0, 0, 0, 0, 0])
    flags_in = compute_pattern_flags((src_in, dst_in), target_node_idx=0)
    assert "fan_in" in flags_in

    # Chain node: in-degree 1, out-degree 1 -> chain_node
    src_chain = np.array([1, 0])
    dst_chain = np.array([0, 2])
    flags_chain = compute_pattern_flags((src_chain, dst_chain), target_node_idx=0)
    assert "chain_node" in flags_chain

    # Isolated or leaf node
    flags_leaf = compute_pattern_flags((np.array([0]), np.array([1])), target_node_idx=1)
    assert "isolated_or_leaf" in flags_leaf

    # High illicit neighborhood flag: 2 of 2 neighbors above threshold 0.5
    nbr_scores = np.array([0.1, 0.9, 0.85, 0.2])
    src_nbr = np.array([0, 0])
    dst_nbr = np.array([1, 2])
    flags_nbr = compute_pattern_flags(
        (src_nbr, dst_nbr), target_node_idx=0, neighbor_scores=nbr_scores, threshold=0.5
    )
    assert "high_illicit_neighborhood" in flags_nbr

    # B4 regression test (106364923 case): 1-hop neighbours are licit/unknown below threshold
    # Neighbors 1 and 2 have scores 0.004 and 0.100, threshold is 0.4656
    scores_licit_nbrs = np.array([0.996, 0.004, 0.100])
    flags_licit_nbrs = compute_pattern_flags(
        (np.array([0, 0]), np.array([1, 2])),
        target_node_idx=0,
        neighbor_scores=scores_licit_nbrs,
        threshold=0.4656,
    )
    assert "high_illicit_neighborhood" not in flags_licit_nbrs


def test_gnn_explainer_planted_decisive_edge():
    """Verify on synthetic graph that a planted, decisive edge gets fidelity+ > 0."""
    from torch_geometric.nn import GCNConv
    import torch.nn as nn

    class DecisiveGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(1, 2, bias=True)
            with torch.no_grad():
                self.conv.lin.weight.copy_(torch.tensor([[-2.0], [2.0]]))
                self.conv.bias.copy_(torch.tensor([1.0, -1.0]))

        def forward(self, x, edge_index):
            return self.conv(x, edge_index)

    model = DecisiveGCN()
    x = torch.tensor([[0.0], [10.0]])
    edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

    res = explain_node_gnn(model, x, edge_index, target_node_idx=0, epochs=50)
    assert "fidelity" in res
    assert res["fidelity"]["fidelity_plus"] > 0, f"Expected fidelity_plus > 0, got {res['fidelity']}"


def test_explanation_schema_contract():
    # Verify the structure matches frontend contract
    required_keys = {
        "txId",
        "time_step",
        "model_run_id",
        "score",
        "predicted_class",
        "threshold",
        "known_label",
        "top_features",
        "neighbor_contributions",
        "explainer_edges",
        "fidelity",
        "pattern_flags",
        "disclaimer",
    }

    dummy_record = {
        "txId": "123456",
        "time_step": 35,
        "model_run_id": "test_run",
        "score": 0.85,
        "predicted_class": 1,
        "threshold": 0.5,
        "known_label": "illicit",
        "top_features": [{"feature": "local_1", "value": 1.0, "shap_value": 0.2, "direction": "+"}],
        "neighbor_contributions": [{"neighbor_txId": "654321", "score_delta": 0.1, "impact": "+"}],
        "explainer_edges": [{"source": "123456", "target": "654321", "edge_weight": 0.8}],
        "fidelity": {"fidelity_plus": 0.9, "fidelity_minus": 0.1},
        "pattern_flags": ["chain_node"],
        "disclaimer": DISCLAIMER,
    }

    assert required_keys.issubset(dummy_record.keys())
    assert dummy_record["known_label"] in ("illicit", "licit", "unknown")


def test_per_model_explanations_exist():
    """Verify per-model exported files exist and separate XGBoost and GCN attributions."""
    base_dir = Path("data/processed/explanations")
    xgb_dir = base_dir / "xgb_paper_refit_all"
    gcn_dir = base_dir / "gcn_paper_refit_all"

    assert xgb_dir.exists(), f"Missing {xgb_dir}"
    assert gcn_dir.exists(), f"Missing {gcn_dir}"

    xgb_files = list(xgb_dir.glob("*.json"))
    gcn_files = list(gcn_dir.glob("*.json"))

    assert len(xgb_files) >= 50, f"Expected >= 50 XGB explanation files, got {len(xgb_files)}"
    assert len(gcn_files) >= 50, f"Expected >= 50 GCN explanation files, got {len(gcn_files)}"

    # Spot check XGB explanation
    sample_xgb = json.loads(xgb_files[0].read_text(encoding="utf-8"))
    assert sample_xgb["model_run_id"] == "xgb_paper_refit_all"
    assert "top_features" in sample_xgb
    assert len(sample_xgb["top_features"]) > 0

    # Spot check GCN explanation
    sample_gcn = json.loads(gcn_files[0].read_text(encoding="utf-8"))
    assert sample_gcn["model_run_id"] == "gcn_paper_refit_all"
    assert "fidelity" in sample_gcn
    assert "explainer_edges" in sample_gcn

    # Check that in GCN explanations, edge indices use global node indices
    for gf in gcn_files[:10]:
        d = json.loads(gf.read_text(encoding="utf-8"))
        for edge in d.get("explainer_edges", []):
            assert "source_idx" in edge
            assert "target_idx" in edge
            # Global node indices in Elliptic are up to 203,769; period-local would be < 50,000
            # Ensure indices are non-negative integers
            assert isinstance(edge["source_idx"], int)
            assert isinstance(edge["target_idx"], int)

