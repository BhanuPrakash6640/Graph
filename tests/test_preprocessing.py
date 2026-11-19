"""Phase 2 tests: chronological splits, train-only scaling, leakage guards."""
import numpy as np
import pytest

from ml.data.loader import FEATURE_COLS
from ml.preprocessing.build import build
from ml.preprocessing.dataset import (
    LeakageError,
    edges_for_steps,
    load_processed,
    prepare_split,
)
from ml.preprocessing.scaler import StepScaler
from ml.preprocessing.splits import Split, all_splits
from tests.conftest import write_synthetic

CFG = {
    "protocols": {
        "paper": {"train_steps": [1, 6], "val_steps": [5, 6], "test_steps": [7, 10]},
        "forward_chaining": {"cuts": [6, 8], "val_width": 2, "test_width": 2},
    }
}


@pytest.fixture
def built(tmp_path):
    raw = tmp_path / "raw"
    write_synthetic(raw, n_steps=10, per_step=40)
    out = tmp_path / "processed"
    build(raw, out, CFG, expected_steps=10)
    return load_processed(out)


# ---------- splits ----------

def test_real_config_splits_are_chronological():
    cfg = {"protocols": {
        "paper": {"train_steps": [1, 34], "val_steps": [30, 34], "test_steps": [35, 49]},
        "forward_chaining": {"cuts": [24, 29, 34, 39, 44], "val_width": 5, "test_width": 5}}}
    splits = {s.name: s for s in all_splits(cfg)}
    p = splits["paper"]
    assert (p.train_steps[0], p.train_steps[-1]) == (1, 29)
    assert (p.val_steps[0], p.val_steps[-1]) == (30, 34)
    assert (p.test_steps[0], p.test_steps[-1]) == (35, 49)
    assert splits["paper_refit"].train_steps == tuple(range(1, 35))
    assert splits["fc_44"].test_steps == tuple(range(45, 50))
    for s in splits.values():
        assert max(s.train_steps) < min(s.test_steps)


def test_overlapping_split_rejected():
    with pytest.raises(ValueError):
        Split("bad", (1, 2, 3), (3, 4), (5,))


def test_unordered_split_rejected():
    with pytest.raises(ValueError):
        Split("bad", (5, 6), (1, 2), (7,))


# ---------- scaler ----------

def test_scaler_ignores_non_train_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    steps = np.repeat(np.arange(1, 11), 10)
    a = StepScaler.fit(X, steps, range(1, 6), ["a", "b", "c"])
    X2 = X.copy()
    X2[steps > 5] *= 1e6  # wreck the future
    b = StepScaler.fit(X2, steps, range(1, 6), ["a", "b", "c"])
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.std, b.std)


def test_constant_feature_does_not_divide_by_zero():
    X = np.ones((10, 2), dtype=np.float32)
    s = StepScaler.fit(X, np.ones(10), [1], ["a", "b"])
    assert np.isfinite(s.transform(X)).all()


# ---------- built artefacts ----------

def test_build_outputs(built):
    assert built.X_raw.shape == (400, len(FEATURE_COLS))
    assert built.edge_index.shape[0] == 2
    assert set(built.splits) == {"paper", "paper_refit", "fc_6", "fc_8"}
    for name in built.splits:
        assert built.scaler_path(name).exists()


def test_train_features_standardised_test_not_refit(built):
    v = prepare_split(built, "paper")
    tr_rows = v.node_mask_for_steps(v.split.train_steps)
    np.testing.assert_allclose(v.X[tr_rows].mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(v.X[tr_rows].std(axis=0), 1, atol=1e-3)


def test_masks_exclude_unknown_and_are_disjoint(built):
    v = prepare_split(built, "paper")
    for m in (v.train_mask, v.val_mask, v.test_mask):
        assert (v.y[m] != -1).all()
    assert not (v.train_mask & v.val_mask).any()
    assert not (v.train_mask & v.test_mask).any()
    assert not (v.val_mask & v.test_mask).any()
    assert set(np.unique(v.time_step[v.test_mask])) <= set(v.split.test_steps)


def test_feature_sets(built):
    assert prepare_split(built, "paper", "local").X.shape[1] == 93
    assert prepare_split(built, "paper", "all").X.shape[1] == 165


def test_leakage_guard_rejects_scaler_fitted_on_test(built):
    bad = StepScaler.fit(built.X_raw, built.time_step, range(1, 11), FEATURE_COLS)
    bad.save(built.scaler_path("paper"))
    with pytest.raises(LeakageError):
        prepare_split(built, "paper")


def test_train_graph_has_no_test_nodes(built):
    v = prepare_split(built, "paper")
    e = edges_for_steps(v.edge_index, v.time_step, v.split.train_steps)
    assert e.shape[1] > 0
    assert set(np.unique(v.time_step[e.ravel()])) <= set(v.split.train_steps)
    # every original within-train edge is retained
    full_train = np.isin(v.time_step[v.edge_index[0]], v.split.train_steps)
    assert e.shape[1] == int(full_train.sum())


def test_edge_index_maps_back_to_txids(built, tmp_path):
    import pandas as pd
    raw_edges = pd.read_csv(tmp_path / "raw" / "elliptic_txs_edgelist.csv")
    src = built.tx_ids[built.edge_index[0]]
    dst = built.tx_ids[built.edge_index[1]]
    np.testing.assert_array_equal(src, raw_edges["txId1"].to_numpy())
    np.testing.assert_array_equal(dst, raw_edges["txId2"].to_numpy())
