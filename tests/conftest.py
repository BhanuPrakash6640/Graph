"""Synthetic Elliptic-format fixture so tests run without the real dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.loader import N_RAW_COLUMNS


def write_synthetic(raw_dir, n_steps=4, per_step=30, seed=0, cross_edges=0):
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tx, steps = [], []
    next_id = 1000
    for s in range(1, n_steps + 1):
        ids = np.arange(next_id, next_id + per_step)
        next_id += per_step
        tx.append(ids)
        steps.append(np.full(per_step, s))
    tx = np.concatenate(tx)
    steps = np.concatenate(steps)
    feats = rng.normal(size=(len(tx), N_RAW_COLUMNS - 2)).round(5)
    pd.DataFrame(np.column_stack([tx, steps, feats])).to_csv(
        raw_dir / "elliptic_txs_features.csv", header=False, index=False,
        float_format="%.5f")
    # integer-format the id/step columns
    df = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df.to_csv(raw_dir / "elliptic_txs_features.csv", header=False, index=False)

    classes = rng.choice(["1", "2", "unknown"], size=len(tx), p=[0.05, 0.25, 0.70])
    pd.DataFrame({"txId": tx, "class": classes}).to_csv(
        raw_dir / "elliptic_txs_classes.csv", index=False)

    edges = []
    for s in range(1, n_steps + 1):
        ids = tx[steps == s]
        for _ in range(per_step):
            a, b = rng.choice(ids, 2, replace=False)
            if a > b:
                a, b = b, a
            edges.append((a, b))
    for _ in range(cross_edges):
        edges.append((tx[steps == 1][0], tx[steps == 2][0]))
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(
        raw_dir / "elliptic_txs_edgelist.csv", index=False)
    return tx, steps, classes


@pytest.fixture
def synthetic_raw(tmp_path):
    raw = tmp_path / "raw"
    tx, steps, classes = write_synthetic(raw)
    return raw, tx, steps, classes
