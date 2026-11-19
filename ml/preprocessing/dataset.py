"""Model-ready views of the processed data for a given split.

``prepare_split`` is the single entry point every model (tabular or graph)
uses, so all models see identical scaling, masks and feature sets.

Masks contain *labelled* nodes only. Unknown nodes (label -1) are kept in
``X`` and in the graph for message passing but never appear in any mask.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.loader import FEATURE_COLS, UNKNOWN, feature_columns
from ml.preprocessing.scaler import StepScaler
from ml.preprocessing.splits import Split, load_splits


class LeakageError(RuntimeError):
    pass


@dataclass
class ProcessedData:
    tx_ids: np.ndarray          # (N,) int64, row i <-> node index i
    time_step: np.ndarray       # (N,) int16
    y: np.ndarray               # (N,) int8 in {1, 0, -1}
    X_raw: np.ndarray           # (N, 165) float32, unscaled, columns = FEATURE_COLS
    edge_index: np.ndarray      # (2, E) int64 node indices, raw direction src -> dst
    splits: dict[str, Split]
    root: Path

    @property
    def num_nodes(self) -> int:
        return len(self.tx_ids)

    def scaler_path(self, split_name: str) -> Path:
        return self.root / "scalers" / f"{split_name}.npz"


def load_processed(root: str | Path) -> ProcessedData:
    root = Path(root)
    nodes = pd.read_parquet(root / "nodes.parquet")
    if list(nodes.columns[3:]) != FEATURE_COLS:
        raise ValueError("nodes.parquet feature columns do not match loader.FEATURE_COLS")
    return ProcessedData(
        tx_ids=nodes["txId"].to_numpy(np.int64),
        time_step=nodes["time_step"].to_numpy(np.int16),
        y=nodes["label"].to_numpy(np.int8),
        X_raw=nodes[FEATURE_COLS].to_numpy(np.float32),
        edge_index=np.load(root / "edge_index.npy"),
        splits=load_splits(root / "splits.json"),
        root=root,
    )


@dataclass
class SplitView:
    split: Split
    feature_set: str
    feature_names: list[str]
    X: np.ndarray               # (N, F) scaled with train-only statistics
    y: np.ndarray               # (N,) int8, -1 = unknown
    time_step: np.ndarray
    train_mask: np.ndarray      # labelled nodes in train steps
    val_mask: np.ndarray
    test_mask: np.ndarray
    edge_index: np.ndarray      # full graph; use edges_for_steps() to restrict

    def node_mask_for_steps(self, steps) -> np.ndarray:
        return np.isin(self.time_step, list(steps))


def edges_for_steps(edge_index: np.ndarray, time_step: np.ndarray, steps) -> np.ndarray:
    """Edges whose endpoints are both in ``steps``. Because snapshots are disjoint
    this equals the union of those snapshots' subgraphs."""
    keep = np.isin(time_step, list(steps))
    return edge_index[:, keep[edge_index[0]] & keep[edge_index[1]]]


def _labelled(y: np.ndarray, step: np.ndarray, steps) -> np.ndarray:
    return np.isin(step, list(steps)) & (y != UNKNOWN)


def prepare_split(data: ProcessedData, split_name: str, feature_set: str = "all") -> SplitView:
    split = data.splits[split_name]
    scaler = StepScaler.load(data.scaler_path(split_name))

    if not set(scaler.fit_steps) <= set(split.train_steps):
        raise LeakageError(
            f"scaler for {split_name} fitted on steps {sorted(set(scaler.fit_steps) - set(split.train_steps))} "
            "outside the training window")
    if tuple(scaler.feature_names) != tuple(FEATURE_COLS):
        raise ValueError("scaler feature names do not match FEATURE_COLS")

    X_all = scaler.transform(data.X_raw)
    cols = feature_columns(feature_set)
    idx = [FEATURE_COLS.index(c) for c in cols]

    return SplitView(
        split=split,
        feature_set=feature_set,
        feature_names=cols,
        X=np.ascontiguousarray(X_all[:, idx]),
        y=data.y,
        time_step=data.time_step,
        train_mask=_labelled(data.y, data.time_step, split.train_steps),
        val_mask=_labelled(data.y, data.time_step, split.val_steps),
        test_mask=_labelled(data.y, data.time_step, split.test_steps),
        edge_index=data.edge_index,
    )


def split_summary(data: ProcessedData) -> list[dict]:
    rows = []
    for name, s in data.splits.items():
        for part, steps in (("train", s.train_steps), ("val", s.val_steps), ("test", s.test_steps)):
            m = _labelled(data.y, data.time_step, steps)
            n_ill = int((data.y[m] == 1).sum())
            rows.append({
                "split": name, "part": part,
                "steps": f"{min(steps)}-{max(steps)}" if steps else "-",
                "labelled": int(m.sum()), "illicit": n_ill,
                "illicit_rate": round(n_ill / m.sum(), 4) if m.sum() else None,
                "all_nodes": int(np.isin(data.time_step, list(steps)).sum()),
            })
    return rows


def write_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
