"""Build processed artefacts from the raw Elliptic CSVs.

Usage:
    python -m ml.preprocessing.build --config experiments/configs/data.yaml

Writes to ``data.processed_dir``:
    nodes.parquet        txId, time_step, label, 165 raw features (row order = node index)
    edge_index.npy       (2, E) int64 node indices
    splits.json          all split definitions
    scalers/<split>.npz  StepScaler fitted on that split's training steps only
    split_summary.csv    labelled / illicit counts per split part
    manifest.json        raw file sha256, sizes, counts, build time

Unscaled features are stored once; scaling is applied per split at load time
by ``ml.preprocessing.dataset.prepare_split`` so no test-period statistic can
be baked into a shared file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.loader import CLASSES_FILE, EDGES_FILE, FEATURE_COLS, FEATURES_FILE, load_elliptic
from ml.data.validate import validate
from ml.preprocessing.dataset import load_processed, split_summary
from ml.preprocessing.scaler import StepScaler
from ml.preprocessing.splits import all_splits, save_splits
from ml.utils.config import load_config


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build(raw_dir: Path, out_dir: Path, cfg: dict, expected_steps: int | None = 49) -> dict:
    t0 = time.perf_counter()
    data = load_elliptic(raw_dir)
    rep = validate(data, expected_steps=expected_steps)
    if not rep.ok:
        raise RuntimeError(f"validation failed: {rep.errors}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scalers").mkdir(exist_ok=True)

    nodes = data.nodes.reset_index(drop=True)
    nodes.to_parquet(out_dir / "nodes.parquet", index=False)

    idx = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes["txId"].to_numpy())
    edge_index = np.vstack([
        idx.loc[data.edges["src"].to_numpy()].to_numpy(),
        idx.loc[data.edges["dst"].to_numpy()].to_numpy(),
    ]).astype(np.int64)
    np.save(out_dir / "edge_index.npy", edge_index)

    max_step = int(nodes["time_step"].max())
    splits = all_splits(cfg, max_step=max_step)
    save_splits(splits, out_dir / "splits.json")

    X = nodes[FEATURE_COLS].to_numpy(np.float32)
    steps = nodes["time_step"].to_numpy()
    for s in splits:
        StepScaler.fit(X, steps, s.train_steps, FEATURE_COLS).save(out_dir / "scalers" / f"{s.name}.npz")

    summary = split_summary(load_processed(out_dir))
    pd.DataFrame(summary).to_csv(out_dir / "split_summary.csv", index=False)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "python": platform.python_version(),
        "raw_files": {
            n: {"sha256": sha256(raw_dir / n), "bytes": (raw_dir / n).stat().st_size}
            for n in (FEATURES_FILE, EDGES_FILE, CLASSES_FILE)
        },
        "num_nodes": int(len(nodes)),
        "num_edges": int(edge_index.shape[1]),
        "num_features": len(FEATURE_COLS),
        "splits": [s.name for s in splits],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "split_summary": summary}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/configs/data.yaml")
    p.add_argument("--raw-dir")
    p.add_argument("--out-dir")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    raw = Path(args.raw_dir or cfg["data"]["raw_dir"])
    out = Path(args.out_dir or cfg["data"]["processed_dir"])
    res = build(raw, out, cfg, expected_steps=cfg["validation"]["expected_time_steps"] or None)
    print(json.dumps(res["manifest"], indent=2))
    print(pd.DataFrame(res["split_summary"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
