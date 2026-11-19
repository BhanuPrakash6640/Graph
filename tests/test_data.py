import json

import numpy as np
import pandas as pd
import pytest

from ml.data import eda
from ml.data.loader import (
    FEATURE_COLS,
    SchemaError,
    feature_columns,
    load_elliptic,
)
from ml.data.validate import validate
from tests.conftest import write_synthetic


def test_shapes_and_columns(synthetic_raw):
    raw, tx, _, _ = synthetic_raw
    data = load_elliptic(raw)
    assert data.num_nodes == len(tx)
    assert list(data.nodes.columns[:3]) == ["txId", "time_step", "label"]
    assert len(FEATURE_COLS) == 165


def test_unknown_is_not_licit(synthetic_raw):
    raw, tx, _, classes = synthetic_raw
    data = load_elliptic(raw)
    lab = data.nodes.set_index("txId")["label"]
    raw_cls = pd.Series(classes, index=tx)
    assert (lab[raw_cls[raw_cls == "unknown"].index] == -1).all()
    assert (lab[raw_cls[raw_cls == "2"].index] == 0).all()
    assert (lab[raw_cls[raw_cls == "1"].index] == 1).all()


def test_time_step_never_a_feature():
    for fs in ("local", "aggregated", "all"):
        assert "time_step" not in feature_columns(fs)
    assert len(feature_columns("local")) == 93
    assert len(feature_columns("aggregated")) == 72


def test_validation_passes_on_clean_data(synthetic_raw):
    raw, *_ = synthetic_raw
    rep = validate(load_elliptic(raw), expected_steps=4)
    assert rep.ok, rep.errors
    assert rep.cross_time_step_edges == 0


def test_cross_step_edges_are_flagged(tmp_path):
    raw = tmp_path / "raw"
    write_synthetic(raw, cross_edges=3)
    rep = validate(load_elliptic(raw), expected_steps=4)
    assert rep.cross_time_step_edges == 3
    assert not rep.ok


def test_bad_class_value_rejected(synthetic_raw):
    raw, *_ = synthetic_raw
    p = raw / "elliptic_txs_classes.csv"
    df = pd.read_csv(p, dtype=str)
    df.loc[0, "class"] = "3"
    df.to_csv(p, index=False)
    with pytest.raises(SchemaError):
        load_elliptic(raw)


def test_wrong_column_count_rejected(synthetic_raw):
    raw, *_ = synthetic_raw
    p = raw / "elliptic_txs_features.csv"
    df = pd.read_csv(p, header=None).iloc[:, :-1]
    df.to_csv(p, header=False, index=False)
    with pytest.raises(SchemaError):
        load_elliptic(raw)


def test_eda_end_to_end(synthetic_raw, tmp_path):
    raw, tx, *_ = synthetic_raw
    out = tmp_path / "eda"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data: {{raw_dir: '{raw.as_posix()}'}}\n"
        "validation: {expected_time_steps: 4}\n"
        "protocols: {paper: {train_steps: [1, 2]}}\n"
        f"eda: {{out_dir: '{out.as_posix()}'}}\n"
    )
    assert eda.main(["--config", str(cfg)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["nodes"] == len(tx)
    assert summary["cross_time_step_edges"] == 0
    for f in ("fig_per_step.png", "fig_degrees.png", "fig_feature_shift.png", "per_step.csv"):
        assert (out / f).stat().st_size > 0
    per_step = pd.read_csv(out / "per_step.csv")
    assert per_step["total"].sum() == len(tx)
    assert np.allclose(per_step[["illicit", "licit", "unknown"]].sum(axis=1), per_step["total"])
