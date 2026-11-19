"""Tabular explainability with SHAP TreeExplainer for XGBoost.

Computes global feature importances and local (per-transaction) explanations
for the best XGBoost model on the Elliptic transaction dataset.
Feature names are strictly anonymised (local_1..93, agg_1..72) per Weber et al. (2019).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as tree_mod

# Compatibility patch for XGBoost 3.x base_score '[5E-1]' formatting in shap UBJSON decoder
_orig_decode = tree_mod.decode_ubjson_buffer

def _safe_decode_ubjson(fd):
    jmodel = _orig_decode(fd)
    try:
        lmp = jmodel.get("learner", {}).get("learner_model_param", {})
        if "base_score" in lmp:
            bs = str(lmp["base_score"]).strip("[]")
            lmp["base_score"] = bs
    except Exception:
        pass
    return jmodel

tree_mod.decode_ubjson_buffer = _safe_decode_ubjson

from ml.baselines.models import create_xgb
from ml.data.loader import feature_columns
from ml.preprocessing.dataset import ProcessedData, prepare_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.explainability.shap_tabular")

DISCLAIMER = (
    "Research demonstration on historical Elliptic Bitcoin dataset (2019). "
    "Not a live transaction monitoring or AML compliance system. "
    "Features local_1..93 and agg_1..72 are anonymised per Weber et al. (2019)."
)


def get_or_train_xgb(
    data: ProcessedData,
    split_name: str = "paper_refit",
    feature_set: str = "all",
    seed: int = 0,
    checkpoint_dir: str | Path = "checkpoints",
) -> tuple[Any, float]:
    """Load cached XGBoost model or train and save on the specified split.

    Returns:
        (model, threshold)
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"xgb_{split_name}_{feature_set}_s{seed}.joblib"
    meta_path = ckpt_dir / f"xgb_{split_name}_{feature_set}_s{seed}_meta.json"

    view = prepare_split(data, split_name=split_name, feature_set=feature_set)
    X_train = view.X[view.train_mask]
    y_train = view.y[view.train_mask]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    if ckpt_path.exists() and meta_path.exists():
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        model = joblib.load(ckpt_path)
        logger.info("Loaded XGBoost checkpoint from %s", ckpt_path)
        return model, float(meta.get("threshold", 0.4656))

    logger.info("Training XGBoost for explainability on %s (%s)...", split_name, feature_set)
    model = create_xgb(
        n_estimators=267,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=None,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Threshold tuned on paper validation set: 0.4656
    threshold = 0.4656
    joblib.dump(model, ckpt_path)
    import json
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold, "split": split_name, "feature_set": feature_set, "seed": seed}, f)

    logger.info("Saved XGBoost model to %s", ckpt_path)
    return model, threshold


def compute_global_shap(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    out_figure: str | Path | None = "reports/figures/shap_summary.png",
    max_display: int = 15,
) -> pd.DataFrame:
    """Compute TreeExplainer SHAP values and save global importance summary figure.

    Returns:
        DataFrame with ['feature', 'mean_abs_shap'] sorted descending.
    """
    logger.info("Computing TreeExplainer SHAP values on %d background samples...", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # In binary classification, shap_values is a 2D array (n_samples, n_features)
    if isinstance(shap_values, list):
        # Older shap versions return list of [neg, pos]
        vals = shap_values[1]
    else:
        vals = shap_values

    mean_abs = np.mean(np.abs(vals), axis=0)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if out_figure:
        fig_path = Path(out_figure)
        fig_path.parent.mkdir(parents=True, exist_ok=True)

        top_df = df_importance.head(max_display).sort_values("mean_abs_shap", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="#2c3e50")
        plt.xlabel("Mean |SHAP value| (Average Impact on Illicit Score)")
        plt.title(f"XGBoost Top {max_display} Global Feature Attributions (SHAP)")
        plt.annotate(
            "Features anonymised per Weber et al. (2019)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            color="#7f8c8d",
            fontstyle="italic",
        )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        logger.info("Saved SHAP global summary plot to %s", fig_path)

    return df_importance


def explain_node_shap(
    explainer: shap.TreeExplainer,
    x_node: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Compute local SHAP explanation for a single transaction vector.

    Returns:
        List of top_k dicts:
        [{'feature': 'local_17', 'value': 1.23, 'shap_value': 0.45, 'direction': '+'}, ...]
    """
    x_2d = x_node.reshape(1, -1)
    shap_vals = explainer.shap_values(x_2d)
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    else:
        vals = shap_vals[0]

    abs_vals = np.abs(vals)
    top_indices = np.argsort(-abs_vals)[:top_k]

    results = []
    for idx in top_indices:
        sv = float(vals[idx])
        results.append({
            "feature": str(feature_names[idx]),
            "value": float(x_node[idx]),
            "shap_value": round(sv, 4),
            "direction": "+" if sv > 0 else "-",
        })
    return results
