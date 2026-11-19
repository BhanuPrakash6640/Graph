"""Statistical comparisons between models and feature configurations.

Loads predictions from experiments/results/<experiment>/, averages scores per node
across random seeds for a given configuration, aligns test nodes, and computes
paired bootstrap 95% confidence intervals on PR-AUC differences (AGENTS.md rule 7).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.data.loader import UNKNOWN
from ml.evaluation.metrics import paired_bootstrap_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.evaluation.compare")


import yaml

from ml.data.loader import UNKNOWN
from ml.evaluation.metrics import paired_block_bootstrap_diff, paired_bootstrap_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.evaluation.compare")


def load_test_predictions(
    results_dir: Path,
    model: str,
    split: str,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find all runs for (model, split, feature_set), load test predictions, and average scores per test node.

    Returns:
        (y_test, mean_scores, time_steps) sorted by node_idx.
    """
    pred_files: list[Path] = []

    for cfg_file in sorted(results_dir.rglob("config.yaml")):
        if "_archive" in cfg_file.parts:
            continue
        d = cfg_file.parent
        try:
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f)
            if (
                cfg.get("model") == model
                and cfg.get("split") == split
                and cfg.get("feature_set") == feature_set
            ):
                pq = d / "predictions.parquet"
                if pq.exists():
                    pred_files.append(pq)
        except Exception:
            continue

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found for model={model}, split={split}, feature_set={feature_set} in {results_dir}"
        )

    dfs: list[pd.DataFrame] = []
    for f in pred_files:
        df = pd.read_parquet(f)
        # Select labelled test nodes only
        test_df = df[df["part"] == "test"].copy()
        test_df = test_df[test_df["y"] != UNKNOWN]
        dfs.append(test_df[["node_idx", "y", "score", "time_step"]])

    all_preds = pd.concat(dfs, ignore_index=True)
    # Average score per node across seeds
    grouped = all_preds.groupby(["node_idx", "y", "time_step"], as_index=False)["score"].mean()
    grouped = grouped.sort_values("node_idx").reset_index(drop=True)

    y_test = grouped["y"].values.astype(np.int8)
    scores = grouped["score"].values.astype(np.float64)
    time_steps = grouped["time_step"].values.astype(np.int64)
    return y_test, scores, time_steps


def compare_configurations(
    results_dir: Path,
    comparisons: list[dict[str, Any]],
    n_bootstraps: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Run paired block-bootstrap (primary) and node-level bootstrap (secondary) comparisons."""
    results = []

    for comp in comparisons:
        name = comp["name"]
        split = comp["split"]
        cfg_a = comp["cfg_a"]  # Baseline / control
        cfg_b = comp["cfg_b"]  # Challenger / treatment

        y_a, s_a, steps_a = load_test_predictions(results_dir, cfg_a["model"], split, cfg_a["feature_set"])
        y_b, s_b, steps_b = load_test_predictions(results_dir, cfg_b["model"], split, cfg_b["feature_set"])

        assert len(y_a) == len(y_b), f"Length mismatch: {len(y_a)} vs {len(y_b)}"
        assert np.array_equal(y_a, y_b), "Test node labels do not match between configurations"
        assert np.array_equal(steps_a, steps_b), "Test node time steps do not match between configurations"

        pr_auc_a = float(average_precision_score(y_a, s_a))
        pr_auc_b = float(average_precision_score(y_b, s_b))

        # 1. Primary: Block bootstrap over time steps (AGENTS.md rule 7)
        mean_diff, block_lo, block_hi = paired_block_bootstrap_diff(
            y=y_a, s_a=s_b, s_b=s_a, groups=steps_a, metric_fn=average_precision_score, n=n_bootstraps, seed=seed
        )

        # 2. Secondary: Node-level bootstrap
        _, node_lo, node_hi = paired_bootstrap_diff(
            y=y_a, s_a=s_b, s_b=s_a, metric_fn=average_precision_score, n=1000, seed=seed
        )

        # 3. Per-step win count
        unique_steps = np.sort(np.unique(steps_a))
        k = 0
        n_eval_steps = 0
        for step in unique_steps:
            mask = steps_a == step
            y_step = y_a[mask]
            if (y_step == 1).sum() > 0 and (y_step == 0).sum() > 0:
                pr_a_step = float(average_precision_score(y_step, s_a[mask]))
                pr_b_step = float(average_precision_score(y_step, s_b[mask]))
                if pr_b_step > pr_a_step:
                    k += 1
                n_eval_steps += 1

        b_better_str = f"{k}/{n_eval_steps} steps"

        if block_lo > 0:
            conclusion = "Significant (B > A)"
        elif block_hi < 0:
            conclusion = "Significant (A > B)"
        else:
            conclusion = "No significant difference"

        if len(unique_steps) < 8:
            conclusion += " (weak evidence: <8 steps)"

        results.append({
            "Comparison": name,
            "Split": split,
            "Config A (Baseline)": f"{cfg_a['model'].upper()} ({cfg_a['feature_set']})",
            "Config B (Challenger)": f"{cfg_b['model'].upper()} ({cfg_b['feature_set']})",
            "PR-AUC (A)": f"{pr_auc_a:.4f}",
            "PR-AUC (B)": f"{pr_auc_b:.4f}",
            "Diff (B - A)": f"{mean_diff:+.4f}",
            "Block 95% CI (Primary)": f"[{block_lo:+.4f}, {block_hi:+.4f}]",
            "Node 95% CI (Secondary)": f"[{node_lo:+.4f}, {node_hi:+.4f}]",
            "B better on k/n steps": b_better_str,
            "Significance (Block CI)": conclusion,
        })

    return pd.DataFrame(results)


def get_comparisons(suite: str = "exp1") -> list[dict[str, Any]]:
    if suite == "exp1":
        return [
            # --- Tabular Baselines Comparisons ---
            {
                "name": "RF: Local vs All features",
                "split": "paper_refit",
                "cfg_a": {"model": "rf", "feature_set": "local"},
                "cfg_b": {"model": "rf", "feature_set": "all"},
            },
            {
                "name": "XGB: Local vs All features",
                "split": "paper_refit",
                "cfg_a": {"model": "xgb", "feature_set": "local"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "RF vs XGB (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "rf", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "RF vs XGB (darknet shutdown / fc_44)",
                "split": "fc_44",
                "cfg_a": {"model": "rf", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            # --- Tabular vs Best GNN Comparisons (paper_refit) ---
            {
                "name": "Best Tabular (XGB) vs Best GNN (SAGE-mean)",
                "split": "paper_refit",
                "cfg_a": {"model": "sage_mean", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "Best Tabular (XGB) vs GCN",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "Best Tabular (XGB) vs GAT",
                "split": "paper_refit",
                "cfg_a": {"model": "gat", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            # --- Neural Architecture vs Graph Inductive Bias ---
            {
                "name": "MLP control vs GCN (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "mlp", "feature_set": "all"},
                "cfg_b": {"model": "gcn", "feature_set": "all"},
            },
            {
                "name": "MLP control vs SAGE-mean (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "mlp", "feature_set": "all"},
                "cfg_b": {"model": "sage_mean", "feature_set": "all"},
            },
            # --- Feature Set Impact on GNNs ---
            {
                "name": "GCN: Local vs All features",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "local"},
                "cfg_b": {"model": "gcn", "feature_set": "all"},
            },
            {
                "name": "SAGE-mean: Local vs All features",
                "split": "paper_refit",
                "cfg_a": {"model": "sage_mean", "feature_set": "local"},
                "cfg_b": {"model": "sage_mean", "feature_set": "all"},
            },
        ]
    elif suite == "exp2":
        return [
            # --- Exp 2 Temporal Comparisons ---
            {
                "name": "Static GCN vs EvolveGCN-Frozen (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "evolvegcn_frozen", "feature_set": "all"},
            },
            {
                "name": "Static GCN vs EvolveGCN-O (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "evolvegcn", "feature_set": "all"},
            },
            {
                "name": "EvolveGCN-O vs EvolveGCN-Frozen (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "evolvegcn", "feature_set": "all"},
                "cfg_b": {"model": "evolvegcn_frozen", "feature_set": "all"},
            },
            {
                "name": "Static GCN vs Control B (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "control_b", "feature_set": "all"},
            },
            {
                "name": "EvolveGCN-O vs Control B (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "evolvegcn", "feature_set": "all"},
                "cfg_b": {"model": "control_b", "feature_set": "all"},
            },
            {
                "name": "Static GCN vs EvolveGCN-O (local features)",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "local"},
                "cfg_b": {"model": "evolvegcn", "feature_set": "local"},
            },
            {
                "name": "Static GCN vs Control B (local features)",
                "split": "paper_refit",
                "cfg_a": {"model": "gcn", "feature_set": "local"},
                "cfg_b": {"model": "control_b", "feature_set": "local"},
            },
            {
                "name": "Best Tabular (XGB) vs EvolveGCN-O (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "evolvegcn", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "Best Tabular (XGB) vs Control B (all features)",
                "split": "paper_refit",
                "cfg_a": {"model": "control_b", "feature_set": "all"},
                "cfg_b": {"model": "xgb", "feature_set": "all"},
            },
            {
                "name": "Static GCN vs EvolveGCN-O (darknet shutdown / fc_44)",
                "split": "fc_44",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "evolvegcn", "feature_set": "all"},
            },
            {
                "name": "Static GCN vs Control B (darknet shutdown / fc_44)",
                "split": "fc_44",
                "cfg_a": {"model": "gcn", "feature_set": "all"},
                "cfg_b": {"model": "control_b", "feature_set": "all"},
            },
        ]
    else:
        raise ValueError(f"Unknown suite: {suite}")


def main():
    parser = argparse.ArgumentParser(description="Compute paired bootstrap statistical comparisons")
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Path to results directory containing run folders",
    )
    parser.add_argument(
        "--suite",
        default="exp1",
        choices=["exp1", "exp2"],
        help="Comparison suite to run (exp1 or exp2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    comparisons = get_comparisons(args.suite)
    df = compare_configurations(results_dir, comparisons)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"reports/tables/{args.suite}_comparisons.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_markdown(out_path, index=False)
    logger.info("Saved comparisons to %s:\n%s", out_path, df.to_markdown(index=False))


if __name__ == "__main__":
    main()

