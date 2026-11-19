"""Experiment 3: Generalisation Analysis Across Forward-Chaining Cuts & Time Steps.

Evaluates performance across chronological forward-chaining splits (fc_24, fc_29, fc_34, fc_39, fc_44),
per-step PR-AUC/F1 trajectories across steps 25-49, base rate overlays, and degradation
at the darknet marketplace shutdown (steps 43-46).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
import yaml

from ml.data.loader import UNKNOWN
from ml.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.experiments.exp3")


FC_SPLITS = ["fc_24", "fc_29", "fc_34", "fc_39", "fc_44"]
MODELS = ["xgb", "rf", "gcn", "sage_mean", "control_b", "evolvegcn"]


def load_fc_predictions(results_dir: Path, models: list[str]) -> pd.DataFrame:
    """Load all test predictions from completed runs for forward-chaining splits."""
    records = []
    for cfg_file in sorted(results_dir.rglob("config.yaml")):
        if "_archive" in cfg_file.parts:
            continue
        run_d = cfg_file.parent
        pq_file = run_d / "predictions.parquet"
        met_file = run_d / "metrics.json"
        if not (pq_file.exists() and met_file.exists()):
            continue

        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            model = cfg.get("model", "").lower()
            split = cfg.get("split", "")
            feat = cfg.get("feature_set", "")
            seed = cfg.get("seed", 0)

            if model in models and split in FC_SPLITS and feat == "all":
                with open(met_file, "r", encoding="utf-8") as f:
                    met = json.load(f)
                thr = met.get("best_threshold") or met.get("threshold_val_f1", 0.5)

                df = pd.read_parquet(pq_file)
                test_df = df[df["part"] == "test"].copy()
                test_df = test_df[test_df["y"] != UNKNOWN]

                test_df["model"] = model.upper()
                test_df["split"] = split
                test_df["seed"] = seed
                test_df["threshold"] = thr
                records.append(test_df)
        except Exception as e:
            logger.warning("Error reading %s: %e", run_d, e)

    if not records:
        raise FileNotFoundError("No matching forward-chaining prediction files found.")
    return pd.concat(records, ignore_index=True)


def compute_generalisation_table(all_preds: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Compute summary metrics per split and model across random seeds."""
    rows = []
    for split in FC_SPLITS:
        split_df = all_preds[all_preds["split"] == split]
        for model in sorted(split_df["model"].unique()):
            m_df = split_df[split_df["model"] == model]
            seed_praucs = []
            seed_f1s = []

            for s in sorted(m_df["seed"].unique()):
                s_df = m_df[m_df["seed"] == s]
                y_true = s_df["y"].values
                scores = s_df["score"].values
                thr = s_df["threshold"].iloc[0]

                if len(np.unique(y_true)) > 1:
                    pr_auc = average_precision_score(y_true, scores)
                    pred_cls = (scores >= thr).astype(int)
                    f1 = f1_score(y_true, pred_cls, zero_division=0)
                    seed_praucs.append(pr_auc)
                    seed_f1s.append(f1)

            if seed_praucs:
                mean_pr = np.mean(seed_praucs)
                std_pr = np.std(seed_praucs)
                mean_f1 = np.mean(seed_f1s)
                std_f1 = np.std(seed_f1s)
                rows.append({
                    "split": split,
                    "model": model,
                    "n_seeds": len(seed_praucs),
                    "pr_auc": f"{mean_pr:.4f} ± {std_pr:.4f}",
                    "f1": f"{mean_f1:.4f} ± {std_f1:.4f}",
                    "mean_pr_auc_val": mean_pr,
                    "mean_f1_val": mean_f1,
                })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp3_generalisation.csv", index=False)
    df[["split", "model", "n_seeds", "pr_auc", "f1"]].to_markdown(
        out_dir / "exp3_generalisation.md", index=False
    )
    return df


def generate_generalisation_plots(all_preds: pd.DataFrame, eda_per_step_path: Path, fig_dir: Path) -> None:
    """Generate publication-ready figures for Exp 3."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load base rates per step
    eda_df = pd.read_csv(eda_per_step_path)
    base_rates = dict(zip(eda_df["time_step"], eda_df["illicit_rate_labelled"]))
    n_illicit_map = dict(zip(eda_df["time_step"], eda_df["illicit"]))

    # 1. Performance across forward-chaining cuts
    summary = all_preds.groupby(["split", "model", "seed"]).apply(
        lambda g: pd.Series({
            "pr_auc": average_precision_score(g["y"], g["score"]) if len(np.unique(g["y"])) > 1 else 0.0,
            "f1": f1_score(g["y"], (g["score"] >= g["threshold"].iloc[0]).astype(int), zero_division=0),
        })
    ).reset_index()

    agg_cut = summary.groupby(["split", "model"]).agg(
        mean_pr=("pr_auc", "mean"),
        std_pr=("pr_auc", "std"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
    ).reset_index()

    plt.figure(figsize=(9, 5))
    cut_order = {s: i for i, s in enumerate(FC_SPLITS)}
    agg_cut["order"] = agg_cut["split"].map(cut_order)
    agg_cut = agg_cut.sort_values("order")

    for model in agg_cut["model"].unique():
        sub = agg_cut[agg_cut["model"] == model]
        plt.plot(sub["split"], sub["mean_pr"], marker="o", linewidth=2, label=model)

    plt.title("Test PR-AUC Across Chronological Forward-Chaining Cuts", fontsize=13, fontweight="bold")
    plt.xlabel("Split Cut", fontsize=11)
    plt.ylabel("Test PR-AUC", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_generalisation_by_cut.png", dpi=300)
    plt.close()

    # 2. Per-step PR-AUC across test steps (25 to 49)
    # Average predicted scores per node across seeds
    node_step = all_preds.groupby(["node_idx", "model", "time_step", "y"], as_index=False)["score"].mean()

    steps = sorted(node_step["time_step"].unique())
    step_records = []
    for step in steps:
        s_df = node_step[node_step["time_step"] == step]
        y_step = s_df["y"].values
        n_pos = int((y_step == 1).sum())
        base_rate = base_rates.get(step, n_pos / max(1, len(y_step)))

        for model in s_df["model"].unique():
            m_s_df = s_df[s_df["model"] == model]
            y_m = m_s_df["y"].values
            scores_m = m_s_df["score"].values

            if len(np.unique(y_m)) > 1:
                pr = average_precision_score(y_m, scores_m)
            else:
                pr = 0.0

            step_records.append({
                "time_step": step,
                "model": model,
                "n_illicit": n_pos,
                "base_rate": base_rate,
                "pr_auc": pr,
                "low_confidence": n_pos < 30,
            })

    step_df = pd.DataFrame(step_records)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot base rate as background grey fill
    step_meta = step_df.drop_duplicates("time_step").sort_values("time_step")
    ax1.bar(
        step_meta["time_step"],
        step_meta["base_rate"],
        color="gray",
        alpha=0.25,
        width=0.8,
        label="Illicit Base Rate (Right Axis)",
    )
    ax1.set_ylabel("Illicit Base Rate", color="gray", fontsize=11)
    ax1.set_ylim(0, max(step_meta["base_rate"].max() * 1.5, 0.4))

    ax2 = ax1.twinx()
    for model in sorted(step_df["model"].unique()):
        sub = step_df[step_df["model"] == model].sort_values("time_step")
        ax2.plot(sub["time_step"], sub["pr_auc"], marker="s", markersize=4, linewidth=1.8, label=model)

    # Highlight regime shift at steps 43-46 (Darknet shutdown)
    ax2.axvspan(43, 46, color="red", alpha=0.15, label="Darknet Shutdown (Steps 43-46)")

    # Mark low-confidence steps (<30 illicit nodes)
    low_conf = step_meta[step_meta["low_confidence"]]
    if not low_conf.empty:
        ax2.scatter(
            low_conf["time_step"],
            np.zeros(len(low_conf)) + 0.02,
            marker="^",
            color="black",
            s=40,
            label="Low Confidence (<30 illicit)",
            zorder=5,
        )

    ax1.set_xlabel("Time Step (Test Period)", fontsize=11)
    ax2.set_ylabel("Test PR-AUC", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    plt.title("Per-Step PR-AUC Trajectory and Darknet Shutdown Regime Break", fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_per_step_prauc.png", dpi=300)
    plt.close()
    logger.info("Saved Exp 3 figures to %s", fig_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3 Generalisation Analysis")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--eda-per-step", default="reports/eda/per_step.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    eda_path = Path(args.eda_per_step)

    all_preds = load_fc_predictions(results_dir, MODELS)
    compute_generalisation_table(all_preds, tables_dir)
    generate_generalisation_plots(all_preds, eda_path, figures_dir)
    logger.info("Experiment 3 completed successfully.")


if __name__ == "__main__":
    main()
