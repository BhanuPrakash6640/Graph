"""Aggregation and report generation for baseline experiments.

Scans experiments/results/<experiment>/, computes summary statistics (mean +/- std)
over seeds, generates reports/tables/exp1_baselines.csv and .md, and creates
PR curves, ROC curves, confusion matrices, and per-step F1 figures with n_illicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def load_experiment_runs(
    results_dir: str | Path,
    experiment: str | list[str] = "exp1_baselines",
) -> list[dict[str, Any]]:
    base = Path(results_dir)
    if isinstance(experiment, str):
        exp_names = [e.strip() for e in experiment.split(",") if e.strip()]
    else:
        exp_names = list(experiment)

    runs = []
    for exp_name in exp_names:
        exp_path = base / exp_name
        if not exp_path.exists():
            continue

        for run_dir in sorted(exp_path.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            cfg_file = run_dir / "config.yaml"
            met_file = run_dir / "metrics.json"
            tim_file = run_dir / "timing.json"
            env_file = run_dir / "env.json"
            pred_file = run_dir / "predictions.parquet"

            if not (cfg_file.exists() and met_file.exists()):
                continue

            try:
                with open(met_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                timing = {}
                if tim_file.exists():
                    with open(tim_file, "r", encoding="utf-8") as f:
                        timing = json.load(f)
                env_info = {}
                if env_file.exists():
                    with open(env_file, "r", encoding="utf-8") as f:
                        env_info = json.load(f)

                import yaml
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                runs.append({
                    "run_dir": run_dir,
                    "config": cfg,
                    "metrics": metrics,
                    "timing": timing,
                    "env": env_info,
                    "has_predictions": pred_file.exists(),
                    "pred_file": pred_file,
                })
            except Exception as e:
                print(f"Skipping corrupt run dir {run_dir}: {e}")

    return runs


def generate_tables(runs: list[dict[str, Any]], out_dir: str | Path = "reports/tables") -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in runs:
        cfg = r["config"]
        met = r["metrics"]
        tim = r["timing"]

        t_opt = met.get("test_at_opt") or met.get("at_val_threshold") or {}
        t_05 = met.get("test_at_05") or met.get("at_05") or {}
        inf = tim.get("inference") or tim.get("inference_latency") or {}

        rows.append({
            "model": cfg.get("model"),
            "split": cfg.get("split"),
            "feature_set": cfg.get("feature_set"),
            "seed": cfg.get("seed"),
            "threshold": met.get("best_threshold") or met.get("threshold_val_f1"),
            "pr_auc": t_opt.get("pr_auc"),
            "f1_opt": t_opt.get("f1"),
            "precision_opt": t_opt.get("precision"),
            "recall_opt": t_opt.get("recall"),
            "roc_auc": t_opt.get("roc_auc"),
            "f1_05": t_05.get("f1"),
            "train_time_s": tim.get("train_time_s") or tim.get("train_s"),
            "node_p50_ms": inf.get("node_p50_ms"),
            "peak_memory_mb": tim.get("peak_memory_mb"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Group by (model, split, feature_set)
    grouped = df.groupby(["split", "model", "feature_set"], sort=False)

    summary_rows = []
    for (split, model, featset), grp in grouped:
        n_seeds = len(grp)

        def _fmt(col: str, decimals: int = 3) -> str:
            vals = grp[col].dropna()
            if len(vals) == 0:
                return "N/A"
            if len(vals) == 1:
                return f"{vals.iloc[0]:.{decimals}f}"
            mean = vals.mean()
            std = vals.std()
            return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

        summary_rows.append({
            "split": split,
            "model": model.upper(),
            "feature_set": featset,
            "n_seeds": n_seeds,
            "pr_auc": _fmt("pr_auc", 4),
            "f1 (val thr)": _fmt("f1_opt", 4),
            "f1 (0.5)": _fmt("f1_05", 4),
            "precision": _fmt("precision_opt", 4),
            "recall": _fmt("recall_opt", 4),
            "roc_auc": _fmt("roc_auc", 4),
            "val_threshold": _fmt("threshold", 3),
            "train_time_s": _fmt("train_time_s", 2),
            "node_p50_ms": _fmt("node_p50_ms", 4),
            "peak_mem_mb": _fmt("peak_memory_mb", 1),
        })

    sum_df = pd.DataFrame(summary_rows)

    # Save CSV and Markdown
    csv_file = out_path / "exp1_baselines.csv"
    md_file = out_path / "exp1_baselines.md"
    sum_df.to_csv(csv_file, index=False)
    sum_df.to_markdown(md_file, index=False)

    has_temporal = any(m.lower() in ("evolvegcn", "control_b") for m in df["model"].unique())
    if has_temporal:
        temp_csv = out_path / "exp2_temporal.csv"
        temp_md = out_path / "exp2_temporal.md"
        # Extract static GCN (Control A), EvolveGCN-O, and Control B
        temp_df = sum_df[sum_df["model"].str.lower().isin(["gcn", "evolvegcn", "control_b"])].copy()
        temp_df.to_csv(temp_csv, index=False)
        temp_df.to_markdown(temp_md, index=False)

        # Also dedicated exp2 paper_refit table
        temp_refit = temp_df[temp_df["split"] == "paper_refit"].copy()
        if not temp_refit.empty:
            temp_refit_csv = out_path / "exp2_paper_refit.csv"
            temp_refit_md = out_path / "exp2_paper_refit.md"
            temp_refit.to_csv(temp_refit_csv, index=False)
            temp_refit.to_markdown(temp_refit_md, index=False)

    # Dedicated paper_refit table
    refit_df = sum_df[sum_df["split"] == "paper_refit"].copy()
    if not refit_df.empty:
        refit_csv = out_path / "exp1_paper_refit.csv"
        refit_md = out_path / "exp1_paper_refit.md"
        refit_df.to_csv(refit_csv, index=False)
        refit_df.to_markdown(refit_md, index=False)

    return sum_df


def generate_figures(runs: list[dict[str, Any]], out_dir: str | Path = "reports/figures") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # We focus figures on the headline paper_refit (or paper) with 'all' features
    headline_runs = [
        r for r in runs
        if r["config"].get("split") == "paper_refit"
        and r["config"].get("feature_set") == "all"
        and r["has_predictions"]
    ]
    if not headline_runs:
        # Fallback to paper split if refit isn't present
        headline_runs = [
            r for r in runs
            if r["config"].get("split") == "paper"
            and r["config"].get("feature_set") == "all"
            and r["has_predictions"]
        ]

    if not headline_runs:
        return

    # Select representative run per model (seed 0)
    rep_runs: dict[str, dict[str, Any]] = {}
    for r in headline_runs:
        m = r["config"].get("model")
        s = r["config"].get("seed", 0)
        if m not in rep_runs or s == 0:
            rep_runs[m] = r

    # 1. PR Curves
    plt.figure(figsize=(8, 6))
    for model_name, r in sorted(rep_runs.items()):
        preds = pd.read_parquet(r["pred_file"])
        test_preds = preds[preds["part"] == "test"]
        y_true = test_preds["y"].to_numpy()
        scores = test_preds["score"].to_numpy()

        p, rc, _ = precision_recall_curve(y_true, scores)
        pr_auc = r["metrics"].get("test_at_opt", {}).get("pr_auc", 0.0)
        plt.plot(rc, p, label=f"{model_name.upper()} (PR-AUC = {pr_auc:.4f})")

    # Plot base rate
    if rep_runs:
        sample_r = next(iter(rep_runs.values()))
        preds = pd.read_parquet(sample_r["pred_file"])
        test_preds = preds[preds["part"] == "test"]
        base_rate = (test_preds["y"] == 1).mean()
        plt.axhline(base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Paper Refit (Test Steps 35–49)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "exp1_pr_curves.png", dpi=150)
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    for model_name, r in sorted(rep_runs.items()):
        preds = pd.read_parquet(r["pred_file"])
        test_preds = preds[preds["part"] == "test"]
        y_true = test_preds["y"].to_numpy()
        scores = test_preds["score"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = r["metrics"].get("test_at_opt", {}).get("roc_auc", 0.0)
        plt.plot(fpr, tpr, label=f"{model_name.upper()} (ROC-AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Paper Refit (Test Steps 35–49)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "exp1_roc_curves.png", dpi=150)
    plt.close()

    # 3. Confusion Matrices
    n_models = len(rep_runs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, sorted(rep_runs.items())):
        cm = np.array(r["metrics"].get("test_at_opt", {}).get("confusion_matrix", [[0, 0], [0, 0]]))
        thr = r["metrics"].get("best_threshold", 0.5)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{model_name.upper()} (thr={thr:.3f})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Licit", "Pred Illicit"])
        ax.set_yticklabels(["True Licit", "True Illicit"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path / "exp1_confusion_matrices.png", dpi=150)
    plt.close()

    # 4. Per-step F1 with n_illicit overlay
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    colors = {"lr": "tab:blue", "rf": "tab:green", "xgb": "tab:orange"}
    n_illicit_plotted = False

    for model_name, r in sorted(rep_runs.items()):
        per_step = r["metrics"].get("per_step_test", [])
        if not per_step:
            continue
        step_df = pd.DataFrame(per_step)
        steps = step_df["step"].to_numpy()
        f1_vals = step_df["f1"].to_numpy()
        n_pos = step_df["n_pos"].to_numpy()

        if not n_illicit_plotted:
            ax2.bar(steps, n_pos, alpha=0.2, color="gray", label="Illicit count (n_pos)")
            n_illicit_plotted = True

        ax1.plot(steps, f1_vals, marker="o", label=f"{model_name.upper()} F1", color=colors.get(model_name))

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("F1 Score")
    ax2.set_ylabel("Number of Illicit Transactions")
    ax1.set_title("Test Period Per-Step F1 and Illicit Node Counts (Steps 35–49)")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path / "exp1_per_step_f1.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment runs and produce reports")
    parser.add_argument("--results-dir", default="experiments/results", help="Directory containing experiment runs")
    parser.add_argument("--experiment", default="exp1_baselines", help="Experiment name")
    parser.add_argument("--tables-dir", default="reports/tables", help="Output directory for tables")
    parser.add_argument("--figures-dir", default="reports/figures", help="Output directory for figures")
    args = parser.parse_args()

    runs = load_experiment_runs(args.results_dir, args.experiment)
    print(f"Loaded {len(runs)} runs from {args.results_dir}/{args.experiment}")
    if runs:
        sum_df = generate_tables(runs, args.tables_dir)
        print(f"Generated summary table with {len(sum_df)} configurations")
        generate_figures(runs, args.figures_dir)
        print(f"Generated figures in {args.figures_dir}")
    else:
        print("No runs found.")


if __name__ == "__main__":
    main()

