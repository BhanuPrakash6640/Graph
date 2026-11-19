"""Master Report Table and Figure Generator.

Regenerates all benchmark tables, statistical comparisons, and figures across
all experimental phases from saved run artifacts under experiments/results/ in a single command.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.evaluation.aggregate import generate_figures, generate_tables, load_experiment_runs
from ml.evaluation.compare import compare_configurations, get_comparisons
from ml.experiments.exp3_generalisation import (
    MODELS,
    compute_generalisation_table,
    generate_generalisation_plots,
    load_fc_predictions,
)
from ml.experiments.exp4_imbalance import run_exp4_imbalance
from ml.experiments.exp5_ablations import run_exp5_ablations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.evaluation.make_report_tables")


def regenerate_all(
    results_dir: str = "experiments/results",
    tables_dir: str = "reports/tables",
    figures_dir: str = "reports/figures",
    eda_per_step: str = "reports/eda/per_step.csv",
) -> None:
    res_path = Path(results_dir)
    tbl_path = Path(tables_dir)
    fig_path = Path(figures_dir)
    eda_path = Path(eda_per_step)

    tbl_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Regenerate Exp 1 & Exp 2 Aggregated Tables and Figures
    logger.info("=== 1. Aggregating Exp 1 (Baselines, GNNs) and Exp 2 (Temporal) ===")
    runs = load_experiment_runs(res_path, ["exp1_baselines", "exp1_gnn", "exp2_temporal"])
    if runs:
        generate_tables(runs, out_dir=tbl_path)
        generate_figures(runs, out_dir=fig_path)
        logger.info("Successfully generated exp1_all_models, exp1_paper_refit, exp2_temporal, exp2_paper_refit.")

    # 2. Regenerate Statistical Comparisons (Exp 1 and Exp 2)
    logger.info("=== 2. Computing Paired Block-Bootstrap Comparison Tables ===")
    exp1_comps = get_comparisons("exp1")
    df_exp1 = compare_configurations(res_path, exp1_comps)
    df_exp1.to_markdown(tbl_path / "exp1_comparisons.md", index=False)

    exp2_comps = get_comparisons("exp2")
    df_exp2 = compare_configurations(res_path, exp2_comps)
    df_exp2.to_markdown(tbl_path / "exp2_comparisons.md", index=False)
    logger.info("Successfully regenerated exp1_comparisons.md and exp2_comparisons.md.")

    # 3. Regenerate Exp 3 Generalisation Analysis
    logger.info("=== 3. Regenerating Exp 3 Generalisation Tables & Figures ===")
    all_fc_preds = load_fc_predictions(res_path, MODELS)
    compute_generalisation_table(all_fc_preds, tbl_path)
    if eda_path.exists():
        generate_generalisation_plots(all_fc_preds, eda_path, fig_path)
    logger.info("Successfully regenerated exp3_generalisation tables and plots.")

    logger.info("All report tables and figures successfully regenerated.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all report tables and figures from results")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--eda-per-step", default="reports/eda/per_step.csv")
    args = parser.parse_args()

    regenerate_all(
        results_dir=args.results_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        eda_per_step=args.eda_per_step,
    )


if __name__ == "__main__":
    main()

