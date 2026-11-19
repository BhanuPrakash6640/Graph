# AGENTS.md — GraphGuard project rules

Read this whole file before every task. The phase-by-phase spec is in
@docs/HANDOFF_PLAN.md — read the section for the phase you are working on.

## What this project is

Offline research + demo platform on the Elliptic Bitcoin dataset asking:
*does graph structure and temporal information improve illicit-transaction
detection over tabular ML?* It is NOT a production AML system and does NOT
monitor any live blockchain. Never describe it as either.

## Current state (do not redo)

- Phase 1 DONE: `ml/data/loader.py`, `ml/data/validate.py`, `ml/data/eda.py`. Findings: `docs/01_phase1_findings.md`.
- Phase 2 DONE: `ml/preprocessing/{splits,scaler,dataset,build}.py`. Findings: `docs/02_phase2_splits.md`.
- Raw data: `data/raw/elliptic_bitcoin_dataset/*.csv`. Processed: `data/processed/` (rebuild: `python -m ml.preprocessing.build`).
- Tests: `pytest` → 20 passing. Keep them passing.
- Next: Phase 3 (baselines). Do phases in order. Do not start the frontend before Phase 7 is done.

## NON-NEGOTIABLE research rules

1. **Never fabricate numbers.** No hardcoded predictions, mock metrics, placeholder
   scores, or "expected" results in code, docs, README, report, or UI. Every number
   shown anywhere must be read from a file under `experiments/results/` or
   `data/processed/` that was produced by code in this repo. If a result does not
   exist yet, write "not yet run".
2. **Unknown ≠ licit.** Labels are `1` illicit, `0` licit, `-1` unknown. `-1` never
   enters a loss, metric, threshold choice, or class-weight calculation. Unknown
   nodes stay in the graph as message-passing participants only.
3. **No temporal leakage.** Always load data via
   `ml.preprocessing.dataset.prepare_split(data, split_name, feature_set)`. Never
   fit scalers, thresholds, early stopping, hyperparameters, oversamplers, or
   feature selection on val/test steps (val is allowed for early stopping /
   HPO / threshold; test is never allowed).
4. **No graph leakage.** During training, the forward pass uses only the
   training-step subgraph: `edges_for_steps(view.edge_index, view.time_step,
   split.train_steps)` and only training-step node rows. This matters for
   BatchNorm statistics and neighbour sampling. Evaluate each period on its own
   snapshots' subgraph.
5. **`time_step` is never a model input.** Use `feature_columns()` / `prepare_split`.
6. **Test set is read once per final config.** Pick hyperparameters and thresholds
   on the `paper` split's val (steps 30–34) or on each `fc_*` split's val. Never
   look at test metrics to choose anything.
7. **No claim that GNNs beat tabular** (or vice versa) without the evidence:
   ≥5 seeds, mean ± std, and a paired bootstrap CI on the PR-AUC difference that
   excludes 0. Otherwise write "no significant difference".
   The CI must come from a **time-step block bootstrap** (resample whole test
   steps, not individual nodes): nodes within a step are correlated, so a
   node-level bootstrap gives CIs that are too narrow. Report the node-level CI
   only as a secondary column. With <8 test steps (e.g. any fc_* split), also
   report per-step wins (e.g. "B better on 4/5 steps") and call the evidence weak.
   This applies to EVERY comparative sentence in docs ("improves", "beats",
   "more resilient"), including tabular-vs-tabular and local-vs-all.
7b. **Seeds must actually change something.** If a model gives identical results
   across seeds (e.g. XGBoost without subsampling), either make it stochastic
   (`subsample`/`colsample_bytree` < 1) or record it as `seed_invariant` with
   n_seeds = 1. Never report "± 0.0000 over 5 seeds".
7c. **Hyperparameters/thresholds carried from `paper` to `paper_refit` are paired
   by seed** (seed k's refit uses seed k's paper threshold), never "last seed wins".
8. **Attention weights / explainer masks are attributions, not causes.** Label them
   so in docs and UI.
9. No LLM API wrappers. No fake real-time feeds. No unnecessary infrastructure
   (no Postgres/Redis/Kafka/Neo4j; SQLite + parquet + .pt files are enough).

## Engineering conventions

- Python ≥3.10, type hints, small modules, `ml.*` imports (repo root on path via
  `pip install -e .`). Each CLI is `python -m ml.<pkg>.<module> --config <yaml>`.
- Configs: YAML in `experiments/configs/`. No magic numbers in code that belong in config.
- Seeds: `ml.utils.config.set_seed(seed)` at the start of every run; seeds `[0,1,2,3,4]`.
- Every training run writes one directory via the run recorder (Phase 3 builds it):
  `experiments/results/<experiment>/<run_id>/` with `config.yaml`, `metrics.json`,
  `predictions.parquet`, `env.json` (git hash, package versions, seed, device),
  `timing.json` (train s, inference p50/p95 ms, peak memory MB). Never overwrite a run dir.
- Positive class for every metric = illicit (1). Primary metric = PR-AUC
  (average precision). Always also report precision, recall, F1, ROC-AUC,
  confusion matrix, FPR, and the base rate of the evaluated set.
- Tests: every new module gets pytest tests that run on the synthetic fixture in
  `tests/conftest.py` (no real data, <30 s total, CPU only). Mark slow/real-data
  tests `@pytest.mark.slow` and skip by default.
- Don't change Phase 1/2 public functions' behaviour. If you must, update their
  tests and log the change.
- Large/generated files (`data/`, `checkpoints/`, `experiments/results/*`) are
  gitignored. Small summary tables/figures that the report uses go in `reports/`.

## Documentation duties (every phase)

- Append new preprocessing/modelling decisions to `docs/preprocessing_decisions.md`
  (D17, D18, …: decision, rationale, where enforced).
- Write `docs/0N_phaseN_<name>.md` with what was built, how to run it, and
  results tables generated from result files (paste generated tables; never type numbers).
- Update the status table in `README.md`.
- End each phase by running `pytest`, then report: files changed, commands run,
  test count, and anything surprising in the results.

## Environment notes

- User is on Windows (`Z:\graph\GraphGuard`). Give commands that work in
  PowerShell; use `pathlib` in code, never hardcoded `/` or `\` paths.
- Dataset is small (~135 MB features). CPU is fine for baselines and GNNs;
  a GPU (local or Colab) only speeds things up. Code must run on both
  (`device = "cuda" if torch.cuda.is_available() else "cpu"`).

## When unsure

Choose the simpler, scientifically valid option and write down why. If a spec
item can't be done honestly (e.g. a metric that would require test labels at
inference), say so in the phase doc instead of faking it.
