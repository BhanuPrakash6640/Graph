# GraphGuard — Temporal Graph Intelligence for Financial Crime Detection

An offline research and demonstration platform that asks one question:

> Does incorporating transaction relationships and temporal information improve
> the detection of illicit activity compared with conventional ML?

It uses the historical Elliptic Bitcoin dataset. It is **not** a production AML
compliance system and does **not** monitor any live blockchain.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Environment, loader, integrity checks, EDA | ✅ done — see [docs/01_phase1_findings.md](docs/01_phase1_findings.md) |
| 2 | Leakage-safe preprocessing & splits | ✅ done — see [docs/02_phase2_splits.md](docs/02_phase2_splits.md) |
| 3 | LR / RF / XGBoost baselines | ✅ done — see [docs/03_phase3_baselines.md](docs/03_phase3_baselines.md) |
| 4 | GCN / GraphSAGE / GAT | ✅ done — see [docs/04_phase4_static_gnns.md](docs/04_phase4_static_gnns.md) |
| 5 | Temporal model (EvolveGCN-O) | ✅ done — see [docs/05_phase5_temporal.md](docs/05_phase5_temporal.md) |
| 6 | Experiments 3–5, ablations | ✅ done — see [docs/06_phase6_ablations.md](docs/06_phase6_ablations.md) |
| 7 | Explainability | ✅ done — see [docs/07_phase7_explainability.md](docs/07_phase7_explainability.md) |
| 8 | FastAPI backend & inference engine | ✅ done — see [docs/08_phase8_fastapi_backend.md](docs/08_phase8_fastapi_backend.md) |
| 9 | Next.js forensic dashboard | ✅ done — see [docs/09_phase9_nextjs_dashboard.md](docs/09_phase9_nextjs_dashboard.md) |
| 10 | Technical report, packaging & CI | ✅ done — see [reports/technical_report.md](reports/technical_report.md) |

No results are reported yet. Results will only ever appear here after they are
produced by code in this repo and saved under `experiments/results/`.

Design rationale, risks, leakage strategy and evaluation protocol:
[`docs/00_design_analysis.md`](docs/00_design_analysis.md).
Preprocessing decisions log: [`docs/preprocessing_decisions.md`](docs/preprocessing_decisions.md).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-ml.txt
pip install -e .
```

## Getting the data

The dataset is not redistributed. With a Kaggle account and API token:

```bash
pip install kaggle
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
```

Expected layout:

```
data/raw/elliptic_bitcoin_dataset/
  elliptic_txs_features.csv
  elliptic_txs_edgelist.csv
  elliptic_txs_classes.csv
```

(Or download the zip from the Kaggle page and extract it there.)

## Phase 1 — run

```bash
pytest                                   # synthetic-data tests, no dataset needed
python -m ml.data.eda                    # validation + EDA on the real data
```

Outputs land in `reports/eda/`: `validation.json`, `summary.json`,
`per_step.csv`, `feature_shift_top.csv`, and three figures. The script exits
non-zero if any integrity check fails (e.g. edges crossing time steps).

## Phase 2 — run

```bash
python -m ml.preprocessing.build         # writes data/processed/ (~20 s)
```

Every model loads data through `ml.preprocessing.dataset.prepare_split(data, split_name, feature_set)`,
which applies the train-only scaler and refuses (`LeakageError`) one fitted outside the training window.

## Phase 3 — run

```bash
pytest                                   # run test suite (29 tests)
python -m ml.baselines.train --config experiments/configs/baselines.yaml
python -m ml.evaluation.aggregate --results-dir experiments/results --experiment exp1_baselines --tables-dir reports/tables --figures-dir reports/figures
```

Outputs land in `reports/tables/exp1_baselines.{csv,md}` and `reports/figures/exp1_*.png`. Each run records full config, environment, predictions parquet, and latency/memory under `experiments/results/exp1_baselines/`.

## Repository layout

```
ml/               data, preprocessing, baselines, gnn, temporal, evaluation, explainability
experiments/      configs/, results/ (one dir per run), notebooks/
backend/          FastAPI app (Phase 8)
frontend/         Next.js dashboard (Phase 9)
reports/          EDA outputs, technical report
docs/             design analysis, decision logs
tests/            pytest suite
```

## Label convention

`1 = illicit`, `0 = licit`, `-1 = unknown`. Unknown transactions stay in the
graph for message passing but are excluded from every loss and metric.
