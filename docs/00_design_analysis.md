# GraphGuard — Pre-implementation Design Analysis

Status: written before any experiment was run. Nothing here is a result. Every
number about model performance in this repository must come from
`experiments/results/`, produced by code in this repo.

---

## 1. What the dataset actually is (and why it shapes everything)

Elliptic (Weber et al., 2019, arXiv:1908.02591), three CSVs:

| File | Content |
|---|---|
| `elliptic_txs_features.csv` | No header. Col 0 = `txId`, col 1 = time step (1–49), cols 2–94 = 93 "local" features, cols 95–166 = 72 "aggregated" features |
| `elliptic_txs_edgelist.csv` | `txId1,txId2` — directed payment flow between transactions |
| `elliptic_txs_classes.csv` | `txId,class` with `1` = illicit, `2` = licit, `unknown` |

Structural facts that drive design (each one is re-checked by `ml/data/validate.py`
on the real files rather than trusted from the paper):

1. **Each time step is its own connected component.** Edges do not cross time
   steps. The graph is 49 disjoint snapshots, not one evolving graph.
2. **A transaction appears in exactly one time step.** Nodes do not persist
   across snapshots, so there is no per-node history to model.
3. **~77% of nodes are unlabelled; among labelled nodes, illicit is roughly 1 in 10.**
4. **The 72 aggregated features are already one-hop neighbourhood statistics.**
   A "non-graph" baseline trained on all 166 features is *not* graph-free.
5. **Features are anonymised.** Explanations can only name `local_feat_17`, not
   "transaction fee". That is a real limitation for the explainability layer.
6. **The paper reports a sharp drop in detection after a dark-market shutdown
   late in the test period.** Temporal distribution shift is built into the
   benchmark, which makes Experiment 3 meaningful.

## 2. Technical risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Treating `unknown` as licit | High | Labels mapped to {1, 0, -1}; -1 masked out of loss and every metric. Unit test enforces it. |
| R2 | Temporal leakage via scaling / imputation / threshold tuning on test steps | High | All fitted transforms (scaler, threshold, early stopping, HPO) use train/val steps only. |
| R3 | Graph leakage: message passing reads test-node features or labels | High | Snapshots are disjoint, so training on steps ≤ T never touches a later node. Labels are never node inputs. |
| R4 | Confounded comparison: "GNN beats tabular" when tabular already has neighbour aggregates | High | Run every model on (a) local features only (93) and (b) all 165. This is the core of Exp 1 & 5. |
| R5 | `time_step` used as a feature | Medium | Excluded from model inputs. Test steps are out of the train range and it encodes period base rates. |
| R6 | TGN is a poor fit: no persistent nodes, no edge timestamps finer than a snapshot | Medium | Pick a snapshot-level temporal model (Sec. 4). Document why TGN was not used. |
| R7 | Single-seed results presented as conclusions | Medium | 5 seeds, mean ± std, paired bootstrap CIs on PR-AUC for model comparisons. |
| R8 | Deployment memory: torch + PyG exceed free-tier hosts (e.g. Render free ≈ 512 MB) | Medium | CPU-only torch, serve k-hop subgraphs, precompute dataset-wide scores offline. Prefer a host with ≥2 GB RAM (e.g. Hugging Face Spaces CPU). |
| R9 | `POST /api/experiments/run` lets a public endpoint start training | Medium | Disabled in deployed mode; local-only, small whitelisted configs, background job. |
| R10 | Attention weights / explainer masks read as causal | Medium | UI and report label them as model-attribution heuristics, with fidelity scores. |
| R11 | Dataset licence/download needs a Kaggle account | Low | Not redistributed in repo. Kaggle CLI instructions + checksum/shape validation. |

## 3. Architecture

```
          offline (Colab / local)                         online (demo)
┌────────────────────────────────────────┐   ┌──────────────────────────────────┐
│ raw CSVs ─► ml/data (load + validate)  │   │ FastAPI                          │
│        ─► ml/preprocessing (splits,    │   │  ├ model registry (load once,    │
│           scaler fit on train only)    │   │  │  LRU cache)                    │
│        ─► data/processed/*.parquet,    │   │  ├ graph store (parquet + CSR    │
│           graph_*.pt                   │   │  │  index, k-hop extraction)      │
│ ml/baselines  ml/gnn  ml/temporal      │──►│  ├ explanation service            │
│ ml/evaluation (metrics, bootstrap)     │   │  └ SQLite: runs, inference log    │
│ ml/explainability (SHAP, GNNExplainer) │   │                                  │
│ experiments/results/<run_id>/          │   │ Next.js + Tailwind + Recharts +   │
│   config.yaml, metrics.json, preds.pq  │   │ Cytoscape.js                      │
│ checkpoints/ (outside git, release     │   └──────────────────────────────────┘
│   assets)                              │
└────────────────────────────────────────┘
```

Simplifications chosen deliberately:

- **Experiment tracking**: YAML config + one directory per run (config, metrics
  JSON, predictions parquet, git hash, seed, wall-clock, peak memory). MLflow is
  optional on top; it is not required to reproduce anything.
- **Storage**: SQLite for app metadata; parquet for features; a single
  `torch.save` of edge index per snapshot. No Postgres, Redis, or Neo4j.
- **No LLM in the loop.**

## 4. Model plan

**Baselines** (Exp 1): Logistic Regression (standardised, class-weighted),
Random Forest, XGBoost (`scale_pos_weight`, early stopping on val).

**Static GNNs**: GCN, GraphSAGE (mean / max aggregators; full-neighbourhood and
sampled fan-outs e.g. [10,5], [25,10]), GAT (multi-head). Trained per snapshot
batch: each training step is one or more time-step subgraphs, which is both
correct (no cross-step edges exist) and memory-light.

Edge direction: payment flow is directed. GCN/GAT will be run on the
symmetrised graph by default and on the directed graph as an ablation.

**Temporal model — recommendation: EvolveGCN-O.**
Because nodes do not persist across snapshots (R6), models that maintain
per-node memory (TGN, JODIE) have nothing to remember. EvolveGCN evolves the
GCN *weights* across snapshots with a GRU/LSTM, which is the form of temporal
information this dataset actually has. It was also evaluated in the original
Elliptic paper, so it is a defensible comparison rather than a name-drop.

A second, cheaper temporal variant is included as a control:
**snapshot-context features** — for each node, append statistics of its own
snapshot (illicit-prediction rate of the previous k snapshots from the model's
own predictions, never labels of the current/test step). If this matches
EvolveGCN, that is a finding worth reporting.

## 5. Preprocessing strategy

1. Load and validate shapes, dtypes, ID uniqueness, edge endpoints, cross-step edges.
2. Map labels `1→1 (illicit)`, `2→0 (licit)`, `unknown→-1`.
3. Drop `time_step` from inputs (kept as metadata).
4. Feature sets: `local` (93), `all` (165 = 93 + 72). Optional `all+emb`
   (node embeddings) for ablations.
5. Standardise with mean/std from **training steps only**; persist the scaler.
6. Unknown nodes stay in the graph as message-passing participants with their
   features; they never contribute to loss or metrics.
7. Every decision is appended to `docs/preprocessing_decisions.md` with rationale.

## 6. Leakage prevention checklist

- Split by `time_step`, never randomly.
- Scaler, imputer, thresholds, early stopping, and hyperparameter selection
  see train/val steps only.
- Validation = last steps of the training window (e.g. 30–34 under protocol A).
- Graph for training contains only training-step snapshots (guaranteed by the
  disjoint-component property, asserted in code).
- No label propagation features; labels are never node inputs.
- Test set is evaluated once per final config. HPO runs log which split they
  touched; a test-set read during HPO fails the run.

## 7. Evaluation methodology

Positive class = illicit. Unknowns excluded.

**Protocol A (paper-compatible):** train 1–29, val 30–34, test 35–49.
Final model may be refit on 1–34 with the val-chosen hyperparameters.

**Protocol B (strict chronological, forward-chaining):** for each cut
T ∈ {24, 29, 34, 39, 44}: train on 1..T−5, val on T−4..T, test on T+1..T+5.
Reports how performance degrades as the gap and regime change.

**Per-step reporting:** illicit F1 and PR-AUC for every test step, plotted, so
the late-period regime break is visible rather than averaged away.

**Metrics:** precision, recall, F1 (at val-optimised threshold and at 0.5),
ROC-AUC, PR-AUC (average precision), confusion matrix, FPR, training time,
inference latency (p50/p95 per node and per snapshot), peak CPU/GPU memory.

**Statistics:** 5 seeds; mean ± std; paired bootstrap (1,000 resamples over
test nodes) for the PR-AUC difference between the best tabular and best graph
model. Claims are made only when the CI excludes 0.

**Experiments mapped:**

| Exp | Comparison | Key control |
|---|---|---|
| 1 | LR / RF / XGB vs GCN / SAGE / GAT | Same feature set (`local` and `all`) |
| 2 | Static GNN vs EvolveGCN-O (+ snapshot-context control) | Same backbone width & features |
| 3 | Protocol B, per-step curves | Gap between train end and test step |
| 4 | Class weights, focal loss, oversampling (tabular only — SMOTE on graph nodes is ill-defined), threshold tuning on val | PR-AUC is threshold-free; F1 is not |
| 5 | Ablations: features only / topology only (constant or degree features) / local vs aggregated / neighbours of 0,1,2 hops / temporal on-off | Each removes exactly one component |

## 8. What runs where (compute estimate)

The graph is small: ~204k nodes × 165 float32 ≈ 135 MB of features and ~234k
edges. Full-batch training on all 34 training snapshots fits in < 2 GB.

| Component | Free Colab T4 | Laptop CPU (8–16 GB) |
|---|---|---|
| EDA, preprocessing | yes | yes |
| LR / RF / XGBoost | not needed | yes (minutes) |
| GCN / SAGE / GAT, full batch | yes, seconds per epoch | yes, slower but feasible |
| GraphSAGE with neighbour sampling grid | yes | feasible |
| EvolveGCN-O | yes | feasible |
| GNNExplainer (per node, 2-hop) | yes | yes (seconds/node) |
| SHAP TreeExplainer on XGBoost | not needed | yes |
| Full experiment grid (≈6 models × 2 feature sets × 2 protocols × 5 seeds + ablations) | a few GPU-hours, split over sessions | overnight |

These are planning estimates. Actual timings and memory are measured and
logged per run from Phase 3 onward and are what the report quotes.

The Cowork VM used to scaffold this repo has 2 CPUs / 3 GB RAM and no GPU and
cannot reach Kaggle, so all real training happens on your machine or Colab.

## 9. Phased plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1 | Repo scaffold, loader, integrity checks, EDA | Validation report + EDA figures generated from the real CSVs |
| 2 | Preprocessing + leakage-safe splits (A and B), scaler, processed artefacts | Leakage tests pass (no test step in any fitted transform) |
| 3 | LR / RF / XGB with run tracking | Results directory per run, 5 seeds |
| 4 | GCN / GraphSAGE / GAT | Same evaluation harness as Phase 3 |
| 5 | EvolveGCN-O + snapshot-context control | Exp 2 table |
| 6 | Exps 3–5, bootstrap CIs, figures | All tables regenerated by one command |
| 7 | SHAP, GNNExplainer, neighbour contribution, fidelity scores | Per-node explanation JSON |
| 8 | FastAPI (registry, k-hop service, endpoints, tests) | OpenAPI docs, API tests green |
| 9 | Next.js dashboard (6 modules) | Every number on screen traces to an API response |
| 10 | Docker, CI, deployment, technical report | Report numbers match `experiments/results/` |

## 10. Where the spec should be simplified

- **TGN → EvolveGCN-O** (reason: R6).
- **"Recent inference results"** on the overview page becomes "inference log":
  predictions the user ran in the app, labelled as such. There is no live feed.
- **`POST /api/experiments/run`** stays local-only (R9).
- **Oversampling** applies to tabular baselines only.
- **Deployment**: precompute scores and explanations for all nodes offline
  (generated by the real checkpoint, versioned with its run id) and keep live
  inference for on-demand k-hop subgraphs. This keeps the backend within a
  small CPU instance without faking anything.
