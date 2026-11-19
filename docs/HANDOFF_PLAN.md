# GraphGuard — Implementation plan for Phases 3–10

Companion to `AGENTS.md` (rules) and `docs/00_design_analysis.md` (rationale).
Each phase lists: goal, files, spec, acceptance criteria. Do not move on until
the acceptance criteria hold.

## Existing APIs you must build on

```python
from ml.preprocessing.dataset import load_processed, prepare_split, edges_for_steps
data = load_processed("data/processed")        # ProcessedData
view = prepare_split(data, "paper", "all")     # or "local" (93) / "aggregated" (72)
view.X            # (N, F) float32, scaled with train-only stats, all 203,769 nodes
view.y            # (N,) int8 {1, 0, -1}
view.time_step    # (N,) int16
view.train_mask / view.val_mask / view.test_mask   # labelled nodes only
view.edge_index   # (2, E) int64, raw direction src -> dst (full graph)
edges_for_steps(view.edge_index, view.time_step, steps)  # subgraph of given steps
view.split.train_steps / val_steps / test_steps
data.tx_ids       # node index -> txId
```

Splits available: `paper` (1–29 / 30–34 / 35–49), `paper_refit` (1–34 / – / 35–49),
`fc_24, fc_29, fc_34, fc_39, fc_44` (train 1..T−5 / val T−4..T / test T+1..T+5).
Snapshots are disjoint: 0 edges cross time steps (verified).

---

## Phase 3 — Evaluation harness + tabular baselines

**Goal:** a shared evaluation and run-tracking layer, then LR / RF / XGBoost results.

**Install:** add to `requirements-ml.txt`: `scikit-learn>=1.3`, `xgboost>=2.0`, `psutil`.

**Files**
- `ml/evaluation/metrics.py`
  - `compute_metrics(y_true, scores, threshold) -> dict`: precision, recall, f1,
    roc_auc, pr_auc (`average_precision_score`), confusion matrix `[[tn, fp], [fn, tp]]`,
    fpr, base_rate, n, n_pos. Inputs must contain no `-1` (assert).
  - `best_f1_threshold(y_val, s_val) -> float` (from the PR curve on val).
  - `per_step_metrics(y, s, steps, threshold) -> DataFrame` (one row per step, with n_pos).
  - `bootstrap_ci(y, s, metric_fn, n=1000, seed=0)` and
    `paired_bootstrap_diff(y, s_a, s_b, metric_fn, n=1000, seed=0)` → (mean, lo, hi).
    Also `paired_block_bootstrap_diff(y, s_a, s_b, groups=time_step, ...)` that resamples
    whole time steps; this is the primary CI for all comparison claims (AGENTS.md rule 7).
- `ml/evaluation/tracking.py` — `RunRecorder(experiment, config)` context manager
  that creates `experiments/results/<experiment>/<YYYYmmdd-HHMMSS>_<model>_<split>_<featset>_s<seed>/`
  and writes config.yaml, env.json (git hash via `git rev-parse HEAD` or "nogit",
  python/numpy/torch/sklearn/xgboost versions, device), metrics.json, timing.json,
  predictions.parquet (`node_idx, txId, time_step, y, score, part` for val+test
  labelled nodes; also store scores for unlabelled test-period nodes with `y=-1`
  for the dashboard). Peak memory: `psutil` RSS high-water mark +
  `torch.cuda.max_memory_allocated()` if CUDA.
- `ml/evaluation/latency.py` — `measure_latency(predict_fn, X, repeats=20)` →
  p50/p95 ms per batch and per node.
- `ml/baselines/models.py` — factories:
  - LR: `LogisticRegression(class_weight="balanced", max_iter=5000)`, tune `C ∈ {0.01, 0.1, 1, 10}` on val PR-AUC.
  - RF: `RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)`, tune `max_depth ∈ {None, 20}`, `max_features ∈ {"sqrt", 0.3}`.
  - XGB: `XGBClassifier(tree_method="hist", eval_metric="aucpr", scale_pos_weight=neg/pos on TRAIN, early_stopping_rounds=50, n_estimators=2000, random_state=seed)`, tune `max_depth ∈ {4, 6, 8}`, `learning_rate ∈ {0.05, 0.1}`.
- `ml/baselines/train.py` — CLI; `experiments/configs/baselines.yaml`.

**Protocol**
- For each model × feature set {local, all} × split in {paper, fc_24..fc_44}:
  HPO on val PR-AUC → threshold = `best_f1_threshold` on val → evaluate test once.
- `paper_refit`: retrain on 1–34 with the HPs and threshold chosen on `paper`
  (for XGB, use the best iteration count from `paper`). This is the headline paper-protocol number.
- Seeds 0–4 for RF/XGB. LR is deterministic: run once, record `seed_invariant: true`.
- Report metrics at the val-chosen threshold AND at 0.5.

**Aggregation:** `ml/evaluation/aggregate.py` → `reports/tables/exp1_baselines.csv/.md`
(mean ± std over seeds), `reports/figures/` PR curves, ROC curves, confusion matrices,
per-step F1 for test steps (show n_illicit per step on the plot).

**Accept when:** all runs recorded; tables regenerate with one command; tests cover
metrics (against hand-computed tiny examples), threshold selection never sees test
(test by passing a val/test with different optimal thresholds), recorder output schema.

---

## Phase 4 — Static GNNs (GCN, GraphSAGE, GAT)

**Install:** torch (CPU wheel is fine) + `torch_geometric` per the PyG install docs.
Do not require `torch_scatter`/`torch_sparse` unless needed.

**Files:** `ml/gnn/models.py`, `ml/gnn/train.py`, `ml/gnn/graph.py`,
`experiments/configs/gnn.yaml`.

**Graph construction (`graph.py`)**
- `build_period_graph(view, steps, undirected=True) -> torch_geometric.data.Data`
  with only the nodes of `steps` (re-indexed), their edges, `x`, `y`, `orig_idx`,
  and train/val/test masks restricted to those nodes.
- Default: symmetrise edges (`to_undirected`) + self-loops handled by the conv.
  Ablation flag: directed.

**Training**
- Train forward pass on the train-period graph only. Val: separate forward pass
  on the val-period graph (model in eval mode). Test: forward on the test-period graph.
- Loss: cross-entropy (or BCE-with-logits) on labelled train nodes, class weights
  from train label counts. Adam, weight decay 5e-4, max 1000 epochs, early stop on
  val PR-AUC (patience 100), restore best weights. Log per-epoch train loss / val PR-AUC
  to `curves.csv` in the run dir.
- Models (2 layers, hidden ∈ {64, 128}, dropout ∈ {0.3, 0.5}, lr ∈ {1e-3, 5e-3}):
  - GCN (`GCNConv`).
  - GraphSAGE (`SAGEConv`): aggregators `mean` and `max`; full-neighbourhood
    and `NeighborLoader` fan-outs `[10, 5]`, `[25, 10]`. The sampler must only
    sample inside the train-period graph.
  - GAT (`GATConv`, heads ∈ {4, 8}, concat for layer 1, mean for output). Save
    attention coefficients of the best model for test nodes (`return_attention_weights=True`)
    to `attention.pt` for Phase 7.
- Also run an **"MLP on the same pipeline"** control (same trainer, no edges). This
  isolates whether gains come from the graph or from the neural training setup.
- Same feature sets (local, all), same splits, seeds 0–4, same recorder and metrics.
- Save best checkpoint to `checkpoints/<run_id>.pt` with model class name + kwargs + feature set.

**Accept when:** Exp 1 table (LR, RF, XGB, MLP, GCN, SAGE-mean, SAGE-max, GAT ×
{local, all}) is generated from results; paired-bootstrap CI for best-tabular vs
best-GNN PR-AUC on `paper_refit`; tests check that the train graph has no non-train
nodes and that a forward pass works on the synthetic fixture.

---

## Phase 5 — Temporal model

**Chosen model: EvolveGCN-O** (see `docs/00_design_analysis.md` §4 — TGN rejected
because nodes do not persist across snapshots).

**Files:** `ml/temporal/evolvegcn.py` (own implementation, ~150 lines; avoid
`torch_geometric_temporal` dependency problems), `ml/temporal/train.py`,
`ml/temporal/snapshot_context.py`, `experiments/configs/temporal.yaml`.

**Spec**
- One `Data` object per time step. Each GCN layer's weight matrix evolves:
  `W_t = GRU/LSTM(W_{t-1})` (EvolveGCN-O). Training: iterate train snapshots in
  order, loss on labelled nodes of each snapshot, truncated BPTT over a window
  (e.g. 5 steps). Inference: roll weights forward through val then test snapshots in
  time order; weight evolution uses no labels.
- Control A: static GCN with identical width/depth (from Phase 4).
- Control B (`snapshot_context.py`): static model + per-snapshot context features
  computed from **the model's own predictions on previous snapshots** (e.g. mean
  predicted illicit score of steps t−1..t−3) and unlabelled structural stats of the
  current snapshot (size, edge density). Never use labels of val/test steps.
- Evaluate on `paper_refit` and all `fc_*` splits, seeds 0–4.

**Accept when:** Exp 2 table (static GCN vs EvolveGCN-O vs Control B), per-step
PR-AUC curves on test steps, CIs. If EvolveGCN does not help, report that plainly.

---

## Phase 6 — Experiments 3–5 and statistics

- **Exp 3 (generalisation):** for each best model per family, plot test PR-AUC
  and F1 vs `fc_*` cut and vs test step; overlay the base rate per step; mark
  steps with <30 illicit nodes as low-confidence. Discuss steps 43–46.
- **Exp 4 (imbalance)** on `paper` → `paper_refit`, best tabular + best GNN:
  none / class weights / focal loss (γ ∈ {1, 2}, α from train) / oversampling
  (tabular only: `RandomOverSampler` and SMOTE fitted on train rows only;
  add `imbalanced-learn`) / threshold tuning (0.5 vs val-optimal). Show PR-AUC is
  unchanged by thresholding while F1/precision/recall change.
- **Exp 5 (ablations)**, each removing one thing:
  1. features: local only vs aggregated only vs all;
  2. topology only: GNN with constant input features (or degree only);
  3. neighbour information: 0 hops (MLP) vs 1 vs 2 layers;
  4. edge direction: undirected vs directed;
  5. temporal: EvolveGCN with weight evolution frozen (= static) vs on.
- `ml/evaluation/make_report_tables.py` regenerates every table and figure in
  `reports/` from `experiments/results/` in one command.

**Accept when:** one command regenerates all report tables/figures; every
comparison claim in docs has a CI next to it.

---

## Phase 7 — Explainability

**Files:** `ml/explainability/{shap_tabular,gnn_explain,neighbors,export}.py`.

- SHAP `TreeExplainer` for the best XGBoost: global importance (mean |SHAP|) and
  per-node top-10 features. Feature names are anonymised (`local_17`, `agg_8`);
  say so wherever they are shown.
- `torch_geometric.explain.Explainer` with `GNNExplainer` (node-level, binary
  classification, `edge_mask_type="object"`, `node_mask_type="attributes"`) on the
  2-hop subgraph of a node within its own snapshot. Report fidelity+/fidelity−
  (PyG `fidelity`) per explanation so users can see how reliable it is.
- Neighbour contribution by occlusion: for each 1-hop neighbour, remove it and
  record the score change. This is model-based, cheap and easy to explain.
- GAT attention: show it but label it "attention weight (not a causal explanation)".
- Simple pattern flags computed from structure (not labels of the target node):
  fan-out (out-degree ≥ k), fan-in, chain length, share of neighbours predicted illicit.
- `export.py`: precompute explanations for the top-N highest-scoring test-period nodes
  + a random sample, to `data/processed/explanations/<model_run_id>/<txId>.json`
  with schema: `{txId, time_step, model_run_id, score, predicted_class, threshold,
  known_label, top_features[], neighbor_contributions[], explainer_edges[],
  fidelity, pattern_flags[], disclaimer}`.

**Accept when:** JSON schema validated by tests; at least one notebook in
`experiments/notebooks/` walks through 3 real flagged nodes (no cherry-picking
claims; also include one false positive).

---

## Phase 8 — FastAPI backend

**Files:** `backend/app/{main.py, api/, services/, schemas/, models/}`,
`backend/requirements.txt`, `backend/tests/`.

- Startup (FastAPI `lifespan`): load `nodes.parquet` (ids, steps, labels, raw
  features), build a CSR adjacency (scipy) for k-hop lookup, load the model
  registry from `backend/models.yaml` (which run ids/checkpoints to serve), load
  each checkpoint once and keep it in memory (LRU if many). Never load per request.
- `services/graph_store.py`: `khop(tx_id, k)` restricted to the node's own time step.
- `services/inference.py`: real inference on the k-hop subgraph (GNN) or the
  feature row (tabular), using the split scaler saved for that model's split.
- Endpoints: `GET /api/health`, `GET /api/models`, `POST /api/predict`
  (`{tx_id, model_id}`), `GET /api/transaction/{id}`, `.../neighbors?k=1|2`,
  `.../explanation?model_id=`, `GET /api/benchmark` (reads `reports/tables`),
  `GET /api/temporal-analysis`, `POST /api/experiments/run` → **403 unless env
  `GG_ENABLE_EXPERIMENTS=1`**, and then only whitelisted configs as a background job.
- Pydantic v2 schemas for every request/response; 404 for unknown txId; 422 validation;
  structured logging (`logging` JSON formatter); OpenAPI docs at `/docs`.
- SQLite (`sqlite3` or SQLModel): `inference_log(id, ts, tx_id, model_id, score, latency_ms)`.
- Every response that includes data says whether it is `"historical_label"` or
  `"model_prediction"`.

**Accept when:** pytest with `TestClient` on synthetic artefacts covers every endpoint,
including 404/422/403; p95 `/api/predict` latency measured and written to the phase doc.

---

## Phase 9 — Next.js dashboard

Next.js (App Router) + TypeScript + Tailwind + Recharts + Cytoscape.js.
`frontend/lib/api.ts` typed client; `NEXT_PUBLIC_API_URL` env var.

Pages: Overview, Investigate (`/transaction/[id]`), Graph Explorer, Benchmark,
Temporal, Explainability (see `docs/PROJECT_SPEC.md` §8).

Rules:
- Every number comes from an API response. No numbers in TSX.
- Persistent banner: "Research demo on historical Elliptic data (2019). Not a
  live monitoring or compliance system."
- Colours: illicit `#c0392b`, licit `#2e86c1`, unknown grey `#95a5a6` with a
  **dashed border** (must be visually distinct from licit). Known labels = filled
  node; model predictions = ring/outline in the predicted colour, so "known" vs
  "predicted" is never confused.
- Overview separates "Dataset statistics (historical)" from "Inference log
  (predictions run in this app)".
- Explainability panel shows the fidelity score and the disclaimer text.
- Loading, empty and error states on every page. Keyboard-accessible; check contrast.

**Accept when:** `npm run build` and `npm run lint` pass; Playwright smoke test
loads each page against a running backend.

---

## Phase 10 — Packaging, CI, deployment, report

- `backend/Dockerfile` (python:3.11-slim, CPU torch wheel), `frontend/Dockerfile`,
  `docker-compose.yml` (backend + frontend; data and checkpoints mounted as volumes).
- `.github/workflows/ci.yml`: ruff + pytest (synthetic fixture only) + frontend
  lint/build. No real data in CI.
- Artefacts: publish `data/processed` (minus raw), chosen checkpoints, and
  precomputed explanations as GitHub Release assets; `scripts/fetch_artifacts.py`
  downloads them. The raw dataset is not redistributed.
- Deploy: frontend on Vercel; backend on a host with ≥2 GB RAM (Hugging Face
  Spaces Docker CPU, or Render paid). The free 512 MB tier will likely fail with torch + PyG.
- `reports/technical_report.md` with the 15 sections from `docs/PROJECT_SPEC.md` §10. All tables and
  figures embedded from `reports/` (generated). References: only papers actually
  used (Weber et al. 2019; Kipf & Welling 2017; Hamilton et al. 2017; Veličković
  et al. 2018; Pareja et al. 2020; Ying et al. 2019; Lundberg & Lee 2017; Lin et al.
  2017 for focal loss). Verify each citation before adding it. No novelty claims.
- Limitations section must cover: anonymised features, label scarcity (77% unknown),
  regime change after the dark-market shutdown, single dataset, offline only.

**Accept when:** `docker compose up` serves the app locally from release artefacts;
CI is green; every number in the report matches a file in `reports/`.
