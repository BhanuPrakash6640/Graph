# Preprocessing decisions log

Each entry: decision, rationale, where enforced. Append-only.

| # | Decision | Rationale | Enforced in |
|---|---|---|---|
| D1 | Map raw class `1→1` (illicit), `2→0` (licit), `unknown→-1` | Unknown ≠ licit; it is missing supervision | `ml/data/loader.py`, `tests/test_data.py::test_unknown_is_not_licit` |
| D2 | Raw feature columns named `local_1..93`, `agg_1..72`; col 1 is `time_step` | Paper describes 94 local features including the time step and 72 aggregated ones; naming separates them for ablations | `ml/data/loader.py` |
| D3 | `time_step` is metadata, never a model input | Test steps are outside the training range; the index would encode period base rates | `feature_columns()`, `test_time_step_never_a_feature` |
| D4 | Load fails on unexpected column counts, class values, duplicate IDs, or missing class rows | Silent coercion hides data problems | `ml/data/loader.py` |
| D5 | Edges crossing time steps are an error in validation | Time-based splits are leak-free only if snapshots are disjoint; if the real data violates this, splitting must cut those edges | `ml/data/validate.py` |
| D6 | Self-loops and duplicate edges are reported, not dropped, in Phase 1 | Handling decided in Phase 2 after seeing counts | `ml/data/validate.py` |
| D7 | Feature-shift EDA uses all nodes, no labels | Describes covariate shift only; no label information from the test period informs any modelling choice | `ml/data/eda.py::feature_shift` |
| D8 | No edge cutting needed for time splits | Validation on real data found 0 cross-time-step edges | `reports/eda/validation.json` |
| D9 | No self-loop / duplicate-edge handling needed | Real data has 0 of each (resolves D6) | `reports/eda/validation.json` |
| D10 | Validation = last 5 steps of each training window; `paper_refit` retrains on 1–34 with validation-chosen settings | HPO, early stopping and thresholds need data from before the test period | `ml/preprocessing/splits.py` |
| D11 | Scaler fitted on *all* nodes (incl. unknown) in training steps only | Features don't depend on labels, and unknown training-period nodes are legitimately available at training time. Test-period nodes are excluded | `ml/preprocessing/scaler.py`, `test_scaler_ignores_non_train_rows` |
| D12 | Raw features stored once; scaling applied per split at load time | A shared pre-scaled file would bake one window's statistics into every split | `ml/preprocessing/build.py` |
| D13 | `prepare_split` raises `LeakageError` if the scaler's fit steps aren't within the training steps | Makes the leak structurally impossible, not just avoided by convention | `ml/preprocessing/dataset.py`, `test_leakage_guard_rejects_scaler_fitted_on_test` |
| D14 | Masks contain labelled nodes only; unknown nodes stay in X and in the graph | Unknown ≠ licit (D1); unknown nodes still carry features for message passing | `test_masks_exclude_unknown_and_are_disjoint` |
| D15 | Training graph = edges with both endpoints in training steps | Equal to the union of training snapshots (0 cross-step edges, D8) | `edges_for_steps`, `test_train_graph_has_no_test_nodes` |
| D16 | Constant-in-training features: std set to 1 (centred, not scaled) | Avoids dividing by zero without dropping columns that may vary later | `test_constant_feature_does_not_divide_by_zero` |
| D17 | Baseline HPO grids tuned on validation PR-AUC (LR uses `lbfgs`) | Optimizes primary competition metric without touching test data; LR uses lbfgs solver with class_weight='balanced' | `ml/baselines/models.py::create_lr`, `ml/baselines/train.py` |
| D18 | Decision threshold chosen by maximizing F1 on validation PR curve | Threshold selection requires validation labels; test set is evaluated once at this tuned threshold | `ml/evaluation/metrics.py::best_f1_threshold`, `test_threshold_selection_never_sees_test_data` |
| D19 | XGBoost `scale_pos_weight = n_neg / n_pos` on train split | Addresses ~1:9 illicit-to-licit class imbalance during gradient boosting | `ml/baselines/models.py::create_xgb` |
| D20 | XGBoost early stopping (patience 50 on val PR-AUC); `paper_refit` uses `paper`'s best iteration | Prevents overfitting while allowing `paper_refit` to reuse optimal stopping point without test leakage | `ml/baselines/train.py` |
| D21 | GPU execution for XGBoost via `device="cuda"` | Accelerates GBDT training and profiling on NVIDIA RTX 3050 Laptop GPU with VRAM tracking | `ml/baselines/models.py`, `ml/evaluation/tracking.py` |
| D22 | Random Forest HPO uses 100 trees as proxy before fitting 500 trees | Keeps grid search fast while retaining fidelity to the final 500-tree model ranking | `ml/baselines/train.py::tune_rf` |
| D23 | `fc_29` validation and test scores (~0.98–0.99 PR-AUC) reflect natural step 30–34 cluster separation | Verified: the `paper` split's validation set covers the exact same steps 30–34 and scores ~0.99 PR-AUC too, confirming no data leakage | `reports/tables/exp1_baselines.md`, `ml/baselines/train.py` |
| D24 | XGBoost stochasticity via `subsample=0.8, colsample_bytree=0.8` | Ensures random seeds produce distinct tree structures and non-zero variance (AGENTS.md rule 7b) | `ml/baselines/models.py::create_xgb` |
| D25 | Seed-to-seed pairing from `paper` to `paper_refit` | Seed k's refit uses seed k's exact validation threshold and best iteration count (AGENTS.md rule 7c) | `ml/baselines/train.py::run_baselines` |
| D26 | Period subgraphs with local re-indexing and symmetrization | Subgraphs for train/val/test strictly isolate nodes and edges of that period with re-indexing to `0..N_period-1` and `to_undirected` symmetrization | `ml/gnn/graph.py::build_period_graph`, `tests/test_gnn.py` |
| D27 | Class weighting for GNN CrossEntropyLoss | Inverse-class frequencies computed strictly on labelled train nodes (`total / (2 * n)`) to counter ~1:9 imbalance | `ml/gnn/train.py::run_single_gnn` |
| D28 | GNN HPO grid on validation PR-AUC | Grid search over hidden (64, 128), dropout (0.3, 0.5), lr (1e-3, 5e-3) on seed 0 with early stopping | `ml/gnn/train.py::tune_gnn` |
| D29 | GNN seed-paired `paper_refit` retraining | Retrains on steps 1–34 using seed k's exact paper threshold, HPs, and best epoch count (AGENTS.md rule 7c) | `ml/gnn/train.py::run_single_gnn` |
| D30 | GAT attention coefficients export | Attentions on test nodes saved with `return_attention_weights=True` to `attention.pt` for Phase 7 attribution | `ml/gnn/models.py::GAT`, `ml/gnn/train.py` |
| D31 | MLP control model on identical pipeline | 2-layer MLP without edges evaluates neural vs graph inductive bias under identical training dynamics | `ml/gnn/models.py::MLP`, `ml/gnn/train.py` |
| D32 | GAT hidden channels (64) & VRAM garbage collection | GAT 4-head attention concat produces 256 channels; restricting hidden to 64 prevents VRAM exhaustion on 4GB GPUs while `torch.cuda.empty_cache()` prevents allocator fragmentation | `ml/gnn/train.py::tune_gnn` |
| D33 | EvolveGCN-O recurrent weight evolution | GCN layer weight matrices evolve via GRU cells across chronological snapshots without requiring node persistence across time steps | `ml/temporal/evolvegcn.py::EvolveGCNO`, `tests/test_temporal.py` |
| D34 | Truncated BPTT windowing ($K=5$) | Backpropagation through time bounded to 5 snapshot windows via tensor detachment, preventing gradient explosion and stabilizing RNN training across 34 snapshots | `ml/temporal/train.py::train_one_epoch`, `tests/test_temporal.py` |
| D35 | Control B leak-free snapshot context tracker | Augments static GCN with 4 unlabelled graph structural stats ($N, E$, density, mean degree) and 4 autoregressive model-own prediction signals ($t-1..t-3$, window mean); never accesses ground-truth labels of current, val, or test snapshots | `ml/temporal/snapshot_context.py::SnapshotContextTracker`, `tests/test_temporal.py` |
| D36 | Temporal seed-paired `paper_refit` retraining | Refit on steps 1–34 uses seed $k$'s exact paper validation threshold, HPs, and best epoch count without looking at test data (AGENTS.md rule 7c) | `ml/temporal/train.py::run_temporal_experiment` |
| D37 | SMOTE & oversampling restricted to training set | Synthetic illicit samples generated strictly on training split nodes after feature scaling; validation and test sets remain unperturbed natural distributions | `ml/experiments/exp4_imbalance.py`, `tests/test_experiments.py` |
| D38 | Numerically stable Focal Loss formulation | Focal Loss with tunable $\gamma \in \{1, 2\}$ and inverse class weights operates on raw logits with log-sum-exp; penalizes easy licit negatives | `ml/experiments/exp4_imbalance.py::FocalLoss`, `tests/test_experiments.py::test_focal_loss_forward_backward` |
| D39 | Topology-only constant feature ablation | Node features replaced by constant scalar ($1.0$) on identical period graphs to test pure graph structural inductive bias in isolation | `ml/experiments/exp5_ablations.py` |
| D40 | Edge symmetrization vs directed flow ablation | Directed edge flow ($u \to v$) compared to symmetrized undirected graph ($u \leftrightarrow v$); confirms bidirectional message propagation is necessary for 2-hop GNNs | `ml/experiments/exp5_ablations.py`, `tests/test_experiments.py::test_directed_vs_undirected_period_graph` |
| D41 | Receptive field hop ablation (0-hop vs 1-hop vs 2-hop) | Evaluates 0-hop MLP, 1-hop GCN, and 2-hop GCN on identical splits and seeds to quantify the value of multi-hop transaction neighborhoods | `ml/experiments/exp5_ablations.py`, `tests/test_experiments.py::test_receptive_field_architectures` |




