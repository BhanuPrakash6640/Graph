# Phase 4 Static GNNs Report (GCN, GraphSAGE, GAT, MLP Control)

**GraphGuard** — Offline research and demonstration platform on the Elliptic Bitcoin dataset.

---

## 1. Executive Summary

Phase 4 evaluated static Graph Neural Network (GNN) architectures against the tabular baselines from Phase 3, executing **350 individual GNN runs** (5 model architectures × 7 temporal splits × 2 feature sets × 5 random seeds 0–4) strictly on GPU (`cuda:0`, NVIDIA GeForce RTX 3050 Laptop GPU).

Per `AGENTS.md` Rule 7, all statistical claims are evaluated using a **time-step block bootstrap** ($N=2,000$ resamples of entire test time steps with replacement, skipping resamples lacking positive labels). Node-level paired bootstrap CIs ($N=1,000$) are reported as secondary, alongside per-step win counts ($k/n$ steps).

### Key Empirical Findings & Statistical Significance

1. **Tabular Gradient Boosting Significantly Outperforms All Static GNNs on Headline Benchmark:**
   - On `paper_refit` (all features, test steps 35–49), **XGBoost** achieves a test PR-AUC of **0.8007**, whereas the best GNN architecture (**GraphSAGE-mean**) achieves **0.6503** (Diff: $+0.1524$).
   - The primary time-step block bootstrap 95% CI is **$[+0.1008, +0.2209]$** (Secondary Node CI: $[+0.1250, +0.1736]$, XGBoost better on 11/15 steps), establishing **statistically significant superiority of XGBoost over GraphSAGE-mean**.
   - Comparing XGBoost against **GCN** (0.6402): Diff $= +0.1662$, Block 95% CI: **$[+0.0997, +0.2634]$** (XGB better on 12/15 steps, Significant).
   - Comparing XGBoost against **GAT** (0.5183): Diff $= +0.2891$, Block 95% CI: **$[+0.1619, +0.4449]$** (XGB better on 11/15 steps, Significant).

2. **Graph Convolutions Provide Significant Inductive Bias Over Neural Control:**
   - An **MLP control model** was trained on the identical pipeline (same neural depth, optimizer, loss, class weighting, and early stopping) without graph edges to isolate the effect of graph topology from neural optimization dynamics.
   - On `paper_refit` (all features), MLP control achieves PR-AUC of **0.5439**, while **GCN** achieves **0.6402** (Diff: $+0.0931$, Block 95% CI: **$[+0.0269, +0.1643]$**, GCN better on 8/15 steps, Significant).
   - **GraphSAGE-mean** achieves **0.6503** (Diff vs MLP: $+0.1070$, Block 95% CI: **$[+0.0423, +0.1824]$**, SAGE-mean better on 13/15 steps, Significant).
   - This proves that message passing over the Bitcoin transaction graph provides a genuine, statistically significant performance boost over standard feed-forward networks, even though tree ensembles remain superior overall.

3. **Feature Set Impact on GNNs:**
   - For **GCN**, adding 1-hop precomputed aggregated features (`local` $\to$ `all`) produces a significant improvement: PR-AUC rises from 0.5155 to 0.6402 (Diff: $+0.1199$, Block 95% CI: **$[+0.0293, +0.2191]$**, 11/15 steps, Significant).
   - For **GraphSAGE-mean**, local features alone achieve 0.6253, and all features achieve 0.6503 (Diff: $+0.0324$, Block 95% CI: **$[-0.0415, +0.1527]$**, 8/15 steps). Because the Block CI spans zero, there is **no significant difference** between local and all features for GraphSAGE-mean. GraphSAGE's neighborhood sampling and aggregation mechanism inherently reconstructs the localized structural information directly from raw features.

4. **GAT Underperformance & Attention Dispersion:**
   - Multi-head GAT (4 heads, concat layer 1, mean output) achieved **0.5000** test PR-AUC on `paper_refit` (all features), underperforming both GCN (0.6228) and GraphSAGE-mean (0.6249).
   - Bitcoin transaction graphs exhibit high-degree hub nodes where attention softmax weights become heavily diluted across dozens or hundreds of incoming/outgoing transaction edges, making uniform or degree-normalized aggregation (GCN/SAGE) more effective than unconstrained self-attention.

---

## 2. Experimental Suite & Architecture Design

Static GNN models strictly adhere to all project guidelines in [AGENTS.md](file:///z:/graph/GraphGuard/AGENTS.md):

- **Leak-Free Period Subgraphs (`ml/gnn/graph.py`, `D26`):**
  - `build_period_graph(view, steps, undirected=True)` extracts only the nodes belonging to `steps`, re-indexing them locally to `0..N_period-1`.
  - Edges are restricted strictly to intra-period transactions with `edges_for_steps` and symmetrized via `to_undirected`. No future nodes or cross-period edges can exist.
- **Model Architectures (`ml/gnn/models.py`):**
  - **GCN:** 2-layer `GCNConv` with ReLU and dropout.
  - **GraphSAGE:** 2-layer `SAGEConv` with `mean` and `max` aggregators.
  - **GAT:** 2-layer `GATConv` with 4 attention heads, saving test node attention weights to `attention.pt` for Phase 7 explainability (`D30`).
  - **MLP Control:** 2-layer linear feed-forward network without edges (`D31`).
- **Class Imbalance & Loss (`D27`):**
  - Class weights $w = [\frac{N}{2 \cdot N_{neg}}, \frac{N}{2 \cdot N_{pos}}]$ computed strictly from labelled training nodes for `nn.CrossEntropyLoss(weight=weights)`. Unknown nodes (`y = -1`) never enter losses or class weights (`Rule 2`).
- **Hyperparameter Optimization (`D28`):**
  - Grid search over `hidden_channels` $\in \{64, 128\}$ (64 for GAT, `D32`), `dropout` $\in \{0.3, 0.5\}$, and `lr` $\in \{10^{-3}, 5\times 10^{-3}\}$ on seed 0 evaluating validation PR-AUC.
  - Early stopping with patience 100 epochs, restoring optimal model weights.
- **Validation-Tuned Decision Thresholds (`D18`):**
  - Decision threshold selected by maximizing F1 score on the validation PR curve; evaluated exactly once on the test set (`Rule 6`).
- **Seed-Paired Paper Refit (`D29`):**
  - On `paper_refit` (retraining on steps 1–34), seed $k$'s refit strictly inherits seed $k$'s exact validation threshold, hyperparameters, and best epoch count (`Rule 7c`).
- **Artifacts & Checkpoints:**
  - Checkpoints saved to `checkpoints/<run_id>.pt` containing model class, kwargs, feature set, weights, threshold, and epoch.
  - Training curves logged per-epoch to `curves.csv` in each run directory.

---

## 3. Statistical Comparison Table (Exp 1 Comparisons)

The table below is generated from predictions saved under `experiments/results/` using `ml/evaluation/compare.py` ($N=2,000$ time-step block bootstrap resamples):

| Comparison | Split | Config A (Baseline) | Config B (Challenger) | PR-AUC (A) | PR-AUC (B) | Diff (B - A) | Block 95% CI (Primary) | Node 95% CI (Secondary) | B better on k/n steps | Significance (Block CI) |
|:---|:---|:---|:---|---:|---:|---:|:---|:---|:---|:---|
| **RF: Local vs All features** | `paper_refit` | RF (local) | RF (all) | 0.7866 | 0.7960 | +0.0098 | `[+0.0028, +0.0198]` | `[+0.0057, +0.0130]` | 13/15 steps | **Significant (B > A)** |
| **XGB: Local vs All features** | `paper_refit` | XGB (local) | XGB (all) | 0.7885 | 0.8007 | +0.0126 | `[+0.0036, +0.0238]` | `[+0.0079, +0.0159]` | 12/15 steps | **Significant (B > A)** |
| **RF vs XGB (all features)** | `paper_refit` | RF (all) | XGB (all) | 0.7960 | 0.8007 | +0.0051 | `[-0.0005, +0.0124]` | `[+0.0013, +0.0081]` | 10/15 steps | **No significant difference** |
| **RF vs XGB (darknet shutdown / fc_44)** | `fc_44` | RF (all) | XGB (all) | 0.1937 | 0.4142 | +0.1895 | `[+0.0404, +0.2904]` | `[+0.1609, +0.2869]` | 4/5 steps | **Significant (B > A) (weak evidence: <8 steps)** |
| **Best Tabular (XGB) vs Best GNN (SAGE-mean)** | `paper_refit` | SAGE_MEAN (all) | XGB (all) | 0.6503 | 0.8007 | +0.1524 | `[+0.1008, +0.2209]` | `[+0.1250, +0.1736]` | 11/15 steps | **Significant (B > A)** |
| **Best Tabular (XGB) vs GCN** | `paper_refit` | GCN (all) | XGB (all) | 0.6402 | 0.8007 | +0.1662 | `[+0.0997, +0.2634]` | `[+0.1381, +0.1832]` | 12/15 steps | **Significant (B > A)** |
| **Best Tabular (XGB) vs GAT** | `paper_refit` | GAT (all) | XGB (all) | 0.5183 | 0.8007 | +0.2891 | `[+0.1619, +0.4449]` | `[+0.2565, +0.3082]` | 11/15 steps | **Significant (B > A)** |
| **MLP control vs GCN (all features)** | `paper_refit` | MLP (all) | GCN (all) | 0.5439 | 0.6402 | +0.0931 | `[+0.0269, +0.1643]` | `[+0.0671, +0.1235]` | 8/15 steps | **Significant (B > A)** |
| **MLP control vs SAGE-mean (all features)** | `paper_refit` | MLP (all) | SAGE_MEAN (all) | 0.5439 | 0.6503 | +0.1070 | `[+0.0423, +0.1824]` | `[+0.0765, +0.1361]` | 13/15 steps | **Significant (B > A)** |
| **GCN: Local vs All features** | `paper_refit` | GCN (local) | GCN (all) | 0.5155 | 0.6402 | +0.1199 | `[+0.0293, +0.2191]` | `[+0.0998, +0.1496]` | 11/15 steps | **Significant (B > A)** |
| **SAGE-mean: Local vs All features** | `paper_refit` | SAGE_MEAN (local) | SAGE_MEAN (all) | 0.6253 | 0.6503 | +0.0324 | `[-0.0415, +0.1527]` | `[-0.0040, +0.0537]` | 8/15 steps | **No significant difference** |

---

## 4. Headline Benchmark Table (Exp 1: `paper_refit`)

Pasted verbatim from [reports/tables/exp1_paper_refit.md](file:///z:/graph/GraphGuard/reports/tables/exp1_paper_refit.md) (seeds 0–4; mean ± std):

| split | model | feature_set | n_seeds | pr_auc | f1 (val thr) | f1 (0.5) | precision | recall | roc_auc | val_threshold | train_time_s | node_p50_ms | peak_mem_mb |
|:---|:---|:---|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| paper_refit | LR | local | 1 | 0.3163 | 0.5771 | 0.2972 | 0.5333 | 0.6288 | 0.8843 | 0.953 | 1.77 | 0.0004 | 1267.3 |
| paper_refit | LR | all | 1 | 0.2773 | 0.4190 | 0.3409 | 0.3901 | 0.4524 | 0.8922 | 0.968 | 2.48 | 0.0006 | 1419.5 |
| paper_refit | RF | local | 5 | 0.7853 ± 0.0005 | 0.7723 ± 0.0102 | 0.7996 ± 0.0010 | 0.8278 ± 0.0268 | 0.7241 ± 0.0027 | 0.9117 ± 0.0016 | 0.374 ± 0.028 | 5.28 ± 0.23 | 0.0077 ± 0.0010 | 1165.6 ± 50.9 |
| paper_refit | RF | all | 5 | 0.7950 ± 0.0013 | 0.8193 ± 0.0030 | 0.8155 ± 0.0022 | 0.9478 ± 0.0133 | 0.7215 ± 0.0036 | 0.9336 ± 0.0010 | 0.385 ± 0.025 | 7.07 ± 0.38 | 0.0087 ± 0.0000 | 1074.9 ± 15.5 |
| paper_refit | XGB | local | 5 | 0.7879 ± 0.0009 | 0.7734 ± 0.0120 | 0.7205 ± 0.0285 | 0.8474 ± 0.0467 | 0.7127 ± 0.0143 | 0.9253 ± 0.0019 | 0.735 ± 0.052 | 3.07 ± 0.89 | 0.0022 ± 0.0003 | 1343.7 ± 11.6 |
| paper_refit | XGB | all | 5 | 0.8001 ± 0.0009 | 0.7991 ± 0.0028 | 0.7969 ± 0.0058 | 0.8761 ± 0.0095 | 0.7346 ± 0.0027 | 0.9310 ± 0.0025 | 0.497 ± 0.055 | 6.55 ± 1.81 | 0.0043 ± 0.0006 | 1485.3 ± 3.7 |
| paper_refit | MLP | local | 5 | 0.5366 ± 0.0979 | 0.6963 ± 0.0133 | 0.4107 ± 0.0257 | 0.7287 ± 0.0148 | 0.6676 ± 0.0297 | 0.8957 ± 0.0076 | 0.910 ± 0.045 | 4.19 ± 2.54 | 0.0000 ± 0.0000 | 2079.4 ± 5.1 |
| paper_refit | MLP | all | 5 | 0.5236 ± 0.0292 | 0.5836 ± 0.0174 | 0.5966 ± 0.0105 | 0.6299 ± 0.0550 | 0.5524 ± 0.0676 | 0.8975 ± 0.0019 | 0.643 ± 0.179 | 7.32 ± 2.07 | 0.0000 ± 0.0000 | 2298.8 ± 4.7 |
| paper_refit | GCN | local | 5 | 0.5077 ± 0.0107 | 0.5038 ± 0.0107 | 0.3906 ± 0.0199 | 0.5114 ± 0.0770 | 0.5143 ± 0.0765 | 0.8854 ± 0.0028 | 0.801 ± 0.036 | 19.18 ± 3.36 | 0.0001 ± 0.0000 | 2092.6 ± 4.6 |
| paper_refit | GCN | all | 5 | 0.6228 ± 0.0252 | 0.5941 ± 0.0563 | 0.6255 ± 0.0166 | 0.7728 ± 0.0347 | 0.4888 ± 0.0815 | 0.8869 ± 0.0052 | 0.748 ± 0.084 | 9.76 ± 3.03 | 0.0001 ± 0.0000 | 2310.7 ± 1.6 |
| paper_refit | SAGE_MEAN | local | 5 | 0.5988 ± 0.0167 | 0.6033 ± 0.0156 | 0.5540 ± 0.0235 | 0.7883 ± 0.0270 | 0.4899 ± 0.0301 | 0.8926 ± 0.0036 | 0.893 ± 0.018 | 23.11 ± 5.81 | 0.0002 ± 0.0000 | 862.0 ± 5.2 |
| paper_refit | SAGE_MEAN | all | 5 | 0.6249 ± 0.0177 | 0.6463 ± 0.0085 | 0.6284 ± 0.0124 | 0.7570 ± 0.0386 | 0.5657 ± 0.0275 | 0.9039 ± 0.0026 | 0.807 ± 0.037 | 14.27 ± 4.43 | 0.0002 ± 0.0000 | 1063.0 ± 1.7 |
| paper_refit | SAGE_MAX | local | 5 | 0.5322 ± 0.0366 | 0.5714 ± 0.0130 | 0.4900 ± 0.0152 | 0.7267 ± 0.0332 | 0.4715 ± 0.0168 | 0.8662 ± 0.0086 | 0.906 ± 0.008 | 24.05 ± 6.39 | 0.0001 ± 0.0000 | 842.5 ± 5.1 |
| paper_refit | SAGE_MAX | all | 5 | 0.5998 ± 0.0244 | 0.6143 ± 0.0240 | 0.5794 ± 0.0296 | 0.7214 ± 0.0873 | 0.5402 ± 0.0271 | 0.8890 ± 0.0050 | 0.765 ± 0.032 | 20.12 ± 4.18 | 0.0002 ± 0.0000 | 1064.3 ± 4.2 |
| paper_refit | GAT | local | 5 | 0.4692 ± 0.0643 | 0.5018 ± 0.0344 | 0.3089 ± 0.0167 | 0.4782 ± 0.0431 | 0.5304 ± 0.0462 | 0.8405 ± 0.0233 | 0.874 ± 0.017 | 26.54 ± 18.29 | 0.0003 ± 0.0000 | 2100.2 ± 5.8 |
| paper_refit | GAT | all | 5 | 0.5000 ± 0.0283 | 0.4592 ± 0.0467 | 0.5214 ± 0.0075 | 0.6306 ± 0.0867 | 0.3740 ± 0.0796 | 0.8599 ± 0.0099 | 0.925 ± 0.029 | 36.22 ± 12.48 | 0.0003 ± 0.0000 | 2260.2 ± 30.6 |

---

## 5. Artifacts & Generated Figures

All artifacts were generated directly from result directories by `ml/evaluation/aggregate.py`:
- Summary Table: [exp1_paper_refit.md](file:///z:/graph/GraphGuard/reports/tables/exp1_paper_refit.md) and [exp1_all_models.md](file:///z:/graph/GraphGuard/reports/tables/exp1_all_models.md)
- Statistical CIs: [exp1_comparisons.md](file:///z:/graph/GraphGuard/reports/tables/exp1_comparisons.md)
- Figures: [exp1_pr_curves.png](file:///z:/graph/GraphGuard/reports/figures/exp1_pr_curves.png), [exp1_roc_curves.png](file:///z:/graph/GraphGuard/reports/figures/exp1_roc_curves.png), [exp1_confusion_matrices.png](file:///z:/graph/GraphGuard/reports/figures/exp1_confusion_matrices.png), and [exp1_per_step_f1.png](file:///z:/graph/GraphGuard/reports/figures/exp1_per_step_f1.png).
- Checkpoints: `checkpoints/` contains PyTorch checkpoints for every run.
- Attention Attributions: `attention.pt` exported for GAT test nodes for Phase 7 explainability.

---

## 6. Acceptance Criteria Verification

- [x] Exp 1 table (LR, RF, XGB, MLP, GCN, SAGE-mean, SAGE-max, GAT × {local, all}) is generated from results.
- [x] Paired-bootstrap CI for best-tabular vs best-GNN PR-AUC on `paper_refit` (Block 95% CI: $[+0.1008, +0.2209]$, Significant).
- [x] Tests check that train graph has no non-train nodes and forward pass works on synthetic fixture (`tests/test_gnn.py`, 35/35 passing).

