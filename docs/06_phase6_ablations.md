# Phase 6: Generalisation Across Cuts, Imbalance Strategies, and System Ablations

## 1. Executive Summary

Phase 6 provides empirical answers to the central questions of transaction graph classification under non-stationary conditions:
1. **Generalisation across temporal regimes (Exp 3):** Performance is strongly non-monotonic across forward-chaining cuts. When the Darknet market shutdown hits in test steps 45–49 (`fc_44`), graph neural networks collapse to near-random PR-AUC (~0.08–0.11), while XGBoost exhibits strong resilience (PR-AUC 0.4042 ± 0.0345).
2. **Class imbalance mitigation (Exp 4):** Standard unweighted loss collapses GCN PR-AUC from 0.6084 to 0.3715. Inverse class frequency weighting is essential for graph neural networks. In contrast, XGBoost handles the ~1:9 imbalance effectively with native `scale_pos_weight`, with SMOTE oversampling pushing PR-AUC to 0.7975 ± 0.0046.
3. **Ablation studies (Exp 5):** Graph topology without node features collapses to 0.0764 PR-AUC (features are indispensable). Bidirectional edge symmetrization outperforms directed message passing by +0.1522 PR-AUC (0.6228 vs 0.4706). Expanding the receptive field from 1-hop (0.2029) to 2-hop (0.6228) is critical to capture multi-hop money laundering aggregation patterns.

All models and ablations are evaluated across 5 random seeds (seeds 0–4) with leak-free chronological preprocessing.

---

## 2. Experiment 3: Generalisation Across Temporal Cuts

We evaluate all six primary architectures across the 5 forward-chaining cuts defined in `ml/preprocessing/splits.py`:
- `fc_24` (train 1–19, val 20–24, test 25–29)
- `fc_29` (train 1–24, val 25–29, test 30–34)
- `fc_34` (train 1–29, val 30–34, test 35–39)
- `fc_39` (train 1–34, val 35–39, test 40–44)
- `fc_44` (train 1–39, val 40–44, test 45–49)

### 2.1 Generalisation Benchmark Table

Below is the verbatim result table from [exp3_generalisation.md](file:///Z:/graph/GraphGuard/reports/tables/exp3_generalisation.md):

| split   | model     |   n_seeds | pr_auc          | f1              |
|:--------|:----------|----------:|:----------------|:----------------|
| fc_24   | CONTROL_B |         5 | 0.7048 ± 0.0027 | 0.6601 ± 0.0065 |
| fc_24   | EVOLVEGCN |         5 | 0.4989 ± 0.0680 | 0.3473 ± 0.1343 |
| fc_24   | GCN       |         5 | 0.6953 ± 0.0046 | 0.6525 ± 0.0047 |
| fc_24   | RF        |         5 | 0.7600 ± 0.0049 | 0.7466 ± 0.0030 |
| fc_24   | SAGE_MEAN |         5 | 0.7306 ± 0.0034 | 0.6808 ± 0.0028 |
| fc_24   | XGB       |         5 | 0.7827 ± 0.0073 | 0.7457 ± 0.0058 |
| fc_29   | CONTROL_B |         5 | 0.7263 ± 0.0202 | 0.7313 ± 0.0112 |
| fc_29   | EVOLVEGCN |         5 | 0.3046 ± 0.0317 | 0.3737 ± 0.0112 |
| fc_29   | GCN       |         5 | 0.7321 ± 0.0331 | 0.7546 ± 0.0137 |
| fc_29   | RF        |         5 | 0.9885 ± 0.0005 | 0.9636 ± 0.0021 |
| fc_29   | SAGE_MEAN |         5 | 0.8706 ± 0.0317 | 0.8505 ± 0.0102 |
| fc_29   | XGB       |         5 | 0.9793 ± 0.0020 | 0.9467 ± 0.0043 |
| fc_34   | CONTROL_B |         5 | 0.6226 ± 0.0649 | 0.6039 ± 0.0415 |
| fc_34   | EVOLVEGCN |         5 | 0.2255 ± 0.0268 | 0.3057 ± 0.0177 |
| fc_34   | GCN       |         5 | 0.6855 ± 0.0329 | 0.6691 ± 0.0236 |
| fc_34   | RF        |         5 | 0.9510 ± 0.0013 | 0.9368 ± 0.0046 |
| fc_34   | SAGE_MEAN |         5 | 0.6705 ± 0.0497 | 0.6439 ± 0.0259 |
| fc_34   | XGB       |         5 | 0.9542 ± 0.0009 | 0.8934 ± 0.0125 |
| fc_39   | CONTROL_B |         5 | 0.6313 ± 0.0279 | 0.6213 ± 0.0139 |
| fc_39   | EVOLVEGCN |         5 | 0.1620 ± 0.0310 | 0.2090 ± 0.0721 |
| fc_39   | GCN       |         5 | 0.6235 ± 0.0193 | 0.6322 ± 0.0136 |
| fc_39   | RF        |         5 | 0.8014 ± 0.0005 | 0.8102 ± 0.0022 |
| fc_39   | SAGE_MEAN |         5 | 0.6573 ± 0.0102 | 0.6734 ± 0.0078 |
| fc_39   | XGB       |         5 | 0.8014 ± 0.0116 | 0.8132 ± 0.0138 |
| fc_44   | CONTROL_B |         5 | 0.1087 ± 0.0147 | 0.0093 ± 0.0076 |
| fc_44   | EVOLVEGCN |         5 | 0.0678 ± 0.0179 | 0.0714 ± 0.0469 |
| fc_44   | GCN       |         5 | 0.1152 ± 0.0056 | 0.0155 ± 0.0002 |
| fc_44   | RF        |         5 | 0.1888 ± 0.0107 | 0.0352 ± 0.0061 |
| fc_44   | SAGE_MEAN |         5 | 0.0888 ± 0.0078 | 0.0282 ± 0.0062 |
| fc_44   | XGB       |         5 | 0.4042 ± 0.0345 | 0.1021 ± 0.0166 |

### 2.2 Regime Shifts and Darknet Market Shutdown Analysis

The visual analysis of model performance across cuts is documented in:
- [exp3_generalisation_by_cut.png](file:///Z:/graph/GraphGuard/reports/figures/exp3_generalisation_by_cut.png)
- [exp3_per_step_prauc.png](file:///Z:/graph/GraphGuard/reports/figures/exp3_per_step_prauc.png)

Key observations:
1. **Easy Separation in Steps 30–34 (`fc_29`):** RF (0.9885) and XGB (0.9793) achieve near-perfect discrimination on `fc_29`. As established in Decision D23, this is genuine non-stationary clustering: the paper split validation window covers the exact same steps and yields 0.99 PR-AUC.
2. **Post-Shutdown Collapse (`fc_44`):** At step 43 (corresponding to the shutdown of major darknet entities), transaction graph topology and illicit transaction characteristics underwent a drastic structural shift.
   - GCN PR-AUC drops from 0.6235 on `fc_39` to 0.1152 on `fc_44`.
   - Control B drops from 0.6313 to 0.1087.
   - GraphSAGE drops to 0.0888.
   - XGBoost drops to 0.4042, maintaining the highest signal among all candidates. Its tree-based axis-aligned splits on engineered local features prove far more resilient to topological disruption than message-passing graph convolutions.

---

## 3. Experiment 4: Class Imbalance Mitigation Strategies

We benchmarked 5 distinct strategies on the primary benchmark split `paper_refit` using identical seed schedules (seeds 0–4):
1. `none`: Standard cross-entropy without weight adjustment.
2. `class_weights`: Balanced inverse class frequencies (`w_1 = N / (2 * N_pos)`).
3. `oversampling_random`: Random duplicate sampling of the illicit class on the training split only.
4. `oversampling_smote`: Synthetic Minority Over-sampling Technique (k_neighbors=5) fitted exclusively on training nodes (Decision D37).
5. `focal_loss`: Dynamic modulation with $\gamma \in \{1, 2\}$ down-weighting well-classified licit negatives (Decision D38).

### 3.1 Imbalance Results Table

Below is the verbatim result table from [exp4_imbalance.md](file:///Z:/graph/GraphGuard/reports/tables/exp4_imbalance.md):

| model   | strategy            |   n_seeds | pr_auc          | f1 (val thr)    | f1 (0.5)        | precision (val thr)   | precision (0.5)   | recall (val thr)   | recall (0.5)    | val_threshold   |
|:--------|:--------------------|----------:|:----------------|:----------------|:----------------|:----------------------|:------------------|:-------------------|:----------------|:----------------|
| XGB     | none                |         5 | 0.7856 ± 0.0066 | 0.7120 ± 0.0833 | 0.7945 ± 0.0071 | 0.7122 ± 0.1823       | 0.8846 ± 0.0146   | 0.7335 ± 0.0148    | 0.7211 ± 0.0024 | 0.367 ± 0.146   |
| XGB     | class_weights       |         5 | 0.7811 ± 0.0043 | 0.7071 ± 0.1073 | 0.7529 ± 0.0223 | 0.7476 ± 0.2394       | 0.7771 ± 0.0471   | 0.7119 ± 0.0498    | 0.7313 ± 0.0033 | 0.497 ± 0.223   |
| XGB     | oversampling_random |         5 | 0.7841 ± 0.0017 | 0.6909 ± 0.1152 | 0.7622 ± 0.0127 | 0.7043 ± 0.2473       | 0.7963 ± 0.0306   | 0.7245 ± 0.0362    | 0.7315 ± 0.0036 | 0.445 ± 0.208   |
| XGB     | oversampling_smote  |         5 | 0.7975 ± 0.0046 | 0.7808 ± 0.0208 | 0.7588 ± 0.0214 | 0.8525 ± 0.0653       | 0.7900 ± 0.0486   | 0.7223 ± 0.0089    | 0.7311 ± 0.0022 | 0.568 ± 0.047   |
| GCN     | none                |         5 | 0.3715 ± 0.1096 | 0.3376 ± 0.1106 | 0.3194 ± 0.0922 | 0.4543 ± 0.1192       | 0.4744 ± 0.1648   | 0.2717 ± 0.1015    | 0.2425 ± 0.0656 | 0.459 ± 0.058   |
| GCN     | class_weights       |         5 | 0.6084 ± 0.0316 | 0.5482 ± 0.0439 | 0.6115 ± 0.0299 | 0.8177 ± 0.0610       | 0.6379 ± 0.0654   | 0.4172 ± 0.0612    | 0.5906 ± 0.0220 | 0.810 ± 0.035   |
| GCN     | focal_loss_gamma1   |         5 | 0.3167 ± 0.0517 | 0.2315 ± 0.1010 | 0.0460 ± 0.0248 | 0.4501 ± 0.1612       | 0.7908 ± 0.2986   | 0.1732 ± 0.1008    | 0.0240 ± 0.0130 | 0.238 ± 0.027   |
| GCN     | focal_loss_gamma2   |         5 | 0.3174 ± 0.0730 | 0.2255 ± 0.0806 | 0.0528 ± 0.0213 | 0.4552 ± 0.1638       | 0.7407 ± 0.3539   | 0.1581 ± 0.0719    | 0.0279 ± 0.0109 | 0.312 ± 0.024   |

### 3.2 Imbalance Analysis & Insights

1. **GCN Sensitivity to Imbalance:**
   - Without class weighting (`none`), GCN training gradients are overwhelmed by the dominant licit majority (~90% of labelled nodes), causing severe under-fitting of illicit patterns (PR-AUC 0.3715 ± 0.1096).
   - Inverse class weighting (`class_weights`) rescues GCN, lifting PR-AUC to 0.6084 ± 0.0316 and precision at val threshold to 0.8177 ± 0.0610.
   - Focal Loss ($\gamma=1, 2$) degraded GCN performance down to 0.3167–0.3174. In sparse graph transaction regimes with noisy licit majority transactions, aggressive focal down-weighting weakens the gradients necessary to separate illicit boundary nodes.
2. **XGBoost Robustness & SMOTE:**
   - XGBoost exhibits high baseline performance across all treatments (0.7811–0.7975).
   - SMOTE achieves the highest overall PR-AUC for XGBoost (0.7975 ± 0.0046) and high precision (0.8525 ± 0.0653), demonstrating that local synthetic interpolation on illicit feature space successfully reinforces decision boundaries without distorting tabular relationships.

---

## 4. Experiment 5: System Ablation Studies

To isolate the individual contribution of features, graph topology, message passing hops, edge directionality, and temporal recurrence, we executed targeted ablations on `paper_refit` across 5 random seeds.

### 4.1 Ablations Benchmark Table

Below is the verbatim result table from [exp5_ablations.md](file:///Z:/graph/GraphGuard/reports/tables/exp5_ablations.md):

| Ablation Dimension   | Variant / Condition            |   Seeds | PR-AUC (mean ± std)   | F1 (mean ± std)   |
|:---------------------|:-------------------------------|--------:|:----------------------|:------------------|
| 1. Features          | XGB (aggregated only, 72d)     |       5 | 0.6364 ± 0.0219       | 0.5716 ± 0.0328   |
| 1. Features          | GCN (aggregated only, 72d)     |       5 | 0.4143 ± 0.0496       | 0.4377 ± 0.0247   |
| 2. Topology Only     | GCN (constant 1s, no features) |       5 | 0.0764 ± 0.0009       | 0.1031 ± 0.0576   |
| 3. Neighbour Hops    | GCN 1-hop (1 layer)            |       5 | 0.2029 ± 0.0130       | 0.3135 ± 0.0182   |
| 4. Edge Direction    | GCN (directed edges)           |       5 | 0.4706 ± 0.0206       | 0.4745 ± 0.0213   |
| 5. Temporal Dynamics | EvolveGCN (frozen weights)     |       5 | 0.0787 ± 0.0036       | 0.1428 ± 0.0050   |

### 4.2 Comprehensive Synthesis of Ablations

1. **Features vs Topology:**
   - Replacing all features with a constant scalar ($1.0$, pure topology) causes GCN PR-AUC to collapse completely to **0.0764 ± 0.0009** (base rate of illicit nodes is ~0.098). Pure Bitcoin transaction graph structure without transaction attributes contains near-zero illicit discriminative signal.
   - Using only aggregated features (72 dimensions) yields 0.6364 for XGBoost and 0.4143 for GCN, substantially below full feature sets (0.8001 for XGBoost and 0.6228 for GCN). Local transaction features (inputs, outputs, transaction fees) are indispensable.
2. **Receptive Field (Hops):**
   - 0-hop MLP: PR-AUC 0.5236 ± 0.0125.
   - 1-hop GCN: PR-AUC 0.2029 ± 0.0130.
   - 2-hop GCN: PR-AUC 0.6228 ± 0.0252.
   - *Key Insight:* A 1-hop GCN blurs immediate neighbor features without sufficient context, performing worse than a 0-hop MLP. A 2-hop GCN aggregates across the intermediate transaction chain (e.g. payer $\to$ intermediate $\to$ payee), yielding a +0.4199 PR-AUC jump over 1-hop.
3. **Edge Symmetrization (Directed vs Undirected):**
   - Symmetrized undirected edges: PR-AUC 0.6228 ± 0.0252.
   - Strictly directed transaction edges ($u \to v$): PR-AUC 0.4706 ± 0.0206.
   - *Key Insight:* In money laundering, flow of funds goes downstream ($u \to v$), but investigation requires upstream tracing ($v \leftarrow u$). Bidirectional propagation allows GNNs to accumulate context from both incoming sources and outgoing destinations.
4. **Temporal Recurrence (EvolveGCN):**
   - Evolving weights (GRUCell): PR-AUC 0.1371 ± 0.0379.
   - Frozen weights: PR-AUC 0.0787 ± 0.0036.
   - *Key Insight:* While weight adaptation via GRU yields a slight lift over frozen weights (+0.0584), both variants are severely crippled by the absence of node identity persistence across snapshots.

---

## 5. Decision Log Cross-Reference

- **D37:** SMOTE oversampling applied only to training set nodes after scaling; validation and test sets remain unperturbed.
- **D38:** Numerically stable Focal Loss formulation with tunable $\gamma$.
- **D39:** Topology-only constant feature baseline protocol.
- **D40:** Edge symmetrization evaluation protocol.
- **D41:** Receptive field hop ablation protocol.

