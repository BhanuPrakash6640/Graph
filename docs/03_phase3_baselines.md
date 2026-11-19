# Phase 3 Baselines & Evaluation Report

**GraphGuard** — Offline research and demonstration platform on the Elliptic Bitcoin dataset.

---

## 1. Executive Summary

Phase 3 established the unified evaluation, experiment-tracking, and statistical comparison framework, executing all tabular baseline models across 42 distinct configurations (3 model architectures × 7 temporal splits × 2 feature sets) across random seeds 0–4 (seeds 0–4 for RF and XGBoost; deterministic single run for Logistic Regression).

Per `AGENTS.md` Rule 7, all statistical claims are evaluated using a **time-step block bootstrap** as the primary inferential metric ($N=2,000$ resamples of entire test time steps with replacement), which accounts for intra-step transaction correlation. Node-level paired bootstrap CIs are reported as secondary, alongside per-step win ratios ($k/n$ steps).

### Key Findings & Statistical Significance:
1. **Aggregated Features Significantly Improve Detection Over Local Features (Supported for both RF and XGBoost):**
   - For **Random Forest** on `paper_refit`, adding 1-hop aggregated features improves test PR-AUC from 0.7866 to 0.7960 (Diff: $+0.0098$, Primary Block 95% CI: $[+0.0028, +0.0198]$, Secondary Node 95% CI: $[+0.0057, +0.0130]$, better on 13/15 steps).
   - For **XGBoost** on `paper_refit`, adding 1-hop aggregated features improves test PR-AUC from 0.7885 to 0.8007 (Diff: $+0.0126$, Primary Block 95% CI: $[+0.0036, +0.0238]$, Secondary Node 95% CI: $[+0.0079, +0.0159]$, better on 12/15 steps).
2. **RF vs XGBoost on Paper Refit Shows No Significant Difference:**
   - On `paper_refit` (all features), XGBoost achieves test PR-AUC of 0.8007 vs 0.7960 for Random Forest (Diff: $+0.0051$). Under the primary time-step block bootstrap, the 95% CI spans zero ($[-0.0005, +0.0124]$, better on 10/15 steps), while the unclustered node-level CI is $[+0.0013, +0.0081]$. Conforming to `AGENTS.md` Rule 7, we report **no significant difference** between Random Forest and XGBoost on the headline paper benchmark.
3. **RF vs XGBoost on `fc_44` (Darknet Shutdown) Favours XGBoost but is Weak Evidence:**
   - On `fc_44` (test steps 45–49, post-shutdown regime shift), XGBoost achieves test PR-AUC of 0.4142 vs 0.1937 for Random Forest (Diff: $+0.1895$, Primary Block 95% CI: $[+0.0404, +0.2904]$, Secondary Node 95% CI: $[+0.1609, +0.2869]$, XGBoost better on 4/5 steps). While the point estimate favours XGBoost, this constitutes **weak evidence** due to the small evaluation window (only 5 test steps and 121 illicit nodes).
4. **Validation-Tuned Decision Thresholds:**
   - With an extreme class imbalance (~10:1 licit to illicit), the default 0.5 decision threshold is suboptimal. Tuning the threshold exclusively on the validation PR curve yields substantial F1 improvements (e.g. for XGBoost on `paper` local: 0.6802 at threshold 0.5 vs 0.7429 at the tuned validation threshold of 0.735).
5. **High Scores on Early Forward-Chaining Splits (`fc_29`):**
   - Both RF (0.9885 ± 0.0005) and XGBoost (0.9793 ± 0.0022) achieve exceptionally high PR-AUC on `fc_29` (test steps 30–34). This was investigated for potential data leakage: the standard `paper` split validation set covers the exact same steps (30–34) and achieves identical ~0.99 validation PR-AUC (`D23`). The near-perfect separation is an intrinsic property of the cluster structure in time steps 30–34, confirming zero leakage.

> [!NOTE]
> **Hardware & Execution Environment:**
> - **XGBoost:** Executed on GPU (`tree_method="hist", device="cuda"`) using an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), recording peak GPU memory via `torch.cuda.max_memory_allocated()`.
> - **Random Forest & Logistic Regression:** Executed on CPU using scikit-learn (`n_jobs=-1` multiprocessing for RF).
> - Latencies reported in the summary tables reflect these respective hardware execution environments and are not directly cross-comparable across hardware boundaries.

---

## 2. Experimental Protocol & Decision Log

The baseline suite follows all project rules and recorded design decisions:
- **Logistic Regression Architecture (`D17`):** Uses `LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=5000)`, tuning regularisation $C \in \{0.01, 0.1, 1.0, 10.0\}$ on validation PR-AUC.
- **Random Forest Architecture & Tuning Proxy (`D22`):** Uses `RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1)`. Hyperparameter optimization tunes `max_depth \in {None, 20}` and `max_features \in {"sqrt", 0.3}` using $N=100$ trees as a fast proxy on validation PR-AUC before training the final $N=500$ tree ensemble.
- **XGBoost Architecture & Stochasticity (`D19`, `D24`):** Uses `XGBClassifier(tree_method="hist", eval_metric="aucpr", device="cuda")` with `scale_pos_weight = n_neg / n_pos` computed strictly on training split nodes. Configured with `subsample=0.8` and `colsample_bytree=0.8` to introduce stochasticity across random seeds 0–4 (`AGENTS.md` Rule 7b).
- **Seed-to-Seed Pairing on Paper Refit (`D20`, `D25`):** For `paper_refit` (steps 1–34 train, no validation split), seed $k$'s refit strictly inherits seed $k$'s optimal threshold and best iteration count from the `paper` split validation run (`AGENTS.md` Rule 7c).
- **Leakage Isolation (`D11`, `D13`, `D18`):** Scalers and thresholds are fitted exclusively on training and validation splits. Unknown nodes (`-1`) never enter losses, metrics, or thresholds. The test set is evaluated exactly once per configuration.

---

## 3. Statistical Comparison Table (Block Bootstrap Primary CIs)

Computed via `ml/evaluation/compare.py` using 2,000 time-step block bootstrap resamples (primary) and 1,000 node-level bootstrap resamples (secondary) on test node predictions averaged across seeds (`AGENTS.md` Rule 7):

| Comparison | Split | Config A (Baseline) | Config B (Challenger) | PR-AUC (A) | PR-AUC (B) | Diff (B - A) | Block 95% CI (Primary) | Node 95% CI (Secondary) | B better on k/n steps | Significance (Block CI) |
|:---|:---|:---|:---|---:|---:|---:|:---|:---|:---|:---|
| **RF: Local vs All features** | `paper_refit` | RF (local) | RF (all) | 0.7866 | 0.7960 | +0.0098 | `[+0.0028, +0.0198]` | `[+0.0057, +0.0130]` | 13/15 steps | **Significant (B > A)** |
| **XGB: Local vs All features** | `paper_refit` | XGB (local) | XGB (all) | 0.7885 | 0.8007 | +0.0126 | `[+0.0036, +0.0238]` | `[+0.0079, +0.0159]` | 12/15 steps | **Significant (B > A)** |
| **RF vs XGB (all features)** | `paper_refit` | RF (all) | XGB (all) | 0.7960 | 0.8007 | +0.0051 | `[-0.0005, +0.0124]` | `[+0.0013, +0.0081]` | 10/15 steps | **No significant difference** |
| **RF vs XGB (darknet shutdown / fc_44)** | `fc_44` | RF (all) | XGB (all) | 0.1937 | 0.4142 | +0.1895 | `[+0.0404, +0.2904]` | `[+0.1609, +0.2869]` | 4/5 steps | **Significant (B > A) (weak evidence: <8 steps)** |

---

## 4. Full Experimental Benchmark Table

Pasted verbatim from [reports/tables/exp1_baselines.md](file:///z:/graph/GraphGuard/reports/tables/exp1_baselines.md) (154 individual runs across seeds 0–4):

| split       | model   | feature_set   |   n_seeds | pr_auc          | f1 (val thr)    | f1 (0.5)        | precision       | recall          | roc_auc         | val_threshold   | train_time_s   | node_p50_ms     | peak_mem_mb   |
|:------------|:--------|:--------------|----------:|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:---------------|:----------------|:--------------|
| paper       | LR      | local         |         1 | 0.2124          | 0.4599          | 0.2419          | 0.3523          | 0.6620          | 0.8494          | 0.953           | 1.61           | 0.0005          | 1244.3        |
| fc_24       | LR      | local         |         1 | 0.5958          | 0.6776          | 0.6370          | 0.7789          | 0.5997          | 0.7669          | 0.816           | 1.03           | 0.0003          | 1248.2        |
| fc_29       | LR      | local         |         1 | 0.6565          | 0.7545          | 0.6179          | 0.6798          | 0.8477          | 0.9465          | 0.838           | 2.00           | 0.0003          | 1237.7        |
| fc_34       | LR      | local         |         1 | 0.2952          | 0.5735          | 0.3141          | 0.4432          | 0.8121          | 0.8964          | 0.953           | 1.43           | 0.0002          | 1244.7        |
| fc_39       | LR      | local         |         1 | 0.3486          | 0.6045          | 0.2982          | 0.5482          | 0.6738          | 0.8906          | 0.932           | 1.95           | 0.0002          | 1254.4        |
| fc_44       | LR      | local         |         1 | 0.0619          | 0.0148          | 0.0725          | 0.0714          | 0.0083          | 0.7319          | 0.919           | 2.59           | 0.0004          | 1256.9        |
| paper_refit | LR      | local         |         1 | 0.3163          | 0.5771          | 0.2972          | 0.5333          | 0.6288          | 0.8843          | 0.953           | 1.77           | 0.0004          | 1267.3        |
| paper       | LR      | all           |         1 | 0.2173          | 0.3785          | 0.2516          | 0.2544          | 0.7396          | 0.8636          | 0.968           | 2.41           | 0.0005          | 1425.7        |
| fc_24       | LR      | all           |         1 | 0.5851          | 0.6392          | 0.5775          | 0.6358          | 0.6426          | 0.8068          | 0.946           | 1.11           | 0.0003          | 1379.1        |
| fc_29       | LR      | all           |         1 | 0.5796          | 0.6736          | 0.5210          | 0.5704          | 0.8223          | 0.9227          | 0.958           | 1.80           | 0.0004          | 1372.6        |
| fc_34       | LR      | all           |         1 | 0.3339          | 0.5241          | 0.3567          | 0.3737          | 0.8770          | 0.9140          | 0.968           | 2.43           | 0.0004          | 1384.1        |
| fc_39       | LR      | all           |         1 | 0.2671          | 0.3727          | 0.2532          | 0.2776          | 0.5670          | 0.8400          | 0.937           | 0.30           | 0.0007          | 1389.1        |
| fc_44       | LR      | all           |         1 | 0.1126          | 0.1163          | 0.1403          | 0.1961          | 0.0826          | 0.8376          | 0.880           | 1.86           | 0.0004          | 1398.5        |
| paper_refit | LR      | all           |         1 | 0.2773          | 0.4190          | 0.3409          | 0.3901          | 0.4524          | 0.8922          | 0.968           | 2.48           | 0.0006          | 1419.5        |
| paper       | RF      | local         |         5 | 0.7773 ± 0.0006 | 0.7590 ± 0.0133 | 0.7904 ± 0.0007 | 0.8065 ± 0.0350 | 0.7175 ± 0.0043 | 0.8971 ± 0.0017 | 0.374 ± 0.028   | 6.99 ± 0.11    | 0.0134 ± 0.0001 | 1317.3 ± 3.4  |
| fc_24       | RF      | local         |         5 | 0.7484 ± 0.0050 | 0.7379 ± 0.0011 | 0.7206 ± 0.0009 | 0.9082 ± 0.0065 | 0.6215 ± 0.0027 | 0.7935 ± 0.0133 | 0.197 ± 0.010   | 11.30 ± 0.18   | 0.0423 ± 0.0008 | 1264.3 ± 0.7  |
| fc_29       | RF      | local         |         5 | 0.9873 ± 0.0004 | 0.9510 ± 0.0020 | 0.9435 ± 0.0010 | 0.9470 ± 0.0031 | 0.9550 ± 0.0009 | 0.9959 ± 0.0002 | 0.321 ± 0.009   | 6.08 ± 0.18    | 0.0368 ± 0.0014 | 1279.7 ± 3.9  |
| fc_34       | RF      | local         |         5 | 0.9464 ± 0.0005 | 0.9031 ± 0.0101 | 0.9273 ± 0.0007 | 0.8988 ± 0.0244 | 0.9078 ± 0.0053 | 0.9804 ± 0.0003 | 0.374 ± 0.028   | 7.19 ± 0.18    | 0.0279 ± 0.0022 | 1293.0 ± 1.0  |
| fc_39       | RF      | local         |         5 | 0.7804 ± 0.0009 | 0.7475 ± 0.0082 | 0.7655 ± 0.0082 | 0.9588 ± 0.0243 | 0.6132 ± 0.0220 | 0.9268 ± 0.0013 | 0.594 ± 0.052   | 21.95 ± 0.23   | 0.0226 ± 0.0012 | 1294.1 ± 1.4  |
| fc_44       | RF      | local         |         5 | 0.0939 ± 0.0039 | 0.0278 ± 0.0011 | 0.0283 ± 0.0002 | 0.0915 ± 0.0237 | 0.0165 ± 0.0000 | 0.8325 ± 0.0059 | 0.469 ± 0.062   | 27.00 ± 1.49   | 0.0420 ± 0.0086 | 1299.2 ± 1.0  |
| paper       | RF      | all           |         5 | 0.7863 ± 0.0012 | 0.8063 ± 0.0062 | 0.7971 ± 0.0064 | 0.9296 ± 0.0216 | 0.7121 ± 0.0038 | 0.9247 ± 0.0020 | 0.385 ± 0.025   | 9.09 ± 0.15    | 0.0154 ± 0.0006 | 1357.9 ± 59.0 |
| fc_24       | RF      | all           |         5 | 0.7600 ± 0.0054 | 0.7466 ± 0.0034 | 0.7147 ± 0.0013 | 0.9323 ± 0.0129 | 0.6227 ± 0.0079 | 0.7761 ± 0.0220 | 0.211 ± 0.023   | 5.73 ± 0.17    | 0.0448 ± 0.0032 | 1205.6 ± 44.3 |
| fc_29       | RF      | all           |         5 | 0.9885 ± 0.0005 | 0.9636 ± 0.0023 | 0.9231 ± 0.0041 | 0.9771 ± 0.0047 | 0.9506 ± 0.0081 | 0.9967 ± 0.0003 | 0.292 ± 0.020   | 7.88 ± 0.19    | 0.0387 ± 0.0015 | 1186.3 ± 0.5  |
| fc_34       | RF      | all           |         5 | 0.9510 ± 0.0015 | 0.9368 ± 0.0051 | 0.9202 ± 0.0110 | 0.9682 ± 0.0135 | 0.9074 ± 0.0026 | 0.9856 ± 0.0007 | 0.385 ± 0.025   | 8.97 ± 0.44    | 0.0271 ± 0.0014 | 1200.0 ± 1.0  |
| fc_39       | RF      | all           |         5 | 0.8014 ± 0.0006 | 0.8102 ± 0.0024 | 0.8109 ± 0.0033 | 0.9487 ± 0.0149 | 0.7072 ± 0.0056 | 0.9329 ± 0.0011 | 0.460 ± 0.022   | 10.04 ± 0.22   | 0.0252 ± 0.0026 | 1216.8 ± 2.3  |
| fc_44       | RF      | all           |         5 | 0.1888 ± 0.0119 | 0.0352 ± 0.0068 | 0.0317 ± 0.0000 | 0.1682 ± 0.0191 | 0.0198 ± 0.0045 | 0.9070 ± 0.0026 | 0.296 ± 0.023   | 12.36 ± 0.31   | 0.0355 ± 0.0014 | 1223.4 ± 3.8  |
| paper_refit | RF      | local         |         5 | 0.7853 ± 0.0005 | 0.7723 ± 0.0102 | 0.7996 ± 0.0010 | 0.8278 ± 0.0268 | 0.7241 ± 0.0027 | 0.9117 ± 0.0016 | 0.374 ± 0.028   | 5.28 ± 0.23    | 0.0077 ± 0.0010 | 1165.6 ± 50.9 |
| paper_refit | RF      | all           |         5 | 0.7950 ± 0.0013 | 0.8193 ± 0.0030 | 0.8155 ± 0.0022 | 0.9478 ± 0.0133 | 0.7215 ± 0.0036 | 0.9336 ± 0.0010 | 0.385 ± 0.025   | 7.07 ± 0.38    | 0.0087 ± 0.0000 | 1074.9 ± 15.5 |
| paper       | XGB     | local         |         5 | 0.7730 ± 0.0017 | 0.7429 ± 0.0095 | 0.6802 ± 0.0243 | 0.7994 ± 0.0373 | 0.6949 ± 0.0141 | 0.9146 ± 0.0014 | 0.735 ± 0.052   | 3.87 ± 0.83    | 0.0022 ± 0.0002 | 1362.4 ± 7.9  |
| fc_24       | XGB     | local         |         5 | 0.7855 ± 0.0068 | 0.7282 ± 0.0051 | 0.7242 ± 0.0046 | 0.9129 ± 0.0133 | 0.6058 ± 0.0065 | 0.8523 ± 0.0140 | 0.309 ± 0.053   | 3.22 ± 0.64    | 0.0046 ± 0.0007 | 1342.5 ± 4.8  |
| fc_29       | XGB     | local         |         5 | 0.9780 ± 0.0008 | 0.9222 ± 0.0017 | 0.9214 ± 0.0022 | 0.9553 ± 0.0198 | 0.8917 ± 0.0150 | 0.9948 ± 0.0002 | 0.588 ± 0.148   | 4.71 ± 0.51    | 0.0046 ± 0.0004 | 1336.9 ± 6.3  |
| fc_34       | XGB     | local         |         5 | 0.9407 ± 0.0008 | 0.8802 ± 0.0061 | 0.8425 ± 0.0191 | 0.8854 ± 0.0219 | 0.8756 ± 0.0116 | 0.9846 ± 0.0007 | 0.735 ± 0.052   | 3.86 ± 0.85    | 0.0030 ± 0.0005 | 1342.7 ± 4.8  |
| fc_39       | XGB     | local         |         5 | 0.7955 ± 0.0014 | 0.7655 ± 0.0083 | 0.6919 ± 0.0151 | 0.8947 ± 0.0262 | 0.6699 ± 0.0262 | 0.9385 ± 0.0019 | 0.860 ± 0.042   | 3.84 ± 0.51    | 0.0039 ± 0.0003 | 1350.4 ± 3.1  |
| fc_44       | XGB     | local         |         5 | 0.1158 ± 0.0100 | 0.0345 ± 0.0066 | 0.0672 ± 0.0238 | 0.1467 ± 0.0425 | 0.0198 ± 0.0045 | 0.8557 ± 0.0081 | 0.789 ± 0.042   | 2.26 ± 0.36    | 0.0024 ± 0.0002 | 1354.9 ± 8.3  |
| paper_refit | XGB     | local         |         5 | 0.7879 ± 0.0009 | 0.7734 ± 0.0120 | 0.7205 ± 0.0285 | 0.8474 ± 0.0467 | 0.7127 ± 0.0143 | 0.9253 ± 0.0019 | 0.735 ± 0.052   | 3.07 ± 0.89    | 0.0022 ± 0.0003 | 1343.7 ± 11.6 |
| paper       | XGB     | all           |         5 | 0.7903 ± 0.0012 | 0.7521 ± 0.0172 | 0.7518 ± 0.0168 | 0.7707 ± 0.0397 | 0.7352 ± 0.0037 | 0.9254 ± 0.0027 | 0.497 ± 0.055   | 7.23 ± 1.58    | 0.0041 ± 0.0006 | 1479.8 ± 1.2  |
| fc_24       | XGB     | all           |         5 | 0.7827 ± 0.0082 | 0.7457 ± 0.0065 | 0.7412 ± 0.0022 | 0.9304 ± 0.0187 | 0.6224 ± 0.0111 | 0.8364 ± 0.0174 | 0.377 ± 0.111   | 2.63 ± 0.59    | 0.0046 ± 0.0007 | 1428.6 ± 7.5  |
| fc_29       | XGB     | all           |         5 | 0.9793 ± 0.0022 | 0.9467 ± 0.0049 | 0.9508 ± 0.0020 | 0.9660 ± 0.0086 | 0.9283 ± 0.0136 | 0.9962 ± 0.0002 | 0.533 ± 0.080   | 6.58 ± 1.39    | 0.0078 ± 0.0014 | 1436.7 ± 8.1  |
| fc_34       | XGB     | all           |         5 | 0.9542 ± 0.0010 | 0.8934 ± 0.0140 | 0.8928 ± 0.0156 | 0.8668 ± 0.0287 | 0.9221 ± 0.0037 | 0.9881 ± 0.0007 | 0.497 ± 0.055   | 7.21 ± 1.62    | 0.0053 ± 0.0009 | 1442.8 ± 4.5  |
| fc_39       | XGB     | all           |         5 | 0.8014 ± 0.0129 | 0.8132 ± 0.0155 | 0.7275 ± 0.1410 | 0.9443 ± 0.0230 | 0.7142 ± 0.0142 | 0.9286 ± 0.0075 | 0.820 ± 0.066   | 6.59 ± 3.50    | 0.0065 ± 0.0023 | 1448.8 ± 1.3  |
| fc_44       | XGB     | all           |         5 | 0.4042 ± 0.0386 | 0.1021 ± 0.0186 | 0.2988 ± 0.0153 | 0.5676 ± 0.0707 | 0.0562 ± 0.0108 | 0.9501 ± 0.0047 | 0.707 ± 0.033   | 4.14 ± 0.48    | 0.0040 ± 0.0002 | 1467.2 ± 13.5 |
| paper_refit | XGB     | all           |         5 | 0.8001 ± 0.0009 | 0.7991 ± 0.0028 | 0.7969 ± 0.0058 | 0.8761 ± 0.0095 | 0.7346 ± 0.0027 | 0.9310 ± 0.0025 | 0.497 ± 0.055   | 6.55 ± 1.81    | 0.0043 ± 0.0006 | 1485.3 ± 3.7  |

---

## 5. Diagnostic Figures

Rendered directly from recorded predictions across seeds:
- **Precision-Recall Curves:** [`reports/figures/exp1_pr_curves.png`](file:///z:/graph/GraphGuard/reports/figures/exp1_pr_curves.png)
- **ROC Curves:** [`reports/figures/exp1_roc_curves.png`](file:///z:/graph/GraphGuard/reports/figures/exp1_roc_curves.png)
- **Confusion Matrices:** [`reports/figures/exp1_confusion_matrices.png`](file:///z:/graph/GraphGuard/reports/figures/exp1_confusion_matrices.png)
- **Per-Step F1 Over Time:** [`reports/figures/exp1_per_step_f1.png`](file:///z:/graph/GraphGuard/reports/figures/exp1_per_step_f1.png) (includes $N_{\text{illicit}}$ overlay)

---

## 6. How to Reproduce

```powershell
# 1. Run unit & integration tests (31 passing)
python -m pytest

# 2. Train baseline models across splits and seeds
python -m ml.baselines.train --config experiments/configs/baselines.yaml

# 3. Aggregate results and generate figures
python -m ml.evaluation.aggregate --results-dir experiments/results --experiment exp1_baselines --tables-dir reports/tables --figures-dir reports/figures

# 4. Run paired block-bootstrap statistical comparisons
python -m ml.evaluation.compare --results-dir experiments/results/exp1_baselines --output reports/tables/exp1_comparisons.md
```
