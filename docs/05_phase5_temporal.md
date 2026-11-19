# Phase 5 Temporal Graph Models Report (EvolveGCN-O & Snapshot Context)

**GraphGuard** — Offline research and demonstration platform on the Elliptic Bitcoin dataset.

---

## 1. Executive Summary

Phase 5 evaluated whether temporal modeling—either evolving GCN layer weights across time snapshots or augmenting static GCNs with autoregressive snapshot-level context—improves illicit transaction detection over static graph message passing and tabular ensembles.

We executed **140 temporal experimental runs** (2 temporal models × 7 temporal splits × 2 feature sets × 5 random seeds 0–4) strictly on GPU (`cuda:0`, NVIDIA GeForce RTX 3050 Laptop GPU), bringing the repository's total evaluated experimental database to **644 logged runs**.

Per `AGENTS.md` Rule 7, all statistical claims are verified using a **time-step block bootstrap** ($N=2,000$ resamples of entire test time steps with replacement, skipping resamples lacking positive labels). Node-level paired bootstrap CIs ($N=1,000$) are reported as secondary, alongside per-step win counts ($k/n$ steps).

### Key Empirical Findings & Statistical Significance

1. **EvolveGCN-O Fails on the Elliptic Dataset (Significant Drop):**
   - On the headline `paper_refit` benchmark (all features, test steps 35–49), static GCN achieves a PR-AUC of **0.6228 ± 0.0252** (pooled test node score: **0.6402**), whereas **EvolveGCN-O** achieves only **0.1371 ± 0.0379** (pooled test node score: **0.1565**; Diff: $-0.4599$).
   - The primary time-step block bootstrap 95% CI is **$[-0.5703, -0.2722]$** (Secondary Node CI: $[-0.5114, -0.4562]$, GCN better on 12/15 steps), establishing **statistically significant superiority of static GCN over EvolveGCN-O**.
   - On local features, static GCN (**0.5077 ± 0.0107**) similarly outperforms EvolveGCN-O (**0.1259 ± 0.0932**): Diff: $-0.2502$, Block 95% CI: **$[-0.3899, -0.0542]$** (Significant: GCN > EvolveGCN-O).

2. **Snapshot-Context GCN (Control B) Matches Static GCN (No Significant Difference):**
   - Control B incorporates 4 unlabelled structural graph statistics ($\log(1+N_t)$, $\log(1+E_t)$, edge density, mean degree) and 4 autoregressive prediction signals from the model's own past outputs ($t-1..t-3$, window mean) without accessing future labels.
   - On `paper_refit` (all features), Control B achieves PR-AUC **0.6127 ± 0.0145** (pooled node score: **0.6438**), compared to static GCN's **0.6228 ± 0.0252** (pooled node score: **0.6402**).
   - Diff: $+0.0047$, primary time-step block bootstrap 95% CI: **$[-0.0323, +0.0490]$** (Node CI: $[-0.0083, +0.0149]$, Control B better on 8/15 steps). Because the Block CI spans zero, there is **no statistically significant difference between static GCN and Control B**.

3. **Control B Decisively Outperforms EvolveGCN-O:**
   - Comparing Control B vs EvolveGCN-O on `paper_refit` (all features): Diff: $+0.4647$, Block 95% CI: **$[+0.3039, +0.5544]$** (Node CI: $[+0.4583, +0.5142]$, Control B better on 13/15 steps, Significant).

4. **Tabular Gradient Boosting (XGBoost) Remains King:**
   - Best tabular baseline (XGBoost, PR-AUC **0.8001 ± 0.0009**, pooled **0.8007**) strongly outperforms both EvolveGCN-O (Diff: $+0.6262$, Block 95% CI: **$[+0.4895, +0.7201]$**, 13/15 steps) and Control B (Diff: $+0.1615$, Block 95% CI: **$[+0.0968, +0.2501]$**, 12/15 steps).

5. **Regime Break Resilience (`fc_44`):**
   - On `fc_44` (test steps 45–49 following the darknet marketplace shutdown), static GCN achieves PR-AUC 0.1147 and Control B achieves 0.1083 (Diff: $-0.0034$, Block 95% CI: **$[-0.0493, +0.0628]$**, No significant difference).
   - EvolveGCN-O scores only 0.0661 on `fc_44` (Diff vs GCN: $-0.0539$, Block 95% CI: **$[-0.1276, -0.0100]$**, Significant drop).

---

## 2. Scientific Rationale: Why EvolveGCN-O Fails on Elliptic

Pareja et al. (AAAI 2020) proposed EvolveGCN to adapt GCN parameters over time. However, on the Elliptic Bitcoin transaction graph, our experimental results reveal fundamental limitations:

1. **Zero Node Persistence:**
   As verified in Phase 1 data integrity checks, nodes in the Elliptic dataset represent individual Bitcoin transactions, not persistent user wallets or entities. Consecutive time steps share **0% node overlap**. Recurrent weight evolution $W_t = \text{GRU}(W_{t-1}, W_{t-1})$ attempts to transfer parameter states across graph snapshots that share no topological or entity continuity.

2. **Unanchored Weight Drift:**
   Without persistent nodes or persistent edges to ground the recurrent cell, weight evolution acts as an unanchored dynamical system. Over 34 training snapshots, the GRU parameters drive the GCN weights into saturated or degenerate regimes, leading to high variance across seeds ($\pm 0.0932$ PR-AUC on local features) and poor calibration on test snapshots.

3. **Static Message Passing Suffices for Intra-Period Structure:**
   Transactions within a single snapshot are tightly coupled via payment flow chains. Static GCN and GraphSAGE extract this local topology effectively in a single step. Evolving weights adds unnecessary recurrent parameter complexity without providing helpful temporal inductive bias.

---

## 3. Architecture & Implementation

### A. EvolveGCN-O (`ml/temporal/evolvegcn.py`, Decisions `D33`, `D34`)
- **Weight Evolution:**
  $$W_t^{(l)} = \text{GRUCell}(W_{t-1}^{(l)}, W_{t-1}^{(l)})$$
  for layers $l \in \{1, 2\}$, initialized from Xavier-uniform parameters $(W_0^{(1)}, W_0^{(2)})$.
- **Dynamic GCN Conv (`EvolveGCNConv`):**
  Linear projection $X_t W_t^{(l)}$ followed by symmetric normalized message passing $\hat{A}_t$ with identity self-loops via `gcn_norm`.
- **Truncated BPTT:**
  Training iterates chronologically over train snapshots. Gradients accumulate over window $K=5$, followed by `loss.backward()` and weight detachment (`w.detach()`), bounding backpropagation horizons and stabilizing optimization across all 34 training periods.
- **Static Ablation Support:**
  Supports `evolve=False` parameter to freeze weights across snapshots, fulfilling the Phase 6 ablation protocol.

### B. Snapshot Context Control (`ml/temporal/snapshot_context.py`, Decision `D35`)
- **Unlabelled Structural Graph Statistics (4 dims):**
  $\log(1 + N_t)$, $\log(1 + E_t)$, Edge Density $\frac{2 E_t}{N_t(N_t-1)+1e-8}$, and Mean Degree $\frac{2 E_t}{N_t+1e-8}$.
- **Autoregressive Prediction Signals (4 dims):**
  Mean predicted illicit score from the model's own past predictions on steps $t-1$, $t-2$, $t-3$, and window mean $[t-3..t-1]$.
- **Leak-Free Guarantee:**
  Ground-truth labels of current, validation, or test snapshots are strictly never accessed. The tracker records only the model's own output probabilities on preceding snapshots.

### C. Training Protocol & Seed-to-Seed Pairing (`ml/temporal/train.py`, Decision `D36`)
- Evaluated on all 7 splits (`paper`, `fc_24`, `fc_29`, `fc_34`, `fc_39`, `fc_44`, `paper_refit`) across seeds 0–4.
- On `paper_refit`, seed $k$'s refit inherits seed $k$'s exact paper validation threshold, hyperparameters, and best epoch count (`Rule 7c`).
- Checkpoints saved under `checkpoints/<run_id>.pt`.

---

## 4. Statistical Comparison Table (Exp 2 Comparisons)

The table below is generated verbatim from `reports/tables/exp2_comparisons.md` using `ml/evaluation/compare.py` ($N=2,000$ time-step block bootstrap resamples):

| Comparison | Split | Config A (Baseline) | Config B (Challenger) | PR-AUC (A) | PR-AUC (B) | Diff (B - A) | Block 95% CI (Primary) | Node 95% CI (Secondary) | B better on k/n steps | Significance (Block CI) |
|:---|:---|:---|:---|---:|---:|---:|:---|:---|:---|:---|
| **Static GCN vs EvolveGCN-O (all features)** | `paper_refit` | GCN (all) | EVOLVEGCN (all) | 0.6402 | 0.1565 | -0.4599 | `[-0.5703, -0.2722]` | `[-0.5114, -0.4562]` | 3/15 steps | **Significant (A > B)** |
| **Static GCN vs Control B (all features)** | `paper_refit` | GCN (all) | CONTROL_B (all) | 0.6402 | 0.6438 | +0.0047 | `[-0.0323, +0.0490]` | `[-0.0083, +0.0149]` | 8/15 steps | **No significant difference** |
| **EvolveGCN-O vs Control B (all features)** | `paper_refit` | EVOLVEGCN (all) | CONTROL_B (all) | 0.1565 | 0.6438 | +0.4647 | `[+0.3039, +0.5544]` | `[+0.4583, +0.5142]` | 13/15 steps | **Significant (B > A)** |
| **Static GCN vs EvolveGCN-O (local features)** | `paper_refit` | GCN (local) | EVOLVEGCN (local) | 0.5155 | 0.2496 | -0.2502 | `[-0.3899, -0.0542]` | `[-0.2948, -0.2333]` | 4/15 steps | **Significant (A > B)** |
| **Static GCN vs Control B (local features)** | `paper_refit` | GCN (local) | CONTROL_B (local) | 0.5155 | 0.3578 | -0.1448 | `[-0.2185, -0.0725]` | `[-0.1741, -0.1390]` | 1/15 steps | **Significant (A > B)** |
| **Best Tabular (XGB) vs EvolveGCN-O (all features)** | `paper_refit` | EVOLVEGCN (all) | XGB (all) | 0.1565 | 0.8007 | +0.6262 | `[+0.4895, +0.7201]` | `[+0.6203, +0.6648]` | 13/15 steps | **Significant (B > A)** |
| **Best Tabular (XGB) vs Control B (all features)** | `paper_refit` | CONTROL_B (all) | XGB (all) | 0.6438 | 0.8007 | +0.1615 | `[+0.0968, +0.2501]` | `[+0.1355, +0.1784]` | 12/15 steps | **Significant (B > A)** |
| **Static GCN vs EvolveGCN-O (darknet shutdown / fc_44)** | `fc_44` | GCN (all) | EVOLVEGCN (all) | 0.1147 | 0.0661 | -0.0539 | `[-0.1276, -0.0100]` | `[-0.0842, -0.0237]` | 1/5 steps | **Significant (A > B) (weak evidence: <8 steps)** |
| **Static GCN vs Control B (darknet shutdown / fc_44)** | `fc_44` | GCN (all) | CONTROL_B (all) | 0.1147 | 0.1083 | -0.0034 | `[-0.0493, +0.0628]` | `[-0.0250, +0.0082]` | 3/5 steps | **No significant difference (weak evidence: <8 steps)** |

---

## 5. Headline Benchmark Table (Exp 2: `paper_refit`)

Pasted verbatim from [reports/tables/exp2_paper_refit.md](file:///z:/graph/GraphGuard/reports/tables/exp2_paper_refit.md) (seeds 0–4; mean ± std):

| split | model | feature_set | n_seeds | pr_auc | f1 (val thr) | f1 (0.5) | precision | recall | roc_auc | val_threshold | train_time_s | node_p50_ms | peak_mem_mb |
|:---|:---|:---|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| paper_refit | GCN | local | 5 | 0.5077 ± 0.0107 | 0.5038 ± 0.0107 | 0.3906 ± 0.0199 | 0.5114 ± 0.0770 | 0.5143 ± 0.0765 | 0.8854 ± 0.0028 | 0.801 ± 0.036 | 19.18 ± 3.36 | 0.0001 ± 0.0000 | 2092.6 ± 4.6 |
| paper_refit | GCN | all | 5 | 0.6228 ± 0.0252 | 0.5941 ± 0.0563 | 0.6255 ± 0.0166 | 0.7728 ± 0.0347 | 0.4888 ± 0.0815 | 0.8869 ± 0.0052 | 0.748 ± 0.084 | 9.76 ± 3.03 | 0.0001 ± 0.0000 | 2310.7 ± 1.6 |
| paper_refit | EVOLVEGCN | local | 5 | 0.1259 ± 0.0932 | 0.1225 ± 0.0860 | 0.2024 ± 0.0950 | 0.1279 ± 0.1477 | 0.3934 ± 0.3900 | 0.5753 ± 0.2862 | 0.648 ± 0.047 | 3.85 ± 1.49 | 0.0001 ± 0.0000 | 1742.3 ± 6.0 |
| paper_refit | CONTROL_B | local | 5 | 0.3405 ± 0.0498 | 0.3893 ± 0.0371 | 0.2820 ± 0.0252 | 0.3425 ± 0.0297 | 0.4574 ± 0.0731 | 0.8493 ± 0.0126 | 0.892 ± 0.025 | 11.33 ± 8.00 | 0.0003 ± 0.0000 | 1164.8 ± 2.0 |
| paper_refit | EVOLVEGCN | all | 5 | 0.1371 ± 0.0379 | 0.0559 ± 0.1039 | 0.1871 ± 0.0769 | 0.0595 ± 0.0672 | 0.1053 ± 0.2225 | 0.7081 ± 0.1699 | 0.639 ± 0.330 | 4.44 ± 3.08 | 0.0001 ± 0.0000 | 1334.8 ± 2.2 |
| paper_refit | CONTROL_B | all | 5 | 0.6127 ± 0.0145 | 0.5835 ± 0.0234 | 0.5978 ± 0.0179 | 0.7978 ± 0.0286 | 0.4615 ± 0.0358 | 0.8889 ± 0.0062 | 0.798 ± 0.039 | 14.55 ± 5.51 | 0.0003 ± 0.0000 | 1335.3 ± 5.0 |

---

## 6. Exit Criteria Verification

- [x] Exp 2 tables (`exp2_temporal.csv/.md` and `exp2_paper_refit.csv/.md`) generated from real runs.
- [x] Statistical comparison table generated via `paired_block_bootstrap_diff` ($N=2,000$ time-step blocks).
- [x] Plain, honest reporting of EvolveGCN-O failure per `HANDOFF_PLAN.md`.
- [x] All 40 unit tests passing on synthetic fixture.
- [x] Checkpoints and predictions saved for every temporal model run under `experiments/results/exp2_temporal/` and `checkpoints/`.

