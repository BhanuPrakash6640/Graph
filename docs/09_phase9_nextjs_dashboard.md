# Phase 9: Next.js Operational & Forensic AML Dashboard

## 1. Overview and Architecture

The GraphGuard frontend provides an interactive, research-grade demonstration interface for forensic transaction analysis, topological graph exploration, temporal regime shift monitoring, and multi-faceted explainability.

Built using:
- **Framework:** Next.js 14 (App Router) + TypeScript
- **Styling:** Tailwind CSS (Dark forensic theme)
- **Topological Graph Visualization:** Cytoscape.js (COSE force-directed layout with directed bezier edges)
- **Time-Series Charts:** Recharts (responsive line and stacked bar charts)
- **Type Safety:** 100% typed API client (`frontend/lib/api.ts`) connecting to FastAPI endpoints

---

## 2. Dashboard Pages & Capabilities

### 2.1 Overview (`/`)
- **Historical Dataset Statistics:** Displays total node count (203,769), edge count (234,355), supervision ratio (~23% labelled, 77% unlabelled), and temporal span (49 snapshots).
- **Deployed Model Registry:** Cards detailing deployed models (`xgb_paper_refit_all`, `gcn_paper_refit_all`, `control_b_paper_refit_all`) with tuned decision thresholds.
- **Inference Audit Log:** Real-time log table fed by SQLite database (`/api/inference-log`) tracking transaction ID, model, predicted score, latency in ms, and timestamps.

### 2.2 Forensic Investigation (`/transaction/[id]`)
- **Interactive Search:** Allows forensic analysts to query any `txId` or select pre-configured test cohorts.
- **Strict Visual Differentiation (Project Spec Compliance):**
  - Historical Illicit: `#c0392b` (Solid Red Badge)
  - Historical Licit: `#2e86c1` (Solid Blue Badge)
  - Historical Unknown: `#95a5a6` (Dashed Grey Border Badge)
  - Model Prediction: Dynamic confidence ring and bar with decision classification.
- **Live Scoring:** Analyst can trigger live inference against any registered model.

### 2.3 Intra-Snapshot Graph Explorer (`/explorer`)
- **Cytoscape Canvas:** Renders localized 1-hop and 2-hop transaction graphs.
- **Intra-Step Disjoint Isolation:** Guarantees no cross-snapshot edges appear in the graph view.
- **Interactive Node Inspection:** Clicking any graph node opens the side metadata card detailing in-degree, out-degree, and direct navigation.

### 2.4 Scientific Benchmark Suite (`/benchmark`)
- **Verbatim Tabular Reporting:** Renders all Markdown tables from `reports/tables/`:
  - Exp 1: Baseline tabular vs static GNNs
  - Exp 2: Temporal EvolveGCN-O vs Control B
  - Exp 3: Generalisation across cuts (`fc_24` to `fc_44`)
  - Exp 4: Class imbalance strategies (weights, SMOTE, focal loss)
  - Exp 5: Isolated system ablations (hops, edge direction, topology only)
  - Statistical Significance tables with paired block bootstrap CIs.

### 2.5 Temporal Dynamics (`/temporal`)
- **Recharts Interactive Visualizations:**
  - Illicit transaction rate line chart across all 49 snapshots.
  - Stacked volume composition bar chart (illicit, licit, unlabelled).
  - Explicit highlight on steps 43–44 marking the darknet market shutdown.

### 2.6 Multi-Faceted Attribution (`/explainability`)
- **Cohort Analysis:** Pre-configured analysis of True Positive 1 (`69746416`), True Positive 2 (`30585812`), and False Positive (`106364923`).
- **Explainability Cards:**
  - SHAP feature attributions (anonymised features `local_53`, `local_90`, etc.).
  - PyG GNNExplainer Subgraph Fidelity metrics (`fidelity+` and `fidelity-`).
  - Model-based leave-one-out neighbour occlusion impact table ($\Delta \text{score}$).
  - Structural transaction pattern flags (`chain_node`, `high_illicit_neighborhood`).

---

## 3. Verification & Build Confirmation

- **Type Check & Build:** Passed via `npm run build` (all 8 static/dynamic routes compiled).
- **ESLint Compliance:** Passed via `npm run lint` (`✔ No ESLint warnings or errors`).
- **Zero Hardcoded Numbers:** All displayed metrics, degrees, scores, and tables are fetched dynamically from backend endpoints.

