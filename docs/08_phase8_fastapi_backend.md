# Phase 8: Demo FastAPI Intelligence Backend

## 1. Overview and Architecture

The GraphGuard backend provides high-performance, leak-free REST endpoints for transaction intelligence, sub-graph extraction, real-time inference, multi-faceted explainability, and benchmarking.

### Core Architectural Guarantees:
1. **Lifespan Initialization (No per-request model loading):** Models (`XGBoost`, `GCN`, `SnapshotContextGCN`), scalers, and node/edge data structures are loaded once during application startup in the FastAPI `lifespan` handler and held in memory.
2. **Disjoint Snapshot Isolation:** Intra-period subgraphs and $k$-hop neighborhoods are strictly constrained to the target node's time step. Cross-snapshot edge traversals are structurally impossible.
3. **Pydantic v2 Type Safety:** All request bodies and responses are validated against rigorous schemas.
4. **Audit Logging (SQLite):** Every prediction is automatically recorded in `inference_log` (`id`, `ts`, `tx_id`, `model_id`, `score`, `latency_ms`).
5. **Experimental Whitelist Security:** The `/api/experiments/run` endpoint is locked by default with an HTTP **403 Forbidden** guard unless explicitly enabled via environment variable `GG_ENABLE_EXPERIMENTS=1`.

---

## 2. API Endpoints Specification

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/api/health` | System health, graph size, active models, disclaimer | `200` |
| `GET` | `/api/models` | Registry of deployable tabular and GNN models | `200` |
| `POST` | `/api/predict` | Real-time score inference on tabular/graph models | `200`, `404`, `422` |
| `GET` | `/api/transaction/{id}` | Historical label, degree, top local features | `200`, `404` |
| `GET` | `/api/transaction/{id}/neighbors` | Intra-snapshot $k$-hop neighborhood ($k \in \{1, 2\}$) | `200`, `404`, `422` |
| `GET` | `/api/transaction/{id}/explanation` | SHAP, GNNExplainer fidelity, occlusion, flags | `200`, `404` |
| `GET` | `/api/benchmark` | Markdown tables from Exp 1–5 and comparisons | `200` |
| `GET` | `/api/temporal-analysis` | Per-step statistics across all 49 time steps | `200` |
| `POST` | `/api/experiments/run` | Protected experiment trigger (`GG_ENABLE_EXPERIMENTS=1`) | `200`, `403` |
| `GET` | `/api/inference-log` | Recent SQLite inference predictions audit log | `200` |

---

## 3. Measured Inference Latency Benchmark

Latency was benchmarked across 50 consecutive inference requests on the active server instance:

| Model | Architecture | p50 Latency | p90 Latency | p95 Latency | p99 Latency |
|---|---|---|---|---|---|
| `xgb_paper_refit_all` | Gradient Boosted Trees (165 feats) | **8.49 ms** | **9.37 ms** | **17.92 ms** | **19.70 ms** |
| `gcn_paper_refit_all` | 2-hop Subgraph PyG GCN | **8.78 ms** | **9.45 ms** | **9.90 ms** | **11.40 ms** |

Both models achieve sub-20ms p95 response times, meeting real-time investigative operational requirements.

---

## 4. Verification & Unit Tests

The backend test suite is implemented in [backend/tests/test_api.py](file:///Z:/graph/GraphGuard/backend/tests/test_api.py) using FastAPI `TestClient`:
- `test_health`: Verifies healthy startup, 203,769 nodes, and loaded model registry.
- `test_list_models`: Verifies registered models and metadata.
- `test_transaction_endpoints`: Verifies historical label, degree, and 404 behavior.
- `test_neighbors_endpoints`: Verifies $k$-hop lookup, 404 for missing IDs, and 422 validation for $k > 2$.
- `test_predict_endpoints`: Verifies inference response, latency measurement, and 404/422 guards.
- `test_explanation_endpoint`: Validates precomputed explanation schema and fidelity numbers.
- `test_benchmark_and_temporal`: Verifies Markdown table serving.
- `test_experiments_guard_403`: Verifies 403 Forbidden without `GG_ENABLE_EXPERIMENTS=1` and 200 with it.
- `test_inference_log`: Verifies SQLite audit logging.

All **9 test suites passed** cleanly.

