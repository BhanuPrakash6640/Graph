"""FastAPI endpoint unit and integration tests using TestClient."""
from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["num_nodes"] > 0
    assert "loaded_models" in data
    assert len(data["loaded_models"]) > 0


def test_list_models(client: TestClient):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) >= 2
    model_ids = [m["id"] for m in models]
    assert "xgb_paper_refit_all" in model_ids


def test_transaction_endpoints(client: TestClient):
    # Known real node: 69746416
    resp = client.get("/api/transaction/69746416")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tx_id"] == "69746416"
    assert data["label"] == "illicit"
    assert data["label_type"] == "historical_label"

    # Unknown node -> 404
    resp_404 = client.get("/api/transaction/9999999999999")
    assert resp_404.status_code == 404


def test_neighbors_endpoints(client: TestClient):
    # Valid 1-hop
    resp = client.get("/api/transaction/69746416/neighbors?k=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tx_id"] == "69746416"
    assert "neighbors" in data
    assert "edges" in data

    # 404 for missing
    resp_404 = client.get("/api/transaction/9999999999999/neighbors?k=1")
    assert resp_404.status_code == 404

    # 422 for invalid k
    resp_422 = client.get("/api/transaction/69746416/neighbors?k=5")
    assert resp_422.status_code == 422


def test_predict_endpoints(client: TestClient):
    # Valid XGBoost prediction
    payload = {"tx_id": "69746416", "model_id": "xgb_paper_refit_all"}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["tx_id"] == "69746416"
    assert data["prediction_type"] == "model_prediction"
    assert 0.0 <= data["score"] <= 1.0
    assert data["latency_ms"] > 0

    # Valid GCN prediction
    payload_gcn = {"tx_id": "69746416", "model_id": "gcn_paper_refit_all"}
    resp_gcn = client.post("/api/predict", json=payload_gcn)
    assert resp_gcn.status_code == 200
    assert 0.0 <= resp_gcn.json()["score"] <= 1.0

    # 404 for unknown transaction
    resp_404 = client.post("/api/predict", json={"tx_id": "9999999999999", "model_id": "xgb_paper_refit_all"})
    assert resp_404.status_code == 404

    # 422 for malformed request
    resp_422 = client.post("/api/predict", json={"invalid_field": 123})
    assert resp_422.status_code == 422


def test_explanation_endpoint(client: TestClient):
    # Precomputed node: 69746416
    resp = client.get("/api/transaction/69746416/explanation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["txId"] == "69746416"
    assert "top_features" in data
    assert "fidelity" in data
    assert "disclaimer" in data

    # 404 for unknown
    resp_404 = client.get("/api/transaction/9999999999999/explanation")
    assert resp_404.status_code == 404


def test_benchmark_and_temporal(client: TestClient):
    resp_bench = client.get("/api/benchmark")
    assert resp_bench.status_code == 200
    assert "tables" in resp_bench.json()

    resp_temp = client.get("/api/temporal-analysis")
    assert resp_temp.status_code == 200
    assert "per_step_stats" in resp_temp.json()


def test_experiments_guard_403(client: TestClient):
    # Without env flag -> must return 403 Forbidden
    if "GG_ENABLE_EXPERIMENTS" in os.environ:
        del os.environ["GG_ENABLE_EXPERIMENTS"]

    resp = client.post("/api/experiments/run", json={"config_name": "exp1_baselines"})
    assert resp.status_code == 403

    # With env flag -> 200
    os.environ["GG_ENABLE_EXPERIMENTS"] = "1"
    resp_allowed = client.post("/api/experiments/run", json={"config_name": "exp1_baselines"})
    assert resp_allowed.status_code == 200
    assert resp_allowed.json()["status"] == "queued"
    del os.environ["GG_ENABLE_EXPERIMENTS"]


def test_inference_log(client: TestClient):
    # Ensure at least one prediction is logged
    client.post("/api/predict", json={"tx_id": "69746416", "model_id": "xgb_paper_refit_all"})

    resp = client.get("/api/inference-log?limit=10")
    assert resp.status_code == 200
    logs = resp.json()["logs"]
    assert len(logs) >= 1
    assert logs[0]["tx_id"] == "69746416"
