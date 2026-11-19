"""FastAPI route handlers for GraphGuard."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.app.schemas import (
    BenchmarkResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
    ExplanationResponse,
    HealthResponse,
    InferenceLogItem,
    InferenceLogResponse,
    ModelInfo,
    ModelsResponse,
    NeighborsResponse,
    PredictRequest,
    PredictResponse,
    TemporalAnalysisResponse,
    TransactionResponse,
)

logger = logging.getLogger("backend.endpoints")
router = APIRouter()

DISCLAIMER = (
    "Research demonstration on historical Elliptic Bitcoin dataset (2019). "
    "Not a live transaction monitoring or AML compliance system."
)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    engine = request.app.state.inference_engine
    store = request.app.state.graph_store
    return HealthResponse(
        status="healthy",
        num_nodes=store.num_nodes,
        num_edges=store.num_edges,
        loaded_models=list(engine.loaded_models.keys()),
        disclaimer=DISCLAIMER,
    )


@router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    engine = request.app.state.inference_engine
    models = [ModelInfo(**m) for m in engine.list_models()]
    return ModelsResponse(models=models)


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    store = request.app.state.graph_store
    engine = request.app.state.inference_engine

    if not store.has_tx(body.tx_id):
        raise HTTPException(status_code=404, detail=f"Transaction ID {body.tx_id} not found in dataset")

    try:
        res = engine.predict(tx_id=body.tx_id, model_id=body.model_id, graph_store=store)
        return PredictResponse(**res)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Inference failed for tx %s: %s", body.tx_id, e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/transaction/{tx_id}", response_model=TransactionResponse)
def get_transaction(tx_id: str, request: Request) -> TransactionResponse:
    store = request.app.state.graph_store
    try:
        info = store.get_node_info(tx_id)
        return TransactionResponse(**info)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Transaction ID {tx_id} not found in dataset")


@router.get("/transaction/{tx_id}/neighbors", response_model=NeighborsResponse)
def get_neighbors(
    tx_id: str,
    request: Request,
    k: int = Query(default=1, ge=1, le=2, description="Neighborhood hop distance"),
) -> NeighborsResponse:
    store = request.app.state.graph_store
    try:
        res = store.khop(tx_id, k=k)
        return NeighborsResponse(**res)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Transaction ID {tx_id} not found in dataset")


@router.get("/transaction/{tx_id}/explanation", response_model=ExplanationResponse)
def get_explanation(
    tx_id: str,
    request: Request,
    model_id: str = Query(default="gcn_paper_refit_all"),
) -> ExplanationResponse:
    store = request.app.state.graph_store
    if not store.has_tx(tx_id):
        raise HTTPException(status_code=404, detail=f"Transaction ID {tx_id} not found in dataset")

    # Map aliases
    target_id = model_id
    if target_id in ("gcn", "static_gcn"):
        target_id = "gcn_paper_refit_all"
    elif target_id in ("xgb", "xgboost"):
        target_id = "xgb_paper_refit_all"

    exp_dir = Path("data/processed/explanations")
    file_path = exp_dir / target_id / f"{tx_id}.json"
    if not file_path.exists():
        # Fallback to other model or root alias
        alt_id = "xgb_paper_refit_all" if "gcn" in target_id else "gcn_paper_refit_all"
        file_path_alt = exp_dir / alt_id / f"{tx_id}.json"
        if file_path_alt.exists():
            file_path = file_path_alt
        else:
            file_path = exp_dir / f"{tx_id}.json"

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Precomputed explanation for transaction {tx_id} (model={model_id}) not found",
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExplanationResponse(**data)


@router.get("/benchmark", response_model=BenchmarkResponse)
def get_benchmark() -> BenchmarkResponse:
    tables_dir = Path("reports/tables")
    tables = {}
    comparisons = {}

    table_files = [
        "exp1_all_models.md",
        "exp1_paper_refit.md",
        "exp2_temporal.md",
        "exp2_paper_refit.md",
        "exp3_generalisation.md",
        "exp4_imbalance.md",
        "exp5_ablations.md",
    ]

    for fname in table_files:
        p = tables_dir / fname
        if p.exists():
            tables[fname.replace(".md", "")] = p.read_text(encoding="utf-8")

    comp_files = ["exp1_comparisons.md", "exp2_comparisons.md"]
    for cname in comp_files:
        p = tables_dir / cname
        if p.exists():
            comparisons[cname.replace(".md", "")] = p.read_text(encoding="utf-8")

    return BenchmarkResponse(tables=tables, comparisons=comparisons, disclaimer=DISCLAIMER)


@router.get("/temporal-analysis", response_model=TemporalAnalysisResponse)
def get_temporal_analysis() -> TemporalAnalysisResponse:
    eda_csv = Path("reports/eda/per_step.csv")
    if not eda_csv.exists():
        return TemporalAnalysisResponse(per_step_stats=[], disclaimer=DISCLAIMER)

    df = pd.read_csv(eda_csv)
    stats = df.to_dict(orient="records")
    return TemporalAnalysisResponse(per_step_stats=stats, disclaimer=DISCLAIMER)


@router.post("/experiments/run", response_model=ExperimentRunResponse)
def run_experiment(body: ExperimentRunRequest) -> ExperimentRunResponse:
    # Whitelist guard: 403 Forbidden unless GG_ENABLE_EXPERIMENTS=1
    enable_flag = os.environ.get("GG_ENABLE_EXPERIMENTS", "0")
    if enable_flag != "1":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Arbitrary experiment execution is disabled on production demo endpoints. Set GG_ENABLE_EXPERIMENTS=1.",
        )

    return ExperimentRunResponse(
        job_id="job_demo_1",
        status="queued",
        message=f"Queued experiment config: {body.config_name}",
    )


@router.get("/inference-log", response_model=InferenceLogResponse)
def get_inference_logs(request: Request, limit: int = 50) -> InferenceLogResponse:
    db = request.app.state.database
    logs = [InferenceLogItem(**r) for r in db.get_recent_logs(limit=limit)]
    return InferenceLogResponse(logs=logs)

