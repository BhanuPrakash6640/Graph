"""Pydantic v2 schemas for GraphGuard API."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"] = "healthy"
    num_nodes: int
    num_edges: int
    loaded_models: list[str]
    disclaimer: str


class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    split: str
    feature_set: str
    threshold: float
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class PredictRequest(BaseModel):
    tx_id: str = Field(..., description="Target transaction ID")
    model_id: str = Field(default="xgb_paper_refit_all", description="Model ID from /api/models")


class PredictResponse(BaseModel):
    tx_id: str
    model_id: str
    score: float
    predicted_class: int
    threshold: float
    latency_ms: float
    prediction_type: Literal["model_prediction"] = "model_prediction"
    disclaimer: str


class TransactionResponse(BaseModel):
    tx_id: str
    time_step: int
    label_type: Literal["historical_label"] = "historical_label"
    label: Literal["illicit", "licit", "unknown"]
    degree: int
    in_degree: int
    out_degree: int
    features_summary: dict[str, float]
    disclaimer: str


class NeighborNode(BaseModel):
    tx_id: str
    time_step: int
    label_type: Literal["historical_label"] = "historical_label"
    label: Literal["illicit", "licit", "unknown"]
    direction: Literal["in", "out", "undirected"]
    distance: int


class NeighborsResponse(BaseModel):
    tx_id: str
    time_step: int
    label: Literal["illicit", "licit", "unknown"] = "unknown"
    k: int
    total_neighbors: int
    neighbors: list[NeighborNode]
    edges: list[dict[str, Any]]
    disclaimer: str


class ExplanationResponse(BaseModel):
    txId: str
    time_step: int
    model_run_id: str
    score: float
    predicted_class: int
    threshold: float
    known_label: Literal["illicit", "licit", "unknown"]
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    top_node_features: list[dict[str, Any]] = Field(default_factory=list)
    neighbor_contributions: list[dict[str, Any]] = Field(default_factory=list)
    explainer_edges: list[dict[str, Any]] = Field(default_factory=list)
    fidelity: dict[str, float] = Field(default_factory=dict)
    pattern_flags: list[str] = Field(default_factory=list)
    disclaimer: str


class BenchmarkResponse(BaseModel):
    tables: dict[str, str]
    comparisons: dict[str, str]
    disclaimer: str


class TemporalAnalysisResponse(BaseModel):
    per_step_stats: list[dict[str, Any]]
    disclaimer: str


class ExperimentRunRequest(BaseModel):
    config_name: str


class ExperimentRunResponse(BaseModel):
    job_id: str
    status: str
    message: str


class InferenceLogItem(BaseModel):
    id: int
    ts: str
    tx_id: str
    model_id: str
    score: float
    latency_ms: float


class InferenceLogResponse(BaseModel):
    logs: list[InferenceLogItem]


class ExampleItem(BaseModel):
    category: str  # "True Positive", "False Positive", "False Negative", "Unknown"
    tx_id: str
    score: float
    true_label: int  # 1, 0, -1
    label_name: str  # "Illicit", "Licit", "Unknown"
    time_step: int
    description: str


class ExamplesResponse(BaseModel):
    examples: list[ExampleItem]
    disclaimer: str


class TemporalStepScore(BaseModel):
    step: int
    illicit: int
    xgb: float
    gcn: float
    evolve: float


class TemporalSeriesResponse(BaseModel):
    steps: list[TemporalStepScore]
    regime_shift_start: int = 43
    disclaimer: str


class BenchmarkSummaryModel(BaseModel):
    name: str
    feature_set: str
    mean: float
    ci_lo: float
    ci_hi: float
    is_headline: bool = False


class BenchmarkSummaryResponse(BaseModel):
    models: list[BenchmarkSummaryModel]
    base_rate: float = 0.065
    key_takeaway: str
    disclaimer: str


class GlobalShapFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    importance_pct: float


class GlobalShapResponse(BaseModel):
    features: list[GlobalShapFeature]
    disclaimer: str

