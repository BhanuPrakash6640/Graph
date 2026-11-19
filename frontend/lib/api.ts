/**
 * Typed API client for GraphGuard Backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface HealthData {
  status: "healthy" | "degraded";
  num_nodes: number;
  num_edges: number;
  loaded_models: string[];
  disclaimer: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  split: string;
  feature_set: string;
  threshold: number;
  description: string;
}

export interface PredictResponse {
  tx_id: string;
  model_id: string;
  score: number;
  predicted_class: number;
  threshold: number;
  latency_ms: number;
  prediction_type: "model_prediction";
  disclaimer: string;
}

export interface TransactionData {
  tx_id: string;
  time_step: number;
  label_type: "historical_label";
  label: "illicit" | "licit" | "unknown";
  degree: number;
  in_degree: number;
  out_degree: number;
  features_summary: Record<string, number>;
  disclaimer: string;
}

export interface NeighborNode {
  tx_id: string;
  time_step: number;
  label_type: "historical_label";
  label: "illicit" | "licit" | "unknown";
  direction: "in" | "out" | "undirected";
  distance: number;
}

export interface NeighborsData {
  tx_id: string;
  time_step: number;
  k: number;
  total_neighbors: number;
  neighbors: NeighborNode[];
  edges: { source: string; target: string }[];
  disclaimer: string;
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  shap_value: number;
  direction: "+" | "-";
}

export interface NeighborContribution {
  neighbor_idx?: number;
  neighbor_txId: string;
  score_delta: number;
  impact: "+" | "-";
}

export interface ExplainerEdge {
  source: string;
  target: string;
  edge_weight: number;
}

export interface ExplanationData {
  txId: string;
  time_step: number;
  model_run_id: string;
  score: number;
  predicted_class: number;
  threshold: number;
  known_label: "illicit" | "licit" | "unknown";
  top_features: FeatureAttribution[];
  neighbor_contributions: NeighborContribution[];
  explainer_edges: ExplainerEdge[];
  fidelity: {
    fidelity_plus: number;
    fidelity_minus: number;
  };
  pattern_flags: string[];
  disclaimer: string;
}

export interface BenchmarkData {
  tables: Record<string, string>;
  comparisons: Record<string, string>;
  disclaimer: string;
}

export interface StepStat {
  step: number;
  total: number;
  illicit: number;
  licit: number;
  unknown: number;
  illicit_rate_labelled: number;
  illicit_rate_all: number;
}

export interface TemporalData {
  per_step_stats: StepStat[];
  disclaimer: string;
}

export interface InferenceLogItem {
  id: number;
  ts: string;
  tx_id: string;
  model_id: string;
  score: number;
  latency_ms: number;
}

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API Error [${res.status}]: ${errorText}`);
  }

  return res.json();
}

export interface ExampleItem {
  category: string;
  tx_id: string;
  score: number;
  true_label: number;
  label_name: string;
  time_step: number;
  description: string;
}

export interface ExamplesData {
  examples: ExampleItem[];
  disclaimer: string;
}

export const api = {
  getHealth: () => fetchJson<HealthData>("/api/health"),
  getModels: () => fetchJson<{ models: ModelInfo[] }>("/api/models"),
  getExamples: () => fetchJson<ExamplesData>("/api/examples"),
  predict: (tx_id: string, model_id: string = "xgb_paper_refit_all") =>
    fetchJson<PredictResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ tx_id, model_id }),
    }),
  getTransaction: (txId: string) => fetchJson<TransactionData>(`/api/transaction/${txId}`),
  getNeighbors: (txId: string, k: number = 1) =>
    fetchJson<NeighborsData>(`/api/transaction/${txId}/neighbors?k=${k}`),
  getExplanation: (txId: string, modelId: string = "gcn_paper_refit_all") =>
    fetchJson<ExplanationData>(`/api/transaction/${txId}/explanation?model_id=${modelId}`),
  getBenchmark: () => fetchJson<BenchmarkData>("/api/benchmark"),
  getTemporalAnalysis: () => fetchJson<TemporalData>("/api/temporal-analysis"),
  getInferenceLogs: (limit: number = 50) =>
    fetchJson<{ logs: InferenceLogItem[] }>(`/api/inference-log?limit=${limit}`),
};
