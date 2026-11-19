"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  api,
  TransactionData,
  PredictResponse,
  ModelInfo,
  NeighborsData,
  ExplanationData,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { ScoreBar } from "@/components/ScoreBar";
import { ClassBadge } from "@/components/ClassBadge";
import { GraphCanvas, GraphNode } from "@/components/GraphCanvas";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import {
  Search,
  ArrowRight,
  Share2,
  Clock,
  Layers,
  Info,
  ExternalLink,
} from "lucide-react";

export default function TransactionCaseFilePage() {
  const params = useParams();
  const router = useRouter();
  const txIdParam = (params?.id as string) || "69746416";

  const [searchInput, setSearchInput] = useState(txIdParam);
  const [txData, setTxData] = useState<TransactionData | null>(null);
  const [neighbors, setNeighbors] = useState<NeighborsData | null>(null);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("xgb_paper_refit_all");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [hopK, setHopK] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadCase = async (id: string, modelId: string, k: number) => {
    try {
      setLoading(true);
      setError(null);
      const [mRes, tRes, nRes, pRes] = await Promise.all([
        api.getModels(),
        api.getTransaction(id),
        api.getNeighbors(id, k),
        api.predict(id, modelId),
      ]);

      setModels(mRes.models);
      setTxData(tRes);
      setNeighbors(nRes);
      setPrediction(pRes);
      setSearchInput(id);

      // Attempt to load precomputed explanation
      try {
        const expRes = await api.getExplanation(id, modelId);
        setExplanation(expRes);
      } catch {
        setExplanation(null);
      }
    } catch (err: any) {
      setError(err.message || `Failed to load case file for payment ${id}.`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCase(txIdParam, selectedModel, hopK);
  }, [txIdParam, selectedModel, hopK]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchInput.trim();
    if (trimmed && trimmed !== txIdParam) {
      router.push(`/transaction/${trimmed}`);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-72 mb-2" />
        <Skeleton className="h-4 w-96 mb-6" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Skeleton className="h-96 w-full" />
          <Skeleton className="h-96 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      </div>
    );
  }

  if (error || !txData) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Case file: Payment lookup"
          technical="Payment lookup error"
        />
        <ErrorState
          title={`Payment ID ${txIdParam} not found`}
          error={error || "Payment ID does not exist in historical Elliptic dataset."}
          retry={() => loadCase(txIdParam, selectedModel, hopK)}
        />
        <div className="text-center pt-4">
          <Link
            href="/investigate"
            className="text-xs font-mono text-accent hover:underline inline-flex items-center gap-1"
          >
            <span>← Back to search & representative examples</span>
          </Link>
        </div>
      </div>
    );
  }

  // Model alert threshold (from prediction or selected model)
  const threshold = prediction?.threshold ?? 0.5;
  const score = prediction?.score ?? 0.0;
  const isFlagged = score >= threshold;

  // Prepare graph canvas nodes from neighbors
  const canvasNodes: GraphNode[] = [
    {
      tx_id: txData.tx_id,
      time_step: txData.time_step,
      label: txData.label,
      direction: "center",
      distance: 0,
    },
    ...(neighbors?.neighbors.map((n) => ({
      tx_id: n.tx_id,
      time_step: n.time_step,
      label: n.label,
      direction: n.direction,
      distance: n.distance,
    })) || []),
  ];

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-rule">
        <div>
          <span className="text-xs font-mono text-ink-muted uppercase tracking-wider">
            Case File
          </span>
          <h1 className="font-serif text-2xl font-bold text-ink">
            Payment {txData.tx_id}
          </h1>
        </div>

        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <div className="relative">
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search other txId..."
              className="px-3 py-1.5 pl-8 bg-paper-card border border-rule font-mono text-xs text-ink w-48 sm:w-60 focus:outline-none focus:border-accent"
            />
            <Search className="w-3.5 h-3.5 text-ink-muted absolute left-2.5 top-2.5" />
          </div>
          <button
            type="submit"
            className="px-3 py-1.5 bg-paper border border-rule hover:bg-paper-muted text-xs font-mono font-medium text-ink transition"
          >
            Go
          </button>
        </form>
      </div>

      {/* 3-Column Forensic Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* COLUMN 1: VERDICT */}
        <div className="p-5 bg-paper-card border border-rule space-y-5">
          <div className="border-b border-rule pb-3">
            <h2 className="font-serif text-base font-semibold text-ink">
              1. Model Verdict
            </h2>
            <p className="font-mono text-xs text-ink-muted mt-0.5">
              Alert decision & ground truth
            </p>
          </div>

          {/* Model Selector */}
          <div>
            <label className="block text-xs font-mono text-ink-muted mb-1">
              Evaluating Model:
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-2.5 py-1.5 bg-paper border border-rule font-mono text-xs text-ink focus:outline-none focus:border-accent"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} (line: {m.threshold.toFixed(3)})
                </option>
              ))}
            </select>
          </div>

          {/* ScoreBar with Threshold Line */}
          <div className="pt-2">
            <ScoreBar score={score} threshold={threshold} showLabels={true} />
          </div>

          {/* Decision Status Badges */}
          <div className="pt-3 border-t border-rule space-y-3">
            <div>
              <div className="text-[11px] font-mono text-ink-muted mb-1">
                Model Prediction:
              </div>
              <ClassBadge
                type="predicted"
                label={isFlagged ? "illicit" : "licit"}
                score={score}
                threshold={threshold}
              />
            </div>

            <div>
              <div className="text-[11px] font-mono text-ink-muted mb-1">
                Historical Record:
              </div>
              <ClassBadge type="known" label={txData.label} />
            </div>
          </div>

          {/* Summary Metadata */}
          <div className="pt-3 border-t border-rule space-y-2 text-xs font-mono text-ink">
            <div className="flex justify-between">
              <span className="text-ink-muted">Time step:</span>
              <span className="font-bold">Step {txData.time_step}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-muted">Alert line (tau):</span>
              <span>{threshold.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ink-muted">Inference latency:</span>
              <span>{prediction?.latency_ms ? `${prediction.latency_ms.toFixed(2)} ms` : "—"}</span>
            </div>
          </div>

          <div className="pt-3 border-t border-rule text-xs text-ink-muted">
            <p className="leading-relaxed">
              Payments scored at or above the model&apos;s alert line ({threshold.toFixed(3)}) are flagged as illicit under the decision threshold.
            </p>
          </div>
        </div>

        {/* COLUMN 2: CONNECTED PAYMENTS (NEIGHBOURHOOD) */}
        <div className="p-5 bg-paper-card border border-rule space-y-4">
          <div className="flex items-center justify-between border-b border-rule pb-3">
            <div>
              <h2 className="font-serif text-base font-semibold text-ink">
                2. Connected Payments
              </h2>
              <p className="font-mono text-xs text-ink-muted mt-0.5">
                Local transaction network
              </p>
            </div>

            {/* Hop selector */}
            <div className="flex items-center gap-1 font-mono text-xs">
              <span className="text-ink-muted mr-1">Hops:</span>
              <button
                onClick={() => setHopK(1)}
                className={`px-2 py-0.5 border text-xs font-semibold ${
                  hopK === 1
                    ? "bg-accent text-white border-accent"
                    : "bg-paper text-ink border-rule hover:bg-paper-muted"
                }`}
              >
                1
              </button>
              <button
                onClick={() => setHopK(2)}
                className={`px-2 py-0.5 border text-xs font-semibold ${
                  hopK === 2
                    ? "bg-accent text-white border-accent"
                    : "bg-paper text-ink border-rule hover:bg-paper-muted"
                }`}
              >
                2
              </button>
            </div>
          </div>

          {/* Graph Visualization */}
          <GraphCanvas
            centerTxId={txData.tx_id}
            nodes={canvasNodes}
            edges={neighbors?.edges || []}
            height="340px"
            onNodeClick={(node) => {
              if (node.tx_id !== txData.tx_id) {
                router.push(`/transaction/${node.tx_id}`);
              }
            }}
          />

          {/* Degree Stats */}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-rule text-center font-mono">
            <div className="p-2 bg-paper-muted border border-rule/60">
              <div className="text-[10px] text-ink-muted">In-degree</div>
              <div className="text-sm font-bold text-ink">{txData.in_degree}</div>
            </div>
            <div className="p-2 bg-paper-muted border border-rule/60">
              <div className="text-[10px] text-ink-muted">Out-degree</div>
              <div className="text-sm font-bold text-ink">{txData.out_degree}</div>
            </div>
            <div className="p-2 bg-paper-muted border border-rule/60">
              <div className="text-[10px] text-ink-muted">Total Flow</div>
              <div className="text-sm font-bold text-ink">{txData.degree}</div>
            </div>
          </div>

          <div className="text-right">
            <Link
              href={`/explorer?txId=${txData.tx_id}`}
              className="text-xs font-mono text-accent hover:underline inline-flex items-center gap-1"
            >
              <span>Open in full Graph Explorer</span>
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>
        </div>

        {/* COLUMN 3: WHY (ATTRIBUTION) */}
        <div className="p-5 bg-paper-card border border-rule space-y-4">
          <div className="border-b border-rule pb-3">
            <h2 className="font-serif text-base font-semibold text-ink">
              3. Why this score?
            </h2>
            <p className="font-mono text-xs text-ink-muted mt-0.5">
              Feature and graph attribution
            </p>
          </div>

          {/* Honesty note per instruction 3 & Section 7 */}
          <div className="p-2.5 bg-paper-muted border border-rule text-xs text-ink leading-relaxed">
            <span className="font-semibold text-ink">Notice: </span>
            This explanation shows what the model paid attention to, not proof of what happened or legal causation.
          </div>

          {/* Pattern Flags */}
          {explanation?.pattern_flags && explanation.pattern_flags.length > 0 && (
            <div>
              <div className="text-xs font-mono text-ink-muted mb-1.5">
                Structural Pattern Flags:
              </div>
              <div className="flex flex-wrap gap-1.5">
                {explanation.pattern_flags.map((flag) => (
                  <span
                    key={flag}
                    className="px-2 py-0.5 text-[11px] font-mono font-medium border border-rule bg-paper text-ink"
                  >
                    #{flag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Top Feature Attributions */}
          <div>
            <div className="flex items-baseline justify-between mb-2">
              <span className="text-xs font-mono font-semibold text-ink">
                Top Feature Drivers
              </span>
              <span className="text-[10px] font-mono text-ink-muted">
                Anonymised attributes
              </span>
            </div>

            {explanation?.top_features && explanation.top_features.length > 0 ? (
              <div className="space-y-2">
                {explanation.top_features.slice(0, 5).map((f) => {
                  const isPositive = f.direction === "+" || f.shap_value > 0;
                  return (
                    <div key={f.feature} className="text-xs font-mono">
                      <div className="flex justify-between text-[11px] mb-0.5">
                        <span className="text-ink">{f.feature}</span>
                        <span className={isPositive ? "text-illicit font-bold" : "text-licit font-bold"}>
                          {isPositive ? "+" : ""}
                          {f.shap_value.toFixed(3)}
                        </span>
                      </div>
                      <div className="h-1.5 bg-paper-muted border border-rule overflow-hidden">
                        <div
                          className={`h-full ${isPositive ? "bg-illicit" : "bg-licit"}`}
                          style={{
                            width: `${Math.min(100, Math.abs(f.shap_value) * 35)}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-ink-muted font-mono">
                Model attribution not precomputed for this node.
              </p>
            )}
          </div>

          {/* Connected Neighbor Contributions (Occlusion Delta) */}
          {explanation?.neighbor_contributions && explanation.neighbor_contributions.length > 0 && (
            <div className="pt-3 border-t border-rule">
              <div className="text-xs font-mono font-semibold text-ink mb-1.5">
                Connected Neighbor Impact (Occlusion)
              </div>
              <div className="space-y-1 text-xs font-mono divide-y divide-rule/60">
                {explanation.neighbor_contributions.slice(0, 3).map((nc) => (
                  <div key={nc.neighbor_txId} className="pt-1 first:pt-0 flex items-center justify-between">
                    <Link
                      href={`/transaction/${nc.neighbor_txId}`}
                      className="text-accent hover:underline text-[11px]"
                    >
                      tx {nc.neighbor_txId}
                    </Link>
                    <span
                      className={`text-[11px] font-bold ${
                        nc.score_delta > 0 ? "text-illicit" : "text-licit"
                      }`}
                    >
                      {nc.score_delta > 0 ? "+" : ""}
                      {nc.score_delta.toFixed(4)} delta
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Anonymisation note */}
          <p className="text-[11px] text-ink-muted leading-relaxed pt-2 border-t border-rule/60">
            Features 1–93 represent local payment metadata; 94–165 represent aggregated 1-hop transaction context. Original Bitcoin transaction hashes and address identities were irreversibly anonymised by Weber et al. (2019).
          </p>
        </div>
      </div>
    </div>
  );
}
