"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api, ExplanationData } from "@/lib/api";
import { Sparkles, ShieldCheck, AlertCircle, Layers, GitBranch, Info } from "lucide-react";

function ExplainabilityContent() {
  const searchParams = useSearchParams();
  const initialTx = searchParams?.get("txId") || "69746416";

  const [selectedTx, setSelectedTx] = useState<string>(initialTx);
  const [selectedModel, setSelectedModel] = useState<"gcn" | "xgb">("gcn");
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const sampleNodes = [
    { txId: "69746416", label: "True Positive 1 (Step 35)", type: "tp" },
    { txId: "30585812", label: "True Positive 2 (Step 37)", type: "tp" },
    { txId: "106364923", label: "False Positive (Step 44 - Shutdown Shift)", type: "fp" },
  ];

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const modelId = selectedModel === "gcn" ? "gcn_paper_refit_all" : "xgb_paper_refit_all";
        const data = await api.getExplanation(selectedTx, modelId);
        setExplanation(data);
      } catch (err: any) {
        setError(err.message || "Failed to load explanation.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [selectedTx, selectedModel]);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Sparkles className="w-6 h-6 text-cyan-400" />
          <span>Multi-Faceted Attribution & Explainability</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Combining SHAP tree attribution, PyG GNNExplainer subgraph masks with fidelity metrics,
          and model-based leave-one-out neighbour occlusion.
        </p>
      </div>

      {/* Controls: Node Selector & Model Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        {/* Node Selector */}
        <div className="flex flex-wrap gap-2.5">
          {sampleNodes.map((n) => (
            <button
              key={n.txId}
              onClick={() => setSelectedTx(n.txId)}
              className={`px-3 py-2 rounded-xl text-xs font-medium transition flex items-center space-x-2 ${
                selectedTx === n.txId
                  ? "bg-cyan-600 text-white font-semibold shadow-lg shadow-cyan-900/30"
                  : "bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800"
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${n.type === "tp" ? "bg-red-400" : "bg-amber-400"}`}></span>
              <span>{n.label}</span>
            </button>
          ))}
        </div>

        {/* Model Toggle */}
        <div className="flex items-center space-x-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 self-start sm:self-auto">
          <button
            onClick={() => setSelectedModel("gcn")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              selectedModel === "gcn"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-900/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            GCN Model
          </button>
          <button
            onClick={() => setSelectedModel("xgb")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              selectedModel === "xgb"
                ? "bg-cyan-600 text-white shadow-md shadow-cyan-900/30"
                : "text-slate-400 hover:text-white"
            }`}
          >
            XGBoost Model
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-4">
          <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-xs text-slate-400">Loading explanation profile for {selectedTx}...</p>
        </div>
      ) : error ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6 text-center text-rose-300 text-sm">
          {error}
        </div>
      ) : explanation ? (
        <div className="space-y-8">
          {/* Executive Overview Banner */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <span className="text-xs uppercase text-slate-400 font-mono">Transaction ID</span>
                <h2 className="text-2xl font-bold font-mono text-cyan-400 mt-0.5">{explanation.txId}</h2>
                <p className="text-xs text-slate-400 mt-1">Snapshot Step: {explanation.time_step}</p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Model Score</span>
                  <span className={`text-base font-bold font-mono ${
                    explanation.score >= explanation.threshold ? "text-[#c0392b]" : "text-[#2e86c1]"
                  }`}>
                    {(explanation.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Decision Threshold</span>
                  <span className="text-base font-bold font-mono text-slate-300">
                    {explanation.threshold}
                  </span>
                </div>
                <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 uppercase block">Historical Label</span>
                  <span className={`text-xs font-bold uppercase ${
                    explanation.known_label === "illicit" ? "text-red-400" : "text-blue-400"
                  }`}>
                    {explanation.known_label}
                  </span>
                </div>
              </div>
            </div>

            {/* Pattern Flags */}
            <div className="mt-5 pt-4 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-500 font-medium">Structural Flags:</span>
              {explanation.pattern_flags.map((flag) => (
                <span
                  key={flag}
                  className="px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-cyan-950 text-cyan-300 border border-cyan-800"
                >
                  {flag}
                </span>
              ))}
            </div>
          </div>

          {/* Grid: SHAP Features vs GNN & Occlusion */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Tabular SHAP */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center space-x-2">
                <Layers className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-semibold text-white">XGBoost SHAP Attributions</h3>
              </div>
              <p className="text-xs text-slate-400">
                Additive contribution of top features toward illicit score. Features are anonymised.
              </p>

              <div className="space-y-2 pt-2">
                {explanation.top_features.map((feat) => {
                  const isPositive = feat.direction === "+";
                  return (
                    <div
                      key={feat.feature}
                      className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between text-xs"
                    >
                      <div className="space-y-0.5">
                        <span className="font-mono text-slate-200 font-semibold">{feat.feature}</span>
                        <p className="text-[11px] text-slate-500 font-mono">Value: {feat.value.toFixed(4)}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span
                          className={`font-mono font-bold px-2 py-0.5 rounded ${
                            isPositive
                              ? "bg-red-500/20 text-red-400 border border-red-500/30"
                              : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                          }`}
                        >
                          {isPositive ? "+" : ""}
                          {feat.shap_value.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right: GNN Subgraph Fidelity & Neighbour Occlusion */}
            <div className="space-y-6">
              {/* PyG Fidelity Card */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-base font-semibold text-white">PyG GNNExplainer Fidelity</h3>
                  </div>
                  {(explanation.fidelity?.fidelity_plus ?? 0) > 0.05 ? (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center space-x-1">
                      <ShieldCheck className="w-3 h-3" />
                      <span>Reliable</span>
                    </span>
                  ) : (
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase bg-amber-500/20 text-amber-400 border border-amber-500/40 flex items-center space-x-1">
                      <AlertCircle className="w-3 h-3" />
                      <span>Not Reliable (≤ 0.05)</span>
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400">
                  Measures how accurately the 2-hop explanation subgraph mirrors full model behavior.
                </p>

                <div className="grid grid-cols-2 gap-4 pt-1">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-[11px] text-slate-500 uppercase block">Fidelity+ (Preservation)</span>
                    <span className={`text-xl font-bold font-mono mt-1 block ${
                      (explanation.fidelity?.fidelity_plus ?? 0) > 0.05 ? "text-emerald-400" : "text-amber-400"
                    }`}>
                      {(explanation.fidelity?.fidelity_plus ?? 0).toFixed(4)}
                    </span>
                    <p className="text-[10px] text-slate-500 mt-1">Accuracy when retaining salient subgraph</p>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-[11px] text-slate-500 uppercase block">Fidelity- (Drop)</span>
                    <span className="text-xl font-bold font-mono text-cyan-400 mt-1 block">
                      {(explanation.fidelity?.fidelity_minus ?? 0).toFixed(4)}
                    </span>
                    <p className="text-[10px] text-slate-500 mt-1">Score drop when salient subgraph is removed</p>
                  </div>
                </div>
              </div>

              {/* Neighbour Occlusion Card */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center space-x-2">
                  <GitBranch className="w-5 h-5 text-cyan-400" />
                  <h3 className="text-base font-semibold text-white">Neighbour Occlusion Impact</h3>
                </div>
                <p className="text-xs text-slate-400">
                  Model-based score difference (Δ score) when isolating 1-hop transaction neighbours.
                </p>

                {explanation.neighbor_contributions.length === 0 ? (
                  <p className="text-xs text-slate-500 italic">No direct 1-hop neighbours (isolated transaction).</p>
                ) : (
                  <div className="space-y-2">
                    {explanation.neighbor_contributions.slice(0, 5).map((nbr) => (
                      <div
                        key={nbr.neighbor_txId}
                        className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between text-xs"
                      >
                        <span className="font-mono text-slate-300">{nbr.neighbor_txId}</span>
                        <span
                          className={`font-mono font-semibold px-2 py-0.5 rounded ${
                            nbr.score_delta > 0
                              ? "bg-red-500/20 text-red-400 border border-red-500/30"
                              : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                          }`}
                        >
                          Δ {nbr.score_delta > 0 ? "+" : ""}
                          {nbr.score_delta.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Legal Disclaimer Card */}
          <div className="p-4 bg-slate-900/40 border border-slate-800 rounded-xl text-xs text-slate-500 flex items-start space-x-3">
            <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
            <p>{explanation.disclaimer}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function ExplainabilityPage() {
  return (
    <Suspense fallback={<div className="text-slate-400 text-sm text-center py-20">Loading Explainability...</div>}>
      <ExplainabilityContent />
    </Suspense>
  );
}
