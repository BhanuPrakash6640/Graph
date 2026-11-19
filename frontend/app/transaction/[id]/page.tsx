"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, TransactionData, PredictResponse, ModelInfo } from "@/lib/api";
import { Search, ShieldAlert, Cpu, Share2, Sparkles, CheckCircle2, AlertCircle } from "lucide-react";

export default function TransactionInvestigatePage() {
  const params = useParams();
  const router = useRouter();
  const txIdParam = (params?.id as string) || "69746416";

  const [txIdInput, setTxIdInput] = useState(txIdParam);
  const [txData, setTxData] = useState<TransactionData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("xgb_paper_refit_all");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        setLoading(true);
        setError(null);
        const [mRes, tRes] = await Promise.all([
          api.getModels(),
          api.getTransaction(txIdParam),
        ]);
        setModels(mRes.models);
        setTxData(tRes);
        setTxIdInput(txIdParam);

        // Run initial prediction
        const pRes = await api.predict(txIdParam, "xgb_paper_refit_all");
        setPrediction(pRes);
      } catch (err: any) {
        setError(err.message || "Failed to load transaction details.");
      } finally {
        setLoading(false);
      }
    }
    init();
  }, [txIdParam]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (txIdInput.trim()) {
      router.push(`/transaction/${txIdInput.trim()}`);
    }
  };

  const handlePredict = async () => {
    if (!txData) return;
    try {
      setPredicting(true);
      const res = await api.predict(txData.tx_id, selectedModel);
      setPrediction(res);
    } catch (err: any) {
      alert("Inference error: " + err.message);
    } finally {
      setPredicting(false);
    }
  };

  const getLabelBadge = (label: string) => {
    if (label === "illicit") {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-[#c0392b] text-white">
          Historical Illicit
        </span>
      );
    }
    if (label === "licit") {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-[#2e86c1] text-white">
          Historical Licit
        </span>
      );
    }
    return (
      <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border border-dashed border-[#95a5a6] text-[#95a5a6] bg-slate-900">
        Historical Unknown
      </span>
    );
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Search Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white">Transaction Investigation</h1>
          <p className="text-xs text-slate-400 mt-1">
            Deep-dive forensic inspection of graph topological placement and feature profiles.
          </p>
        </div>

        <form onSubmit={handleSearch} className="flex items-center space-x-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
            <input
              type="text"
              value={txIdInput}
              onChange={(e) => setTxIdInput(e.target.value)}
              placeholder="Search txId (e.g. 69746416)..."
              className="pl-9 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-64"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm font-medium transition"
          >
            Inspect
          </button>
        </form>
      </div>

      {/* Quick Test Cohort Chips */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-500 font-medium">Sample Test Nodes:</span>
        <button
          onClick={() => router.push("/transaction/69746416")}
          className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-mono"
        >
          69746416 (True Positive)
        </button>
        <button
          onClick={() => router.push("/transaction/30585812")}
          className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-mono"
        >
          30585812 (True Positive)
        </button>
        <button
          onClick={() => router.push("/transaction/106364923")}
          className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-amber-400 font-mono"
        >
          106364923 (False Positive)
        </button>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-4">
          <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm text-slate-400">Loading transaction {txIdParam}...</p>
        </div>
      ) : error ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6 text-center">
          <ShieldAlert className="w-10 h-10 text-rose-400 mx-auto mb-2" />
          <p className="text-rose-300 font-medium">{error}</p>
        </div>
      ) : txData ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Card: Identity & Status */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <span className="text-xs text-slate-400 uppercase font-mono">Transaction ID</span>
                  <h2 className="text-2xl font-bold font-mono text-cyan-400 mt-0.5">{txData.tx_id}</h2>
                </div>
                <div>{getLabelBadge(txData.label)}</div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-slate-800/80">
                <div>
                  <span className="text-xs text-slate-500 uppercase">Time Step</span>
                  <p className="text-lg font-bold text-white mt-1">Step {txData.time_step}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500 uppercase">Total Degree</span>
                  <p className="text-lg font-bold text-white mt-1">{txData.degree}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500 uppercase">In-Degree</span>
                  <p className="text-lg font-bold text-white mt-1">{txData.in_degree}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500 uppercase">Out-Degree</span>
                  <p className="text-lg font-bold text-white mt-1">{txData.out_degree}</p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-slate-800/80">
                <Link
                  href={`/explorer?txId=${txData.tx_id}`}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 rounded-lg text-xs font-semibold flex items-center space-x-2 transition"
                >
                  <Share2 className="w-3.5 h-3.5" />
                  <span>View in Graph Explorer</span>
                </Link>
                <Link
                  href={`/explainability?txId=${txData.tx_id}`}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-blue-300 border border-slate-700 rounded-lg text-xs font-semibold flex items-center space-x-2 transition"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>View Full Explainability</span>
                </Link>
              </div>
            </div>

            {/* Feature Attributes Summary */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">
                Raw Feature Profile (Anonymised)
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.entries(txData.features_summary).map(([k, v]) => (
                  <div key={k} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <span className="text-xs font-mono text-slate-400">{k}</span>
                    <p className="text-sm font-mono font-medium text-slate-200 mt-1">{v.toFixed(4)}</p>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-4 italic">
                Local transaction characteristics anonymised per Weber et al. (2019).
              </p>
            </div>
          </div>

          {/* Right Column: Live Model Inference */}
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-5">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-semibold text-white">Real-Time Model Scoring</h3>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-400 block mb-1.5">
                  Select Model:
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} (Thr: {m.threshold})
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handlePredict}
                disabled={predicting}
                className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold uppercase tracking-wider transition disabled:opacity-50"
              >
                {predicting ? "Running Inference..." : "Score Transaction"}
              </button>

              {prediction && (
                <div className="pt-4 border-t border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">Predicted Illicit Risk:</span>
                    <span className={`text-lg font-bold font-mono ${
                      prediction.score >= prediction.threshold ? "text-[#c0392b]" : "text-[#2e86c1]"
                    }`}>
                      {(prediction.score * 100).toFixed(2)}%
                    </span>
                  </div>

                  {/* Visual Confidence Bar */}
                  <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full ${
                        prediction.score >= prediction.threshold ? "bg-[#c0392b]" : "bg-[#2e86c1]"
                      }`}
                      style={{ width: `${Math.min(100, prediction.score * 100)}%` }}
                    ></div>
                  </div>

                  <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3 space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Model Decision:</span>
                      <span className={`font-semibold ${
                        prediction.predicted_class === 1 ? "text-red-400" : "text-blue-400"
                      }`}>
                        {prediction.predicted_class === 1 ? "FLAGGED ILLICIT" : "LICIT"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Tuned Decision Threshold:</span>
                      <span className="font-mono text-slate-300">{prediction.threshold}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Inference Latency:</span>
                      <span className="font-mono text-slate-300">{prediction.latency_ms} ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Data Type:</span>
                      <span className="text-cyan-400 font-semibold">{prediction.prediction_type}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

