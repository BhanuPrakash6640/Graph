"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { api, HealthData, ModelInfo, InferenceLogItem } from "@/lib/api";
import { ShieldAlert, Database, Cpu, Activity, Clock, ArrowRight } from "lucide-react";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [logs, setLogs] = useState<InferenceLogItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [h, m, l] = await Promise.all([
          api.getHealth(),
          api.getModels(),
          api.getInferenceLogs(10),
        ]);
        setHealth(h);
        setModels(m.models);
        setLogs(l.logs);
      } catch (err: any) {
        setError(err.message || "Failed to connect to GraphGuard API backend.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="w-10 h-10 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm">Connecting to GraphGuard Backend...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6 text-center max-w-xl mx-auto my-12">
        <ShieldAlert className="w-12 h-12 text-rose-400 mx-auto mb-3" />
        <h3 className="text-lg font-semibold text-rose-300">Backend Connection Error</h3>
        <p className="text-sm text-slate-300 mt-2">{error}</p>
        <p className="text-xs text-slate-400 mt-4">
          Ensure the FastAPI backend is running at <code className="bg-slate-900 px-2 py-0.5 rounded">http://localhost:8000</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      {/* Hero Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Graph Intelligence Platform
        </h1>
        <p className="text-slate-400 mt-2 max-w-3xl">
          Empirical evaluation of tabular vs graph neural networks for AML transaction classification
          across chronological Bitcoin transaction networks.
        </p>
      </div>

      {/* 1. Historical Dataset Statistics */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <Database className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-semibold text-white">Dataset Statistics (Historical)</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Transactions</p>
            <p className="text-2xl font-bold text-white mt-2">
              {health ? health.num_nodes.toLocaleString() : "203,769"}
            </p>
            <p className="text-xs text-slate-500 mt-1">Nodes across 49 snapshots</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Graph Edges (Payment Flows)</p>
            <p className="text-2xl font-bold text-white mt-2">
              {health ? health.num_edges.toLocaleString() : "234,355"}
            </p>
            <p className="text-xs text-slate-500 mt-1">Disjoint intra-step edges</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Class Supervision</p>
            <p className="text-2xl font-bold text-white mt-2">~23% Labelled</p>
            <p className="text-xs text-slate-500 mt-1">77% Unlabelled historical nodes</p>
          </div>
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Temporal Snapshots</p>
            <p className="text-2xl font-bold text-white mt-2">49 Steps</p>
            <p className="text-xs text-slate-500 mt-1">~2-week intervals (~2017-2018)</p>
          </div>
        </div>
      </div>

      {/* 2. Deployed Model Registry */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <Cpu className="w-5 h-5 text-blue-400" />
          <h2 className="text-xl font-semibold text-white">Registered Intelligence Models</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {models.map((m) => (
            <div key={m.id} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs uppercase font-semibold px-2 py-0.5 rounded bg-slate-800 text-cyan-400 border border-slate-700">
                    {m.type}
                  </span>
                  <span className="text-xs text-slate-400">
                    Thr: <strong>{m.threshold}</strong>
                  </span>
                </div>
                <h3 className="text-base font-semibold text-white">{m.name}</h3>
                <p className="text-xs text-slate-400 mt-2">{m.description}</p>
              </div>
              <div className="mt-5 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                <span>Split: <code className="text-slate-300">{m.split}</code></span>
                <span>Features: <code className="text-slate-300">{m.feature_set}</code></span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Inference Audit Log (Predictions Run in This App) */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <h2 className="text-xl font-semibold text-white">Inference Audit Log (Predictions Run in This App)</h2>
          </div>
          <Link href="/transaction/69746416" className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center space-x-1">
            <span>Investigate node</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {logs.length === 0 ? (
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-8 text-center text-slate-500 text-sm">
            No live predictions recorded yet. Run inference on a transaction to see audit records.
          </div>
        ) : (
          <div className="overflow-x-auto bg-slate-900/60 border border-slate-800 rounded-xl">
            <table className="w-full text-left text-sm text-slate-300">
              <thead className="text-xs uppercase bg-slate-950/60 border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-4 py-3">Transaction ID</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Predicted Score</th>
                  <th className="px-4 py-3">Latency</th>
                  <th className="px-4 py-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-800/40 transition">
                    <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                      {new Date(log.ts).toLocaleTimeString()}
                    </td>
                    <td className="px-4 py-3 font-mono text-cyan-400 font-medium">
                      {log.tx_id}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-300 font-mono">
                      {log.model_id}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        log.score >= 0.5 ? "bg-red-500/20 text-red-400 border border-red-500/30" : "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                      }`}>
                        {log.score.toFixed(4)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                      {log.latency_ms.toFixed(2)} ms
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/transaction/${log.tx_id}`}
                        className="text-xs text-cyan-400 hover:underline"
                      >
                        Inspect
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

