"use client";

import React, { useEffect, useState } from "react";
import { api, BenchmarkData } from "@/lib/api";
import { BarChart3, CheckCircle2, ShieldAlert, FileText } from "lucide-react";

function MarkdownTableViewer({ markdown }: { markdown: string }) {
  if (!markdown) {
    return <div className="text-xs text-slate-500 italic p-4">Table data not available.</div>;
  }

  const lines = markdown
    .trim()
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && l.startsWith("|"));

  if (lines.length < 2) {
    return <pre className="text-xs text-slate-300 font-mono p-4 overflow-x-auto">{markdown}</pre>;
  }

  const headerCells = lines[0]
    .split("|")
    .map((c) => c.trim())
    .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);

  const bodyRows = lines.slice(2).map((line) =>
    line
      .split("|")
      .map((c) => c.trim())
      .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1)
  );

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60">
      <table className="w-full text-left text-xs text-slate-300">
        <thead className="bg-slate-950/80 border-b border-slate-800 uppercase font-semibold text-slate-400">
          <tr>
            {headerCells.map((h, i) => (
              <th key={i} className="px-4 py-3 whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {bodyRows.map((row, rIdx) => (
            <tr key={rIdx} className="hover:bg-slate-800/40 transition">
              {row.map((cell, cIdx) => (
                <td key={cIdx} className="px-4 py-2.5 font-mono whitespace-nowrap text-slate-200">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [activeTab, setActiveTab] = useState<string>("exp1_all_models");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const res = await api.getBenchmark();
        setData(res);
      } catch (err: any) {
        setError(err.message || "Failed to load benchmark tables.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const tabOptions = [
    { id: "exp1_all_models", label: "Exp 1: Baselines & GNNs" },
    { id: "exp1_paper_refit", label: "Exp 1: Paper Refit" },
    { id: "exp2_temporal", label: "Exp 2: Temporal (EvolveGCN)" },
    { id: "exp2_paper_refit", label: "Exp 2: Temporal Refit" },
    { id: "exp3_generalisation", label: "Exp 3: Generalisation (Cuts)" },
    { id: "exp4_imbalance", label: "Exp 4: Class Imbalance" },
    { id: "exp5_ablations", label: "Exp 5: System Ablations" },
    { id: "exp1_comparisons", label: "Statistical CIs (Exp 1)" },
    { id: "exp2_comparisons", label: "Statistical CIs (Exp 2)" },
  ];

  const getTableContent = (tabId: string) => {
    if (!data) return "";
    if (tabId in data.tables) return data.tables[tabId];
    if (tabId in data.comparisons) return data.comparisons[tabId];
    return "";
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
          <span>Scientific Benchmark & Evaluation Suite</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Every table below is produced verbatim from offline experimental artifacts under{" "}
          <code className="text-slate-300">experiments/results/</code>. Statistical confidence intervals use
          paired time-step block bootstrap with 2,000 resamples.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
        {tabOptions.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
              activeTab === t.id
                ? "bg-cyan-600 text-white font-semibold"
                : "bg-slate-900 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-4">
          <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-xs text-slate-400">Loading benchmark tables from API...</p>
        </div>
      ) : error ? (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6 text-center text-rose-300 text-sm">
          {error}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span>Viewing: <strong className="text-white">{activeTab}.md</strong></span>
            <span className="text-emerald-400 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Verified Leak-Free Preprocessing</span>
            </span>
          </div>
          <MarkdownTableViewer markdown={getTableContent(activeTab)} />
        </div>
      )}
    </div>
  );
}

