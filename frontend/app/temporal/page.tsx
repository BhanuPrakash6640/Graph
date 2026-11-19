"use client";

import React, { useEffect, useState } from "react";
import { api, StepStat } from "@/lib/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
  ReferenceArea,
} from "recharts";
import { Clock, TrendingUp, AlertTriangle } from "lucide-react";

export default function TemporalAnalysisPage() {
  const [stats, setStats] = useState<StepStat[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const res = await api.getTemporalAnalysis();
        setStats(res.per_step_stats);
      } catch (err: any) {
        setError(err.message || "Failed to load temporal data.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-xs text-slate-400">Loading temporal statistics across 49 snapshots...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6 text-center text-rose-300 text-sm">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
          <Clock className="w-6 h-6 text-cyan-400" />
          <span>Temporal Dynamics & Regime Shift Analysis</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Historical analysis of Bitcoin transaction distributions across all 49 chronological snapshots.
          Notice the sharp regime shift at steps 43–44 during the darknet market shutdown.
        </p>
      </div>

      {/* Shutdown Callout Alert */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start space-x-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-300 space-y-1">
          <p className="font-semibold text-amber-300">
            Structural Regime Shift: Darknet Market Shutdown (Steps 43–44)
          </p>
          <p>
            Law enforcement operations in 2017 caused large-scale shutdown of prominent darknet markets.
            Illicit transaction volume plummeted while topological structures shifted radically, causing GNN models
            trained on steps 1–34 to collapse from ~0.62 PR-AUC down to ~0.11 on cut fc_44.
          </p>
        </div>
      </div>

      {/* Chart 1: Illicit Rate Across 49 Steps */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">Illicit Transaction Rate by Step</h3>
            <p className="text-xs text-slate-400">Ratio of illicit transactions among labelled nodes</p>
          </div>
          <span className="text-xs font-mono text-cyan-400">Steps 1 → 49</span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={stats} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }}
                formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, "Illicit Rate"]}
              />
              <ReferenceArea x1={43} x2={49} stroke="#c0392b" strokeOpacity={0.4} fill="#c0392b" fillOpacity={0.1} />
              <Line
                type="monotone"
                dataKey="illicit_rate_labelled"
                stroke="#e74c3c"
                strokeWidth={2}
                dot={{ r: 2, fill: "#e74c3c" }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Transaction Volume Breakdown */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-white">Transaction Volume Composition</h3>
          <p className="text-xs text-slate-400">Total transaction count partitioned by class supervision</p>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stats} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: "12px" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar dataKey="illicit" name="Illicit" stackId="a" fill="#c0392b" />
              <Bar dataKey="licit" name="Licit" stackId="a" fill="#2e86c1" />
              <Bar dataKey="unknown" name="Unlabelled" stackId="a" fill="#64748b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

