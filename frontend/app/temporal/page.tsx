"use client";

import React, { useEffect, useState } from "react";
import { api, StepStat, TemporalSeriesData, BenchmarkData } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { DataTable, Column } from "@/components/DataTable";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
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
  ReferenceArea,
} from "recharts";

// Helper to parse markdown table from reports/tables/exp3_generalisation.md
function parseMarkdownTable(markdown: string): { columns: Column<any>[]; rows: any[] } {
  if (!markdown) return { columns: [], rows: [] };

  const lines = markdown
    .trim()
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && l.startsWith("|"));

  if (lines.length < 2) return { columns: [], rows: [] };

  const rawHeaders = lines[0]
    .split("|")
    .map((c) => c.trim())
    .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);

  const columns: Column<any>[] = rawHeaders.map((h, i) => ({
    key: `col_${i}`,
    header: h,
    sortable: true,
    align: i === 0 ? "left" : "right",
  }));

  const rows = lines.slice(2).map((line, rIdx) => {
    const cells = line
      .split("|")
      .map((c) => c.trim())
      .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);

    const rowObj: Record<string, string> = { id: `row_${rIdx}` };
    cells.forEach((cell, cIdx) => {
      rowObj[`col_${cIdx}`] = cell;
    });
    return rowObj;
  });

  return { columns, rows };
}

export default function TemporalPage() {
  const [stats, setStats] = useState<StepStat[]>([]);
  const [series, setSeries] = useState<TemporalSeriesData | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null);
  const [viewTab, setViewTab] = useState<"per_step" | "forward_chaining">("per_step");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [tRes, sRes, bRes] = await Promise.all([
          api.getTemporalAnalysis(),
          api.getTemporalSeries(),
          api.getBenchmark(),
        ]);
        setStats(tRes.per_step_stats);
        setSeries(sRes);
        setBenchmark(bRes);
      } catch (err: any) {
        setError(err.message || "Failed to load temporal snapshot statistics.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-72 mb-2" />
        <Skeleton className="h-4 w-96 mb-6" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    );
  }

  if (error) {
    return <ErrorState title="Failed to load temporal data" error={error} />;
  }

  const stepsData = series?.steps || [];
  const exp3Markdown = benchmark?.tables?.["exp3_generalisation"] || "";
  const { columns: fcColumns, rows: fcRows } = parseMarkdownTable(exp3Markdown);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Temporal Dynamics & Regime Shift"
        technical="Performance evaluation across 49 chronological snapshots and 5 expanding temporal cuts"
        inShort="Model performance drops markedly after time step 43, coinciding with the late-period regime change where illicit instance counts drop significantly across the final snapshots."
      />

      {/* View Switcher Tabs */}
      <div className="flex items-center gap-2 border-b border-rule pb-3 font-mono text-xs">
        <button
          onClick={() => setViewTab("per_step")}
          className={`px-3 py-1.5 border font-semibold ${
            viewTab === "per_step"
              ? "bg-accent text-white border-accent"
              : "bg-paper text-ink border-rule hover:bg-paper-muted"
          }`}
        >
          Per-Step Performance (Steps 35–49)
        </button>
        <button
          onClick={() => setViewTab("forward_chaining")}
          className={`px-3 py-1.5 border font-semibold ${
            viewTab === "forward_chaining"
              ? "bg-accent text-white border-accent"
              : "bg-paper text-ink border-rule hover:bg-paper-muted"
          }`}
        >
          Forward-Chaining Cuts (fc_24 to fc_44)
        </button>
      </div>

      {viewTab === "per_step" ? (
        <div className="space-y-8">
          {/* Main Line Chart: PR-AUC by Step */}
          <Section
            title="PR-AUC by Evaluation Time Step (Test Snapshots 35–49)"
            technical="Comparison of XGBoost, Static GCN, and EvolveGCN-O across test snapshots. Shaded area indicates the late-period regime shift (steps 43–49)."
          >
            <div className="p-4 bg-paper-card border border-rule space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-accent inline-block" />
                    <span className="font-bold text-accent">XGBoost (all)</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-licit inline-block" />
                    <span className="font-bold text-licit">Static GCN (all)</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-ink-muted inline-block" />
                    <span className="font-bold text-ink-muted">EvolveGCN-O (evolving)</span>
                  </span>
                </div>
                <span className="text-[11px] text-ink-muted">
                  Shaded area: steps 43–49 (regime shift)
                </span>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stepsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
                    <XAxis
                      dataKey="step"
                      tickLine={false}
                      stroke="#9CA3AF"
                      fontSize={11}
                      fontFamily="IBM Plex Mono"
                      tickFormatter={(v) => `t${v}`}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickLine={false}
                      stroke="#9CA3AF"
                      fontSize={11}
                      fontFamily="IBM Plex Mono"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#FAFAF7",
                        borderColor: "#E5E7EB",
                        fontFamily: "IBM Plex Mono",
                        fontSize: "11px",
                      }}
                      formatter={(val: any) => [`${parseFloat(val).toFixed(3)}`, "PR-AUC"]}
                    />
                    {/* Shaded regime change area */}
                    <ReferenceArea
                      x1={43}
                      x2={49}
                      fill="#F3F4F6"
                      stroke="#D1D5DB"
                      strokeDasharray="2 2"
                    />
                    <Line
                      type="monotone"
                      dataKey="xgb"
                      stroke="#0F766E"
                      strokeWidth={2}
                      dot={{ r: 3, fill: "#0F766E" }}
                      name="XGBoost"
                    />
                    <Line
                      type="monotone"
                      dataKey="gcn"
                      stroke="#1D4ED8"
                      strokeWidth={1.5}
                      dot={{ r: 2.5, fill: "#1D4ED8" }}
                      name="Static GCN"
                    />
                    <Line
                      type="monotone"
                      dataKey="evolve"
                      stroke="#9CA3AF"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      dot={{ r: 2, fill: "#9CA3AF" }}
                      name="EvolveGCN-O"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Sub-panel: Illicit Count per Step */}
              <div className="pt-4 border-t border-rule space-y-2">
                <div className="flex items-center justify-between text-xs font-mono text-ink">
                  <span className="font-semibold">Illicit Class Volume per Snapshot</span>
                  <span className="text-[11px] text-ink-muted">
                    Dashed line: low-count threshold (&lt;30 positive instances)
                  </span>
                </div>

                <div className="h-32 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={stepsData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
                      <XAxis
                        dataKey="step"
                        tickLine={false}
                        stroke="#9CA3AF"
                        fontSize={11}
                        fontFamily="IBM Plex Mono"
                        tickFormatter={(v) => `t${v}`}
                      />
                      <YAxis
                        tickLine={false}
                        stroke="#9CA3AF"
                        fontSize={11}
                        fontFamily="IBM Plex Mono"
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#FAFAF7",
                          borderColor: "#E5E7EB",
                          fontFamily: "IBM Plex Mono",
                          fontSize: "11px",
                        }}
                        formatter={(val: any) => [`${val} transactions`, "Illicit Nodes"]}
                      />
                      <ReferenceArea
                        x1={43}
                        x2={49}
                        fill="#F3F4F6"
                        stroke="#D1D5DB"
                        strokeDasharray="2 2"
                      />
                      <Bar
                        dataKey="illicit"
                        fill="#B42318"
                        name="Illicit count"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <details className="pt-2 border-t border-rule/60 text-xs text-ink-muted group cursor-pointer">
                <summary className="font-medium text-ink hover:text-accent flex items-center justify-between">
                  <span>How to read this per-step chart</span>
                  <span className="text-[10px] text-ink-muted group-open:rotate-180 transition">▼</span>
                </summary>
                <div className="mt-2 text-ink text-xs leading-relaxed space-y-1">
                  <p>
                    • <strong>Steps 35–42:</strong> Baseline chronological period where tabular models and GNNs operate under standard illicit payment volumes.
                  </p>
                  <p>
                    • <strong>Steps 43–49 (Shaded):</strong> Late-period regime change where illicit payment volume drops to low sample sizes per step.
                  </p>
                  <p>
                    • <strong>Sample Size Warning:</strong> The final steps contain very few illicit instances in total (&lt;30 in multiple steps). Evaluating model superiority on low-count steps presents weak statistical evidence.
                  </p>
                </div>
              </details>
            </div>
          </Section>
        </div>
      ) : (
        /* Forward-Chaining Cuts View */
        <Section
          title="Forward-Chaining Temporal Splits (fc_24 to fc_44)"
          technical="Evaluating model resilience by expanding training windows across chronological cuts"
        >
          <div className="p-4 bg-paper-card border border-rule">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-rule">
              <div>
                <h3 className="font-serif text-sm font-semibold text-ink">
                  EXP3 GENERALISATION CUTS
                </h3>
                <p className="text-[11px] font-mono text-ink-muted">
                  Forward chaining evaluation across increasing historical training windows
                </p>
              </div>
              <span className="text-[11px] font-mono text-ink-muted bg-paper-muted px-2 py-0.5 border border-rule">
                reports/tables/exp3_generalisation.md
              </span>
            </div>

            {fcRows.length > 0 ? (
              <DataTable columns={fcColumns} data={fcRows} keyField="id" />
            ) : (
              <pre className="text-xs font-mono text-ink-muted p-4 overflow-x-auto whitespace-pre-wrap">
                {exp3Markdown || "Loading generalisation cuts..."}
              </pre>
            )}
          </div>

          <div className="mt-3 p-3 bg-paper-muted border border-rule text-xs text-ink leading-relaxed">
            <p>
              <strong>Temporal Finding:</strong> All models experience performance drops on cut fc_44 (test steps 45–49) during the late-period regime change where illicit instance counts drop significantly.
            </p>
          </div>
        </Section>
      )}
    </div>
  );
}
