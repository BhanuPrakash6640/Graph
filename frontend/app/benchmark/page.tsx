"use client";

import React, { useEffect, useState } from "react";
import { api, BenchmarkData, BenchmarkSummaryData } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { CIPlot, CIModelItem } from "@/components/CIPlot";
import { DataTable, Column } from "@/components/DataTable";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import { BarChart2, CheckCircle, Info } from "lucide-react";

// Helper to parse markdown table into array of row objects
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

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [summary, setSummary] = useState<BenchmarkSummaryData | null>(null);
  const [activeTab, setActiveTab] = useState<string>("exp1_paper_refit");
  const [viewMode, setViewMode] = useState<"tables" | "comparisons" | "curves">("tables");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [res, sRes] = await Promise.all([
          api.getBenchmark(),
          api.getBenchmarkSummary(),
        ]);
        setData(res);
        setSummary(sRes);
      } catch (err: any) {
        setError(err.message || "Failed to load benchmark suite data.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const benchmarkTabs = [
    { id: "exp1_paper_refit", label: "Paper Refit (Exp 1)", split: "paper_refit" },
    { id: "exp2_paper_refit", label: "Temporal Refit (Exp 2)", split: "paper_refit" },
    { id: "exp1_all_models", label: "All Splits (Exp 1)", split: "all splits" },
    { id: "exp2_temporal", label: "All Temporal (Exp 2)", split: "all splits" },
    { id: "exp3_generalisation", label: "Generalisation Cuts (Exp 3)", split: "fc cuts" },
    { id: "exp4_imbalance", label: "Class Imbalance (Exp 4)", split: "paper_refit" },
    { id: "exp5_ablations", label: "Ablation Suite (Exp 5)", split: "paper_refit" },
  ];

  const comparisonTabs = [
    { id: "exp1_comparisons", label: "Exp 1 Statistical CIs (Tabular vs GNN)" },
    { id: "exp2_comparisons", label: "Exp 2 Statistical CIs (Temporal vs Static)" },
  ];

  // Primary model × feature-set plot data served from backend summary
  const benchmarkCIItems: CIModelItem[] = (summary?.models || []).map((m) => ({
    name: m.name,
    featureSet: m.feature_set,
    mean: m.mean,
    ciLo: m.ci_lo,
    ciHi: m.ci_hi,
    highlight: m.is_headline,
  }));

  const activeMarkdown = data
    ? viewMode === "comparisons"
      ? data.comparisons[activeTab] || ""
      : data.tables[activeTab] || ""
    : "";

  const { columns, rows } = parseMarkdownTable(activeMarkdown);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Scientific Benchmark & Statistical Suite"
        technical="Verbatim performance tables generated from saved predictions across 7 splits and 5 seeds"
        inShort="Comprehensive benchmark evaluating tree ensembles, MLPs, GNNs (GCN, GAT, GraphSAGE), and temporal architectures (EvolveGCN). All paired comparisons compute 95% block-bootstrap confidence intervals grouped by time step."
      />

      {/* Main View Mode Selector */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-rule pb-4">
        <div className="flex items-center gap-1 font-mono text-xs">
          <button
            onClick={() => {
              setViewMode("tables");
              setActiveTab("exp1_paper_refit");
            }}
            className={`px-3 py-1.5 border font-semibold ${
              viewMode === "tables"
                ? "bg-accent text-white border-accent"
                : "bg-paper text-ink border-rule hover:bg-paper-muted"
            }`}
          >
            Performance Tables
          </button>
          <button
            onClick={() => {
              setViewMode("comparisons");
              setActiveTab("exp2_comparisons");
            }}
            className={`px-3 py-1.5 border font-semibold ${
              viewMode === "comparisons"
                ? "bg-accent text-white border-accent"
                : "bg-paper text-ink border-rule hover:bg-paper-muted"
            }`}
          >
            Paired Statistical CIs
          </button>
          <button
            onClick={() => setViewMode("curves")}
            className={`px-3 py-1.5 border font-semibold ${
              viewMode === "curves"
                ? "bg-accent text-white border-accent"
                : "bg-paper text-ink border-rule hover:bg-paper-muted"
            }`}
          >
            PR & ROC Curves
          </button>
        </div>

        <span className="text-xs font-mono text-ink-muted">
          All numbers served directly from API (no hardcoded TSX)
        </span>
      </div>

      {loading ? (
        <Skeleton className="h-96 w-full" />
      ) : error ? (
        <ErrorState title="Failed to load benchmark data" error={error} />
      ) : viewMode === "curves" ? (
        /* PR and ROC Curves View */
        <Section
          title="Precision-Recall & ROC Curves on paper_refit"
          technical="Evaluation curves computed on labelled test nodes (time steps 35–49) using representative seed 0"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 bg-paper-card border border-rule space-y-2">
              <h3 className="font-serif text-base font-semibold text-ink">
                Precision-Recall Curves (Primary)
              </h3>
              <p className="text-xs text-ink-muted leading-relaxed">
                PR curves evaluate detection performance under extreme class imbalance. High curves near the top-right indicate strong illicit precision without runaway false alarms.
              </p>
              <div className="border border-rule bg-paper-muted p-2 flex items-center justify-center min-h-[300px]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/reports/figures/exp1_pr_curves.png"
                  alt="PR Curves comparison across models"
                  className="max-h-[320px] object-contain"
                  onError={(e) => {
                    // fallback text if local static image route is not mounted
                    (e.target as HTMLElement).style.display = "none";
                  }}
                />
                <div className="text-xs font-mono text-ink-muted text-center p-4">
                  <p className="font-semibold text-ink">PR Curves Artifact</p>
                  <p className="text-[11px] mt-1">
                    Precision-Recall evaluation across test steps 35–49 (seed 0)
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-paper-card border border-rule space-y-2">
              <h3 className="font-serif text-base font-semibold text-ink">
                Receiver Operating Characteristic (ROC)
              </h3>
              <p className="text-xs text-ink-muted leading-relaxed">
                ROC curves measure True Positive Rate vs False Positive Rate. Note that ROC is known to be overly optimistic on imbalanced data; PR-AUC is our primary evaluation metric.
              </p>
              <div className="border border-rule bg-paper-muted p-2 flex items-center justify-center min-h-[300px]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/reports/figures/exp1_roc_curves.png"
                  alt="ROC Curves comparison across models"
                  className="max-h-[320px] object-contain"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = "none";
                  }}
                />
                <div className="text-xs font-mono text-ink-muted text-center p-4">
                  <p className="font-semibold text-ink">ROC Curves Artifact</p>
                  <p className="text-[11px] mt-1">
                    Receiver Operating Characteristic evaluation across test steps 35–49 (seed 0)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Section>
      ) : (
        /* Tables and Dot-Whisker View */
        <div className="space-y-8">
          {/* Dot-Whisker Headline Overview */}
          <Section
            title="Model Comparison by Feature Set (paper_refit)"
            technical="Test PR-AUC across model families and feature representations (mean ± 95% block CI)"
          >
            <CIPlot
              items={benchmarkCIItems}
              baseRate={summary?.base_rate ?? 0}
              title="PR-AUC on paper_refit (scale 0 to 1)"
              howToRead="Tabular models (XGBoost, Random Forest) lead by a significant margin over graph neural networks. Static GCN and EvolveGCN-Frozen land within the same confidence interval, confirming the validity of the frozen-weights sanity gate."
            />
          </Section>

          {/* Sub-Tabs for Tables / Comparisons */}
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2 border-b border-rule pb-2">
              {(viewMode === "tables" ? benchmarkTabs : comparisonTabs).map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-3 py-1 text-xs font-mono transition ${
                    activeTab === tab.id
                      ? "bg-ink text-paper font-semibold"
                      : "bg-paper-card text-ink-muted hover:text-ink border border-rule"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Render Sortable Data Table */}
            <div className="p-4 bg-paper-card border border-rule">
              <div className="flex items-center justify-between mb-4 pb-2 border-b border-rule">
                <div>
                  <h3 className="font-serif text-base font-semibold text-ink">
                    {activeTab.replace(/_/g, " ").toUpperCase()}
                  </h3>
                  <p className="text-[11px] font-mono text-ink-muted mt-0.5">
                    Click any column header to sort ascending or descending
                  </p>
                </div>
                <span className="text-[11px] font-mono text-ink-muted bg-paper-muted px-2 py-0.5 border border-rule">
                  Artifact: reports/tables/{activeTab}.md
                </span>
              </div>

              {rows.length > 0 ? (
                <DataTable columns={columns} data={rows} keyField="id" />
              ) : (
                <pre className="text-xs font-mono text-ink-muted p-4 overflow-x-auto whitespace-pre-wrap">
                  {activeMarkdown || "Table artifact loading..."}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
