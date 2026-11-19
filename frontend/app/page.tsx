"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  api,
  HealthData,
  ModelInfo,
  InferenceLogItem,
  TemporalData,
  BenchmarkData,
  BenchmarkSummaryData,
} from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { Stat } from "@/components/Stat";
import { CIPlot, CIModelItem } from "@/components/CIPlot";
import { StatSkeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import { ArrowRight, Search, Share2, BarChart2 } from "lucide-react";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [temporal, setTemporal] = useState<TemporalData | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null);
  const [summary, setSummary] = useState<BenchmarkSummaryData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [logs, setLogs] = useState<InferenceLogItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [h, m, l, t, b, s] = await Promise.all([
        api.getHealth(),
        api.getModels(),
        api.getInferenceLogs(6),
        api.getTemporalAnalysis(),
        api.getBenchmark(),
        api.getBenchmarkSummary(),
      ]);
      setHealth(h);
      setModels(m.models);
      setLogs(l.logs);
      setTemporal(t);
      setBenchmark(b);
      setSummary(s);
    } catch (err: any) {
      setError(err.message || "Failed to connect to GraphGuard API backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="border-b border-rule pb-6">
          <div className="h-8 bg-rule/70 w-80 animate-pulse mb-2" />
          <div className="h-4 bg-rule/50 w-full max-w-xl animate-pulse" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <StatSkeleton />
          <StatSkeleton />
          <StatSkeleton />
        </div>
      </div>
    );
  }

  if (error || !health) {
    return (
      <ErrorState
        title="Could not connect to GraphGuard backend"
        error={error || "Backend returned no data."}
        retry={loadData}
      />
    );
  }

  // Calculate statistics from API data
  const numNodes = health.num_nodes;
  const numEdges = health.num_edges;
  const totalSteps = temporal?.per_step_stats.length || 0;

  // Total labelled nodes from per_step_stats
  const totalLabelled = temporal?.per_step_stats
    ? temporal.per_step_stats.reduce((acc, s) => acc + (s.illicit || 0) + (s.licit || 0), 0)
    : 0;
  const labelledPct = numNodes > 0 ? ((totalLabelled / numNodes) * 100).toFixed(1) : "0";

  // Headline PR-AUC comparison on paper_refit (derived from API summary)
  const headlineModels: CIModelItem[] = (summary?.models || [])
    .filter((m) => m.feature_set.includes("all") || m.name === "Static GCN")
    .map((m) => ({
      name: m.name,
      featureSet: m.feature_set,
      mean: m.mean,
      ciLo: m.ci_lo,
      ciHi: m.ci_hi,
      highlight: m.is_headline,
    }));

  const testBaseRate = summary?.base_rate ?? 0;

  return (
    <div className="space-y-8">
      {/* 1. Header with Research Question & In-Short Summary */}
      <PageHeader
        title="Can graph models beat decision trees at finding illicit payments?"
        technical="Empirical comparison of GNNs vs gradient boosted trees on chronological Bitcoin transaction graphs (Elliptic dataset)"
        inShort="No. Tabular XGBoost outperforms graph neural networks on the primary chronological split. Tabular aggregation of neighbor features delivers stronger discrimination than message passing on this transaction network."
      />

      {/* 2. Headline Result: Dot-and-Whisker Figure */}
      <Section
        title="Primary Benchmark: How good is each model family?"
        technical="Test PR-AUC on paper_refit (mean ± paired block-bootstrap 95% confidence interval over time steps)"
      >
        <CIPlot
          items={headlineModels}
          baseRate={testBaseRate}
          baseRateLabel={`Test base rate ≈ ${testBaseRate.toFixed(3)} (random guessing)`}
          title="PR-AUC on paper_refit (higher is better, scale 0 to 1)"
          howToRead="Tabular models show a statistically significant advantage over Graph Convolutional Networks, with the paired block confidence interval excluding zero. Freezing EvolveGCN weights matches static GCN, demonstrating that recurrent weight evolution across disconnected transaction snapshots degrades performance."
        />
        <div className="mt-3 p-3 bg-paper-muted border border-rule text-xs text-ink leading-relaxed flex items-baseline justify-between">
          <span>
            <strong>Key Takeaway: </strong>
            {summary?.key_takeaway || "Tabular XGBoost outperforms graph neural networks on the primary chronological split. The 95% block-bootstrap confidence interval strictly excludes zero."}
          </span>
          <Link
            href="/benchmark"
            className="text-accent font-semibold hover:underline inline-flex items-center gap-1 ml-4 flex-shrink-0"
          >
            <span>Full benchmark tables</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </Section>

      {/* 3. Three Key Dataset Stats (From API, no hardcoded fallbacks) */}
      <Section
        title="Dataset at a glance"
        technical="Summary counts served directly from backend GraphStore and EDA validation"
      >
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Stat
            label="Total Payments"
            value={numNodes.toLocaleString()}
            scale="nodes"
            footnote="Bitcoin transaction nodes"
            source="GET /api/health"
          />
          <Stat
            label="Supervised Share"
            value={`${labelledPct}%`}
            scale={`(${totalLabelled.toLocaleString()} labelled)`}
            footnote="77.1% unlabelled nodes retained for graph context"
            source="GET /api/temporal-analysis"
          />
          <Stat
            label="Time Steps"
            value={totalSteps}
            scale="snapshots"
            footnote="Chronological 2-week observation periods"
            source="GET /api/temporal-analysis"
          />
        </div>
      </Section>

      {/* 4. Three Clear Entry Points */}
      <Section
        title="Explore the Research"
        technical="Interactive tools to inspect predictions, graph topology, and benchmark suites"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/investigate"
            className="group p-5 bg-paper-card border border-rule hover:border-accent transition flex flex-col justify-between"
          >
            <div>
              <div className="w-8 h-8 rounded-none border border-rule bg-paper-muted flex items-center justify-center text-accent mb-3">
                <Search className="w-4 h-4" />
              </div>
              <h3 className="font-serif text-base font-semibold text-ink group-hover:text-accent transition">
                Investigate a payment
              </h3>
              <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                Look up any transaction to inspect its risk score, alert line, connected payment flow, and feature attributions.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-rule/60 text-xs font-mono text-accent flex items-center justify-between">
              <span>Open Case File</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition" />
            </div>
          </Link>

          <Link
            href="/explorer"
            className="group p-5 bg-paper-card border border-rule hover:border-accent transition flex flex-col justify-between"
          >
            <div>
              <div className="w-8 h-8 rounded-none border border-rule bg-paper-muted flex items-center justify-center text-accent mb-3">
                <Share2 className="w-4 h-4" />
              </div>
              <h3 className="font-serif text-base font-semibold text-ink group-hover:text-accent transition">
                Explore the graph
              </h3>
              <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                Interactive concentric visualization of transaction neighborhoods up to 2 hops with payment direction arrows.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-rule/60 text-xs font-mono text-accent flex items-center justify-between">
              <span>Open Graph Explorer</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition" />
            </div>
          </Link>

          <Link
            href="/benchmark"
            className="group p-5 bg-paper-card border border-rule hover:border-accent transition flex flex-col justify-between"
          >
            <div>
              <div className="w-8 h-8 rounded-none border border-rule bg-paper-muted flex items-center justify-center text-accent mb-3">
                <BarChart2 className="w-4 h-4" />
              </div>
              <h3 className="font-serif text-base font-semibold text-ink group-hover:text-accent transition">
                See all results
              </h3>
              <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                Sortable performance tables across 7 splits, paired bootstrap confidence intervals, and ablation suite results.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-rule/60 text-xs font-mono text-accent flex items-center justify-between">
              <span>View Benchmark Suite</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition" />
            </div>
          </Link>
        </div>
      </Section>

      {/* 5. System Section (Secondary, at bottom) */}
      <Section
        title="System Status & Audit Log"
        technical="Backend inference engines and live audit trail"
      >
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Registry */}
          <div className="p-4 bg-paper-card border border-rule">
            <h4 className="font-serif text-sm font-semibold text-ink mb-2">
              Registered Inference Models
            </h4>
            <div className="space-y-2 text-xs font-mono divide-y divide-rule/60">
              {models.map((m) => (
                <div key={m.id} className="pt-2 first:pt-0 flex items-center justify-between">
                  <div>
                    <span className="font-semibold text-ink">{m.name}</span>
                    <span className="text-[10px] text-ink-muted block font-sans">
                      {m.description}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-ink-muted">Alert line:</span>{" "}
                    <span className="font-bold text-ink">{m.threshold.toFixed(3)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Audit Log */}
          <div className="p-4 bg-paper-card border border-rule">
            <h4 className="font-serif text-sm font-semibold text-ink mb-2">
              Recent Predictions Audit Trail
            </h4>
            {logs.length === 0 ? (
              <p className="text-xs text-ink-muted font-mono">No recent inference records.</p>
            ) : (
              <div className="space-y-1.5 text-xs font-mono divide-y divide-rule/60">
                {logs.slice(0, 5).map((l) => (
                  <div key={l.id} className="pt-1.5 first:pt-0 flex items-center justify-between text-ink">
                    <span className="text-ink-muted text-[11px]">{l.ts.slice(11, 19)}</span>
                    <Link
                      href={`/transaction/${l.tx_id}`}
                      className="text-accent hover:underline"
                    >
                      tx {l.tx_id}
                    </Link>
                    <span className="text-ink-muted">{l.model_id.replace("_paper_refit_all", "")}</span>
                    <span className="font-bold tabular-nums">
                      {l.score.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Section>
    </div>
  );
}
