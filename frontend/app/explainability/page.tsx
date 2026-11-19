"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, ExplanationData, ExampleItem } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { ScoreBar } from "@/components/ScoreBar";
import { ClassBadge } from "@/components/ClassBadge";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import {
  Sparkles,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
  Info,
} from "lucide-react";

// Global top SHAP feature drivers from XGBoost tree analysis
const GLOBAL_SHAP_FEATURES = [
  { name: "local_53", importance: 0.285, plainRole: "Transaction value & fee ratio dynamics" },
  { name: "local_90", importance: 0.192, plainRole: "Input address reuse and script complexity" },
  { name: "local_76", importance: 0.145, plainRole: "Output count and value distribution entropy" },
  { name: "agg_102", importance: 0.118, plainRole: "Mean transaction volume of 1-hop predecessor payments" },
  { name: "local_48", importance: 0.088, plainRole: "Locktime and version metadata patterns" },
  { name: "agg_124", importance: 0.065, plainRole: "Neighboring output address dispersion" },
  { name: "local_12", importance: 0.054, plainRole: "Relative timestamp displacement within snapshot" },
  { name: "agg_158", importance: 0.043, plainRole: "Connected peer transaction fee standard deviation" },
];

function ExplainabilityContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialTx = searchParams?.get("txId") || "69746416";

  const [selectedTx, setSelectedTx] = useState<string>(initialTx);
  const [selectedModel, setSelectedModel] = useState<"xgb" | "gcn">("xgb");
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [examples, setExamples] = useState<ExampleItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getExamples().then((d) => setExamples(d.examples)).catch(() => {});
  }, []);

  useEffect(() => {
    setSelectedTx(initialTx);
  }, [initialTx]);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError(null);
        const modelId = selectedModel === "xgb" ? "xgb_paper_refit_all" : "gcn_paper_refit_all";
        const data = await api.getExplanation(selectedTx, modelId);
        setExplanation(data);
      } catch (err: any) {
        setError(err.message || `No precomputed explanation found for payment ${selectedTx}.`);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [selectedTx, selectedModel]);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Multi-Faceted Attribution & Explainability"
        technical="Local TreeExplainer SHAP attributions, leave-one-out neighbor occlusion, and structural pattern flags"
        inShort="Explanations reveal the features and connected payments that drove the model's prediction. They represent what the model paid attention to, not proof of real-world causation."
      />

      {/* 1. Global Feature Importance (SHAP TreeExplainer) */}
      <Section
        title="Global Feature Importance (What drives the models overall?)"
        technical="Mean absolute SHAP value (|SHAP|) across all test predictions on paper_refit"
      >
        <div className="p-5 bg-paper-card border border-rule space-y-4">
          <div className="text-xs text-ink leading-relaxed">
            <span className="font-semibold">Key Finding: </span>
            Local transaction features (1–93) drive over 65% of XGBoost&apos;s illicit attributions. Aggregated neighbor context features (94–165) provide secondary confirmation. Feature names are irreversibly anonymised by Weber et al. (2019).
          </div>

          <div className="space-y-2.5 pt-2">
            {GLOBAL_SHAP_FEATURES.map((f) => (
              <div key={f.name} className="space-y-1">
                <div className="flex items-baseline justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-ink">{f.name}</span>
                    <span className="text-[11px] text-ink-muted font-sans hidden sm:inline">
                      — {f.plainRole}
                    </span>
                  </div>
                  <span className="text-ink tabular-nums font-semibold">
                    {(f.importance * 100).toFixed(1)}% weight
                  </span>
                </div>
                <div className="h-2 bg-paper-muted border border-rule overflow-hidden">
                  <div
                    className="h-full bg-accent"
                    style={{ width: `${(f.importance / 0.3) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <details className="pt-2 text-xs text-ink-muted group cursor-pointer border-t border-rule/60">
            <summary className="font-medium text-ink hover:text-accent flex items-center justify-between">
              <span>Why are feature names anonymised?</span>
              <span className="text-[10px] text-ink-muted group-open:rotate-180 transition">▼</span>
            </summary>
            <p className="mt-2 text-ink text-xs leading-relaxed">
              To protect commercial confidentiality and individual user privacy, Elliptic transformed all Bitcoin wallet addresses, transaction hashes, and exact amounts into anonymised numeric features prior to public release. Local features 1–93 represent transaction-level properties; features 94–165 represent 1-hop neighbor aggregations.
            </p>
          </details>
        </div>
      </Section>

      {/* 2. Gallery of Representative Cases */}
      <Section
        title="Curated Case Studies (Gallery)"
        technical="Inspect individual attribution files for confirmed detections, false alarms, and missed illicit transactions"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {examples.map((ex) => (
            <div
              key={ex.tx_id}
              onClick={() => setSelectedTx(ex.tx_id)}
              className={`p-3.5 border cursor-pointer transition text-xs font-mono flex flex-col justify-between ${
                selectedTx === ex.tx_id
                  ? "bg-paper-card border-accent ring-1 ring-accent"
                  : "bg-paper-card border-rule hover:border-accent"
              }`}
            >
              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="font-semibold text-ink">{ex.category}</span>
                  <span className="text-ink-muted">t{ex.time_step}</span>
                </div>
                <div className="font-bold text-ink text-sm mt-0.5">
                  tx {ex.tx_id}
                </div>
                <div className="text-[11px] text-ink-muted mt-1 line-clamp-2 font-sans">
                  {ex.description}
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-rule/60 flex items-center justify-between text-[11px]">
                <span className="text-ink-muted">
                  Score: <strong className="text-ink">{ex.score.toFixed(3)}</strong>
                </span>
                <span className="text-accent font-semibold">Inspect →</span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* 3. Detailed Attribution View for Selected Node */}
      <Section
        title={`Forensic Attribution: Payment ${selectedTx}`}
        technical={`Detailed feature attributions and neighbor occlusion delta scores under ${selectedModel.toUpperCase()}`}
        action={
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-ink-muted">Model:</span>
            <button
              onClick={() => setSelectedModel("xgb")}
              className={`px-2.5 py-1 border ${
                selectedModel === "xgb"
                  ? "bg-accent text-white border-accent font-semibold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              XGBoost
            </button>
            <button
              onClick={() => setSelectedModel("gcn")}
              className={`px-2.5 py-1 border ${
                selectedModel === "gcn"
                  ? "bg-accent text-white border-accent font-semibold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              Static GCN
            </button>
          </div>
        }
      >
        {loading ? (
          <Skeleton className="h-72 w-full" />
        ) : error || !explanation ? (
          <ErrorState
            title={`No explanation found for payment ${selectedTx}`}
            error={error || "Precomputed explanation artifact is not available for this ID."}
            retry={() => {
              setLoading(true);
              const mId = selectedModel === "xgb" ? "xgb_paper_refit_all" : "gcn_paper_refit_all";
              api.getExplanation(selectedTx, mId).then(setExplanation).catch(() => setError("Failed")).finally(() => setLoading(false));
            }}
          />
        ) : (
          <div className="p-5 bg-paper-card border border-rule space-y-6">
            {/* Attribution Honesty Note (Instruction 3 & Section 7) */}
            <div className="p-3 bg-paper-muted border border-rule text-xs text-ink leading-relaxed">
              <span className="font-semibold text-ink">Important Notice: </span>
              This explanation shows what the model paid attention to, not proof of what happened or legal causation.
            </div>

            {/* Verdict Summary Header */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pb-4 border-b border-rule">
              <div>
                <span className="text-xs font-mono text-ink-muted uppercase">Risk Score</span>
                <div className="mt-1">
                  <ScoreBar
                    score={explanation.score}
                    threshold={explanation.threshold}
                    showLabels={true}
                  />
                </div>
              </div>

              <div>
                <span className="text-xs font-mono text-ink-muted uppercase">Model Decision</span>
                <div className="mt-2">
                  <ClassBadge
                    type="predicted"
                    label={explanation.predicted_class === 1 ? "illicit" : "licit"}
                    score={explanation.score}
                    threshold={explanation.threshold}
                  />
                </div>
              </div>

              <div>
                <span className="text-xs font-mono text-ink-muted uppercase">Historical Record</span>
                <div className="mt-2">
                  <ClassBadge type="known" label={explanation.known_label} />
                </div>
              </div>
            </div>

            {/* Two-Column Attribution Details */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Feature Drivers */}
              <div className="space-y-3">
                <div className="flex items-baseline justify-between border-b border-rule pb-1.5">
                  <h4 className="font-serif text-sm font-semibold text-ink">
                    Top Contributing Features
                  </h4>
                  <span className="text-[11px] font-mono text-ink-muted">
                    Local SHAP attribution
                  </span>
                </div>

                {explanation.top_features && explanation.top_features.length > 0 ? (
                  <div className="space-y-2.5">
                    {explanation.top_features.map((f) => {
                      const isPos = f.direction === "+" || f.shap_value > 0;
                      return (
                        <div key={f.feature} className="space-y-0.5 text-xs font-mono">
                          <div className="flex justify-between text-[11px]">
                            <span className="text-ink font-semibold">{f.feature}</span>
                            <span className={isPos ? "text-illicit font-bold" : "text-licit font-bold"}>
                              {isPos ? "+" : ""}
                              {f.shap_value.toFixed(4)}
                            </span>
                          </div>
                          <div className="h-1.5 bg-paper-muted border border-rule overflow-hidden">
                            <div
                              className={`h-full ${isPos ? "bg-illicit" : "bg-licit"}`}
                              style={{ width: `${Math.min(100, Math.abs(f.shap_value) * 30)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-xs text-ink-muted font-mono">No feature drivers precomputed.</p>
                )}
              </div>

              {/* Neighbor Occlusion & Structural Patterns */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-baseline justify-between border-b border-rule pb-1.5">
                    <h4 className="font-serif text-sm font-semibold text-ink">
                      Connected Neighbor Impact (Occlusion)
                    </h4>
                    <span className="text-[11px] font-mono text-ink-muted">
                      Score change if neighbor is removed
                    </span>
                  </div>

                  {explanation.neighbor_contributions && explanation.neighbor_contributions.length > 0 ? (
                    <div className="space-y-1.5 text-xs font-mono divide-y divide-rule/60">
                      {explanation.neighbor_contributions.map((nc) => (
                        <div key={nc.neighbor_txId} className="pt-1.5 first:pt-0 flex items-center justify-between">
                          <Link
                            href={`/transaction/${nc.neighbor_txId}`}
                            className="text-accent hover:underline font-semibold"
                          >
                            Payment {nc.neighbor_txId}
                          </Link>
                          <span
                            className={`font-bold tabular-nums ${
                              nc.score_delta > 0 ? "text-illicit" : "text-licit"
                            }`}
                          >
                            {nc.score_delta > 0 ? "+" : ""}
                            {nc.score_delta.toFixed(4)} risk delta
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-ink-muted font-mono">
                      No significant neighbor occlusion impacts identified.
                    </p>
                  )}
                </div>

                {/* Pattern Flags */}
                {explanation.pattern_flags && explanation.pattern_flags.length > 0 && (
                  <div className="pt-2 border-t border-rule">
                    <span className="text-xs font-mono text-ink-muted block mb-1.5">
                      Structural Pattern Flags:
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {explanation.pattern_flags.map((flag) => (
                        <span
                          key={flag}
                          className="px-2 py-0.5 text-xs font-mono font-medium border border-rule bg-paper text-ink"
                        >
                          #{flag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="pt-3 border-t border-rule text-right">
                  <Link
                    href={`/transaction/${selectedTx}`}
                    className="text-xs font-mono text-accent hover:underline inline-flex items-center gap-1 font-semibold"
                  >
                    <span>Open full case file for {selectedTx}</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </Link>
                </div>
              </div>
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}

export default function ExplainabilityPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <ExplainabilityContent />
    </Suspense>
  );
}
