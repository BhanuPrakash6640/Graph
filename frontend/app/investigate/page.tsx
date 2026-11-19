"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, ExampleItem } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import { Search, ArrowRight, ShieldCheck, AlertTriangle, HelpCircle } from "lucide-react";

export default function InvestigateSearchPage() {
  const router = useRouter();
  const [searchId, setSearchId] = useState("");
  const [examples, setExamples] = useState<ExampleItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const data = await api.getExamples();
        setExamples(data.examples);
      } catch (err: any) {
        setError(err.message || "Failed to load example cases from backend.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchId.trim();
    if (trimmed) {
      router.push(`/transaction/${trimmed}`);
    }
  };

  const getCategoryIcon = (category: string) => {
    if (category.includes("True Positive")) {
      return <AlertTriangle className="w-4 h-4 text-illicit" />;
    }
    if (category.includes("False Positive")) {
      return <ShieldCheck className="w-4 h-4 text-licit" />;
    }
    if (category.includes("False Negative")) {
      return <AlertTriangle className="w-4 h-4 text-ink-muted" />;
    }
    return <HelpCircle className="w-4 h-4 text-ink-muted" />;
  };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Investigate a payment"
        technical="Search individual transaction nodes to inspect model scores, connected payments, and feature attributions"
        inShort="Enter any Bitcoin payment ID from the 203,769 transactions in the historical dataset, or choose a curated case study below to open its forensic file."
      />

      {/* Search Input Box */}
      <Section title="Search by Payment ID" technical="Lookup txId in historical dataset">
        <form onSubmit={handleSearch} className="max-w-2xl">
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
                placeholder="Enter transaction ID (e.g. 69746416 or 106364923)"
                className="w-full px-4 py-3 bg-paper-card border border-rule font-mono text-sm text-ink placeholder:text-ink-muted focus:outline-none focus:border-accent"
              />
              <Search className="w-4 h-4 text-ink-muted absolute right-3.5 top-3.5" />
            </div>
            <button
              type="submit"
              disabled={!searchId.trim()}
              className="px-6 py-3 bg-accent text-white font-mono text-xs font-semibold hover:bg-accent-hover transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <span>Open Case File</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </form>
      </Section>

      {/* Curated Case Files Gallery */}
      <Section
        title="Representative Case Studies"
        technical="Pre-selected case files demonstrating model detection patterns, false alarms, and evasion"
      >
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        ) : error ? (
          <ErrorState
            title="Failed to load example cases"
            error={error}
            retry={() => {
              setLoading(true);
              api.getExamples().then((d) => setExamples(d.examples)).finally(() => setLoading(false));
            }}
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {examples.map((item) => (
              <div
                key={item.tx_id}
                onClick={() => router.push(`/transaction/${item.tx_id}`)}
                className="p-5 bg-paper-card border border-rule hover:border-accent cursor-pointer transition flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="inline-flex items-center gap-1.5 text-xs font-mono font-semibold px-2 py-0.5 border border-rule bg-paper-muted text-ink">
                      {getCategoryIcon(item.category)}
                      <span>{item.category}</span>
                    </span>
                    <span className="text-[11px] font-mono text-ink-muted">
                      Step {item.time_step}
                    </span>
                  </div>

                  <h3 className="font-mono text-sm font-bold text-ink group-hover:text-accent transition mt-2">
                    Payment ID: {item.tx_id}
                  </h3>
                  <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                    {item.description}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-rule/60 flex items-center justify-between text-xs font-mono">
                  <span className="text-ink-muted">
                    Score: <strong className="text-ink font-bold">{item.score.toFixed(4)}</strong>
                  </span>
                  <span className="text-accent flex items-center gap-1 group-hover:translate-x-0.5 transition font-semibold">
                    <span>Inspect</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>
    </div>
  );
}

