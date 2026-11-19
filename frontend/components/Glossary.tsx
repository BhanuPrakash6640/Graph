"use client";

import React, { useState } from "react";
import { BookOpen, X } from "lucide-react";

export interface GlossaryTerm {
  term: string;
  plain: string;
  technical: string;
}

export const GLOSSARY_TERMS: GlossaryTerm[] = [
  {
    term: "Payment",
    plain: "A Bitcoin transaction in the dataset transferring value between addresses.",
    technical: "A graph node (txId) with 165 numeric attributes and a discrete time step (1–49).",
  },
  {
    term: "Alert line",
    plain: "The model's decision boundary. A payment scored above this line triggers an alert for manual review.",
    technical: "The decision threshold (tau) tuned on validation set to maximize F1 score (typically 0.45–0.85).",
  },
  {
    term: "Illicit",
    plain: "Payments linked by forensic analysts to illegal entities (ransomware, dark markets, scams).",
    technical: "Class 1 in ground truth labels (4,545 nodes total, ~2.2% base rate across all nodes).",
  },
  {
    term: "Licit",
    plain: "Payments linked to legitimate businesses, exchanges, miners, or wallet providers.",
    technical: "Class 0 in ground truth labels (42,019 nodes total).",
  },
  {
    term: "Unlabelled / Unknown",
    plain: "Payments where real-world entity identity is unconfirmed by forensic investigators.",
    technical: "Unlabelled nodes (157,205 nodes, ~77% of graph). Kept in graph for message passing.",
  },
  {
    term: "PR-AUC",
    plain: "Precision-Recall Area Under Curve. A score from 0 to 1 measuring how well the model catches illicit payments without raising excessive false alarms.",
    technical: "Primary benchmark metric robust to extreme class imbalance; random chance equals the illicit prevalence (~0.07 on labelled test set).",
  },
  {
    term: "Connected payments",
    plain: "Other transactions that either sent money directly into this payment (inputs) or received money from it (outputs).",
    technical: "Directed edges (payment flows) and k-hop neighborhood subgraphs.",
  },
  {
    term: "False alarm (False positive)",
    plain: "A legitimate payment mistakenly flagged with a high illicit risk score.",
    technical: "True label = licit (0), but model prediction score >= alert threshold.",
  },
  {
    term: "Missed illicit (False negative)",
    plain: "An illicit payment that evaded detection with an alert score below the threshold.",
    technical: "True label = illicit (1), but model prediction score < alert threshold.",
  },
  {
    term: "Attribution",
    plain: "What features or neighbors the model paid attention to when calculating its score. This is an indicator of model reasoning, not legal proof of guilt or real-world causation.",
    technical: "Local SHAP feature importances and GNNExplainer edge masks.",
  },
  {
    term: "Block bootstrap",
    plain: "A statistical test that accounts for temporal clustering by resampling entire time steps together rather than individual transactions in isolation.",
    technical: "Paired block-bootstrap 95% confidence intervals (2,000 iterations grouped by time step).",
  },
];

export function GlossaryModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 backdrop-blur-sm p-4"
    >
      <div className="bg-paper-card border border-rule w-full max-w-2xl max-h-[85vh] flex flex-col shadow-xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-rule">
          <div>
            <h2 className="font-serif text-xl font-semibold text-ink">
              Words used on this site
            </h2>
            <p className="text-xs text-ink-muted mt-0.5">
              Plain-language guide to terms and metrics in GraphGuard
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-ink-muted hover:text-ink hover:bg-paper-muted"
            aria-label="Close glossary"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="overflow-y-auto px-6 py-4 space-y-4 divide-y divide-rule">
          {GLOSSARY_TERMS.map((g, idx) => (
            <div key={g.term} className={idx > 0 ? "pt-4" : ""}>
              <h3 className="font-mono text-sm font-semibold text-ink">
                {g.term}
              </h3>
              <p className="text-sm text-ink mt-1 leading-relaxed">{g.plain}</p>
              <p className="text-xs text-ink-muted mt-1 font-mono">
                Technical: {g.technical}
              </p>
            </div>
          ))}
        </div>

        <div className="border-t border-rule px-6 py-3 bg-paper-muted flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-ink bg-paper border border-rule hover:bg-paper-card transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export function GlossaryButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 text-xs text-ink-muted hover:text-accent font-medium transition"
        title="View plain language definitions"
      >
        <BookOpen className="w-3.5 h-3.5" />
        <span>Glossary</span>
      </button>
      <GlossaryModal isOpen={open} onClose={() => setOpen(false)} />
    </>
  );
}

