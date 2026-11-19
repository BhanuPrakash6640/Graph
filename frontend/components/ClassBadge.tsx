import React from "react";

export type LabelType = "illicit" | "licit" | "unknown" | 1 | 0 | -1 | 2;

interface ClassBadgeProps {
  type: "known" | "predicted";
  label: LabelType;
  score?: number;
  threshold?: number;
  className?: string;
}

function normalizeLabel(l: LabelType): "illicit" | "licit" | "unknown" {
  if (l === "illicit" || l === 1) return "illicit";
  if (l === "licit" || l === 0) return "licit";
  return "unknown";
}

export function ClassBadge({
  type,
  label,
  score,
  threshold,
  className = "",
}: ClassBadgeProps) {
  const norm = normalizeLabel(label);

  if (type === "known") {
    // Known label = filled shape with solid semantic color or hatched for unknown
    if (norm === "illicit") {
      return (
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-semibold text-white bg-illicit border border-illicit ${className}`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-white" />
          <span>Known Illicit</span>
        </span>
      );
    }
    if (norm === "licit") {
      return (
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-semibold text-white bg-licit border border-licit ${className}`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-white" />
          <span>Known Licit</span>
        </span>
      );
    }
    return (
      <span
        className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium text-ink hatched-unknown border border-dashed border-rule-dark ${className}`}
      >
        <span className="w-1.5 h-1.5 rounded-full border border-ink-muted bg-paper" />
        <span>Known: Unknown</span>
      </span>
    );
  }

  // Prediction: outline ring in the predicted colour + score
  const isHighRisk =
    score !== undefined && threshold !== undefined
      ? score >= threshold
      : norm === "illicit";

  const borderColor = isHighRisk ? "border-illicit text-illicit" : "border-licit text-licit";
  const statusText = isHighRisk ? "Alert: Flagged" : "Alert: Low Risk";

  return (
    <span
      className={`inline-flex items-center gap-2 px-2.5 py-0.5 text-xs font-mono font-medium bg-paper border ${borderColor} ${className}`}
    >
      <span className="w-2 h-2 rounded-full border-2 border-current bg-transparent" />
      <span>{statusText}</span>
      {score !== undefined && (
        <span className="font-bold tabular-nums">
          ({(score * 100).toFixed(1)}%)
        </span>
      )}
    </span>
  );
}

