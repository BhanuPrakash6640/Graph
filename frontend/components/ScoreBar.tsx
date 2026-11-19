import React from "react";

interface ScoreBarProps {
  score: number; // 0 to 1
  threshold: number; // 0 to 1 (the model's alert line)
  showLabels?: boolean;
  className?: string;
}

export function ScoreBar({
  score,
  threshold,
  showLabels = true,
  className = "",
}: ScoreBarProps) {
  const clampedScore = Math.max(0, Math.min(1, score));
  const clampedThreshold = Math.max(0, Math.min(1, threshold));
  const isAlert = clampedScore >= clampedThreshold;

  const barColor = isAlert ? "bg-illicit" : "bg-licit";

  return (
    <div className={`w-full font-mono text-xs ${className}`}>
      {showLabels && (
        <div className="flex items-center justify-between mb-1.5 text-xs text-ink">
          <span>
            Risk score:{" "}
            <strong className={isAlert ? "text-illicit font-bold" : "text-licit font-bold"}>
              {clampedScore.toFixed(4)}
            </strong>{" "}
            <span className="text-ink-muted">out of 1.0000</span>
          </span>
          <span className="text-ink-muted">
            Alert line: <span className="font-semibold text-ink">{clampedThreshold.toFixed(3)}</span>
          </span>
        </div>
      )}

      {/* Track container */}
      <div className="relative h-4 bg-paper-muted border border-rule w-full overflow-visible">
        {/* Fill bar */}
        <div
          className={`h-full ${barColor} transition-all duration-300`}
          style={{ width: `${clampedScore * 100}%` }}
        />

        {/* Threshold tick line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-ink z-10"
          style={{ left: `${clampedThreshold * 100}%` }}
          title={`Alert line: ${clampedThreshold.toFixed(3)}`}
        >
          <div className="absolute -top-1 -translate-x-1/2 w-1.5 h-1 bg-ink" />
          <div className="absolute -bottom-1 -translate-x-1/2 w-1.5 h-1 bg-ink" />
        </div>
      </div>

      {showLabels && (
        <div className="flex justify-between text-[10px] text-ink-muted mt-1 tabular-nums">
          <span>0.00 (Safe)</span>
          <span
            className="text-ink font-semibold"
            style={{ marginLeft: `${clampedThreshold * 100 - 15}%` }}
          >
            ▲ Alert line ({clampedThreshold.toFixed(2)})
          </span>
          <span>1.00 (Illicit)</span>
        </div>
      )}
    </div>
  );
}

