import React from "react";

export interface CIModelItem {
  name: string;
  mean: number;
  ciLo: number;
  ciHi: number;
  highlight?: boolean;
  featureSet?: string;
  detail?: string;
}

interface CIPlotProps {
  items: CIModelItem[];
  baseRate?: number; // e.g. 0.065 (random chance)
  baseRateLabel?: string;
  minScale?: number;
  maxScale?: number;
  title?: string;
  technicalNote?: string;
  howToRead?: string;
  className?: string;
}

export function CIPlot({
  items,
  baseRate = 0.065,
  baseRateLabel = "Test base rate ≈ 0.065 (random chance)",
  minScale = 0.0,
  maxScale = 1.0,
  title,
  technicalNote,
  howToRead,
  className = "",
}: CIPlotProps) {
  const range = maxScale - minScale;
  const toPct = (val: number) => {
    const clamped = Math.max(minScale, Math.min(maxScale, val));
    return ((clamped - minScale) / range) * 100;
  };

  return (
    <div className={`p-4 bg-paper-card border border-rule ${className}`}>
      {title && (
        <div className="mb-3">
          <h3 className="font-serif text-base font-semibold text-ink">{title}</h3>
          {technicalNote && (
            <p className="font-mono text-xs text-ink-muted mt-0.5">
              Technical: {technicalNote}
            </p>
          )}
        </div>
      )}

      {/* Axis and plot area */}
      <div className="pt-2 pb-2">
        {/* Model Rows */}
        <div className="space-y-3.5">
          {items.map((m) => {
            const leftPct = toPct(m.ciLo);
            const rightPct = toPct(m.ciHi);
            const centerPct = toPct(m.mean);
            const widthPct = Math.max(0.5, rightPct - leftPct);

            return (
              <div key={`${m.name}_${m.featureSet}`} className="flex flex-col sm:flex-row sm:items-center text-xs">
                {/* Model Label */}
                <div className="w-48 flex-shrink-0 font-mono text-ink mb-1 sm:mb-0">
                  <span className={m.highlight ? "font-bold text-accent" : "font-medium"}>
                    {m.name}
                  </span>
                  {m.featureSet && (
                    <span className="text-[10px] text-ink-muted ml-1.5">
                      ({m.featureSet})
                    </span>
                  )}
                </div>

                {/* Plot track */}
                <div className="relative flex-1 h-6 bg-paper-muted border border-rule">
                  {/* Base rate reference line inside track */}
                  {baseRate !== undefined && (
                    <div
                      className="absolute top-0 bottom-0 w-px border-l border-dashed border-ink-muted/80 z-0 pointer-events-none"
                      style={{ left: `${toPct(baseRate)}%` }}
                    />
                  )}

                  {/* Whisker line */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 h-0.5 bg-ink"
                    style={{
                      left: `${leftPct}%`,
                      width: `${widthPct}%`,
                    }}
                  />
                  {/* Whisker left cap */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-0.5 h-3 bg-ink"
                    style={{ left: `${leftPct}%` }}
                  />
                  {/* Whisker right cap */}
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-0.5 h-3 bg-ink"
                    style={{ left: `${rightPct}%` }}
                  />
                  {/* Center Dot */}
                  <div
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full border-2 ${
                      m.highlight
                        ? "bg-accent border-ink"
                        : "bg-ink border-paper"
                    }`}
                    style={{ left: `${centerPct}%` }}
                    title={`${m.name}: ${m.mean.toFixed(4)} [${m.ciLo.toFixed(4)}, ${m.ciHi.toFixed(4)}]`}
                  />
                </div>

                {/* Values numeric */}
                <div className="w-28 flex-shrink-0 font-mono text-right text-ink tabular-nums mt-1 sm:mt-0 sm:ml-3">
                  <span className="font-bold">{m.mean.toFixed(3)}</span>
                  <span className="text-[10px] text-ink-muted block sm:inline sm:ml-1">
                    [{m.ciLo.toFixed(2)}, {m.ciHi.toFixed(2)}]
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Ticks & Base-rate aligned with flex-1 track column */}
        <div className="flex flex-col sm:flex-row text-[10px] font-mono text-ink-muted border-t border-rule pt-2 mt-3">
          <div className="w-48 flex-shrink-0 hidden sm:block" />
          <div className="relative flex-1 h-5">
            {/* Base rate tick label */}
            {baseRate !== undefined && (
              <div
                className="absolute top-0 bottom-0 w-px border-l border-dashed border-ink-muted/80"
                style={{ left: `${toPct(baseRate)}%` }}
              >
                <span className="absolute top-2.5 -translate-x-1/2 whitespace-nowrap text-[9px] font-mono text-ink-muted bg-paper px-0.5">
                  base rate ({baseRate.toFixed(3)})
                </span>
              </div>
            )}
            <span className="absolute left-0 -translate-x-0">0.00</span>
            <span className="absolute left-1/4 -translate-x-1/2">0.25</span>
            <span className="absolute left-1/2 -translate-x-1/2">0.50</span>
            <span className="absolute left-3/4 -translate-x-1/2">0.75</span>
            <span className="absolute right-0 translate-x-0">1.00</span>
          </div>
          <div className="w-28 flex-shrink-0 hidden sm:block text-right">
            <span>(Higher is better)</span>
          </div>
        </div>
      </div>

      {/* How to read this details */}
      <details className="mt-4 pt-3 border-t border-rule/60 text-xs text-ink-muted group cursor-pointer">
        <summary className="font-medium text-ink hover:text-accent flex items-center justify-between">
          <span>How to read this plot</span>
          <span className="text-[10px] text-ink-muted group-open:rotate-180 transition">▼</span>
        </summary>
        <div className="mt-2 text-ink text-xs leading-relaxed space-y-1">
          <p>
            • <strong>Dot:</strong> The model&apos;s average PR-AUC score across test evaluation time steps.
          </p>
          <p>
            • <strong>Whisker bar:</strong> The 95% confidence interval from paired block bootstrapping over time steps. If two bars do not overlap, their difference is statistically significant.
          </p>
          <p>
            • <strong>Dashed line:</strong> Random guessing baseline ({baseRate.toFixed(3)}). Any useful model must comfortably exceed this line.
          </p>
          {howToRead && <p className="pt-1 text-ink-muted">{howToRead}</p>}
        </div>
      </details>
    </div>
  );
}
