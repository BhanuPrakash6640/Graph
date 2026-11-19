import React from "react";

interface StatProps {
  label: string;
  value: string | number | React.ReactNode;
  scale?: string;
  footnote?: string;
  source?: string;
  className?: string;
}

export function Stat({
  label,
  value,
  scale,
  footnote,
  source,
  className = "",
}: StatProps) {
  return (
    <div className={`p-4 bg-paper-card border border-rule ${className}`}>
      <div className="text-xs uppercase tracking-wider font-mono text-ink-muted">
        {label}
      </div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="font-mono text-2xl font-bold text-ink tabular-nums">
          {value}
        </span>
        {scale && (
          <span className="text-xs font-mono text-ink-muted">
            {scale}
          </span>
        )}
      </div>
      {(footnote || source) && (
        <div className="mt-2 text-xs text-ink-muted flex items-center justify-between border-t border-rule/60 pt-1.5">
          {footnote && <span>{footnote}</span>}
          {source && (
            <span className="font-mono text-[10px] text-ink-muted/80 bg-paper-muted px-1.5 py-0.5 border border-rule/50">
              {source}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

