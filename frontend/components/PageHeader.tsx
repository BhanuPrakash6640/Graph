import React from "react";

interface PageHeaderProps {
  title: string;
  inShort?: string;
  technical?: string;
  action?: React.ReactNode;
}

export function PageHeader({ title, inShort, technical, action }: PageHeaderProps) {
  return (
    <header className="mb-8 pb-6 border-b border-rule">
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-4">
        <div>
          <h1 className="font-serif text-3xl font-semibold text-ink tracking-tight">
            {title}
          </h1>
          {technical && (
            <p className="font-mono text-xs text-ink-muted mt-1">
              Technical: {technical}
            </p>
          )}
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>

      {inShort && (
        <div className="mt-4 p-3.5 bg-paper-muted border border-rule text-sm text-ink leading-relaxed">
          <span className="font-semibold text-ink">In short: </span>
          {inShort}
        </div>
      )}
    </header>
  );
}

