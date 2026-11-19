import React from "react";

interface SectionProps {
  title: string;
  technical?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export function Section({
  title,
  technical,
  action,
  children,
  className = "",
}: SectionProps) {
  return (
    <section className={`py-6 border-b border-rule last:border-b-0 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-2 mb-4">
        <div>
          <h2 className="font-serif text-xl font-semibold text-ink">
            {title}
          </h2>
          {technical && (
            <p className="font-mono text-xs text-ink-muted mt-0.5">
              Technical: {technical}
            </p>
          )}
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>
      <div>{children}</div>
    </section>
  );
}

