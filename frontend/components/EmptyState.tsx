import React from "react";
import { Inbox } from "lucide-react";

interface EmptyStateProps {
  title?: string;
  message: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  title = "No data found",
  message,
  action,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`p-8 text-center bg-paper-card border border-rule flex flex-col items-center justify-center ${className}`}
    >
      <Inbox className="w-8 h-8 text-ink-muted/50 mb-2 stroke-[1.5]" />
      <h3 className="font-serif text-base font-semibold text-ink">{title}</h3>
      <p className="text-xs text-ink-muted mt-1 max-w-sm">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

