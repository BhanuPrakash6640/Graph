import React from "react";

export function Disclaimer({ className = "" }: { className?: string }) {
  return (
    <div
      className={`border-b border-rule bg-paper-muted px-4 py-2 text-center text-xs text-ink-muted ${className}`}
    >
      <span>
        Research demo on historical Elliptic data (Weber et al. 2019). Not a live
        monitoring or compliance system.
      </span>
    </div>
  );
}

