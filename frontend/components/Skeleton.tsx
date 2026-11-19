import React from "react";

export function Skeleton({
  className = "",
}: {
  className?: string;
}) {
  return (
    <div
      className={`animate-pulse bg-rule/70 rounded-none ${className}`}
      aria-hidden="true"
    />
  );
}

export function StatSkeleton() {
  return (
    <div className="p-4 bg-paper-card border border-rule">
      <Skeleton className="h-3 w-20 mb-2" />
      <Skeleton className="h-8 w-28 mb-2" />
      <Skeleton className="h-3 w-36" />
    </div>
  );
}

