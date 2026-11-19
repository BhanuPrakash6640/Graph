import React from "react";
import { AlertCircle } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  error?: string | Error;
  retry?: () => void;
  className?: string;
}

export function ErrorState({
  title = "Failed to load data",
  error,
  retry,
  className = "",
}: ErrorStateProps) {
  const errorMsg =
    typeof error === "string"
      ? error
      : error?.message || "An unexpected error occurred while communicating with the backend.";

  return (
    <div
      className={`p-6 bg-paper-card border border-illicit/30 text-ink flex flex-col items-center text-center ${className}`}
    >
      <AlertCircle className="w-6 h-6 text-illicit mb-2 stroke-[1.5]" />
      <h3 className="font-serif text-base font-semibold text-ink">{title}</h3>
      <p className="font-mono text-xs text-illicit mt-1 max-w-md">{errorMsg}</p>
      {retry && (
        <button
          onClick={retry}
          className="mt-4 px-3 py-1.5 text-xs font-mono font-medium text-ink bg-paper border border-rule hover:bg-paper-muted transition"
        >
          Try again
        </button>
      )}
    </div>
  );
}

