"use client";

import React, { useState } from "react";
import { ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";

export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => React.ReactNode;
  sortable?: boolean;
  align?: "left" | "center" | "right";
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyField: keyof T | ((row: T) => string);
  caption?: string;
  className?: string;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data,
  keyField,
  caption,
  className = "",
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sortedData = React.useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => {
      let va = a[sortKey];
      let vb = b[sortKey];

      // Clean string numbers (e.g. "0.8001 ± 0.0008")
      if (typeof va === "string" && !isNaN(parseFloat(va))) {
        va = parseFloat(va);
      }
      if (typeof vb === "string" && !isNaN(parseFloat(vb))) {
        vb = parseFloat(vb);
      }

      if (va < vb) return sortDir === "asc" ? -1 : 1;
      if (va > vb) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
  }, [data, sortKey, sortDir]);

  return (
    <div className={`overflow-x-auto ${className}`}>
      {caption && (
        <div className="text-xs font-mono text-ink-muted mb-2">{caption}</div>
      )}
      <table className="w-full text-left text-xs border-collapse">
        <thead>
          <tr className="border-b-2 border-rule">
            {columns.map((col) => {
              const alignClass =
                col.align === "right"
                  ? "text-right"
                  : col.align === "center"
                  ? "text-center"
                  : "text-left";

              return (
                <th
                  key={col.key}
                  className={`py-2 px-3 font-mono font-semibold text-ink uppercase tracking-wider text-[11px] whitespace-nowrap ${alignClass}`}
                >
                  {col.sortable !== false ? (
                    <button
                      onClick={() => handleSort(col.key)}
                      className="inline-flex items-center gap-1 hover:text-accent font-semibold focus:outline-none"
                    >
                      <span>{col.header}</span>
                      {sortKey === col.key ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="w-3 h-3 text-accent" />
                        ) : (
                          <ArrowDown className="w-3 h-3 text-accent" />
                        )
                      ) : (
                        <ArrowUpDown className="w-3 h-3 text-ink-muted/50" />
                      )}
                    </button>
                  ) : (
                    col.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody className="divide-y divide-rule/60">
          {sortedData.map((row, idx) => {
            const rowKey =
              typeof keyField === "function" ? keyField(row) : String(row[keyField] ?? idx);

            return (
              <tr
                key={rowKey}
                className="hover:bg-paper-muted/60 transition-colors font-mono"
              >
                {columns.map((col) => {
                  const alignClass =
                    col.align === "right"
                      ? "text-right"
                      : col.align === "center"
                      ? "text-center"
                      : "text-left";

                  return (
                    <td
                      key={col.key}
                      className={`py-2.5 px-3 text-ink tabular-nums whitespace-nowrap ${alignClass}`}
                    >
                      {col.render ? col.render(row) : String(row[col.key] ?? "—")}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

