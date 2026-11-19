import "./globals.css";
import Link from "next/link";
import React from "react";
import { Disclaimer } from "@/components/Disclaimer";
import { GlossaryButton } from "@/components/Glossary";

export const metadata = {
  title: "GraphGuard — Financial Crime Graph Intelligence",
  description:
    "Offline research instrument evaluating Graph Neural Networks vs tabular baselines on historical Elliptic Bitcoin data.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-paper text-ink flex flex-col font-sans selection:bg-accent/15 selection:text-ink">
        {/* Navigation Bar */}
        <header className="border-b border-rule bg-paper sticky top-0 z-40">
          <div className="max-w-[1200px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Link href="/" className="flex items-baseline space-x-2">
                <span className="font-serif text-xl font-bold tracking-tight text-ink">
                  GraphGuard
                </span>
                <span className="text-[11px] font-mono uppercase tracking-wider px-1.5 py-0.5 border border-rule text-ink-muted">
                  Research
                </span>
              </Link>
            </div>

            <div className="flex items-center space-x-4">
              <nav className="flex space-x-1 sm:space-x-3 text-xs font-medium">
                <Link
                  href="/"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Overview
                </Link>
                <Link
                  href="/investigate"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Investigate
                </Link>
                <Link
                  href="/explorer"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Graph Explorer
                </Link>
                <Link
                  href="/benchmark"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Benchmark
                </Link>
                <Link
                  href="/temporal"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Temporal
                </Link>
                <Link
                  href="/explainability"
                  className="px-2.5 py-1.5 text-ink hover:text-accent transition"
                >
                  Explainability
                </Link>
              </nav>

              <div className="border-l border-rule pl-3 hidden sm:block">
                <GlossaryButton />
              </div>
            </div>
          </div>
        </header>

        {/* Quiet Research Disclaimer Line */}
        <Disclaimer />

        {/* Main Content Area */}
        <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 sm:px-6 py-8">
          {children}
        </main>

        {/* Minimalist Footer */}
        <footer className="border-t border-rule bg-paper-muted py-6 text-xs text-ink-muted">
          <div className="max-w-[1200px] mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-center sm:text-left">
            <div>
              <p className="font-medium text-ink">
                GraphGuard • Research Instrument
              </p>
              <p className="text-[11px] mt-0.5 text-ink-muted">
                Evaluated on the Elliptic Bitcoin dataset (Weber et al., 2019).
                Features are strictly anonymised.
              </p>
            </div>
            <div className="flex items-center space-x-4 text-[11px]">
              <GlossaryButton />
              <span className="text-rule">•</span>
              <span className="font-mono">Offline Environment</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
