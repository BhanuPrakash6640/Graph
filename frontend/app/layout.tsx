import "./globals.css";
import Link from "next/link";
import React from "react";

export const metadata = {
  title: "GraphGuard — Financial Crime Graph Intelligence",
  description: "Offline research demonstration on historical Elliptic dataset.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
        {/* Persistent Compliance & Ethics Banner */}
        <div className="bg-amber-500/10 border-b border-amber-500/30 px-4 py-2 text-center text-xs font-medium text-amber-300">
          ⚠️ <strong>Research Demonstration Only:</strong> Built on historical Elliptic Bitcoin data (2019).
          Not a live transaction monitoring or AML regulatory compliance system. Features are strictly anonymised.
        </div>

        {/* Navigation Bar */}
        <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Link href="/" className="flex items-center space-x-2">
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  GraphGuard
                </span>
                <span className="text-xs uppercase tracking-wider px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 font-semibold">
                  Intelligence Demo
                </span>
              </Link>
            </div>

            <nav className="flex space-x-1 sm:space-x-4 text-sm font-medium">
              <Link href="/" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Overview
              </Link>
              <Link href="/transaction/69746416" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Investigate
              </Link>
              <Link href="/explorer" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Graph Explorer
              </Link>
              <Link href="/benchmark" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Benchmark
              </Link>
              <Link href="/temporal" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Temporal
              </Link>
              <Link href="/explainability" className="px-3 py-2 rounded-md hover:bg-slate-800 transition text-slate-300 hover:text-white">
                Explainability
              </Link>
            </nav>
          </div>
        </header>

        {/* Main Body */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
          <p>GraphGuard AML Demonstration • Historical Elliptic Dataset (Weber et al. 2019)</p>
          <p className="mt-1">All numbers served directly from backend API endpoints with paired block bootstrap validation.</p>
        </footer>
      </body>
    </html>
  );
}

