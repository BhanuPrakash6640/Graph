"use client";

import React, { useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, NeighborsData, NeighborNode, ExampleItem } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Section } from "@/components/Section";
import { ClassBadge } from "@/components/ClassBadge";
import { Skeleton } from "@/components/Skeleton";
import { ErrorState } from "@/components/ErrorState";
import cytoscape from "cytoscape";
import {
  Search,
  ZoomIn,
  ZoomOut,
  RefreshCw,
  ExternalLink,
  ArrowRight,
  Layers,
} from "lucide-react";

function ExplorerContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const txParam = searchParams?.get("txId") || "69746416";

  const [txId, setTxId] = useState<string>(txParam);
  const [searchInput, setSearchInput] = useState<string>(txParam);
  const [hopK, setHopK] = useState<number>(1);
  const [layoutName, setLayoutName] = useState<"concentric" | "breadthfirst">("concentric");
  const [graphData, setGraphData] = useState<NeighborsData | null>(null);
  const [selectedNode, setSelectedNode] = useState<NeighborNode | null>(null);
  const [examples, setExamples] = useState<ExampleItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const loadGraph = async (id: string, k: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getNeighbors(id, k);
      setGraphData(data);
      setSelectedNode(null);
    } catch (err: any) {
      setError(err.message || `Failed to load graph neighbors for payment ${id}.`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    api.getExamples().then((d) => setExamples(d.examples)).catch(() => {});
  }, []);

  useEffect(() => {
    setTxId(txParam);
    setSearchInput(txParam);
  }, [txParam]);

  useEffect(() => {
    loadGraph(txId, hopK);
  }, [txId, hopK]);

  // Build Cytoscape graph
  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    const elements: cytoscape.ElementDefinition[] = [];

    // Focus / Center Node
    const centerLabel = graphData.label || "unknown";
    const centerColor =
      centerLabel === "illicit"
        ? "#B42318"
        : centerLabel === "licit"
        ? "#1D4ED8"
        : "#9CA3AF";

    elements.push({
      data: {
        id: graphData.tx_id,
        label: `★ ${graphData.tx_id}`,
        full_tx: graphData.tx_id,
        label_class: centerLabel,
        step: graphData.time_step,
        color: centerColor,
        isCenter: true,
        distance: 0,
      },
    });

    // Neighbor Nodes
    for (const nbr of graphData.neighbors) {
      let color = "#6B7280";
      if (nbr.label === "illicit") color = "#B42318";
      if (nbr.label === "licit") color = "#1D4ED8";

      elements.push({
        data: {
          id: nbr.tx_id,
          label: nbr.tx_id,
          full_tx: nbr.tx_id,
          label_class: nbr.label,
          step: nbr.time_step,
          color: color,
          isCenter: false,
          distance: nbr.distance || 1,
          rawNode: nbr,
        },
      });
    }

    // Edges
    for (let i = 0; i < graphData.edges.length; i++) {
      const e = graphData.edges[i];
      elements.push({
        data: {
          id: `e_${e.source}_${e.target}_${i}`,
          source: e.source,
          target: e.target,
        },
      });
    }

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-family": "IBM Plex Mono, Menlo, monospace",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 6,
            color: "#111827",
            "background-color": "data(color)",
            width: "mapData(distance, 0, 2, 32, 20)",
            height: "mapData(distance, 0, 2, 32, 20)",
            "border-width": 1.5,
            "border-color": "#FFFFFF",
          },
        },
        {
          selector: 'node[label_class = "unknown"]',
          style: {
            "border-style": "dashed",
            "border-color": "#9CA3AF",
            "background-color": "#E5E7EB",
          },
        },
        {
          selector: "node[isCenter]",
          style: {
            width: 38,
            height: 38,
            "border-width": 3.5,
            "border-color": "#000000",
            "border-style": "solid",
            "background-color": "data(color)",
            "font-weight": "bold",
            "z-index": 999,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#D1D5DB",
            "target-arrow-color": "#9CA3AF",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "arrow-scale": 0.8,
          },
        },
        {
          selector: ":selected",
          style: {
            "border-color": "#0F766E",
            "border-width": 3,
            "line-color": "#0F766E",
            "target-arrow-color": "#0F766E",
          },
        },
      ],
      layout:
        layoutName === "concentric"
          ? {
              name: "concentric",
              concentric: (node: any) => 10 - (node.data("distance") || 0) * 3,
              levelWidth: () => 1,
              minNodeSpacing: 50,
              avoidOverlap: true,
              animate: false,
              padding: 40,
            }
          : {
              name: "breadthfirst",
              directed: true,
              spacingFactor: 1.5,
              roots: [`node[id = "${graphData.tx_id}"]`],
              animate: false,
              padding: 40,
            } as any,
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      const raw = n.data("rawNode");
      if (raw) {
        setSelectedNode(raw);
      } else if (n.data("full_tx") === graphData.tx_id) {
        setSelectedNode({
          tx_id: graphData.tx_id,
          time_step: graphData.time_step,
          label_type: "historical_label",
          label: "illicit",
          direction: "undirected",
          distance: 0,
        });
      }
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graphData, layoutName]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = searchInput.trim();
    if (trimmed) {
      router.push(`/explorer?txId=${trimmed}`);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Intra-Snapshot Graph Explorer"
        technical="Centred concentric visualization of local payment networks within single 2-week observation periods"
        inShort="Explore connected payments up to 2 hops away from any focus transaction. Directed arrows show Bitcoin value flow from input payments to output payments."
      />

      {/* Search & Layout Toolbar */}
      <div className="p-4 bg-paper-card border border-rule flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <form onSubmit={handleSearch} className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative flex-1">
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search focus payment ID..."
              className="w-full px-3 py-1.5 pl-8 bg-paper border border-rule font-mono text-xs text-ink focus:outline-none focus:border-accent"
            />
            <Search className="w-3.5 h-3.5 text-ink-muted absolute left-2.5 top-2.5" />
          </div>
          <button
            type="submit"
            className="px-3 py-1.5 bg-paper border border-rule hover:bg-paper-muted text-xs font-mono font-medium text-ink transition"
          >
            Inspect
          </button>
        </form>

        <div className="flex items-center gap-4 text-xs font-mono">
          {/* Hop toggle */}
          <div className="flex items-center gap-1.5">
            <span className="text-ink-muted">Hops:</span>
            <button
              onClick={() => setHopK(1)}
              className={`px-2 py-0.5 border ${
                hopK === 1
                  ? "bg-accent text-white border-accent font-bold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              1-hop
            </button>
            <button
              onClick={() => setHopK(2)}
              className={`px-2 py-0.5 border ${
                hopK === 2
                  ? "bg-accent text-white border-accent font-bold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              2-hop
            </button>
          </div>

          {/* Layout toggle */}
          <div className="flex items-center gap-1.5">
            <span className="text-ink-muted">Layout:</span>
            <button
              onClick={() => setLayoutName("concentric")}
              className={`px-2 py-0.5 border ${
                layoutName === "concentric"
                  ? "bg-accent text-white border-accent font-bold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              Concentric
            </button>
            <button
              onClick={() => setLayoutName("breadthfirst")}
              className={`px-2 py-0.5 border ${
                layoutName === "breadthfirst"
                  ? "bg-accent text-white border-accent font-bold"
                  : "bg-paper text-ink border-rule hover:bg-paper-muted"
              }`}
            >
              Directed
            </button>
          </div>
        </div>
      </div>

      {/* Example Chips */}
      {examples.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <span className="text-ink-muted">Examples:</span>
          {examples.map((ex) => (
            <button
              key={ex.tx_id}
              onClick={() => router.push(`/explorer?txId=${ex.tx_id}`)}
              className={`px-2 py-0.5 border text-[11px] ${
                ex.tx_id === txId
                  ? "bg-accent text-white border-accent font-semibold"
                  : "bg-paper-card text-ink border-rule hover:border-accent"
              }`}
            >
              {ex.tx_id} ({ex.category})
            </button>
          ))}
        </div>
      )}

      {/* Main Canvas & Inspector Layout */}
      {loading ? (
        <Skeleton className="h-[500px] w-full" />
      ) : error ? (
        <ErrorState
          title={`Could not load graph for ${txId}`}
          error={error}
          retry={() => loadGraph(txId, hopK)}
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
          {/* Canvas View */}
          <div className="lg:col-span-3 border border-rule bg-paper-card relative">
            <div
              ref={containerRef}
              style={{ width: "100%", height: "520px" }}
              className="cursor-grab active:cursor-grabbing"
            />

            {/* Canvas Controls */}
            <div className="absolute top-3 right-3 flex items-center gap-1 z-10 bg-paper/90 border border-rule p-1">
              <button
                onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 1.25)}
                className="p-1 hover:bg-paper-muted text-ink"
                title="Zoom in"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 0.8)}
                className="p-1 hover:bg-paper-muted text-ink"
                title="Zoom out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => cyRef.current?.fit()}
                className="p-1 hover:bg-paper-muted text-ink"
                title="Fit to view"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Always Visible Legend */}
            <div className="absolute bottom-3 left-3 bg-paper/95 border border-rule px-3 py-2 text-xs font-mono text-ink flex flex-wrap items-center gap-4 z-10 shadow-sm pointer-events-none">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-illicit inline-block" />
                <span>Known Illicit</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-licit inline-block" />
                <span>Known Licit</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full bg-paper-muted border border-dashed border-ink-muted inline-block" />
                <span>Unknown</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-full border-2 border-ink inline-block" />
                <span>Focus (Center)</span>
              </span>
              <span className="text-ink-muted text-[11px]">
                Arrow: Payment flow (Input → Output)
              </span>
            </div>
          </div>

          {/* Node Inspector Side Panel */}
          <div className="p-4 bg-paper-card border border-rule space-y-4">
            <div className="border-b border-rule pb-2">
              <h3 className="font-serif text-sm font-semibold text-ink">
                Payment Inspector
              </h3>
              <p className="font-mono text-[11px] text-ink-muted mt-0.5">
                {selectedNode ? "Selected node details" : "Click any node in graph"}
              </p>
            </div>

            {selectedNode ? (
              <div className="space-y-3 font-mono text-xs">
                <div>
                  <span className="text-ink-muted text-[11px] block">Payment ID:</span>
                  <span className="font-bold text-ink text-sm break-all">
                    {selectedNode.tx_id}
                  </span>
                </div>

                <div>
                  <span className="text-ink-muted text-[11px] block mb-1">
                    Historical Record:
                  </span>
                  <ClassBadge type="known" label={selectedNode.label} />
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-rule">
                  <div>
                    <span className="text-ink-muted text-[11px]">Time Step:</span>
                    <p className="font-bold text-ink">Step {selectedNode.time_step}</p>
                  </div>
                  <div>
                    <span className="text-ink-muted text-[11px]">Hop Distance:</span>
                    <p className="font-bold text-ink">{selectedNode.distance ?? 1}</p>
                  </div>
                </div>

                <div className="pt-3 border-t border-rule space-y-2">
                  <Link
                    href={`/transaction/${selectedNode.tx_id}`}
                    className="w-full py-2 bg-accent text-white text-xs font-semibold hover:bg-accent-hover transition flex items-center justify-center gap-1.5"
                  >
                    <span>Open Case File</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>

                  {selectedNode.tx_id !== txId && (
                    <button
                      onClick={() => setTxId(selectedNode.tx_id)}
                      className="w-full py-1.5 bg-paper border border-rule hover:bg-paper-muted text-ink text-xs transition"
                    >
                      Center graph on this node
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-xs text-ink-muted font-mono">
                Click any node in the graph to inspect its properties, flow direction, and case file.
              </div>
            )}

            <div className="pt-3 border-t border-rule text-[11px] text-ink-muted leading-relaxed">
              <p>
                Nodes in this snapshot ({graphData?.neighbors.length || 0} visible) occur strictly within time step {graphData?.time_step || 1}.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ExplorerPage() {
  return (
    <Suspense fallback={<Skeleton className="h-[500px] w-full" />}>
      <ExplorerContent />
    </Suspense>
  );
}
