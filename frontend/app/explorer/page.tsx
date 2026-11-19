"use client";

import React, { useEffect, useRef, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { api, NeighborsData, NeighborNode } from "@/lib/api";
import cytoscape from "cytoscape";
import { Network, ZoomIn, ZoomOut, RefreshCw, Info, ExternalLink } from "lucide-react";

function ExplorerContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const defaultTx = searchParams?.get("txId") || "69746416";

  const [txId, setTxId] = useState<string>(defaultTx);
  const [hopK, setHopK] = useState<number>(1);
  const [graphData, setGraphData] = useState<NeighborsData | null>(null);
  const [selectedNode, setSelectedNode] = useState<NeighborNode | null>(null);
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
    } catch (err: any) {
      setError(err.message || "Failed to load graph neighbors.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph(txId, hopK);
  }, [txId, hopK]);

  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    // Build Cytoscape elements
    const elements: cytoscape.ElementDefinition[] = [];

    // Center Node
    elements.push({
      data: {
        id: graphData.tx_id,
        label: `${graphData.tx_id.slice(0, 6)}... (Center)`,
        full_tx: graphData.tx_id,
        label_class: "center",
        step: graphData.time_step,
        color: "#f39c12",
        border_style: "solid",
      },
    });

    // Neighbor Nodes
    for (const nbr of graphData.neighbors) {
      let color = "#95a5a6";
      let borderStyle = "dashed";
      if (nbr.label === "illicit") {
        color = "#c0392b";
        borderStyle = "solid";
      } else if (nbr.label === "licit") {
        color = "#2e86c1";
        borderStyle = "solid";
      }

      elements.push({
        data: {
          id: nbr.tx_id,
          label: `${nbr.tx_id.slice(0, 6)}...`,
          full_tx: nbr.tx_id,
          label_class: nbr.label,
          step: nbr.time_step,
          color: color,
          border_style: borderStyle,
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
            "background-color": "data(color)",
            label: "data(label)",
            color: "#e2e8f0",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-margin-y": 4,
            width: 28,
            height: 28,
            "border-width": 2,
            "border-color": "#475569",
            "border-style": "data(border_style)" as any,
          },
        },
        {
          selector: 'node[label_class = "center"]',
          style: {
            width: 36,
            height: 36,
            "border-width": 3,
            "border-color": "#f1c40f",
            "font-weight": "bold",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#475569",
            "target-arrow-color": "#64748b",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
      ],
      layout: {
        name: "cose",
        animate: true,
        animationDuration: 500,
        nodeRepulsion: () => 4500,
        idealEdgeLength: () => 60,
      },
    });

    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      const fullId = n.data("full_tx");
      const matched = graphData.neighbors.find((item) => item.tx_id === fullId);
      if (matched) {
        setSelectedNode(matched);
      } else if (fullId === graphData.tx_id) {
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
  }, [graphData]);

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Network className="w-6 h-6 text-cyan-400" />
            <span>Intra-Snapshot Graph Explorer</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Visualizing localized payment flows strictly isolated within chronological time steps.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <input
            type="text"
            value={txId}
            onChange={(e) => setTxId(e.target.value)}
            placeholder="txId (e.g. 69746416)"
            className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white w-40 font-mono"
          />
          <div className="flex items-center space-x-1 bg-slate-900 border border-slate-700 rounded-lg p-1 text-xs">
            <button
              onClick={() => setHopK(1)}
              className={`px-2.5 py-1 rounded font-medium transition ${
                hopK === 1 ? "bg-cyan-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              1-Hop
            </button>
            <button
              onClick={() => setHopK(2)}
              className={`px-2.5 py-1 rounded font-medium transition ${
                hopK === 2 ? "bg-cyan-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              2-Hop
            </button>
          </div>
          <button
            onClick={() => loadGraph(txId, hopK)}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg"
            title="Refresh graph"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs bg-slate-900/60 border border-slate-800 rounded-xl px-4 py-2.5">
        <span className="text-slate-500 font-semibold uppercase">Legend:</span>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 rounded-full bg-[#c0392b]"></div>
          <span className="text-slate-300">Illicit (Solid Red)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 rounded-full bg-[#2e86c1]"></div>
          <span className="text-slate-300">Licit (Solid Blue)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 rounded-full bg-slate-900 border border-dashed border-[#95a5a6]"></div>
          <span className="text-slate-300">Unlabelled (Dashed Grey)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <div className="w-3 h-3 rounded-full bg-[#f39c12] ring-2 ring-yellow-400/50"></div>
          <span className="text-slate-300">Target Node</span>
        </div>
      </div>

      {/* Graph Area */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 bg-slate-950 border border-slate-800 rounded-2xl h-[560px] relative overflow-hidden">
          {loading && (
            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center space-y-3">
              <div className="w-8 h-8 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-xs text-slate-400">Rendering graph topology...</p>
            </div>
          )}
          {error && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 text-center text-rose-400 text-sm">
              <p>{error}</p>
            </div>
          )}
          <div ref={containerRef} className="w-full h-full" />
        </div>

        {/* Selected Node Details Side Pane */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-slate-300 mb-3">
              <Info className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">Inspected Node</h3>
            </div>

            {selectedNode ? (
              <div className="space-y-3 text-xs">
                <div>
                  <span className="text-slate-500 uppercase font-mono">txId</span>
                  <p className="font-mono text-cyan-300 font-bold break-all mt-0.5">
                    {selectedNode.tx_id}
                  </p>
                </div>
                <div>
                  <span className="text-slate-500 uppercase">Historical Label</span>
                  <p className="mt-1 font-semibold">
                    {selectedNode.label === "illicit" ? (
                      <span className="text-red-400 font-bold uppercase">Illicit</span>
                    ) : selectedNode.label === "licit" ? (
                      <span className="text-blue-400 font-bold uppercase">Licit</span>
                    ) : (
                      <span className="text-slate-400 italic">Unlabelled</span>
                    )}
                  </p>
                </div>
                <div>
                  <span className="text-slate-500 uppercase">Snapshot Step</span>
                  <p className="text-slate-200 font-semibold mt-0.5">Step {selectedNode.time_step}</p>
                </div>
                <div>
                  <span className="text-slate-500 uppercase">Hop Distance</span>
                  <p className="text-slate-200 font-semibold mt-0.5">{selectedNode.distance} Hops</p>
                </div>

                <div className="pt-3 border-t border-slate-800">
                  <button
                    onClick={() => router.push(`/transaction/${selectedNode.tx_id}`)}
                    className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-cyan-400 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1.5 transition"
                  >
                    <span>Full Investigation</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">
                Click any node in the graph canvas to inspect its metadata and flow properties.
              </p>
            )}
          </div>

          <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg text-[11px] text-slate-400 space-y-1">
            <p className="font-semibold text-slate-300">Intra-Step Isolation</p>
            <p>Edges never cross temporal snapshots. Graph convolutions are strictly contained.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function GraphExplorerPage() {
  return (
    <Suspense fallback={<div className="text-slate-400 text-sm text-center py-20">Loading Explorer...</div>}>
      <ExplorerContent />
    </Suspense>
  );
}
