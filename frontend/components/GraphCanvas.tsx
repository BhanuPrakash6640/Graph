"use client";

import React, { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

export interface GraphNode {
  tx_id: string;
  time_step: number;
  label: "illicit" | "licit" | "unknown";
  direction?: "in" | "out" | "undirected" | "center";
  distance?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
}

interface GraphCanvasProps {
  centerTxId: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (node: GraphNode) => void;
  className?: string;
  height?: string;
}

export function GraphCanvas({
  centerTxId,
  nodes,
  edges,
  onNodeClick,
  className = "",
  height = "380px",
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const elements: cytoscape.ElementDefinition[] = [];

    // Ensure center node is present
    const hasCenter = nodes.some((n) => n.tx_id === centerTxId);
    if (!hasCenter) {
      elements.push({
        data: {
          id: centerTxId,
          label: `${centerTxId.slice(0, 7)}...`,
          fullId: centerTxId,
          nodeLabel: "unknown",
          isCenter: true,
          distance: 0,
          color: "#0F766E",
          borderColor: "#111827",
          borderWidth: 3,
        },
      });
    }

    nodes.forEach((n) => {
      const isCenter = n.tx_id === centerTxId;
      let color = "#6B7280"; // unknown
      if (n.label === "illicit") color = "#B42318";
      if (n.label === "licit") color = "#1D4ED8";
      if (isCenter) color = n.label === "illicit" ? "#B42318" : n.label === "licit" ? "#1D4ED8" : "#0F766E";

      elements.push({
        data: {
          id: n.tx_id,
          label: isCenter ? `★ ${n.tx_id.slice(0, 7)}...` : `${n.tx_id.slice(0, 6)}...`,
          fullId: n.tx_id,
          nodeLabel: n.label,
          isCenter: isCenter,
          distance: isCenter ? 0 : n.distance || 1,
          color: color,
          borderColor: isCenter ? "#111827" : "#FFFFFF",
          borderWidth: isCenter ? 3 : 1.5,
          rawNode: n,
        },
      });
    });

    edges.forEach((e, idx) => {
      elements.push({
        data: {
          id: `e_${e.source}_${e.target}_${idx}`,
          source: e.source,
          target: e.target,
        },
      });
    });

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
            "text-margin-y": 4,
            color: "#111827",
            "background-color": "data(color)",
            "border-width": "data(borderWidth)",
            "border-color": "data(borderColor)",
            width: "mapData(distance, 0, 2, 28, 18)",
            height: "mapData(distance, 0, 2, 28, 18)",
          },
        },
        {
          selector: "node[isCenter]",
          style: {
            width: 32,
            height: 32,
            "font-weight": "bold",
            "border-color": "#111827",
            "border-width": 3,
          },
        },
        {
          selector: "node[nodeLabel = 'unknown']",
          style: {
            "border-style": "dashed",
            "border-color": "#9CA3AF",
            "background-color": "#E5E7EB",
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
      layout: {
        name: "concentric",
        concentric: (node: any) => {
          return 10 - (node.data("distance") || 0) * 3;
        },
        levelWidth: () => 1,
        minNodeSpacing: 40,
        avoidOverlap: true,
        animate: false,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
    });

    cy.on("tap", "node", (evt) => {
      const nodeData = evt.target.data("rawNode");
      if (nodeData && onNodeClick) {
        onNodeClick(nodeData);
      }
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [centerTxId, nodes, edges, onNodeClick]);

  return (
    <div className={`relative border border-rule bg-paper-card ${className}`}>
      <div
        ref={containerRef}
        style={{ width: "100%", height: height }}
        className="cursor-grab active:cursor-grabbing"
      />

      {/* Embedded Clean Legend */}
      <div className="absolute bottom-2 left-2 bg-paper/95 border border-rule px-2.5 py-1.5 text-[10px] font-mono text-ink flex items-center gap-3 shadow-sm z-10 pointer-events-none">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-illicit inline-block" />
          <span>Illicit</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-licit inline-block" />
          <span>Licit</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-paper-muted border border-dashed border-ink-muted inline-block" />
          <span>Unknown</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-full border-2 border-ink inline-block" />
          <span>Focus Node</span>
        </span>
      </div>
    </div>
  );
}

