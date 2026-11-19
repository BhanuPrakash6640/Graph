"""Graph Store for fast k-hop topological lookup and transaction metadata."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ml.data.loader import FEATURE_COLS, UNKNOWN

logger = logging.getLogger("backend.graph_store")


class GraphStore:
    def __init__(self, data_dir: str | Path = "data/processed") -> None:
        self.data_dir = Path(data_dir)
        self.tx_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self.tx_str_to_idx: dict[str, int] = {}
        self.time_steps: np.ndarray = np.empty(0, dtype=np.int16)
        self.labels: np.ndarray = np.empty(0, dtype=np.int8)
        self.features_df: pd.DataFrame | None = None
        self.edge_index: np.ndarray = np.empty((2, 0), dtype=np.int64)

        # Adjacency lists per time step for fast intra-snapshot traversal
        self.step_adj_out: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.step_adj_in: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._loaded: bool = False

    def load(self) -> None:
        nodes_path = self.data_dir / "nodes.parquet"
        edges_path = self.data_dir / "edge_index.npy"

        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f"Missing data files in {self.data_dir}")

        logger.info("Loading transaction nodes from %s...", nodes_path)
        nodes_df = pd.read_parquet(nodes_path)
        self.tx_ids = nodes_df["txId"].to_numpy(np.int64)
        self.time_steps = nodes_df["time_step"].to_numpy(np.int16)
        self.labels = nodes_df["label"].to_numpy(np.int8)
        self.features_df = nodes_df[FEATURE_COLS]

        # Fast string lookup
        for idx, tx_val in enumerate(self.tx_ids):
            self.tx_str_to_idx[str(tx_val)] = idx

        logger.info("Loading graph edges from %s...", edges_path)
        self.edge_index = np.load(edges_path)
        src = self.edge_index[0]
        dst = self.edge_index[1]

        # Build per-step adjacency
        for s, d in zip(src, dst):
            step = int(self.time_steps[s])
            self.step_adj_out[step][int(s)].append(int(d))
            self.step_adj_in[step][int(d)].append(int(s))

        self._loaded = True
        logger.info("GraphStore loaded: %d nodes, %d edges.", len(self.tx_ids), self.edge_index.shape[1])

    @property
    def num_nodes(self) -> int:
        return len(self.tx_ids)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def has_tx(self, tx_id: str) -> bool:
        return tx_id in self.tx_str_to_idx

    def get_node_idx(self, tx_id: str) -> int:
        if tx_id not in self.tx_str_to_idx:
            raise KeyError(f"Transaction ID {tx_id} not found in dataset")
        return self.tx_str_to_idx[tx_id]

    def get_tx_id_str(self, node_idx: int) -> str:
        return str(self.tx_ids[node_idx])

    def get_node_info(self, tx_id: str) -> dict[str, Any]:
        idx = self.get_node_idx(tx_id)
        step = int(self.time_steps[idx])
        raw_label = int(self.labels[idx])

        if raw_label == 1:
            label_str = "illicit"
        elif raw_label == 0:
            label_str = "licit"
        else:
            label_str = "unknown"

        out_nbrs = self.step_adj_out[step].get(idx, [])
        in_nbrs = self.step_adj_in[step].get(idx, [])
        out_degree = len(out_nbrs)
        in_degree = len(in_nbrs)

        # Summary of top 5 local features
        row = self.features_df.iloc[idx]
        feat_summary = {
            col: round(float(row[col]), 4)
            for col in ["local_1", "local_2", "local_3", "local_53", "local_90"]
            if col in row
        }

        return {
            "tx_id": tx_id,
            "node_idx": idx,
            "time_step": step,
            "label_type": "historical_label",
            "label": label_str,
            "degree": in_degree + out_degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "features_summary": feat_summary,
            "disclaimer": "Research demo on historical Elliptic data (2019).",
        }

    def khop(self, tx_id: str, k: int = 1) -> dict[str, Any]:
        """Perform intra-snapshot k-hop neighborhood search around tx_id.

        Restricted strictly to the node's own time step (disjoint snapshots).
        """
        center_idx = self.get_node_idx(tx_id)
        step = int(self.time_steps[center_idx])

        adj_out = self.step_adj_out[step]
        adj_in = self.step_adj_in[step]

        # BFS up to distance k
        visited: dict[int, int] = {center_idx: 0}  # node_idx -> distance
        directions: dict[int, str] = {center_idx: "center"}
        queue: deque[tuple[int, int]] = deque([(center_idx, 0)])

        while queue:
            curr, dist = queue.popleft()
            if dist >= k:
                continue

            # Outgoing neighbors
            for nxt in adj_out.get(curr, []):
                if nxt not in visited:
                    visited[nxt] = dist + 1
                    directions[nxt] = "out"
                    queue.append((nxt, dist + 1))

            # Incoming neighbors
            for prv in adj_in.get(curr, []):
                if prv not in visited:
                    visited[prv] = dist + 1
                    directions[prv] = "in"
                    queue.append((prv, dist + 1))

        # Build neighbor list (excluding center)
        neighbors_list = []
        for n_idx, dist in visited.items():
            if n_idx == center_idx:
                continue
            y_val = int(self.labels[n_idx])
            lbl = "illicit" if y_val == 1 else ("licit" if y_val == 0 else "unknown")
            neighbors_list.append({
                "tx_id": str(self.tx_ids[n_idx]),
                "time_step": step,
                "label_type": "historical_label",
                "label": lbl,
                "direction": directions.get(n_idx, "undirected"),
                "distance": dist,
            })

        # Find all edges between visited nodes
        visited_set = set(visited.keys())
        sub_edges = []
        for u in visited_set:
            for v in adj_out.get(u, []):
                if v in visited_set:
                    sub_edges.append({
                        "source": str(self.tx_ids[u]),
                        "target": str(self.tx_ids[v]),
                    })

        center_label_raw = int(self.labels[center_idx])
        center_label_str = "illicit" if center_label_raw == 1 else "licit" if center_label_raw == 0 else "unknown"

        return {
            "tx_id": tx_id,
            "time_step": step,
            "label": center_label_str,
            "k": k,
            "total_neighbors": len(neighbors_list),
            "neighbors": sorted(neighbors_list, key=lambda n: (n["distance"], n["tx_id"])),
            "edges": sub_edges,
            "disclaimer": "Research demo on historical Elliptic data (2019).",
        }

