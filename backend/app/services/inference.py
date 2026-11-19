"""Inference engine for real-time model scoring and registry management."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import k_hop_subgraph

from backend.app.services.database import Database
from backend.app.services.graph_store import GraphStore
from ml.data.loader import FEATURE_COLS
from ml.gnn.models import GCN
from ml.preprocessing.scaler import StepScaler
from ml.temporal.snapshot_context import (
    CONTEXT_DIM,
    SnapshotContextGCN,
    compute_snapshot_structural_stats,
)

logger = logging.getLogger("backend.inference")

DISCLAIMER = (
    "Research demonstration on historical Elliptic Bitcoin dataset (2019). "
    "Not a live monitoring or compliance system. Features are anonymised."
)


class InferenceEngine:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.models_meta: dict[str, dict[str, Any]] = {}
        self.loaded_models: dict[str, Any] = {}
        self.scalers: dict[str, StepScaler] = {}
        self._initialized: bool = False

    def load(
        self,
        config_path: str | Path = "backend/models.yaml",
        scalers_dir: str | Path = "data/processed/scalers",
    ) -> None:
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Missing model registry file: {cfg_file}")

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)

        s_dir = Path(scalers_dir)

        for m_info in cfg.get("models", []):
            m_id = m_info["id"]
            m_type = m_info["type"]
            ckpt_path = Path(m_info["checkpoint"])
            split = m_info["split"]

            logger.info("Registering model: %s (type: %s, split: %s)...", m_id, m_type, split)
            self.models_meta[m_id] = m_info

            # Load split scaler if not already cached
            if split not in self.scalers:
                scaler_file = s_dir / f"{split}.npz"
                if scaler_file.exists():
                    self.scalers[split] = StepScaler.load(scaler_file)
                    logger.info("Loaded scaler for split: %s", split)
                else:
                    logger.warning("Scaler file %s not found. Proceeding unscaled if needed.", scaler_file)

            # Load model checkpoint
            if not ckpt_path.exists():
                logger.warning("Checkpoint %s does not exist. Skipping preload.", ckpt_path)
                continue

            if m_type == "tabular":
                booster = joblib.load(ckpt_path)
                self.loaded_models[m_id] = booster
                logger.info("Loaded tabular model: %s", m_id)
            elif m_type == "gnn":
                raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                gcn_kwargs = raw_ckpt.get("model_kwargs", {"in_channels": 165, "hidden_channels": 128, "out_channels": 2})
                m_class = raw_ckpt.get("model_class", "GCN")
                if m_class == "SnapshotContextGCN":
                    model = SnapshotContextGCN(
                        in_channels=gcn_kwargs.get("in_channels", 165),
                        hidden_channels=gcn_kwargs.get("hidden_channels", 64),
                        out_channels=gcn_kwargs.get("out_channels", 2),
                        dropout=0.0,
                    )
                else:
                    model = GCN(
                        in_channels=gcn_kwargs.get("in_channels", 165),
                        hidden_channels=gcn_kwargs.get("hidden_channels", 128),
                        out_channels=gcn_kwargs.get("out_channels", 2),
                        dropout=0.0,
                    )
                model.load_state_dict(raw_ckpt.get("state_dict", raw_ckpt))
                model.eval()
                self.loaded_models[m_id] = model
                logger.info("Loaded GNN model: %s", m_id)

        self._initialized = True

    def list_models(self) -> list[dict[str, Any]]:
        return list(self.models_meta.values())

    def get_model_meta(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.models_meta:
            raise KeyError(f"Unknown model_id: {model_id}. Available: {list(self.models_meta.keys())}")
        return self.models_meta[model_id]

    def predict(
        self,
        tx_id: str,
        model_id: str,
        graph_store: GraphStore,
    ) -> dict[str, Any]:
        meta = self.get_model_meta(model_id)
        if model_id not in self.loaded_models:
            raise RuntimeError(f"Model {model_id} is registered but not loaded in memory")

        node_idx = graph_store.get_node_idx(tx_id)
        split = meta["split"]
        threshold = float(meta["threshold"])
        scaler = self.scalers.get(split)

        t_start = time.perf_counter()

        # 1. Tabular Prediction
        if meta["type"] == "tabular":
            raw_x = graph_store.features_df.iloc[node_idx].to_numpy(dtype=np.float32).reshape(1, -1)
            if scaler is not None:
                scaled_x = scaler.transform(raw_x)
            else:
                scaled_x = raw_x

            model = self.loaded_models[model_id]
            score = float(model.predict_proba(scaled_x)[0, 1])

        # 2. GNN Prediction
        elif meta["type"] == "gnn":
            step = int(graph_store.time_steps[node_idx])
            # Intra-step nodes
            in_step_mask = (graph_store.time_steps == step)
            step_indices = np.where(in_step_mask)[0]
            step_local_map = {int(g): loc for loc, g in enumerate(step_indices)}

            # Edges within step
            step_edge_mask = in_step_mask[graph_store.edge_index[0]] & in_step_mask[graph_store.edge_index[1]]
            sub_edges = graph_store.edge_index[:, step_edge_mask]
            sub_src = np.array([step_local_map[int(s)] for s in sub_edges[0]])
            sub_dst = np.array([step_local_map[int(d)] for d in sub_edges[1]])
            step_edge_tensor = torch.tensor(np.stack([sub_src, sub_dst], axis=0), dtype=torch.long)

            raw_step_x = graph_store.features_df.iloc[step_indices].to_numpy(dtype=np.float32)
            if scaler is not None:
                scaled_step_x = scaler.transform(raw_step_x)
            else:
                scaled_step_x = raw_step_x

            step_x_tensor = torch.from_numpy(scaled_step_x)
            target_local = step_local_map[node_idx]

            # 2-hop local subgraph
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=target_local,
                num_hops=2,
                edge_index=step_edge_tensor,
                relabel_nodes=True,
            )

            model = self.loaded_models[model_id]
            with torch.no_grad():
                if isinstance(model, SnapshotContextGCN):
                    num_n = len(step_indices)
                    num_e = step_edge_tensor.size(1)
                    s_stats = compute_snapshot_structural_stats(num_n, num_e)
                    p_stats = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
                    c_vec = torch.from_numpy(np.concatenate([s_stats, p_stats], axis=0))
                    logits = model(step_x_tensor[subset], sub_edge_index, c_vec)
                else:
                    logits = model(step_x_tensor[subset], sub_edge_index)
                score = float(F.softmax(logits, dim=-1)[int(mapping), 1].item())
        else:
            raise NotImplementedError(f"Model type {meta['type']} inference not supported")

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        pred_class = int(score >= threshold)

        # Log to SQLite
        self.db.log_inference(
            tx_id=tx_id,
            model_id=model_id,
            score=round(score, 4),
            latency_ms=round(latency_ms, 2),
        )

        return {
            "tx_id": tx_id,
            "model_id": model_id,
            "score": round(score, 4),
            "predicted_class": pred_class,
            "threshold": threshold,
            "latency_ms": round(latency_ms, 2),
            "prediction_type": "model_prediction",
            "disclaimer": DISCLAIMER,
        }
