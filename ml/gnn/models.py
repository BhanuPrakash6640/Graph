"""GNN Model Architectures (GCN, GraphSAGE, GAT) and MLP Control Model.

All models adhere to a standard 2-layer inductive architecture with dropout,
activation functions, and binary classification logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE network with configurable aggregator (mean or max)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """2-layer Graph Attention Network with multi-head attention.

    Supports returning attention coefficients from the output or intermediate layer
    for explainability in Phase 7.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.heads = heads

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if return_attention_weights:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            return x, att2
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x


class MLP(nn.Module):
    """2-layer Multi-Layer Perceptron control model (no graph edges)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def create_gnn_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    **kwargs,
) -> nn.Module:
    """Factory creating GNN models or MLP control."""
    name = model_name.lower()
    if name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    elif name in ("sage", "graphsage", "sage-mean", "sage_mean"):
        return GraphSAGE(in_channels, hidden_channels, out_channels, dropout, aggr="mean")
    elif name in ("sage-max", "sage_max"):
        return GraphSAGE(in_channels, hidden_channels, out_channels, dropout, aggr="max")
    elif name == "gat":
        heads = kwargs.get("heads", 4)
        return GAT(in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout)
    elif name == "mlp":
        return MLP(in_channels, hidden_channels, out_channels, dropout)
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

