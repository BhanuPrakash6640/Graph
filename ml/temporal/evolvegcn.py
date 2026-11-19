"""EvolveGCN-O (Output Version) Implementation for Temporal Dynamic Graphs.

Reference:
    Pareja et al. (2020) "EvolveGCN: Evolving Graph Convolutional Networks
    for Dynamic Graphs", AAAI 2020.

EvolveGCN-O evolves the GCN weight matrices over time using an RNN (GRU or LSTM)
without requiring node identity persistence across snapshots.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class EvolveGCNConv(MessagePassing):
    """GCN message-passing operator accepting dynamic external weight matrix."""

    def __init__(self, add_self_loops: bool = True) -> None:
        super().__init__(aggr="add")
        self.add_self_loops = add_self_loops

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GCN message passing with external weight W.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph edge indices of shape (2, E).
            weight: Dynamic weight matrix of shape (in_channels, out_channels).
            bias: Optional bias of shape (out_channels,).
        """
        # Linear projection
        x_trans = torch.matmul(x, weight)
        num_nodes = x.size(0)

        # Compute symmetric normalized adjacency with self-loops
        edge_index_norm, edge_weight = gcn_norm(
            edge_index,
            num_nodes=num_nodes,
            add_self_loops=self.add_self_loops,
            dtype=x.dtype,
        )

        out = self.propagate(edge_index_norm, x=x_trans, edge_weight=edge_weight)
        if bias is not None:
            out = out + bias
        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


class EvolveGCNO(nn.Module):
    """2-layer EvolveGCN-O model.

    Evolves layer weight matrices across snapshots:
        W_t^{(l)} = GRU(W_{t-1}^{(l)}, W_{t-1}^{(l)})

    Supports:
        - Autonomous weight evolution across snapshot sequences.
        - Truncated BPTT (via detaching weights at window boundaries).
        - Frozen mode (evolve=False) for static ablation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.3,
        evolve: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.evolve = evolve

        # Initial weight matrices as learnable parameters
        self.initial_w1 = nn.Parameter(torch.empty(in_channels, hidden_channels))
        self.initial_w2 = nn.Parameter(torch.empty(hidden_channels, out_channels))

        # Biases
        self.bias1 = nn.Parameter(torch.zeros(hidden_channels))
        self.bias2 = nn.Parameter(torch.zeros(out_channels))

        # Recurrent cells for weight evolution:
        # Layer 1: input_size=hidden, hidden_size=hidden (batch_size=in_channels)
        # Layer 2: input_size=out, hidden_size=out (batch_size=hidden_channels)
        self.recurrent_cell1 = nn.GRUCell(hidden_channels, hidden_channels)
        self.recurrent_cell2 = nn.GRUCell(out_channels, out_channels)

        # Graph convolution layers
        self.conv1 = EvolveGCNConv(add_self_loops=True)
        self.conv2 = EvolveGCNConv(add_self_loops=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_w1)
        nn.init.xavier_uniform_(self.initial_w2)
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        self.recurrent_cell1.reset_parameters()
        self.recurrent_cell2.reset_parameters()

    def get_initial_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial layer weights (W_0^(1), W_0^(2))."""
        return self.initial_w1, self.initial_w2

    def evolve_weights(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evolve weights from step t-1 to step t.

        If self.evolve is False, returns w1, w2 unchanged (static ablation).
        """
        if not self.evolve:
            return w1, w2

        next_w1 = self.recurrent_cell1(w1, w1)
        next_w2 = self.recurrent_cell2(w2, w2)
        return next_w1, next_w2

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        """Run 2-layer GCN message passing on a single snapshot using weights w1, w2.

        Args:
            x: Node features (N_t, in_channels).
            edge_index: Symmetrized edges within snapshot t (2, E_t).
            w1: Weight matrix for layer 1 (in_channels, hidden_channels).
            w2: Weight matrix for layer 2 (hidden_channels, out_channels).

        Returns:
            Logits of shape (N_t, out_channels).
        """
        h = self.conv1(x, edge_index, w1, bias=self.bias1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, edge_index, w2, bias=self.bias2)
        return logits

