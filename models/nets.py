"""The two baselines.

GraphMLP    reads only the graph-level feature block.  With `precheck_x`
            included this is the rules baseline the GNN has to beat: those ten
            numbers decide every fatal precheck class on their own.
HeteroGNN   message passing over the four node types, pooled and concatenated
            with the same graph-level block.  Run without `precheck_x` it
            answers the actual question: does the design *structure* carry
            routability information beyond the counting features?

Both share one multi-task head set, so the comparison is like for like.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_max_pool, global_mean_pool

from cadna.graph import NODE_TYPES, RELATIONS

from .data import N_CLASSES


class Heads(nn.Module):
    """One shared trunk output -> the four prediction targets."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.routable = nn.Linear(hidden, 1)
        self.hamilton = nn.Linear(hidden, 1)
        self.failure = nn.Linear(hidden, N_CLASSES)
        self.cost = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "routable": self.routable(h).squeeze(-1),
            "hamilton": self.hamilton(h).squeeze(-1),
            "failure": self.failure(h),
            "cost": self.cost(h).squeeze(-1),
        }


class GraphMLP(nn.Module):
    """Graph-level features only: no message passing, no node information."""

    def __init__(self, graph_dim: int, hidden: int = 128):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(graph_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = Heads(hidden, hidden)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(batch.graph_x))


class HeteroGNN(nn.Module):
    def __init__(self, in_dims: dict[str, int], graph_dim: int,
                 hidden: int = 64, layers: int = 3):
        super().__init__()
        self.encode = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden) for t in NODE_TYPES})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv(
                {rel: SAGEConv((hidden, hidden), hidden) for rel in RELATIONS},
                aggr="sum",
            ))
            self.norms.append(nn.ModuleDict({t: nn.LayerNorm(hidden) for t in NODE_TYPES}))
        # mean + max pooling over four node types, then the graph-level block
        self.heads = Heads(2 * hidden * len(NODE_TYPES) + graph_dim, 128)

    def forward(self, batch) -> dict[str, torch.Tensor]:
        n_graphs = batch.graph_x.size(0)
        x = {t: F.relu(self.encode[t](batch[t].x)) for t in NODE_TYPES}
        edge_index = {rel: batch[rel].edge_index for rel in RELATIONS}

        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, edge_index)
            # a node type that received no message this layer keeps its state
            x = {
                t: F.relu(norm[t](x[t] + out[t])) if t in out else x[t]
                for t in NODE_TYPES
            }

        pooled = []
        for t in NODE_TYPES:
            b = batch[t].batch
            pooled.append(global_mean_pool(x[t], b, size=n_graphs))
            pooled.append(global_max_pool(x[t], b, size=n_graphs))
        return self.heads(torch.cat(pooled + [batch.graph_x], dim=-1))
