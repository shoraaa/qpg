"""GNN components for neural SeeA* over oriented tangle graphs.

This adapts the useful part of the DyNACO network family to QPG:
edge/node embeddings, repeated edge-aware message passing, and cheap heads for
edge/action scoring plus state-value prediction.  It intentionally does not
depend on torch-geometric because the QPG environment only needs sparse directed
GFA graphs and dynamic prefix features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import networkx as nx

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


NODE_FEATURES = 12
EDGE_FEATURES = 8
GLOBAL_FEATURES = 6


def require_torch():
    if torch is None:
        raise ImportError(
            "QPG neural SeeA* requires torch. Install the repo environment with "
            "`uv sync` or install torch before importing the neural model."
        ) from _TORCH_IMPORT_ERROR


@dataclass
class QPGGraphTensor:
    x: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    global_attr: "torch.Tensor"
    node_names: list[str]
    edge_pairs: list[tuple[int, int]]
    end_index: int
    start_index: int

    def to(self, device):
        return QPGGraphTensor(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            global_attr=self.global_attr.to(device),
            node_names=self.node_names,
            edge_pairs=self.edge_pairs,
            end_index=self.end_index,
            start_index=self.start_index,
        )


def _biological_index(oriented_index: int) -> int:
    return oriented_index // 2


def _orientation_flags(oriented_index: int) -> tuple[float, float]:
    return (1.0, 0.0) if oriented_index % 2 == 0 else (0.0, 1.0)


def _as_counts(counts: Sequence[int] | None, biological_nodes: int) -> list[float]:
    if counts is None:
        return [0.0] * biological_nodes
    if len(counts) != biological_nodes:
        raise ValueError(f"counts length {len(counts)} does not match V={biological_nodes}")
    return [float(value) for value in counts]


def build_qpg_graph_tensor(
    graph: nx.DiGraph,
    counts: Sequence[int] | None = None,
    current_index: int | None = None,
    depth: int = 0,
    horizon: int | None = None,
    device=None,
) -> QPGGraphTensor:
    """Build dynamic GNN features for one search prefix.

    The node order follows the QUBO graph node order and appends one artificial
    `end` node.  Biological node counts are orientation-merged, matching the
    QUBO/tangle objective.
    """
    require_torch()
    node_names = list(graph.nodes)
    biological_nodes = len(node_names) // 2
    end_index = len(node_names)
    start_index = end_index + 1
    weights = [float(graph.nodes[node_names[2 * i]]["weight"]) for i in range(biological_nodes)]
    lengths = [float(graph.nodes[node_names[2 * i]].get("length", 1.0) or 1.0) for i in range(biological_nodes)]
    counts_list = _as_counts(counts, biological_nodes)
    total_weight = max(sum(max(weight, 0.0) for weight in weights), 1.0)
    max_log_len = max(math.log1p(length) for length in lengths) if lengths else 1.0
    horizon_value = max(float(horizon if horizon is not None else max(int(total_weight), 1)), 1.0)

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    max_degree = max(
        [1.0]
        + [float(in_degree.get(node, 0) + out_degree.get(node, 0)) for node in node_names]
    )

    features = []
    for oriented_index, node in enumerate(node_names):
        bio = _biological_index(oriented_index)
        plus_flag, minus_flag = _orientation_flags(oriented_index)
        weight = weights[bio]
        count = counts_list[bio]
        residual = weight - count
        over = max(0.0, count - weight)
        features.append(
            [
                weight / total_weight,
                count / (1.0 + total_weight),
                residual / (1.0 + total_weight),
                max(0.0, residual) / (1.0 + total_weight),
                over / (1.0 + total_weight),
                math.log1p(lengths[bio]) / max_log_len,
                plus_flag,
                minus_flag,
                1.0 if oriented_index == current_index else 0.0,
                1.0 if weight <= 0.0 else 0.0,
                float(out_degree.get(node, 0)) / max_degree,
                float(in_degree.get(node, 0)) / max_degree,
            ]
        )

    features.append(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if current_index == end_index else 0.0,
            1.0,
            0.0,
            1.0,
        ]
    )
    features.append(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 if current_index == start_index else 0.0,
            1.0,
            1.0,
            0.0,
        ]
    )

    node_to_index = {node: index for index, node in enumerate(node_names)}
    edge_pairs: list[tuple[int, int]] = []
    for source, target in graph.edges:
        edge_pairs.append((node_to_index[source], node_to_index[target]))
    for source_index in range(end_index):
        edge_pairs.append((source_index, end_index))
    edge_pairs.append((end_index, end_index))
    for target_index in range(end_index + 1):
        edge_pairs.append((start_index, target_index))

    edge_features = []
    for source, target in edge_pairs:
        source_is_end = source == end_index
        target_is_end = target == end_index
        source_is_start = source == start_index
        if target_is_end:
            dst_residual = 0.0
            dst_weight = 0.0
            dst_count = 0.0
        else:
            dst_bio = _biological_index(target)
            dst_weight = weights[dst_bio] / total_weight
            dst_count = counts_list[dst_bio] / (1.0 + total_weight)
            dst_residual = max(0.0, weights[dst_bio] - counts_list[dst_bio]) / (1.0 + total_weight)
        if source_is_end or source_is_start:
            src_weight = 0.0
        else:
            src_weight = weights[_biological_index(source)] / total_weight
        edge_features.append(
            [
                1.0 if source == current_index else 0.0,
                1.0 if target == current_index else 0.0,
                1.0 if target_is_end else 0.0,
                1.0 if source_is_end or source_is_start else 0.0,
                1.0 if source == target else 0.0,
                src_weight,
                dst_weight,
                dst_residual + dst_count,
            ]
        )

    x = torch.tensor(features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)
    residual_l1 = sum(abs(weights[i] - counts_list[i]) for i in range(biological_nodes))
    over_l1 = sum(max(0.0, counts_list[i] - weights[i]) for i in range(biological_nodes))
    under_l1 = sum(max(0.0, weights[i] - counts_list[i]) for i in range(biological_nodes))
    global_attr = torch.tensor(
        [
            float(depth) / horizon_value,
            total_weight,
            residual_l1 / (1.0 + total_weight),
            over_l1 / (1.0 + total_weight),
            under_l1 / (1.0 + total_weight),
            float(len(edge_pairs)) / max(float(len(features) ** 2), 1.0),
        ],
        dtype=torch.float32,
        device=device,
    )
    return QPGGraphTensor(
        x,
        edge_index,
        edge_attr,
        global_attr,
        node_names + ["end", "start"],
        edge_pairs,
        end_index,
        start_index,
    )


class QPGGNNLayer(nn.Module):
    """DyNACO-style edge-aware message passing implemented with index_add."""

    def __init__(self, units: int):
        super().__init__()
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.e_lin0 = nn.Linear(units, units)
        self.v_norm = nn.LayerNorm(units)
        self.e_norm = nn.LayerNorm(units)

    def forward(self, x, edge_index, edge_state):
        source, target = edge_index
        gate = torch.sigmoid(edge_state)
        messages = gate * self.v_lin2(x[source])
        agg = torch.zeros_like(x)
        agg.index_add_(0, target, messages)
        degree = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones((target.numel(), 1), device=x.device, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0)

        x_next = x + F.silu(self.v_norm(self.v_lin1(x) + agg))
        edge_next = edge_state + F.silu(
            self.e_norm(self.e_lin0(edge_state) + self.v_lin3(x[source]) + self.v_lin4(x[target]))
        )
        return x_next, edge_next


class QPGSeeAGNN(nn.Module):
    """State-value and action-prior model for QPG SeeA*.

    Output:
      - `value`: non-negative cost-to-go estimate for the current prefix.
      - `edge_logits`: one logit per edge in `QPGGraphTensor.edge_pairs`.
      - `embedding`: graph/state embedding usable for future SeeA* clustering.
    """

    def __init__(
        self,
        node_features: int = NODE_FEATURES,
        edge_features: int = EDGE_FEATURES,
        global_features: int = GLOBAL_FEATURES,
        units: int = 64,
        depth: int = 6,
    ):
        super().__init__()
        if torch is None:
            require_torch()
        self.node_in = nn.Linear(node_features, units)
        self.edge_in = nn.Linear(edge_features, units)
        self.layers = nn.ModuleList([QPGGNNLayer(units) for _ in range(depth)])
        self.edge_head = nn.Sequential(
            nn.Linear(units * 3 + global_features, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(units * 3 + global_features, units),
            nn.SiLU(),
            nn.Linear(units, 1),
            nn.Softplus(),
        )

    def forward(self, graph_tensor: QPGGraphTensor):
        x = F.silu(self.node_in(graph_tensor.x))
        edge_state = F.silu(self.edge_in(graph_tensor.edge_attr))
        for layer in self.layers:
            x, edge_state = layer(x, graph_tensor.edge_index, edge_state)

        current_mask = graph_tensor.x[:, 8] > 0.5
        if current_mask.any():
            current_emb = x[current_mask].mean(dim=0)
        else:
            current_emb = x.new_zeros(x.size(1))
        graph_mean = x.mean(dim=0)
        graph_max = x.max(dim=0).values
        state_emb = torch.cat([graph_mean, graph_max, current_emb, graph_tensor.global_attr], dim=0)
        value = self.value_head(state_emb).squeeze(-1)

        source, target = graph_tensor.edge_index
        global_expand = graph_tensor.global_attr.unsqueeze(0).expand(edge_state.size(0), -1)
        edge_inputs = torch.cat([edge_state, x[source], x[target], global_expand], dim=-1)
        edge_logits = self.edge_head(edge_inputs).squeeze(-1)
        return {"value": value, "edge_logits": edge_logits, "embedding": state_emb}
