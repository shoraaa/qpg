#!/usr/bin/env python3
"""Smoke test the QPG neural SeeA* GNN on one GFA graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qubo_solvers.oriented_tangle.neural_gnn import (  # noqa: E402
    QPGSeeAGNN,
    build_qpg_graph_tensor,
)
from qubo_solvers.oriented_tangle.utils.graph_utils import (  # noqa: E402
    oriented_graph_with_copy_numbers,
)


def count_segments(gfa: Path) -> int:
    with gfa.open() as handle:
        return sum(1 for line in handle if line.startswith("S\t"))


def parse_copy_numbers(value: str, gfa: Path) -> list[float]:
    if value == "ones":
        return [1.0] * count_segments(gfa)
    return [float(item) for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--gfa", default=REPO_ROOT / "examples" / "tiny_line.gfa", type=Path)
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument("--units", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    args = parser.parse_args()

    graph = oriented_graph_with_copy_numbers(args.gfa, parse_copy_numbers(args.copy_numbers, args.gfa))
    tensor = build_qpg_graph_tensor(
        graph,
        counts=[0] * (len(graph.nodes) // 2),
        current_index=0,
        depth=0,
        horizon=max(len(graph.nodes) // 2, 1),
    )
    model = QPGSeeAGNN(units=args.units, depth=args.depth)
    output = model(tensor)
    print(f"GFA: {args.gfa}")
    print(f"nodes: {tensor.x.shape[0]}, edges: {tensor.edge_attr.shape[0]}")
    print(f"value_shape: {tuple(output['value'].shape)}, value: {float(output['value'].detach()):.6f}")
    print(f"edge_logits_shape: {tuple(output['edge_logits'].shape)}")
    print(f"embedding_shape: {tuple(output['embedding'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
