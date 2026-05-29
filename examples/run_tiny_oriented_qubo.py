#!/usr/bin/env python3
"""Run a tiny exact QUBO example for the paper's oriented tangle problem.

This enumerates one-hot assignments for a five-node graph, so it is only for
smoke tests and explanation. It avoids external solvers such as MQLib, Gurobi,
and D-Wave while still using the repository's QUBO construction code.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qubo_solvers.oriented_tangle.utils.graph_utils import (  # noqa: E402
    oriented_graph_with_copy_numbers,
)
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402


def local_state_names(graph, original_node_count: int) -> list[str]:
    return list(graph.nodes)[: original_node_count * 2] + ["end"]


def one_hot_solution(choices: tuple[int, ...], states_per_time: int) -> np.ndarray:
    x = np.zeros(len(choices) * states_per_time, dtype=float)
    for t, choice in enumerate(choices):
        x[t * states_per_time + choice] = 1.0
    return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--gfa",
        default=REPO_ROOT / "examples" / "tiny_line.gfa",
        type=Path,
        help="GFA graph to solve.",
    )
    parser.add_argument(
        "-c",
        "--copy-numbers",
        default="1,1,1,1,1",
        help="Comma-separated copy numbers, one per GFA segment.",
    )
    parser.add_argument(
        "-p",
        "--penalties",
        default="200,50,1",
        help="Comma-separated QUBO penalties: one-node, graph-step, node-weight.",
    )
    parser.add_argument(
        "--alpha",
        default=1.1,
        type=float,
        help="Walk-length multiplier passed to qubo_matrix_from_graph.",
    )
    args = parser.parse_args()

    copy_numbers = [float(value) for value in args.copy_numbers.split(",")]
    penalties = [int(value) for value in args.penalties.split(",")]
    graph = oriented_graph_with_copy_numbers(args.gfa, copy_numbers)
    q_matrix, offset, t_max, original_node_count = qubo_matrix_from_graph(
        graph,
        alpha=args.alpha,
        penalties=penalties,
    )

    states = local_state_names(graph, original_node_count)
    states_per_time = len(states)

    best_energy = np.inf
    best_choices: tuple[int, ...] | None = None
    for choices in product(range(states_per_time), repeat=t_max):
        x = one_hot_solution(choices, states_per_time)
        energy = float(x @ q_matrix @ x + offset)
        if energy < best_energy:
            best_energy = energy
            best_choices = choices

    assert best_choices is not None
    path = [states[choice] for choice in best_choices]
    visits = {node: path.count(f"{node}_+") + path.count(f"{node}_-") for node in "ABCDE"}

    print(f"GFA: {args.gfa}")
    print(f"copy_numbers: {copy_numbers}")
    print(f"alpha: {args.alpha}")
    print(f"QUBO shape: {q_matrix.shape}, T={t_max}, states_per_time={states_per_time}")
    print(f"best_energy: {best_energy:.3f}")
    print("path:", " -> ".join(path))
    print("visits:", ", ".join(f"{node}={count}" for node, count in visits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
