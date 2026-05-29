#!/usr/bin/env python3
"""Benchmark oriented QUBO solvers on one instance.

The intended comparison is:
  - exact: reference energy when the instance is small enough;
  - local and simple graph heuristics: low-cost baselines;
  - astar/seea/metaheuristics: should improve energy and/or runtime.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qubo_solvers.definitions import QuboDescription, Solver  # noqa: E402
from qubo_solvers.oriented_tangle.utils.graph_utils import (  # noqa: E402
    oriented_graph_with_copy_numbers,
)
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    aco_sample_qubo,
    astar_sample_qubo,
    beam_search_sample_qubo,
    exact_sample_qubo,
    greedy_residual_sample_qubo,
    local_sample_qubo,
    neural_aco_sample_qubo,
    print_path,
    random_residual_sample_qubo,
    seea_sample_qubo,
)


SOLVER_FUNCS = {
    Solver.EXACT: exact_sample_qubo,
    Solver.LOCAL: local_sample_qubo,
    Solver.GREEDY_RESIDUAL: greedy_residual_sample_qubo,
    Solver.RANDOM_RESIDUAL: random_residual_sample_qubo,
    Solver.BEAM: beam_search_sample_qubo,
    Solver.ACO: aco_sample_qubo,
    Solver.NEURAL_ACO: neural_aco_sample_qubo,
    Solver.ASTAR: astar_sample_qubo,
    Solver.SEEA: seea_sample_qubo,
}

DEFAULT_SOLVERS = (
    "exact,local,greedy_residual,random_residual_walk,"
    "beam_search,aco,astar,seea"
)


def count_segments(gfa: Path) -> int:
    count = 0
    with gfa.open() as handle:
        for line in handle:
            if line.startswith("S\t"):
                count += 1
    return count


def parse_copy_numbers(value: str, gfa: Path) -> list[float]:
    if value == "ones":
        return [1.0] * count_segments(gfa)
    return [float(item) for item in value.split(",") if item]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def best_result(paths: dict[int, list[tuple]]) -> tuple[float, list]:
    best_energy = np.inf
    best_path = []
    for runs in paths.values():
        for _, energy, path in runs:
            if energy < best_energy:
                best_energy = float(energy)
                best_path = path
    return best_energy, best_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--gfa",
        default=REPO_ROOT / "examples" / "tiny_line.gfa",
        type=Path,
        help="GFA graph to benchmark.",
    )
    parser.add_argument(
        "-c",
        "--copy-numbers",
        default="ones",
        help="Comma-separated copy numbers, one per GFA segment, or 'ones'.",
    )
    parser.add_argument(
        "-p",
        "--penalties",
        default="200,50,1",
        help="Comma-separated QUBO penalties.",
    )
    parser.add_argument("-t", "--time-limit", default=1, type=int)
    parser.add_argument("-j", "--local-jobs", default=1, type=int)
    parser.add_argument(
        "--max-expansions",
        type=int,
        help="Expansion cap for astar/seea. Defaults to QPG_ASTAR_MAX_EXPANSIONS or solver default.",
    )
    parser.add_argument(
        "--neural-model",
        type=Path,
        help="Checkpoint for neural SeeA*. Required when solver list includes seea.",
    )
    parser.add_argument(
        "--neural-beta",
        type=float,
        default=1.0,
        help="Weight beta in neural SeeA* priority g(n)+beta*h_theta(n).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for neural SeeA* model evaluation.",
    )
    parser.add_argument("--no-paths", action="store_true", help="Do not print decoded paths.")
    parser.add_argument(
        "--solvers",
        default=DEFAULT_SOLVERS,
        help=f"Comma-separated subset of: {','.join(item.value for item in SOLVER_FUNCS)}.",
    )
    parser.add_argument(
        "--alpha",
        default=1.1,
        type=float,
        help="Walk-length multiplier passed to qubo_matrix_from_graph.",
    )
    args = parser.parse_args()
    if args.max_expansions is not None:
        os.environ["QPG_ASTAR_MAX_EXPANSIONS"] = str(args.max_expansions)
    if args.neural_model is not None:
        os.environ["QPG_SEEA_MODEL"] = str(args.neural_model)
        os.environ["QPG_ACO_MODEL"] = str(args.neural_model)
    os.environ["QPG_SEEA_BETA"] = str(args.neural_beta)
    os.environ["QPG_SEEA_DEVICE"] = args.device
    os.environ["QPG_ACO_DEVICE"] = args.device

    graph = oriented_graph_with_copy_numbers(args.gfa, parse_copy_numbers(args.copy_numbers, args.gfa))
    q_matrix, offset, t_max, original_node_count = qubo_matrix_from_graph(
        graph,
        alpha=args.alpha,
        penalties=parse_csv_ints(args.penalties),
    )
    requested = [Solver(name) for name in args.solvers.split(",") if name]

    print(f"GFA: {args.gfa}")
    print(f"QUBO shape: {q_matrix.shape}, T={t_max}, states_per_time={len(q_matrix) // t_max}")
    print(f"time_limit: {args.time_limit}s")
    print()

    results: dict[Solver, dict[str, object]] = {}
    exact_energy: float | None = None

    for solver in requested:
        jobs = args.local_jobs if solver in {Solver.LOCAL, Solver.RANDOM_RESIDUAL, Solver.ACO} else 1
        description = QuboDescription(
            filename=args.gfa.name,
            data_dir=str(args.gfa.parent),
            graph=graph,
            time_limits=[args.time_limit],
            jobs=jobs,
            Q=q_matrix,
            offset=offset,
            T=t_max,
            V=original_node_count,
            solver=solver,
        )
        started = time.perf_counter()
        try:
            paths = SOLVER_FUNCS[solver](description)
            elapsed = time.perf_counter() - started
            energy, path = best_result(paths)
            results[solver] = {"ok": True, "energy": energy, "path": path, "elapsed": elapsed}
            if solver == Solver.EXACT:
                exact_energy = energy
        except Exception as exc:  # exact intentionally refuses large spaces
            elapsed = time.perf_counter() - started
            results[solver] = {"ok": False, "error": str(exc), "elapsed": elapsed}

    local_energy = results.get(Solver.LOCAL, {}).get("energy")
    print("solver\tenergy\tgap_to_exact\tgap_to_local\truntime_s\tstatus")
    for solver in requested:
        result = results[solver]
        if not result["ok"]:
            print(f"{solver.value}\tNA\tNA\tNA\t{result['elapsed']:.4f}\t{result['error']}")
            continue
        energy = float(result["energy"])
        gap_to_exact = "NA" if exact_energy is None else f"{energy - exact_energy:.12g}"
        gap_to_local = "NA" if local_energy is None else f"{energy - float(local_energy):.12g}"
        print(
            f"{solver.value}\t{energy:.12g}\t{gap_to_exact}\t"
            f"{gap_to_local}\t{result['elapsed']:.4f}\tok"
        )

    if not args.no_paths:
        print()
        for solver in requested:
            result = results[solver]
            if result["ok"]:
                print(f"{solver.value} best path:")
                print_path(result["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
