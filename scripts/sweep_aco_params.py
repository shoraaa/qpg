#!/usr/bin/env python3
"""Sweep C++ ACO parameters before training Neural ACO.

This uses the same `qpg_aco_cpp.sample_batch` backend and pheromone update rule
as the online DyNACO trainer, but with a zero neural prior.  The goal is to pick
the base ACO settings for the target benchmark before spending time on neural
training.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "qubo"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from examples.train_qpg_dynaco_online import (  # noqa: E402
    build_instance,
    collect_gfas,
    config_defaults,
    generate_synthetic_gfas,
    load_config,
)
from qubo_solvers.oriented_tangle import qpg_aco_cpp  # noqa: E402


def split_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def split_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def aco_run(instance, *, ants: int, alpha: float, beta: float, evaporation: float, min_iterations: int, time_limit: float, seed: int, parallel_traced: bool) -> tuple[float, float, int]:
    started = time.perf_counter()
    deadline = max(float(time_limit), 0.001)
    pheromone = np.ones_like(instance.heuristic, dtype=np.float32)
    prior = np.zeros_like(instance.heuristic, dtype=np.float32)
    q_float = np.asarray(instance.description.Q, dtype=np.float32)
    best = float("inf")
    iterations = 0

    while iterations < min_iterations or time.perf_counter() - started < deadline:
        batch = qpg_aco_cpp.sample_batch(
            instance.offsets,
            instance.targets,
            pheromone,
            instance.heuristic,
            prior,
            instance.weights_array,
            instance.lengths_array,
            q_float,
            float(instance.description.offset),
            instance.states_per_time,
            instance.description.T,
            ants,
            instance.start_source,
            instance.end_index,
            alpha,
            beta,
            0.0,
            seed + iterations,
            parallel_traced,
        )
        energies = np.asarray(batch["energies"], dtype=np.float32)
        best = min(best, float(np.min(energies)))

        pheromone *= 1.0 - evaporation
        np.maximum(pheromone, 1e-6, out=pheromone)
        worst_energy = float(np.max(energies))
        trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
        trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
        for ant_index in np.argsort(energies)[: max(1, ants // 4)]:
            deposit = (worst_energy - float(energies[ant_index]) + 1.0) / (abs(worst_energy) + 1.0)
            begin = int(trace_starts[ant_index])
            end = int(trace_starts[ant_index + 1])
            for edge_id in trace_edges[begin:end]:
                pheromone[int(edge_id)] += deposit

        iterations += 1
        if time.perf_counter() - started >= deadline and iterations >= min_iterations:
            break

    return best, time.perf_counter() - started, iterations


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float, float, float, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["ants"]),
            float(row["alpha"]),
            float(row["beta"]),
            float(row["evaporation"]),
            int(row["min_iterations"]),
        )
        grouped.setdefault(key, []).append(row)

    best_by_case: dict[tuple[str, int], float] = {}
    for row in rows:
        case = (str(row["gfa"]), int(row["seed"]))
        best_by_case[case] = min(best_by_case.get(case, float("inf")), float(row["energy"]))

    summary = []
    for (ants, alpha, beta, evaporation, min_iterations), group in sorted(grouped.items()):
        energies = [float(row["energy"]) for row in group]
        runtimes = [float(row["runtime_s"]) for row in group]
        gaps = []
        wins = 0
        for row in group:
            best = best_by_case[(str(row["gfa"]), int(row["seed"]))]
            energy = float(row["energy"])
            gaps.append(energy - best)
            if abs(energy - best) <= 1e-9:
                wins += 1
        summary.append(
            {
                "ants": ants,
                "alpha": alpha,
                "beta": beta,
                "evaporation": evaporation,
                "min_iterations": min_iterations,
                "rows": len(group),
                "mean_energy": float(np.mean(energies)),
                "median_energy": float(np.median(energies)),
                "best_energy": float(np.min(energies)),
                "mean_gap_to_sweep_best": float(np.mean(gaps)),
                "wins_or_ties": wins,
                "mean_runtime_s": float(np.mean(runtimes)),
            }
        )
    summary.sort(key=lambda row: (float(row["mean_gap_to_sweep_best"]), -int(row["wins_or_ties"]), float(row["mean_energy"])))
    return summary


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dynaco_online_hard.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "aco_param_sweep" / timestamp)
    parser.add_argument("--gfas", nargs="+", help="Explicit GFA files or globs.")
    parser.add_argument("--test-glob", action="append", help="Additional test GFA glob.")
    parser.add_argument("--max-gfas", type=int, default=16)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ants", default="64,128,256")
    parser.add_argument("--alpha", default="0.5,1.0,1.5")
    parser.add_argument("--beta", default="1.0,2.0,3.0")
    parser.add_argument("--evaporation", default="0.1,0.2,0.4")
    parser.add_argument("--min-iterations", default="4,6")
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for qpg_aco_cpp.")
    parser.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    parser.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--generate-test", type=int, help="Generate this many held-out synthetic GFAs into the sweep output directory.")
    args = parser.parse_args()

    config = config_defaults(load_config(args.config))
    config.update(
        {
            "synthetic_dir": args.out_dir / "generated",
            "device": "cpu",
            "eval_time_limit": args.time_limit,
        }
    )
    cfg = argparse.Namespace(**config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads is not None:
        qpg_aco_cpp.set_num_threads(args.threads)

    if args.generate_test is not None:
        gfas = generate_synthetic_gfas(cfg, "test", args.generate_test)
    else:
        test_globs = args.test_glob if args.gfas else (args.test_glob or config.get("test_glob"))
        gfas = collect_gfas(args.gfas, test_globs, required=False)
        if not gfas:
            cfg.generate_synthetic_test = int(config.get("generate_synthetic_test", 100))
            gfas = generate_synthetic_gfas(cfg, "test", cfg.generate_synthetic_test)

    random.Random(args.seed).shuffle(gfas)
    if args.max_gfas > 0:
        gfas = gfas[: args.max_gfas]
    if not gfas:
        raise ValueError("No GFA instances available for ACO sweep.")

    instances = {str(path): build_instance(path, cfg) for path in gfas}
    grid = list(
        itertools.product(
            split_ints(args.ants),
            split_floats(args.alpha),
            split_floats(args.beta),
            split_floats(args.evaporation),
            split_ints(args.min_iterations),
        )
    )
    rows = []
    total = len(grid) * len(gfas) * args.seed_count
    completed = 0
    for ants, alpha, beta, evaporation, min_iterations in grid:
        for gfa in gfas:
            instance = instances[str(gfa)]
            for seed_index in range(args.seed_count):
                run_seed = args.seed + seed_index * 1000003
                energy, runtime_s, iterations = aco_run(
                    instance,
                    ants=ants,
                    alpha=alpha,
                    beta=beta,
                    evaporation=evaporation,
                    min_iterations=min_iterations,
                    time_limit=args.time_limit,
                    seed=run_seed,
                    parallel_traced=args.parallel_traced,
                )
                rows.append(
                    {
                        "gfa": str(gfa),
                        "segments": instance.description.V,
                        "horizon": instance.description.T,
                        "seed": run_seed,
                        "ants": ants,
                        "alpha": alpha,
                        "beta": beta,
                        "evaporation": evaporation,
                        "min_iterations": min_iterations,
                        "time_limit_s": args.time_limit,
                        "iterations": iterations,
                        "energy": energy,
                        "runtime_s": runtime_s,
                    }
                )
                completed += 1
                print(
                    f"{completed}/{total}\tants={ants}\talpha={alpha}\tbeta={beta}\t"
                    f"evap={evaporation}\tmin_iter={min_iterations}\t{Path(gfa).name}\t"
                    f"seed={run_seed}\tenergy={energy:.12g}\t{runtime_s:.3f}s",
                    flush=True,
                )

    summary = summarize(rows)
    write_csv(
        args.out_dir / "aco_sweep_raw.csv",
        [
            "gfa",
            "segments",
            "horizon",
            "seed",
            "ants",
            "alpha",
            "beta",
            "evaporation",
            "min_iterations",
            "time_limit_s",
            "iterations",
            "energy",
            "runtime_s",
        ],
        rows,
    )
    write_csv(
        args.out_dir / "aco_sweep_summary.csv",
        [
            "ants",
            "alpha",
            "beta",
            "evaporation",
            "min_iterations",
            "rows",
            "mean_energy",
            "median_energy",
            "best_energy",
            "mean_gap_to_sweep_best",
            "wins_or_ties",
            "mean_runtime_s",
        ],
        summary,
    )
    manifest = {
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "gfas": [str(path) for path in gfas],
        "grid_size": len(grid),
        "seed_count": args.seed_count,
        "parallel_traced": args.parallel_traced,
        "threads": qpg_aco_cpp.get_max_threads(),
        "best": summary[0] if summary else None,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"best: {summary[0] if summary else None}")
    print(f"wrote: {args.out_dir / 'aco_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
