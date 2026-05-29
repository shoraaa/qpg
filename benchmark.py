#!/usr/bin/env python3
"""Run reusable one-time baseline benchmarks on all QPG GFA datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

from qpg_dynaco_workflow import (
    BASELINE_SOLVER_FUNCS,
    BASELINE_SOLVERS,
    DEFAULT_DATASET_GLOBS,
    add_gap_columns,
    append_rows,
    collect_gfas,
    load_config,
    read_completed,
    run_solver_row,
)
from qubo_solvers.definitions import Solver


FIELDNAMES = [
    "gfa",
    "solver",
    "segments",
    "horizon",
    "qubo_variables",
    "energy",
    "gap_to_exact",
    "gap_to_local",
    "runtime_s",
    "status",
    "error",
    "path",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Optional config; test_glob is reused when present.")
    parser.add_argument("--gfas", nargs="+", help="GFA files or shell patterns to benchmark.")
    parser.add_argument("--test-glob", action="append", help="Additional recursive glob for GFA files.")
    parser.add_argument("--out-csv", type=Path, default=Path("results/baselines/qubo_baselines.csv"))
    parser.add_argument(
        "--solvers",
        default=BASELINE_SOLVERS,
        help=f"Comma-separated baseline solvers. Default: {BASELINE_SOLVERS}",
    )
    parser.add_argument("-c", "--copy-numbers", default=None)
    parser.add_argument("-p", "--penalties", default=None)
    parser.add_argument("--alpha-qubo", "--qubo-alpha", dest="alpha_qubo", type=float, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--local-jobs", type=int, default=1)
    parser.add_argument("--n_ants", "--aco-ants", dest="aco_ants", type=int, default=None)
    parser.add_argument("--aco-min-iterations", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--mini_H", type=int, default=None)
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", type=float, default=None)
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", type=float, default=None)
    parser.add_argument("--rho", "--aco-evaporation", dest="aco_evaporation", type=float, default=None)
    parser.add_argument("--max-expansions", type=int, help="Expansion cap for astar.")
    parser.add_argument("--force", action="store_true", help="Re-run rows already present in the output CSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    copy_numbers = args.copy_numbers or str(config.get("copy_numbers", "ones"))
    penalties = args.penalties or str(config.get("penalties", "200,50,1"))
    alpha = args.alpha_qubo if args.alpha_qubo is not None else float(config.get("alpha_qubo", 1.1))
    time_limit = args.time_limit if args.time_limit is not None else int(config.get("eval_time_limit", 1))
    test_globs = args.test_glob if args.gfas else (args.test_glob or config.get("test_glob") or DEFAULT_DATASET_GLOBS)
    gfas = collect_gfas(args.gfas, test_globs)
    if not gfas:
        raise ValueError("No GFA files matched the benchmark dataset inputs.")

    solvers = [Solver(name) for name in args.solvers.split(",") if name]
    unsupported = [solver.value for solver in solvers if solver not in BASELINE_SOLVER_FUNCS]
    if unsupported:
        raise ValueError(f"Unsupported baseline solvers in benchmark.py: {','.join(unsupported)}")

    env = {}
    if args.aco_ants is not None or config.get("n_ants") is not None or config.get("eval_ants") is not None:
        env["QPG_ACO_ANTS"] = str(
            args.aco_ants if args.aco_ants is not None else config.get("n_ants", config.get("eval_ants"))
        )
    config_h = config.get("H", config.get("online_steps"))
    config_mini_h = config.get("mini_H", config.get("mini_h"))
    if args.aco_min_iterations is not None or config.get("eval_min_iterations") is not None:
        env["QPG_ACO_MIN_ITERATIONS"] = str(
            args.aco_min_iterations if args.aco_min_iterations is not None else config.get("eval_min_iterations")
        )
    elif args.H is not None or args.mini_H is not None or config_h is not None or config_mini_h is not None:
        H = int(args.H if args.H is not None else config_h if config_h is not None else 10)
        mini_H = int(args.mini_H if args.mini_H is not None else config_mini_h if config_mini_h is not None else 100)
        env["QPG_ACO_MIN_ITERATIONS"] = str(H * mini_H)
    if args.aco_alpha is not None or config.get("aco_alpha") is not None:
        env["QPG_ACO_ALPHA"] = str(args.aco_alpha if args.aco_alpha is not None else config.get("aco_alpha"))
    elif config.get("alpha") is not None:
        env["QPG_ACO_ALPHA"] = str(config.get("alpha"))
    if args.aco_beta is not None or config.get("aco_beta") is not None:
        env["QPG_ACO_BETA"] = str(args.aco_beta if args.aco_beta is not None else config.get("aco_beta"))
    elif config.get("beta") is not None:
        env["QPG_ACO_BETA"] = str(config.get("beta"))
    if args.aco_evaporation is not None or config.get("rho") is not None or config.get("evaporation") is not None:
        env["QPG_ACO_EVAPORATION"] = str(
            args.aco_evaporation if args.aco_evaporation is not None else config.get("rho", config.get("evaporation"))
        )
    if args.max_expansions is not None:
        env["QPG_ASTAR_MAX_EXPANSIONS"] = str(args.max_expansions)

    completed = set() if args.force else read_completed(args.out_csv)
    for gfa in tqdm(gfas, desc="benchmark GFAs", unit="gfa"):
        rows = []
        for solver in tqdm(solvers, desc=f"{gfa.name} solvers", unit="solver", leave=False):
            key = (str(gfa), solver.value)
            if key in completed:
                print(f"skip\t{gfa}\t{solver.value}")
                continue
            jobs = args.local_jobs if solver in {Solver.LOCAL, Solver.RANDOM_RESIDUAL, Solver.ACO} else args.jobs
            row = run_solver_row(
                gfa,
                solver,
                BASELINE_SOLVER_FUNCS[solver],
                copy_numbers=copy_numbers,
                penalties=penalties,
                alpha=alpha,
                time_limit=time_limit,
                jobs=jobs,
                env=env,
            )
            rows.append(row)
            print(f"{row['status']}\t{gfa}\t{solver.value}\t{row['energy']}\t{float(row['runtime_s']):.3f}s")
        add_gap_columns(rows)
        if rows:
            append_rows(args.out_csv, FIELDNAMES, rows)

    print(f"wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
