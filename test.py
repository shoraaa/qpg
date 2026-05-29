#!/usr/bin/env python3
"""Evaluate a trained QPG DyNACO model on every configured GFA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from qpg_dynaco_workflow import (
    DEFAULT_DATASET_GLOBS,
    NEURAL_SOLVER_FUNCS,
    append_rows,
    collect_gfas,
    load_config,
    read_completed,
    run_solver_row,
)
from qubo_solvers.definitions import Solver


FIELDNAMES = [
    "gfa",
    "checkpoint",
    "solver",
    "segments",
    "horizon",
    "qubo_variables",
    "energy",
    "runtime_s",
    "status",
    "error",
    "path",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Training config; test_glob is reused when present.")
    parser.add_argument("--model", type=Path, help="DyNACO checkpoint. Defaults to config out.")
    parser.add_argument("--gfas", nargs="+", help="GFA files or shell patterns to evaluate.")
    parser.add_argument("--test-glob", action="append", help="Additional recursive glob for GFA files.")
    parser.add_argument("--out-csv", type=Path, default=Path("results/dynaco_online/test_results.csv"))
    parser.add_argument("-c", "--copy-numbers", default=None)
    parser.add_argument("-p", "--penalties", default=None)
    parser.add_argument("--alpha-qubo", "--qubo-alpha", dest="alpha_qubo", type=float, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_ants", "--aco-ants", dest="aco_ants", type=int, default=None)
    parser.add_argument("--aco-min-iterations", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--mini_H", type=int, default=None)
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", type=float, default=None)
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", type=float, default=None)
    parser.add_argument("--gamma", "--aco-gamma", dest="aco_gamma", type=float, default=None)
    parser.add_argument(
        "--prior-mode",
        choices=["learned", "zero", "shuffle", "random"],
        default=None,
        help="Neural ACO prior control for ablations.",
    )
    parser.add_argument("--rho", "--aco-evaporation", dest="aco_evaporation", type=float, default=None)
    traced_group = parser.add_mutually_exclusive_group()
    traced_group.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    traced_group.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for the C++ ACO backend.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-run rows already present in the output CSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    model = args.model or (Path(str(config["out"])) if config.get("out") else None)
    if model is None:
        raise ValueError("Provide --model or a config with out: <checkpoint>.")

    copy_numbers = args.copy_numbers or str(config.get("copy_numbers", "ones"))
    penalties = args.penalties or str(config.get("penalties", "200,50,1"))
    alpha = args.alpha_qubo if args.alpha_qubo is not None else float(config.get("alpha_qubo", 1.1))
    time_limit = args.time_limit if args.time_limit is not None else int(config.get("eval_time_limit", 1))
    device = args.device or str(config.get("device", "cpu"))
    test_globs = args.test_glob if args.gfas else (args.test_glob or config.get("test_glob") or DEFAULT_DATASET_GLOBS)
    gfas = collect_gfas(args.gfas, test_globs)
    if not gfas:
        raise ValueError("No GFA files matched the test dataset inputs.")

    env = {
        "QPG_ACO_MODEL": str(model),
        "QPG_ACO_DEVICE": device,
        "QPG_ACO_SEED": str(args.seed),
        "QPG_ACO_PARALLEL_TRACED": "1" if args.parallel_traced else "0",
    }
    if args.threads is not None:
        env["QPG_ACO_THREADS"] = str(args.threads)
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
    if args.aco_gamma is not None or config.get("gamma") is not None:
        env["QPG_ACO_GAMMA"] = str(args.aco_gamma if args.aco_gamma is not None else config.get("gamma"))
    if args.prior_mode is not None:
        env["QPG_ACO_PRIOR_MODE"] = args.prior_mode
    if args.aco_evaporation is not None or config.get("rho") is not None or config.get("evaporation") is not None:
        env["QPG_ACO_EVAPORATION"] = str(
            args.aco_evaporation if args.aco_evaporation is not None else config.get("rho", config.get("evaporation"))
        )

    completed = set() if args.force else read_completed(args.out_csv)
    rows = []
    for gfa in gfas:
        key = (str(gfa), Solver.NEURAL_ACO.value)
        if key in completed:
            print(f"skip\t{gfa}\tneural_aco")
            continue
        row = run_solver_row(
            gfa,
            Solver.NEURAL_ACO,
            NEURAL_SOLVER_FUNCS[Solver.NEURAL_ACO],
            copy_numbers=copy_numbers,
            penalties=penalties,
            alpha=alpha,
            time_limit=time_limit,
            jobs=args.jobs,
            env=env,
        )
        row["checkpoint"] = str(model)
        rows.append(row)
        append_rows(args.out_csv, FIELDNAMES, [row])
        print(f"{row['status']}\t{gfa}\tneural_aco\t{row['energy']}\t{float(row['runtime_s']):.3f}s")

    print(f"wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
