#!/usr/bin/env python3
"""Reproduce paper.md-style QPG assembly benchmarks.

This is a thin, logged orchestration layer over the original shell pipeline:

    genome_create -> minigraph pangenome -> annotator -> solver -> path_seq
    -> candidate_stats / consensus evaluation -> aggregate text tables

By default the script prints the commands it would run. Use --run for the full
benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


PAPER_TABLE2_SOLVERS = "pathfinder,mqlib,gurobi"
FUTURE_QUBO_SOLVERS = "local,beam_search,aco,neural_aco,astar"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, execute: bool) -> int:
    printable = shell_join(command)
    print(printable)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"$ {printable}\n")
    if not execute:
        return 0
    with log_path.open("a") as log:
        process = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    return int(process.returncode)


def make_env(args, out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["QDIR"] = str(REPO_ROOT)
    env["PYTHON"] = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable
    env["PYTHONPATH"] = str(REPO_ROOT / "qubo") + os.pathsep + env.get("PYTHONPATH", "")
    if DEFAULT_PYTHON.exists():
        env["VIRTUAL_ENV"] = str((REPO_ROOT / ".venv").resolve())
    env["PATHFINDER"] = str(REPO_ROOT / ".tools" / "bin" / "pathfinder")
    env["BWA"] = str(REPO_ROOT / ".tools" / "bwa" / "bwa")
    tool_paths = [
        REPO_ROOT / ".venv" / "bin",
        REPO_ROOT,
        REPO_ROOT / ".tools" / "bin",
        REPO_ROOT / ".tools" / "minigraph",
        REPO_ROOT / ".tools" / "minimap2",
        REPO_ROOT / ".tools" / "bwa",
        REPO_ROOT / ".tools" / "samtools",
        REPO_ROOT / ".tools" / "samtools" / "build" / "bin",
        REPO_ROOT / ".tools" / "htslib" / "build" / "bin",
    ]
    env["PATH"] = ":".join(str(path) for path in tool_paths) + f":{env.get('PATH', '')}"
    env["SHRED_DEPTH"] = str(args.shred_depth)
    env["SHUF_RANDOM_SOURCE"] = args.shuf_random_source
    if args.neural_model:
        env["QPG_ACO_MODEL"] = str(Path(args.neural_model).resolve())
        env["QPG_SEEA_MODEL"] = str(Path(args.neural_model).resolve())
    env["QPG_ACO_DEVICE"] = args.device
    env["QPG_SEEA_DEVICE"] = args.device
    env["QPG_ACO_ANTS"] = str(args.aco_ants)
    env["QPG_ACO_MIN_ITERATIONS"] = str(args.aco_min_iterations)
    env["QPG_ACO_ALPHA"] = str(args.aco_alpha)
    env["QPG_ACO_BETA"] = str(args.aco_beta)
    env["QPG_ACO_EVAPORATION"] = str(args.aco_evaporation)
    env["QPG_ASTAR_MAX_EXPANSIONS"] = str(args.max_expansions)
    env["SOLVERS"] = " ".join(split_csv(args.solvers))
    env["QPG_REPRO_OUT"] = str(out_dir)
    htslib = REPO_ROOT / ".tools" / "htslib" / "build" / "lib"
    env["LD_LIBRARY_PATH"] = str(htslib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def pathfinder_compatible(env: dict[str, str]) -> tuple[bool, str]:
    pathfinder = shutil.which("pathfinder", path=env.get("PATH"))
    if pathfinder is None:
        return False, "missing"
    process = subprocess.run(
        [pathfinder, "--X50"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = (process.stdout + process.stderr).lower()
    if "unknown option" in output:
        return False, f"{pathfinder} rejects QPG-era --X50"
    return True, pathfinder


def gurobi_compatible(env: dict[str, str]) -> tuple[bool, str]:
    python = env.get("PYTHON", sys.executable)
    probe = (
        "import gurobipy as gp\n"
        "from gurobipy import GRB\n"
        "with gp.Env(empty=True) as env:\n"
        "    env.setParam('OutputFlag', 0)\n"
        "    env.start()\n"
        "    with gp.Model(env=env) as model:\n"
        "        model.addMVar(shape=2501, vtype=GRB.BINARY, name='x')\n"
        "        model.setObjective(0, GRB.MINIMIZE)\n"
        "        model.optimize()\n"
    )
    process = subprocess.run(
        [python, "-c", probe],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = (process.stdout + process.stderr).strip()
    if process.returncode != 0:
        if "size-limited license" in output:
            return False, "installed Gurobi is size-limited and rejects paper-scale QUBO models"
        return False, output or "gurobipy probe failed"
    return True, "available"


def preflight(args, env: dict[str, str]) -> list[str]:
    failures = []
    solvers = split_csv(args.solvers)
    if args.pathfinder_graph or "pathfinder" in solvers:
        ok, message = pathfinder_compatible(env)
        if not ok:
            failures.append(
                "compatible QPG-era pathfinder unavailable; "
                f"{message}. The paper method uses Pathfinder options from config_base.sh "
                "and Pathfinder preprocessing for QUBO runs when --pathfinder-graph is enabled."
            )
    if "mqlib" in solvers and shutil.which("MQLib", path=env.get("PATH")) is None:
        failures.append("MQLib is unavailable on PATH, so the paper Table 2 solver set cannot run.")
    if "gurobi" in solvers:
        ok, message = gurobi_compatible(env)
        if not ok:
            failures.append(f"Gurobi is unavailable for paper-scale runs: {message}.")
    return failures


def paper_table2_command(args) -> list[str]:
    solvers = " ".join(split_csv(args.solvers))
    return [
        str(REPO_ROOT / "tangle_resolution_benchmark.sh"),
        str(args.seeds),
        args.time_limits,
        str(args.jobs),
        str(args.test_sequences),
        solvers,
        "1" if args.pathfinder_graph else "0",
        " ".join(split_csv(args.annotators)),
    ]


def paired_data_command(args, seed: int, annotator: str, out_prefix: str) -> list[str]:
    return [
        str(REPO_ROOT / "run_gfa_sim.sh"),
        "-s",
        str(seed),
        "-c",
        str(REPO_ROOT / f"config_illumina_{annotator}.sh"),
        "-a",
        annotator,
        "--solver",
        "aco",
        "-p",
        out_prefix,
        "-n",
        str(args.test_sequences),
        "--data-only",
    ]


def paired_solver_command(args, seed: int, annotator: str, solver: str, data_dir: Path) -> list[str]:
    command = [
        str(REPO_ROOT / "run_gfa_sim.sh"),
        "-s",
        str(seed),
        "-c",
        str(REPO_ROOT / f"config_illumina_{annotator}.sh"),
        "-a",
        annotator,
        "--solver",
        solver,
        "-p",
        f"{solver}.{annotator}.",
        "-t",
        args.time_limits,
        "-j",
        str(args.jobs),
        "-n",
        str(args.test_sequences),
        "--from-data",
        str(data_dir),
    ]
    if args.pathfinder_graph:
        command.append("--pathfinder_graph")
    return command


def paired_full_commands(args, out_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    data_root = out_dir / "paired_data"
    for seed in range(1, args.seeds + 1):
        for annotator in split_csv(args.annotators):
            prefix = str(data_root / f"data.{annotator}.")
            commands.append(paired_data_command(args, seed, annotator, prefix))
            data_dir = data_root / f"data.{annotator}.{seed:05d}"
            for solver in split_csv(args.solvers):
                commands.append(paired_solver_command(args, seed, annotator, solver, data_dir))
    commands.extend(
        [
            [
                str(REPO_ROOT / "tangle_resolution_benchmark_stats.sh"),
                str(args.seeds),
                args.time_limits,
                str(args.jobs),
                str(args.test_sequences),
                " ".join(split_csv(args.solvers)),
                "cons",
            ],
            parse_stats_command("cons"),
        ]
    )
    return commands


def parse_stats_command(data_type: str) -> list[str]:
    return [str(REPO_ROOT / "tangle_resolution_benchmark_parse_stats.sh"), data_type]


def parsed_metric_rows(out_dir: Path) -> int:
    rows = 0
    for path in out_dir.glob("*.cons.avg.parsed.txt"):
        text = path.read_text(errors="replace").strip()
        if text:
            rows += len(text.splitlines())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute commands. Default is dry-run.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to results/paper_repro/<timestamp>.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of synthetic pangenome seeds.")
    parser.add_argument("--test-sequences", type=int, default=5, help="Held-out sequences per seed.")
    parser.add_argument("--jobs", type=int, default=3, help="Runs per QUBO solver per instance.")
    parser.add_argument("--time-limits", default="5,300", help="Comma-separated QUBO solver time limits.")
    parser.add_argument("--annotators", default="mg", help="Comma-separated annotation routes to run. Default is MG only.")
    parser.add_argument(
        "--solvers",
        default=PAPER_TABLE2_SOLVERS,
        help=f"Comma-separated solvers. Paper Table 2 default: {PAPER_TABLE2_SOLVERS}. "
        f"Future comparison set: {FUTURE_QUBO_SOLVERS}.",
    )
    parser.add_argument(
        "--pathfinder-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Pathfinder subgraphs/copy-number preprocessing for QUBO solvers, matching the paper method text.",
    )
    parser.add_argument(
        "--paired-data",
        action="store_true",
        help="Generate each seed/annotator once, then reuse the same annotated GFAs for every solver.",
    )
    parser.add_argument("--shred-depth", type=int, default=30, help="Simulated short-read coverage depth.")
    parser.add_argument("--shuf-random-source", default="/usr/bin/emacs", help="Stable shuf source for train/test split.")
    parser.add_argument("--neural-model", type=Path, help="Checkpoint for neural_aco or seea comparison runs.")
    parser.add_argument("--device", default="cpu", help="Torch device for neural solvers.")
    parser.add_argument("--n_ants", "--aco-ants", dest="aco_ants", type=int, default=32, help="ACO/neural_aco ants per iteration.")
    parser.add_argument("--H", type=int, default=10, help="DyNACO outer steps.")
    parser.add_argument("--mini_H", type=int, default=10, help="DyNACO inner steps per outer step.")
    parser.add_argument("--aco-min-iterations", type=int, default=None, help="Minimum ACO iterations. Defaults to H * mini_H.")
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", type=float, default=1.0, help="Pheromone exponent.")
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", type=float, default=1.0, help="Residual heuristic exponent.")
    parser.add_argument("--rho", "--aco-evaporation", dest="aco_evaporation", type=float, default=0.1, help="Pheromone evaporation rate.")
    parser.add_argument("--max-expansions", type=int, default=100000, help="A*/SeeA* expansion cap.")
    args = parser.parse_args()
    if args.aco_min_iterations is None:
        args.aco_min_iterations = args.H * args.mini_H

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPO_ROOT / "results" / "paper_repro" / timestamp)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    env = make_env(args, out_dir)
    preflight_failures = preflight(args, env)
    if preflight_failures:
        for failure in preflight_failures:
            print(f"preflight failed: {failure}", file=sys.stderr)
        if args.run:
            return 2

    manifest = {
        "created_at": timestamp,
        "repo_root": str(REPO_ROOT),
        "paper_scope": {
            "population": "synthetic haploid pangenomes from genome_create",
            "annotations": split_csv(args.annotators),
            "default_solvers": split_csv(PAPER_TABLE2_SOLVERS),
            "metrics": ["covered", "used", "ncontig", "nbreaks", "nindel", "ndiff", "n50", "identity"],
            "consensus": "candidate sequence and read-realigned consensus are both produced by run_sim_evaluate_path.sh",
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment_overrides": {
            key: env[key]
            for key in [
                "QDIR",
                "SHRED_DEPTH",
                "SHUF_RANDOM_SOURCE",
                "QPG_ACO_MODEL",
                "QPG_ACO_DEVICE",
                "QPG_ACO_ANTS",
                "QPG_ACO_MIN_ITERATIONS",
                "QPG_ASTAR_MAX_EXPANSIONS",
                "SOLVERS",
                "PYTHON",
                "PYTHONPATH",
                "PATHFINDER",
                "BWA",
                "LD_LIBRARY_PATH",
            ]
            if key in env
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    if args.paired_data:
        commands = paired_full_commands(args, out_dir)
    else:
        commands = [
            paper_table2_command(args),
            parse_stats_command("cons"),
        ]
    for command in commands:
        code = run_command(command, cwd=out_dir, env=env, log_path=log_path, execute=args.run)
        if code != 0:
            print(f"command failed with exit code {code}: {shell_join(command)}", file=sys.stderr)
            return code

    if not args.run:
        print(f"dry-run only; add --run to execute. Manifest: {out_dir / 'manifest.json'}")
    else:
        metric_rows = parsed_metric_rows(out_dir)
        if metric_rows == 0:
            print(
                "paper reproduction finished without parsed metric rows; "
                "treat this run as invalid and inspect run.log plus *.err files.",
                file=sys.stderr,
            )
            return 3
        print(f"completed paper reproduction pipeline: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
