#!/usr/bin/env python3
"""Plan or run the next reviewer-facing QPG paper experiments.

This launcher is deliberately focused on the evidence gaps in `method.tex`:

* wall-clock-oriented QUBO baselines on the same held-out GFAs;
* learned-prior ablations through gamma and stochastic seeds;
* paired full-assembly baselines under the same minigraph pathfinder-graph route.

By default the script writes commands and a manifest without executing them.
Add `--execute` to spend compute.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = (
    REPO_ROOT
    / "results"
    / "overnight_dynaco_paper"
    / "20260522_013319"
    / "dynaco_overnight_best.pt"
)
DEFAULT_SELECTED_GFAS = (
    REPO_ROOT
    / "results"
    / "dynaco_claims"
    / "final_qubo_80_heldout_fixed"
    / "qubo_scale"
    / "selected_gfas.csv"
)
DEFAULT_EXISTING_NEURAL_QUBO = (
    REPO_ROOT
    / "results"
    / "dynaco_claims"
    / "final_qubo_80_heldout_fixed"
    / "qubo_scale"
)
EVAL_RE = re.compile(
    r"^(?P<seq>\S+)\s+\d+\s+\d+\s+(?P<covered>[\d.]+)%\s+(?P<used>[\d.]+)%\s+"
    r"(?P<contigs>\d+)\s+(?P<breaks>\d+)\s+(?P<indels>\d+)\s+"
    r"(?P<diffs>\d+)\s+(?P<identity>[\d.]+)%"
)
EVAL_NAME_RE = re.compile(r"(?P<seq>.+)\.eval_cons\.(?P<budget>\d+)\.(?P<job>\d+)$")
NEURAL_QUBO_RE = re.compile(r"dynaco_budget_(?P<budget>\d+)s\.csv$")
GAMMA_RE = re.compile(
    r"neural_(?:prior_(?P<prior_mode>[^_]+)_)?gamma_(?P<gamma>.+)_seed_(?P<seed>\d+)_budget_(?P<budget>\d+)s\.csv$"
)
BASELINE_RE = re.compile(r"baselines_budget_(?P<budget>\d+)s\.csv$")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable).resolve())


def command_env() -> dict[str, str]:
    import os

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
    env = dict(os.environ)
    env["QDIR"] = str(REPO_ROOT)
    env["PYTHON"] = python_executable()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT / "qubo") + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.pathsep.join(str(path) for path in tool_paths) + os.pathsep + env.get("PATH", "")
    htslib = REPO_ROOT / ".tools" / "htslib" / "build" / "lib"
    env["LD_LIBRARY_PATH"] = str(htslib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PATHFINDER"] = str(REPO_ROOT / ".tools" / "bin" / "pathfinder")
    env["BWA"] = str(REPO_ROOT / ".tools" / "bwa" / "bwa")
    return env


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("")
        return
    fields = fieldnames
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ok_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def compact_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def gamma_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def ablation_control_label(prior_mode: str, gamma: float) -> str:
    if gamma == 0.0:
        return "zero-weight prior"
    if prior_mode == "zero":
        return "zero prior"
    return "active learned prior" if prior_mode == "learned" else f"{prior_mode} prior"


def load_selected_gfas(path: Path, limit: int | None) -> list[Path]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    gfas = [Path(row["gfa"]).resolve() for row in rows if row.get("gfa")]
    if limit is not None and limit > 0:
        gfas = gfas[:limit]
    missing = [path for path in gfas if not path.exists()]
    if missing:
        raise FileNotFoundError(f"selected GFA does not exist: {missing[0]}")
    return gfas


def run_command(
    command: list[str],
    *,
    execute: bool,
    cwd: Path,
    log_path: Path,
    env_updates: dict[str, str] | None = None,
) -> tuple[int, float]:
    printable = shell_join(command)
    print(printable)
    with log_path.open("a") as handle:
        handle.write(f"$ {printable}\n")
    if not execute:
        return 0, 0.0

    started = time.perf_counter()
    env = command_env()
    if env_updates:
        env.update(env_updates)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("a") as handle:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
    code = int(process.wait())
    elapsed = time.perf_counter() - started
    with log_path.open("a") as handle:
        handle.write(f"# exit={code} elapsed_s={elapsed:.3f}\n")
    return code, elapsed


def qubo_neural_command(
    args: argparse.Namespace,
    gfas: list[Path],
    out_csv: Path,
    *,
    budget: int,
    gamma: float,
    seed: int,
    prior_mode: str,
) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "test.py"),
        "--model",
        str(args.model),
        "--out-csv",
        str(out_csv),
        "--time-limit",
        str(budget),
        "--jobs",
        str(args.qubo_jobs),
        "--n_ants",
        str(args.n_ants),
        "--aco-min-iterations",
        str(args.aco_min_iterations),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--aco-gamma",
        str(gamma),
        "--rho",
        str(args.aco_evaporation),
        "--device",
        args.device,
        "--prior-mode",
        prior_mode,
        "--seed",
        str(seed),
        "--parallel-traced" if args.parallel_traced else "--no-parallel-traced",
        "--force",
        "--gfas",
    ]
    command.extend(str(path) for path in gfas)
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    return command


def qubo_baseline_command(args: argparse.Namespace, gfas: list[Path], out_csv: Path, *, budget: int) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "benchmark.py"),
        "--out-csv",
        str(out_csv),
        "--solvers",
        args.baselines,
        "--time-limit",
        str(budget),
        "--jobs",
        str(args.qubo_jobs),
        "--local-jobs",
        str(args.qubo_local_jobs),
        "--n_ants",
        str(args.n_ants),
        "--aco-min-iterations",
        str(args.aco_min_iterations),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--rho",
        str(args.aco_evaporation),
        "--max-expansions",
        str(args.max_expansions),
        "--force",
        "--gfas",
    ]
    command.extend(str(path) for path in gfas)
    return command


def full_assembly_command(args: argparse.Namespace, *, solver: str, budget: int, seed: int, out_prefix: str) -> list[str]:
    command = [
        str(REPO_ROOT / "run_gfa_sim.sh"),
        "--seed",
        str(seed),
        "--config",
        str(REPO_ROOT / f"config_illumina_{args.annotator}.sh"),
        "--annotate",
        args.annotator,
        "--solver",
        solver,
        "--prefix",
        f"{out_prefix}.{args.annotator}.",
        "--times",
        str(budget),
        "--jobs",
        str(args.full_jobs),
        "--training",
        str(args.test_sequences),
    ]
    if solver == "neural_aco":
        command.extend(["--neural-model", str(args.model), "--device", args.device])
    if args.pathfinder_graph:
        command.append("--pathfinder_graph")
    return command


def run_qubo_wallclock(args: argparse.Namespace, gfas: list[Path], log_path: Path, command_rows: list[dict[str, object]]) -> int:
    for budget in args.wallclock_budgets:
        out_csv = args.out_dir / "qubo_wallclock" / f"baselines_budget_{budget:04d}s.csv"
        command = qubo_baseline_command(args, gfas, out_csv, budget=budget)
        code, elapsed = run_command(command, execute=args.execute, cwd=REPO_ROOT, log_path=log_path)
        command_rows.append({"suite": "qubo_wallclock", "budget_s": budget, "command": shell_join(command), "exit_code": code, "elapsed_s": elapsed})
        if code != 0:
            return code
        neural_out_csv = args.out_dir / "qubo_wallclock" / f"neural_budget_{budget:04d}s.csv"
        neural_command = qubo_neural_command(
            args,
            gfas,
            neural_out_csv,
            budget=budget,
            gamma=args.wallclock_neural_gamma,
            seed=args.wallclock_neural_seed,
            prior_mode=args.wallclock_prior_mode,
        )
        code, elapsed = run_command(neural_command, execute=args.execute, cwd=REPO_ROOT, log_path=log_path)
        command_rows.append(
            {
                "suite": "qubo_wallclock_neural",
                "budget_s": budget,
                "seed": args.wallclock_neural_seed,
                "prior_mode": args.wallclock_prior_mode,
                "gamma": args.wallclock_neural_gamma,
                "command": shell_join(neural_command),
                "exit_code": code,
                "elapsed_s": elapsed,
            }
        )
        if code != 0:
            return code
    return 0


def run_qubo_ablation(args: argparse.Namespace, gfas: list[Path], log_path: Path, command_rows: list[dict[str, object]]) -> int:
    for budget in args.ablation_budgets:
        for seed in range(args.ablation_seed_start, args.ablation_seed_start + args.ablation_seed_count):
            for prior_mode in args.prior_modes:
                for gamma in args.gamma_values:
                    gamma_name = gamma_label(gamma)
                    out_csv = (
                        args.out_dir
                        / "qubo_ablation"
                        / f"neural_prior_{prior_mode}_gamma_{gamma_name}_seed_{seed:04d}_budget_{budget:04d}s.csv"
                    )
                    command = qubo_neural_command(
                        args,
                        gfas,
                        out_csv,
                        budget=budget,
                        gamma=gamma,
                        seed=seed,
                        prior_mode=prior_mode,
                    )
                    code, elapsed = run_command(command, execute=args.execute, cwd=REPO_ROOT, log_path=log_path)
                    command_rows.append(
                        {
                            "suite": "qubo_ablation",
                            "budget_s": budget,
                            "seed": seed,
                            "prior_mode": prior_mode,
                            "gamma": gamma,
                            "command": shell_join(command),
                            "exit_code": code,
                            "elapsed_s": elapsed,
                        }
                    )
                    if code != 0:
                        return code
    return 0


def run_full_assembly(args: argparse.Namespace, log_path: Path, command_rows: list[dict[str, object]]) -> int:
    for solver in split_csv(args.full_solvers):
        for budget in args.full_budgets:
            budget_dir = args.out_dir / "full_assembly" / f"{solver}_budget_{budget:04d}s"
            budget_dir.mkdir(parents=True, exist_ok=True)
            for seed in range(args.full_seed_start, args.full_seed_start + args.full_seed_count):
                for repeat in range(args.full_repeat_count):
                    out_prefix = solver if args.full_repeat_count == 1 else f"{solver}_r{repeat + 1:02d}"
                    command = full_assembly_command(args, solver=solver, budget=budget, seed=seed, out_prefix=out_prefix)
                    solver_seed = args.full_solver_seed_start + repeat
                    env_updates = {}
                    if solver == "neural_aco":
                        env_updates = {
                            "QPG_ACO_MODEL": str(args.model),
                            "QPG_ACO_DEVICE": args.device,
                            "QPG_ACO_SEED": str(solver_seed),
                            "QPG_ACO_GAMMA": str(args.full_neural_gamma),
                            "QPG_ACO_PRIOR_MODE": args.full_prior_mode,
                        }
                    code, elapsed = run_command(
                        command,
                        execute=args.execute,
                        cwd=budget_dir,
                        log_path=log_path,
                        env_updates=env_updates,
                    )
                    command_rows.append(
                        {
                            "suite": "full_assembly",
                            "solver": solver,
                            "budget_s": budget,
                            "seed": seed,
                            "repeat": repeat + 1,
                            "solver_seed": solver_seed if solver == "neural_aco" else "",
                            "prior_mode": args.full_prior_mode if solver == "neural_aco" else "",
                            "gamma": args.full_neural_gamma if solver == "neural_aco" else "",
                            "command": shell_join(command),
                            "exit_code": code,
                            "elapsed_s": elapsed,
                        }
                    )
                    if code != 0:
                        return code
    return 0


def summarize_solver_rows(rows: list[dict[str, object]], *, extra_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(name, "") for name in [*extra_keys, "solver"])
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items()):
        oks = ok_rows(group)
        energies = [as_float(row, "energy") for row in oks]
        runtimes = [as_float(row, "runtime_s") for row in oks]
        energies_f = [item for item in energies if item is not None]
        runtimes_f = [item for item in runtimes if item is not None]
        out = {name: value for name, value in zip([*extra_keys, "solver"], key, strict=False)}
        out.update(
            {
                "rows": len(group),
                "ok": len(oks),
                "mean_energy": mean(energies_f) if energies_f else "",
                "mean_runtime_s": mean(runtimes_f) if runtimes_f else "",
            }
        )
        summaries.append(out)
    return summaries


def summarize_qubo_wallclock(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    wallclock_dir = args.out_dir / "qubo_wallclock"
    for path in sorted(wallclock_dir.glob("baselines_budget_*s.csv")):
        match = BASELINE_RE.search(path.name)
        budget = int(match.group("budget")) if match else ""
        for row in read_csv_rows(path):
            row["budget_s"] = budget
            row["source_csv"] = str(path)
            rows.append(row)
    for path in sorted(wallclock_dir.glob("neural_budget_*s.csv")):
        match = re.search(r"neural_budget_(?P<budget>\d+)s\.csv$", path.name)
        budget = int(match.group("budget")) if match else ""
        for row in read_csv_rows(path):
            row["budget_s"] = budget
            row["source_csv"] = str(path)
            rows.append(row)

    existing_dir = args.existing_neural_qubo_dir
    if existing_dir is not None and existing_dir.exists():
        for path in sorted(existing_dir.glob("dynaco_budget_*s.csv")):
            match = NEURAL_QUBO_RE.search(path.name)
            budget = int(match.group("budget")) if match else ""
            for row in read_csv_rows(path):
                row["budget_s"] = budget
                row["source_csv"] = str(path)
                rows.append(row)

    for row in rows:
        row["instance"] = str(row.get("gfa", ""))
        row["budget_s"] = int(row.get("budget_s", 0) or 0)

    best_by_budget_instance: dict[tuple[int, str], float] = {}
    for row in ok_rows(rows):
        energy = as_float(row, "energy")
        if energy is None:
            continue
        key = (int(row["budget_s"]), str(row["instance"]))
        best_by_budget_instance[key] = min(best_by_budget_instance.get(key, energy), energy)

    for row in rows:
        energy = as_float(row, "energy")
        best = best_by_budget_instance.get((int(row["budget_s"]), str(row["instance"])))
        row["gap_to_budget_best"] = "" if energy is None or best is None else energy - best

    raw_path = args.out_dir / "summary_qubo_wallclock_rows.csv"
    if rows:
        write_csv(raw_path, rows)

    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget_s"]), str(row.get("solver", "")))].append(row)

    summary = []
    for (budget, solver), group in sorted(grouped.items()):
        oks = ok_rows(group)
        energies = [as_float(row, "energy") for row in oks]
        runtimes = [as_float(row, "runtime_s") for row in oks]
        gaps = [as_float(row, "gap_to_budget_best") for row in oks]
        wins = sum(1 for row in oks if as_float(row, "gap_to_budget_best") == 0.0)
        energy_values = [x for x in energies if x is not None]
        runtime_values = [x for x in runtimes if x is not None]
        gap_values = [x for x in gaps if x is not None]
        summary.append(
            {
                "budget_s": budget,
                "solver": solver,
                "rows": len(group),
                "ok": len(oks),
                "mean_energy": mean(energy_values) if energy_values else "",
                "mean_gap_to_budget_best": mean(gap_values) if gap_values else "",
                "wins_or_ties": wins,
                "mean_runtime_s": mean(runtime_values) if runtime_values else "",
            }
        )
    if summary:
        write_csv(args.out_dir / "summary_qubo_wallclock.csv", summary)
    return summary


def summarize_qubo_ablation(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ablation_dir = args.out_dir / "qubo_ablation"
    for path in sorted(ablation_dir.glob("neural_*_gamma_*_seed_*_budget_*s.csv")):
        match = GAMMA_RE.search(path.name)
        if match is None:
            continue
        gamma = match.group("gamma").replace("p", ".")
        for row in read_csv_rows(path):
            row["prior_mode"] = match.group("prior_mode") or "legacy"
            row["gamma"] = float(gamma)
            row["control"] = ablation_control_label(str(row["prior_mode"]), float(row["gamma"]))
            row["seed"] = int(match.group("seed"))
            row["budget_s"] = int(match.group("budget"))
            row["source_csv"] = str(path)
            rows.append(row)

    best_by_budget_seed_instance: dict[tuple[int, int, str], float] = {}
    for row in ok_rows(rows):
        energy = as_float(row, "energy")
        if energy is None:
            continue
        key = (int(row["budget_s"]), int(row["seed"]), str(row.get("gfa", "")))
        best_by_budget_seed_instance[key] = min(best_by_budget_seed_instance.get(key, energy), energy)

    for row in rows:
        energy = as_float(row, "energy")
        best = best_by_budget_seed_instance.get((int(row["budget_s"]), int(row["seed"]), str(row.get("gfa", ""))))
        row["gap_to_ablation_best"] = "" if energy is None or best is None else energy - best

    if rows:
        write_csv(args.out_dir / "summary_qubo_ablation_rows.csv", rows)

    grouped: dict[tuple[int, str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget_s"]), str(row["prior_mode"]), float(row["gamma"]), str(row.get("control", "")))].append(row)

    summary = []
    for (budget, prior_mode, gamma, control), group in sorted(grouped.items()):
        oks = ok_rows(group)
        energies = [as_float(row, "energy") for row in oks]
        runtimes = [as_float(row, "runtime_s") for row in oks]
        gaps = [as_float(row, "gap_to_ablation_best") for row in oks]
        wins = sum(1 for row in oks if as_float(row, "gap_to_ablation_best") == 0.0)
        energy_values = [x for x in energies if x is not None]
        runtime_values = [x for x in runtimes if x is not None]
        gap_values = [x for x in gaps if x is not None]
        summary.append(
            {
                "budget_s": budget,
                "prior_mode": prior_mode,
                "gamma": gamma,
                "control": control,
                "rows": len(group),
                "ok": len(oks),
                "mean_energy": mean(energy_values) if energy_values else "",
                "mean_gap_to_ablation_best": mean(gap_values) if gap_values else "",
                "wins_or_ties": wins,
                "mean_runtime_s": mean(runtime_values) if runtime_values else "",
            }
        )
    if summary:
        write_csv(args.out_dir / "summary_qubo_ablation.csv", summary)
    return summary


def parse_eval_file(path: Path) -> dict[str, object] | None:
    name_match = EVAL_NAME_RE.search(path.name)
    if name_match is None:
        return None
    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.match(line)
        if match is None:
            continue
        parts = path.parent.name.split(".")
        if len(parts) < 3:
            return None
        graph = parts[-2]
        seed = parts[-1]
        solver = ".".join(parts[:-2])
        repeat = ""
        repeat_match = re.search(r"_r(?P<repeat>\d+)$", solver)
        if repeat_match is not None:
            repeat = repeat_match.group("repeat")
            solver = solver[: repeat_match.start()]
        return {
            "solver": solver,
            "graph": graph,
            "seed": seed,
            "repeat": repeat,
            "sequence": match.group("seq"),
            "budget_s": int(name_match.group("budget")),
            "job": int(name_match.group("job")),
            "covered": float(match.group("covered")),
            "used": float(match.group("used")),
            "contigs": int(match.group("contigs")),
            "breaks": int(match.group("breaks")),
            "indels": int(match.group("indels")),
            "diffs": int(match.group("diffs")),
            "identity": float(match.group("identity")),
            "source": str(path),
        }
    return None


def summarize_full_assembly(args: argparse.Namespace) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    full_dir = args.out_dir / "full_assembly"
    for path in full_dir.rglob("*.eval_cons.*"):
        row = parse_eval_file(path)
        if row is None:
            continue
        key = (
            str(row["solver"]),
            str(row["graph"]),
            str(row["seed"]),
            str(row.get("repeat", "")),
            str(row["sequence"]),
            int(row["budget_s"]),
        )
        grouped[key].append(row)

    best_rows = []
    for candidates in grouped.values():
        best_rows.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["covered"]),
                    float(row["used"]),
                    -int(row["breaks"]),
                    -int(row["indels"]),
                    -int(row["diffs"]),
                    float(row["identity"]),
                ),
            )
        )
    best_rows.sort(
        key=lambda row: (
            str(row["solver"]),
            str(row["graph"]),
            int(row["budget_s"]),
            str(row["seed"]),
            str(row.get("repeat", "")),
            str(row["sequence"]),
        )
    )
    if best_rows:
        write_csv(args.out_dir / "summary_full_assembly_rows.csv", best_rows)

    by_solver: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        by_solver[(str(row["solver"]), str(row["graph"]), int(row["budget_s"]))].append(row)

    summary = []
    for (solver, graph, budget), group in sorted(by_solver.items()):
        seeds = sorted({str(row["seed"]) for row in group})
        repeats = sorted({str(row.get("repeat") or "1") for row in group})
        summary.append(
            {
                "solver": solver,
                "graph": graph,
                "budget_s": budget,
                "seqs": len(group),
                "seeds": f"{seeds[0]}-{seeds[-1]}" if seeds else "",
                "repeats": len(repeats),
                "repeat_ids": ",".join(repeats),
                "covered": mean(float(row["covered"]) for row in group),
                "used": mean(float(row["used"]) for row in group),
                "contigs": mean(float(row["contigs"]) for row in group),
                "breaks": mean(float(row["breaks"]) for row in group),
                "indels": mean(float(row["indels"]) for row in group),
                "diffs": mean(float(row["diffs"]) for row in group),
                "identity": mean(float(row["identity"]) for row in group),
            }
        )
    if summary:
        write_csv(args.out_dir / "summary_full_assembly.csv", summary)
    return summary


def write_markdown_report(
    args: argparse.Namespace,
    wallclock: list[dict[str, object]],
    ablation: list[dict[str, object]],
    full: list[dict[str, object]],
) -> None:
    lines = [
        "# Next Paper Experiment Summary",
        "",
        f"- Executed commands: `{args.execute}`",
        f"- Output directory: `{args.out_dir}`",
        f"- Selected GFAs: `{args.selected_gfas}`",
        "",
    ]
    if wallclock:
        lines.extend(["## QUBO Wall-Clock Summary", "", "| budget_s | solver | rows | ok | mean_energy | gap | wins | runtime_s |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in wallclock:
            lines.append(
                f"| {row['budget_s']} | {row['solver']} | {row['rows']} | {row['ok']} | "
                f"{compact_float(as_float(row, 'mean_energy'))} | {compact_float(as_float(row, 'mean_gap_to_budget_best'))} | "
                f"{row['wins_or_ties']} | {compact_float(as_float(row, 'mean_runtime_s'))} |"
            )
        lines.append("")
    if ablation:
        lines.extend(["## Learned-Prior Ablation Summary", "", "| budget_s | prior | gamma | control | rows | ok | mean_energy | gap | wins | runtime_s |", "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in ablation:
            lines.append(
                f"| {row['budget_s']} | {row['prior_mode']} | {row['gamma']} | {row.get('control', '')} | {row['rows']} | {row['ok']} | "
                f"{compact_float(as_float(row, 'mean_energy'))} | {compact_float(as_float(row, 'mean_gap_to_ablation_best'))} | "
                f"{row['wins_or_ties']} | {compact_float(as_float(row, 'mean_runtime_s'))} |"
            )
        lines.append("")
    if full:
        lines.extend(["## Full-Assembly Summary", "", "| solver | graph | budget_s | seqs | seeds | repeats | covered | used | contigs | breaks | identity |", "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"])
        for row in full:
            lines.append(
                f"| {row['solver']} | {row['graph']} | {row['budget_s']} | {row['seqs']} | {row['seeds']} | {row['repeats']} | "
                f"{compact_float(as_float(row, 'covered'))} | {compact_float(as_float(row, 'used'))} | "
                f"{compact_float(as_float(row, 'contigs'))} | {compact_float(as_float(row, 'breaks'))} | "
                f"{compact_float(as_float(row, 'identity'))} |"
            )
        lines.append("")
    if not (wallclock or ablation or full):
        lines.append("No measurement rows found yet. Run with `--execute`, or point `--out-dir` at a completed run.")
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def summarize_outputs(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    wallclock = summarize_qubo_wallclock(args)
    ablation = summarize_qubo_ablation(args)
    full = summarize_full_assembly(args)
    write_markdown_report(args, wallclock, ablation, full)
    return wallclock, ablation, full


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run commands. Default writes only plan files.")
    parser.add_argument(
        "--suite",
        choices=["all", "qubo-wallclock", "qubo-ablation", "full-assembly", "summarize"],
        default="all",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "next_paper_experiments" / timestamp)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--selected-gfas", type=Path, default=DEFAULT_SELECTED_GFAS)
    parser.add_argument(
        "--existing-neural-qubo-dir",
        type=Path,
        default=None,
        help="Optional completed neural QUBO directory to merge into summaries. Defaults to current run only.",
    )
    parser.add_argument("--max-gfas", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baselines", default="mqlib,aco,beam_search,local,random_residual_walk")
    parser.add_argument("--wallclock-budgets", type=lambda value: [int(item) for item in split_csv(value)], default=[30, 60, 120, 180])
    parser.add_argument("--wallclock-prior-mode", choices=["learned", "zero", "shuffle", "random"], default="learned")
    parser.add_argument("--wallclock-neural-gamma", type=float, default=1.0)
    parser.add_argument("--wallclock-neural-seed", type=int, default=1)
    parser.add_argument("--ablation-budgets", type=lambda value: [int(item) for item in split_csv(value)], default=[5, 10])
    parser.add_argument("--prior-modes", type=split_csv, default=["learned", "zero", "shuffle", "random"])
    parser.add_argument("--gamma-values", type=lambda value: [float(item) for item in split_csv(value)], default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--ablation-seed-start", type=int, default=1)
    parser.add_argument("--ablation-seed-count", type=int, default=3)
    parser.add_argument("--qubo-jobs", type=int, default=3)
    parser.add_argument("--qubo-local-jobs", type=int, default=1)
    parser.add_argument("--n-ants", dest="n_ants", type=int, default=32)
    parser.add_argument("--aco-min-iterations", type=int, default=100)
    parser.add_argument("--aco-alpha", type=float, default=1.0)
    parser.add_argument("--aco-beta", type=float, default=1.0)
    parser.add_argument("--aco-evaporation", type=float, default=0.1)
    parser.add_argument("--max-expansions", type=int, default=100000)
    parser.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    parser.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--full-solvers", default="mqlib,aco,beam_search,neural_aco")
    parser.add_argument("--full-budgets", type=lambda value: [int(item) for item in split_csv(value)], default=[10])
    parser.add_argument("--full-seed-start", type=int, default=1)
    parser.add_argument("--full-seed-count", type=int, default=8)
    parser.add_argument("--full-repeat-count", type=int, default=1)
    parser.add_argument("--full-solver-seed-start", type=int, default=1)
    parser.add_argument("--full-neural-gamma", type=float, default=1.0)
    parser.add_argument("--full-prior-mode", choices=["learned", "zero", "shuffle", "random"], default="learned")
    parser.add_argument("--full-jobs", type=int, default=1)
    parser.add_argument("--test-sequences", type=int, default=5)
    parser.add_argument("--annotator", default="mg")
    parser.add_argument("--pathfinder-graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.model = args.model.resolve()
    args.selected_gfas = args.selected_gfas.resolve()
    args.existing_neural_qubo_dir = args.existing_neural_qubo_dir.resolve() if args.existing_neural_qubo_dir else None
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "commands.log"
    command_rows: list[dict[str, object]] = []
    gfas = load_selected_gfas(args.selected_gfas, args.max_gfas)
    write_csv(args.out_dir / "selected_gfas.csv", [{"gfa": str(path)} for path in gfas], ["gfa"])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "execute": args.execute,
        "suite": args.suite,
        "model": str(args.model),
        "selected_gfas": str(args.selected_gfas),
        "out_dir": str(args.out_dir),
        "purpose": [
            "wall-clock-oriented QUBO baselines",
            "learned-prior gamma and stochastic ablations",
            "paired full-assembly baselines",
        ],
        "wallclock_neural": {
            "prior_mode": args.wallclock_prior_mode,
            "gamma": args.wallclock_neural_gamma,
            "seed": args.wallclock_neural_seed,
        },
        "existing_neural_qubo_dir": str(args.existing_neural_qubo_dir) if args.existing_neural_qubo_dir else "",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.execute and not args.model.exists():
        print(f"model does not exist: {args.model}", file=sys.stderr)
        return 2

    if args.suite in {"all", "qubo-wallclock"}:
        code = run_qubo_wallclock(args, gfas, log_path, command_rows)
        if code != 0:
            write_csv(args.out_dir / "command_summary.csv", command_rows)
            return code
    if args.suite in {"all", "qubo-ablation"}:
        code = run_qubo_ablation(args, gfas, log_path, command_rows)
        if code != 0:
            write_csv(args.out_dir / "command_summary.csv", command_rows)
            return code
    if args.suite in {"all", "full-assembly"}:
        code = run_full_assembly(args, log_path, command_rows)
        if code != 0:
            write_csv(args.out_dir / "command_summary.csv", command_rows)
            return code

    write_csv(args.out_dir / "command_summary.csv", command_rows)
    _wallclock_summary, _ablation_summary, full_summary = summarize_outputs(args)
    if args.execute and args.suite in {"all", "full-assembly"} and split_csv(args.full_solvers) and not full_summary:
        print(
            "full-assembly suite produced no parsed consensus rows; inspect commands.log and sim.err files.",
            file=sys.stderr,
        )
        return 3
    print(f"wrote plan: {args.out_dir}")
    if not args.execute:
        print("dry-run only; add --execute to run measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
