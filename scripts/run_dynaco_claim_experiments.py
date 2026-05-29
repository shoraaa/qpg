#!/usr/bin/env python3
"""Run DyNACO-first experiments for runtime/scale and budgeted-inference claims.

The default plan is intentionally not a generic leaderboard rerun:

* full-budget: run only DyNACO/neural_aco through the full assembly pipeline on
  the retained minigraph annotation route, across several solver budgets.
* qubo-scale: run DyNACO on larger or user-supplied GFA instances and summarize
  runtime by graph size; MQLib is optional and off by default.

By default this script writes a manifest and prints commands without running
them. Add --execute to spend compute.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import glob
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MODEL = (
    REPO_ROOT
    / "results"
    / "overnight_dynaco_paper"
    / "20260522_013319"
    / "dynaco_overnight_best.pt"
)
DEFAULT_MQLIB_CACHE = (
    REPO_ROOT
    / "results"
    / "overnight_dynaco_paper"
    / "20260522_013319"
    / "analytics"
    / "partial_full_assembly_best_consensus.csv"
)
DEFAULT_GFA_GLOBS = [
    "results/overnight_dynaco_paper/*/generated/paper_pipeline_cache/train/mg.*/*.gfa",
    "results/overnight_dynaco_paper/*/generated/paper_pipeline_cache/val/mg.*/*.gfa",
    "results/overnight_dynaco_paper/*/generated/paper_pipeline_cache/test/mg.*/*.gfa",
    "results/dynaco_online/generated_larger/test/*.gfa",
    "results/dynaco_online/paper_pipeline_online_cache/**/*.gfa",
    "examples/*.gfa",
]

EVAL_RE = re.compile(r"^(?P<seq>\S+)\s+\d+\s+\d+\s+(?P<covered>[\d.]+)%\s+(?P<used>[\d.]+)%\s+"
                     r"(?P<contigs>\d+)\s+(?P<breaks>\d+)\s+(?P<indels>\d+)\s+"
                     r"(?P<diffs>\d+)\s+(?P<identity>[\d.]+)%")
EVAL_NAME_RE = re.compile(r"(?P<seq>.+)\.eval_cons\.(?P<budget>\d+)\.(?P<job>\d+)$")
QUBO_BUDGET_RE = re.compile(r"_budget_(?P<budget>\d+)s\.csv$")
DEFAULT_QUBO_BASELINES = "aco,beam_search,astar,local"


def python_executable() -> str:
    return str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else str(Path(sys.executable).resolve())


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text.rstrip() + "\n")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, execute: bool) -> tuple[int, float]:
    printable = shell_join(command)
    print(printable)
    append_log(log_path, f"$ {printable}")
    if not execute:
        return 0, 0.0

    started = time.perf_counter()
    child_env = dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = int(process.wait())
    elapsed = time.perf_counter() - started
    append_log(log_path, f"# exit={code} elapsed_s={elapsed:.3f}")
    return code, elapsed


def make_env(args: argparse.Namespace, out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["QDIR"] = str(REPO_ROOT)
    env["PYTHON"] = python_executable()
    env["PYTHONPATH"] = str(REPO_ROOT / "qubo") + os.pathsep + env.get("PYTHONPATH", "")
    if DEFAULT_PYTHON.exists():
        env["VIRTUAL_ENV"] = str((REPO_ROOT / ".venv").resolve())
    env["QPG_REPRO_OUT"] = str(out_dir)
    env["QPG_ACO_MODEL"] = str(args.model.resolve())
    env["QPG_SEEA_MODEL"] = str(args.model.resolve())
    env["QPG_ACO_DEVICE"] = args.device
    env["QPG_SEEA_DEVICE"] = args.device
    env["QPG_ACO_ANTS"] = str(args.n_ants)
    env["QPG_ACO_MIN_ITERATIONS"] = str(args.aco_min_iterations)
    env["QPG_ACO_ALPHA"] = str(args.aco_alpha)
    env["QPG_ACO_BETA"] = str(args.aco_beta)
    env["QPG_ACO_EVAPORATION"] = str(args.aco_evaporation)
    env["QPG_ACO_GAMMA"] = str(args.aco_gamma)
    env["QPG_ACO_PARALLEL_TRACED"] = "1" if args.parallel_traced else "0"
    env["SHRED_DEPTH"] = str(args.shred_depth)
    env["SHUF_RANDOM_SOURCE"] = args.shuf_random_source
    env["PATHFINDER"] = str(REPO_ROOT / ".tools" / "bin" / "pathfinder")
    env["BWA"] = str(REPO_ROOT / ".tools" / "bwa" / "bwa")
    if args.threads is not None:
        env["QPG_ACO_THREADS"] = str(args.threads)

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
    env["PATH"] = os.pathsep.join(str(path) for path in tool_paths) + os.pathsep + env.get("PATH", "")
    htslib = REPO_ROOT / ".tools" / "htslib" / "build" / "lib"
    env["LD_LIBRARY_PATH"] = str(htslib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def count_segments(gfa: Path) -> int:
    with gfa.open(errors="replace") as handle:
        return sum(1 for line in handle if line.startswith("S\t"))


def collect_gfas(
    patterns: list[str],
    *,
    max_gfas: int | None,
    min_segments: int,
    max_segments: int | None,
    largest_first: bool,
) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        search_pattern = pattern if Path(pattern).is_absolute() else str(REPO_ROOT / pattern)
        matches = sorted(Path(match).resolve() for match in glob.glob(search_pattern, recursive=True) if Path(match).is_file())
        paths.extend(matches)

    unique: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.exists():
            continue
        if path.name == "pop.gfa":
            continue
        seen.add(path)
        segments = count_segments(path)
        if segments >= min_segments and (max_segments is None or segments <= max_segments):
            unique.append((path, segments))

    unique.sort(key=lambda item: (item[1], str(item[0])), reverse=largest_first)
    if max_gfas is not None:
        unique = unique[:max_gfas]
    return [path for path, _segments in unique]


def full_budget_command(args: argparse.Namespace, budget: int, annotator: str, seed: int) -> list[str]:
    command = [
        str(REPO_ROOT / "run_gfa_sim.sh"),
        "--seed",
        str(seed),
        "--config",
        str(REPO_ROOT / f"config_illumina_{annotator}.sh"),
        "--annotate",
        annotator,
        "--solver",
        "neural_aco",
        "--prefix",
        f"neural_aco.{annotator}.",
        "--times",
        str(budget),
        "--jobs",
        str(args.full_jobs),
        "--training",
        str(args.test_sequences),
        "--neural-model",
        str(args.model.resolve()),
        "--device",
        args.device,
    ]
    if args.pathfinder_graph:
        command.append("--pathfinder_graph")
    return command


def parse_eval_file(path: Path) -> dict[str, object] | None:
    name_match = EVAL_NAME_RE.search(path.name)
    if name_match is None:
        return None
    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.match(line)
        if match is None:
            continue
        run_dir = path.parent.name.split(".")
        if len(run_dir) < 3:
            return None
        return {
            "solver": run_dir[0],
            "graph": run_dir[1],
            "seed": run_dir[2],
            "sequence": match.group("seq"),
            "budget": int(name_match.group("budget")),
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


def best_consensus_rows(full_dir: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for path in full_dir.rglob("*.eval_cons.*"):
        row = parse_eval_file(path)
        if row is not None:
            key = (str(row["solver"]), str(row["graph"]), str(row["seed"]), str(row["sequence"]), int(row["budget"]))
            grouped[key].append(row)

    best = []
    for rows in grouped.values():
        best.append(
            max(
                rows,
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
    return sorted(best, key=lambda row: (row["solver"], row["graph"], int(row["budget"]), row["seed"], row["sequence"]))


def summarize_best_consensus(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["solver"]), str(row["graph"]), int(row["budget"]))].append(row)

    summary = []
    for (solver, graph, budget), group in sorted(grouped.items()):
        seeds = sorted({str(row["seed"]) for row in group})
        summary.append(
            {
                "solver": solver,
                "graph": graph,
                "budget_s": budget,
                "seqs": len(group),
                "seeds": f"{seeds[0]}-{seeds[-1]}" if seeds else "",
                "covered": mean(float(row["covered"]) for row in group),
                "used": mean(float(row["used"]) for row in group),
                "contigs": mean(float(row["contigs"]) for row in group),
                "breaks": mean(float(row["breaks"]) for row in group),
                "indels": mean(float(row["indels"]) for row in group),
                "diffs": mean(float(row["diffs"]) for row in group),
                "identity": mean(float(row["identity"]) for row in group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_full_budget_suite(args: argparse.Namespace, env: dict[str, str], out_dir: Path, log_path: Path) -> int:
    suite_dir = out_dir / "full_budget"
    command_rows = []
    if args.pathfinder_graph and not Path(env["PATHFINDER"]).exists():
        print(f"Pathfinder preprocessing requested but PATHFINDER does not exist: {env['PATHFINDER']}", file=sys.stderr)
        return 2
    for budget in args.budgets:
        budget_dir = suite_dir / f"budget_{budget:04d}s"
        budget_dir.mkdir(parents=True, exist_ok=True)
        for annotator in split_csv(args.annotators):
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                command = full_budget_command(args, budget, annotator, seed)
                code, elapsed = run_command(command, cwd=budget_dir, env=env, log_path=log_path, execute=args.execute)
                command_rows.append(
                    {
                        "suite": "full_budget",
                        "budget_s": budget,
                        "annotator": annotator,
                        "seed": seed,
                        "exit_code": code,
                        "elapsed_s": elapsed,
                        "cwd": str(budget_dir),
                        "command": shell_join(command),
                    }
                )
                if code != 0:
                    write_csv(suite_dir / "command_times.csv", command_rows, list(command_rows[0]))
                    return code

    if command_rows:
        write_csv(suite_dir / "command_times.csv", command_rows, list(command_rows[0]))
    best_rows = best_consensus_rows(suite_dir)
    if best_rows:
        write_csv(suite_dir / "best_consensus_rows.csv", best_rows, list(best_rows[0]))
        summary = summarize_best_consensus(best_rows)
        write_csv(suite_dir / "best_consensus_summary.csv", summary, list(summary[0]))
    elif args.execute:
        print(
            "Full-budget suite produced no parsed consensus rows; inspect sim.err files before using this run.",
            file=sys.stderr,
        )
        return 3
    return 0


def qubo_command(args: argparse.Namespace, budget: int, gfas: list[Path], solver_kind: str, out_csv: Path) -> list[str]:
    if solver_kind == "dynaco":
        command = [
            python_executable(),
            str(REPO_ROOT / "test.py"),
            "--model",
            str(args.model.resolve()),
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
            str(args.aco_gamma),
            "--rho",
            str(args.aco_evaporation),
            "--device",
            args.device,
            "--parallel-traced" if args.parallel_traced else "--no-parallel-traced",
            "--force",
            "--gfas",
        ]
        command.extend(str(path) for path in gfas)
        if args.threads is not None:
            command.extend(["--threads", str(args.threads)])
        return command

    command = [
        python_executable(),
        str(REPO_ROOT / "benchmark.py"),
        "--out-csv",
        str(out_csv),
        "--solvers",
        solver_kind,
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


def summarize_qubo_scale(
    scale_dir: Path,
    command_rows: list[dict[str, object]],
    out_csv: Path,
    *,
    bucket_width: int,
) -> list[dict[str, object]]:
    rows = []
    for path in sorted(scale_dir.glob("*.csv")):
        if path.name in {"command_times.csv", "scale_summary.csv", "selected_gfas.csv"}:
            continue
        budget_match = QUBO_BUDGET_RE.search(path.name)
        budget_s = int(budget_match.group("budget")) if budget_match else ""
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row["segments"] = int(row.get("segments", "0") or 0)
                    row["runtime_s"] = float(row.get("runtime_s", "0") or 0)
                    row["energy"] = float(row.get("energy", "nan"))
                except ValueError:
                    continue
                row["budget_s"] = budget_s
                row["source_csv"] = str(path)
                rows.append(row)

    best_by_instance: dict[tuple[object, str], float] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["budget_s"], str(row.get("gfa", "")))
        energy = float(row["energy"])
        best_by_instance[key] = min(best_by_instance.get(key, energy), energy)

    grouped: dict[tuple[str, object, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = f"{(int(row['segments']) // bucket_width) * bucket_width:04d}+"
        grouped[(str(row.get("solver", "")), row["budget_s"], bucket)].append(row)

    summary = []
    for (solver, budget_s, bucket), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        ok = [row for row in group if row.get("status") == "ok"]
        gaps = []
        wins_or_ties = 0
        for row in ok:
            best = best_by_instance.get((row["budget_s"], str(row.get("gfa", ""))))
            if best is None:
                continue
            gap = float(row["energy"]) - best
            gaps.append(gap)
            if abs(gap) <= 1e-9:
                wins_or_ties += 1
        summary.append(
            {
                "solver": solver,
                "budget_s": budget_s,
                "segment_bucket": bucket,
                "rows": len(group),
                "ok": len(ok),
                "mean_segments": mean(int(row["segments"]) for row in ok) if ok else "",
                "mean_runtime_s": mean(float(row["runtime_s"]) for row in ok) if ok else "",
                "mean_energy": mean(float(row["energy"]) for row in ok) if ok else "",
                "mean_gap_to_best": mean(gaps) if gaps else "",
                "wins_or_ties": wins_or_ties,
            }
        )

    if summary:
        write_csv(out_csv, summary, list(summary[0]))
    if command_rows:
        write_csv(scale_dir / "command_times.csv", command_rows, list(command_rows[0]))
    return summary


def run_qubo_scale_suite(args: argparse.Namespace, env: dict[str, str], out_dir: Path, log_path: Path) -> int:
    scale_dir = out_dir / "qubo_scale"
    scale_dir.mkdir(parents=True, exist_ok=True)
    gfas = collect_gfas(
        args.gfa_globs,
        max_gfas=args.max_gfas,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        largest_first=True,
    )
    gfa_rows = [{"gfa": str(path), "segments": count_segments(path)} for path in gfas]
    if gfa_rows:
        write_csv(scale_dir / "selected_gfas.csv", gfa_rows, ["gfa", "segments"])
    if not gfas:
        print("No GFA files matched --gfa-glob/--min-segments/--max-segments for qubo-scale.", file=sys.stderr)
        return 2 if args.execute else 0

    baseline_solvers = split_csv(args.qubo_baselines)
    if args.run_mqlib and "mqlib" not in baseline_solvers:
        baseline_solvers.insert(0, "mqlib")
    baseline_solvers = [solver for solver in baseline_solvers if solver not in {"", "none", "neural_aco"}]

    command_rows = []
    for budget in args.budgets:
        solver_jobs = [("dynaco", "neural_aco")]
        if baseline_solvers:
            solver_jobs.append((",".join(baseline_solvers), ",".join(baseline_solvers)))
        for solver_kind, solver_label in solver_jobs:
            out_label = "dynaco" if solver_kind == "dynaco" else "baselines"
            out_csv = scale_dir / f"{out_label}_budget_{budget:04d}s.csv"
            command = qubo_command(args, budget, gfas, solver_kind, out_csv)
            code, elapsed = run_command(command, cwd=REPO_ROOT, env=env, log_path=log_path, execute=args.execute)
            command_rows.append(
                {
                    "suite": "qubo_scale",
                    "budget_s": budget,
                    "solver": solver_label,
                    "gfas": len(gfas),
                    "exit_code": code,
                    "elapsed_s": elapsed,
                    "command": shell_join(command),
                }
            )
            if code != 0:
                write_csv(scale_dir / "command_times.csv", command_rows, list(command_rows[0]))
                return code

    summarize_qubo_scale(scale_dir, command_rows, scale_dir / "scale_summary.csv", bucket_width=args.segment_bucket_width)
    return 0


def load_mqlib_cache(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("solver") == "mqlib" and row.get("graph") == "mg"]


def write_report(args: argparse.Namespace, out_dir: Path) -> None:
    lines = [
        "# DyNACO Claim Experiments",
        "",
        "This run is scoped to DyNACO-first evidence for two possible claims:",
        "",
        "- Budgeted inference: full assembly quality as the neural_aco budget changes on minigraph-annotated graphs.",
        "- Runtime/scale: QUBO-stage runtime and energy as graph size grows, compared with configured baseline solvers.",
        "",
        f"- Executed commands: `{args.execute}`",
        f"- Model: `{args.model}`",
        f"- Annotators: `{args.annotators}`",
        f"- Budgets: `{','.join(str(item) for item in args.budgets)}` seconds",
        f"- MQLib cache: `{args.mqlib_cache}`",
        "",
    ]

    mqlib_rows = load_mqlib_cache(args.mqlib_cache)
    if mqlib_rows:
        lines.extend(
            [
                "## Cached MQLib Target",
                "",
                "| graph | seqs | seeds | covered | used | contigs | breaks | identity |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mqlib_rows:
            lines.append(
                f"| {row.get('graph', '')} | {row.get('seqs', '')} | {row.get('seeds', '')} | "
                f"{row.get('covered', '')} | {row.get('used', '')} | {row.get('contigs', '')} | "
                f"{row.get('breaks', '')} | {row.get('identity', '')} |"
            )
        lines.append("")

    full_summary = out_dir / "full_budget" / "best_consensus_summary.csv"
    if full_summary.exists():
        lines.extend(["## Full-Budget Summary", ""])
        with full_summary.open() as handle:
            lines.extend("    " + line.rstrip() for line in handle)
        lines.append("")

    scale_summary = out_dir / "qubo_scale" / "scale_summary.csv"
    if scale_summary.exists():
        lines.extend(["## QUBO-Scale Summary", ""])
        with scale_summary.open() as handle:
            lines.extend("    " + line.rstrip() for line in handle)
        lines.append("")

    if not args.execute:
        lines.append("Dry run only. Re-run with `--execute` to create measurement rows.")
    out_dir.joinpath("CLAIM_REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run commands. Default only writes the plan and manifest.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "dynaco_claims" / timestamp)
    parser.add_argument("--suite", choices=["both", "full-budget", "qubo-scale"], default="both")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="DyNACO checkpoint for neural_aco.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--annotators", default="mg", help="Comma-separated graph annotations. Keep this mg for the current paper picture.")
    parser.add_argument("--budgets", type=lambda value: [int(item) for item in split_csv(value)], default=[5, 10, 30, 100])
    parser.add_argument("--seeds", type=int, default=8, help="Full-assembly synthetic seeds to run.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--test-sequences", type=int, default=5)
    parser.add_argument("--full-jobs", type=int, default=3)
    parser.add_argument("--pathfinder-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shred-depth", type=int, default=30)
    parser.add_argument("--shuf-random-source", default="/usr/bin/emacs")
    parser.add_argument("--qubo-jobs", type=int, default=3)
    parser.add_argument("--qubo-local-jobs", type=int, default=1)
    parser.add_argument(
        "--qubo-baselines",
        default=DEFAULT_QUBO_BASELINES,
        help=(
            "Comma-separated benchmark.py baselines for the QUBO-scale suite. "
            "Use 'none' for DyNACO only. Default: "
            f"{DEFAULT_QUBO_BASELINES}."
        ),
    )
    parser.add_argument(
        "--gfa-glob",
        dest="gfa_globs",
        action="append",
        default=None,
        help=(
            "Glob for QUBO-scale GFAs. Repeat to provide multiple globs. "
            "When set, explicit globs replace the built-in defaults."
        ),
    )
    parser.add_argument("--max-gfas", type=int, default=24)
    parser.add_argument("--min-segments", type=int, default=30)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=100,
        help=(
            "Largest GFA segment count to include in qubo-scale. The current QUBO builder uses "
            "dense matrices, so very large minigraph queries can exhaust 32GB RAM. Use 0 to disable."
        ),
    )
    parser.add_argument("--segment-bucket-width", type=int, default=50)
    parser.add_argument("--run-mqlib", action="store_true", help="Also run MQLib in the QUBO-scale suite. Off by default.")
    parser.add_argument("--mqlib-cache", type=Path, default=DEFAULT_MQLIB_CACHE)
    parser.add_argument("--n_ants", type=int, default=32)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--mini_H", type=int, default=10)
    parser.add_argument("--aco-min-iterations", type=int, default=None)
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", type=float, default=1.0)
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", type=float, default=1.0)
    parser.add_argument("--rho", "--aco-evaporation", dest="aco_evaporation", type=float, default=0.1)
    parser.add_argument("--gamma", "--aco-gamma", dest="aco_gamma", type=float, default=1.0)
    parser.add_argument("--max-expansions", type=int, default=100000, help="Expansion cap for astar in QUBO-scale baselines.")
    traced = parser.add_mutually_exclusive_group()
    traced.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    traced.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for the C++ ACO backend.")
    args = parser.parse_args()
    if args.gfa_globs is None:
        args.gfa_globs = list(DEFAULT_GFA_GLOBS)
    if args.aco_min_iterations is None:
        args.aco_min_iterations = args.H * args.mini_H
    if args.max_segments is not None and args.max_segments <= 0:
        args.max_segments = None
    args.out_dir = args.out_dir.resolve()
    args.model = args.model.resolve()
    args.mqlib_cache = args.mqlib_cache.resolve()
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "commands.log"
    env = make_env(args, args.out_dir)
    tool_status = {
        tool: shutil.which(tool, path=env.get("PATH"))
        for tool in ["parallel", "minigraph", "GraphAligner", "bwa", "minimap2", "samtools", "kmer2node4", "MQLib"]
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "out_dir": str(args.out_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "tool_status": tool_status,
        "environment_overrides": {
            "PATHFINDER": env.get("PATHFINDER", ""),
            "BWA": env.get("BWA", ""),
            "QPG_ACO_MODEL": env.get("QPG_ACO_MODEL", ""),
            "QPG_ACO_DEVICE": env.get("QPG_ACO_DEVICE", ""),
        },
        "policy": {
            "active_annotators": split_csv(args.annotators),
            "default_solver": "neural_aco",
            "mqlib": "cached full-stage target by default; include in qubo-scale with --run-mqlib",
            "qubo_baselines": split_csv(args.qubo_baselines),
            "excluded": ["non-minigraph annotation routes", "pathfinder baseline rows"],
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)

    if args.execute and not args.model.exists():
        print(f"DyNACO model does not exist: {args.model}", file=sys.stderr)
        return 2

    if args.suite in {"both", "full-budget"}:
        code = run_full_budget_suite(args, env, args.out_dir, log_path)
        if code != 0:
            return code

    if args.suite in {"both", "qubo-scale"}:
        code = run_qubo_scale_suite(args, env, args.out_dir, log_path)
        if code != 0:
            return code

    write_report(args, args.out_dir)
    print(f"wrote report: {args.out_dir / 'CLAIM_REPORT.md'}")
    if not args.execute:
        print("dry-run only; add --execute to run measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
