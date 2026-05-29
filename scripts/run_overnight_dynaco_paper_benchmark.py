#!/usr/bin/env python3
"""Run an overnight DyNACO-vs-baseline QPG benchmark.

The run has two evaluation stages:

1. QUBO-scope benchmark on held-out GFA instances.
2. Full synthetic assembly benchmark on the original paper-shaped dataset.

The full assembly stage defaults to the original paper's 20 seeds x 5 held-out
sequences x three annotation strategies, but uses a 100 second solver budget,
one third of the paper's 300 second QUBO budget.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUBO_BASELINES = "mqlib,gurobi,beam_search,aco,astar"
DEFAULT_FULL_SOLVERS = "pathfinder,mqlib,aco,neural_aco"
ORIGINAL_PAPER_SECONDS = 300
DEFAULT_FULL_TIME_LIMIT = ORIGINAL_PAPER_SECONDS // 3
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_DYNACO_CONFIG = REPO_ROOT / "configs" / "dynaco_paper_pipeline_mg.yaml"

# Match references/DyNACO defaults where this QPG runner has equivalent knobs.
DYNACO_REFERENCE_EPOCHS = 10
DYNACO_REFERENCE_STEPS_PER_EPOCH = 16
DYNACO_REFERENCE_H = 10
DYNACO_REFERENCE_MINI_H = 10
DYNACO_REFERENCE_ANTS = 32
DYNACO_REFERENCE_ALPHA = 1.0
DYNACO_REFERENCE_BETA = 1.0
DYNACO_REFERENCE_RHO = 0.1
DYNACO_REFERENCE_GAMMA = 1.0


def python_executable() -> str:
    return str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else str(Path(sys.executable).resolve())


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, execute: bool) -> int:
    printable = shell_join(command)
    print(printable)
    append_log(log_path, f"$ {printable}")
    if not execute:
        return 0
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
        return_code = process.wait()
    append_log(log_path, f"# exit={return_code} elapsed_s={time.perf_counter() - started:.3f}")
    return int(return_code)


def make_env(args, out_dir: Path, model_path: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["QDIR"] = str(REPO_ROOT)
    env["PYTHONPATH"] = str(REPO_ROOT / "qubo") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHON"] = python_executable()
    if DEFAULT_PYTHON.exists():
        env["VIRTUAL_ENV"] = str((REPO_ROOT / ".venv").resolve())
    env["SHUF_RANDOM_SOURCE"] = args.shuf_random_source
    env["SHRED_DEPTH"] = str(args.shred_depth)
    env["QPG_REPRO_OUT"] = str(out_dir)
    if model_path is not None:
        env["QPG_ACO_MODEL"] = str(model_path.resolve())
        env["QPG_SEEA_MODEL"] = str(model_path.resolve())
    env["QPG_ACO_DEVICE"] = args.device
    env["QPG_SEEA_DEVICE"] = args.device
    env["QPG_ACO_ANTS"] = str(args.eval_ants)
    env["QPG_ACO_MIN_ITERATIONS"] = str(args.eval_min_iterations)
    env["QPG_ACO_ALPHA"] = str(args.aco_alpha)
    env["QPG_ACO_BETA"] = str(args.aco_beta)
    env["QPG_ACO_EVAPORATION"] = str(args.aco_evaporation)
    env["QPG_ACO_GAMMA"] = str(args.aco_gamma)
    env["QPG_ACO_PARALLEL_TRACED"] = "1" if args.parallel_traced else "0"
    if args.threads is not None:
        env["QPG_ACO_THREADS"] = str(args.threads)
    env["QPG_ASTAR_MAX_EXPANSIONS"] = str(args.max_expansions)
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


def tool_status(env: dict[str, str]) -> dict[str, str | None]:
    tools = ["parallel", "minigraph", "GraphAligner", "bwa", "minimap2", "samtools", "kmer2node4", "pathfinder"]
    return {tool: shutil.which(tool, path=env.get("PATH")) for tool in tools}


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
        return False, "installed pathfinder rejects QPG-era --X50 option"
    return True, "available"


def train_command(args, model_path: Path, generated_dir: Path) -> list[str]:
    if args.smoke:
        return [
            python_executable(),
            str(REPO_ROOT / "train.py"),
            "--synthetic-dir",
            str(generated_dir),
            "--out",
            str(model_path),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "1",
            "--H",
            "1",
            "--mini_H",
            "1",
            "--n_ants",
            "8",
            "--eval-time-limit",
            "1",
            "--device",
            "cpu",
            "--parallel-traced",
            "--generate-synthetic-train",
            "2",
            "--generate-synthetic-test",
            "1",
            "--synthetic-min-segments",
            "4",
            "--synthetic-max-segments",
            "6",
        ]
    command = [
        python_executable(),
        str(REPO_ROOT / "train.py"),
        "--config",
        str(args.config),
        "--out",
        str(model_path),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
        "--n_ants",
        str(args.train_ants),
        "--eval-time-limit",
        str(args.qubo_time_limit),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--rho",
        str(args.aco_evaporation),
        "--gamma",
        str(args.aco_gamma),
        "--device",
        args.device,
        "--parallel-traced" if args.parallel_traced else "--no-parallel-traced",
    ]
    if uses_paper_pipeline_config(args.config):
        command.extend(["--paper-pipeline-cache-dir", str(generated_dir / "paper_pipeline_cache")])
    else:
        command.extend(["--synthetic-dir", str(generated_dir)])
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    return command


def trained_validation_gfas(model_path: Path) -> list[str]:
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return []
    return [str(path) for path in metadata.get("test_gfas", []) if Path(path).exists()]


def benchmark_command(args, model_path: Path, out_csv: Path) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "benchmark.py"),
        "--out-csv",
        str(out_csv),
        "--solvers",
        args.qubo_baselines,
        "--time-limit",
        str(args.qubo_time_limit),
        "--jobs",
        str(args.qubo_jobs),
        "--local-jobs",
        str(args.qubo_local_jobs),
        "--n_ants",
        str(args.eval_ants),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--rho",
        str(args.aco_evaporation),
        "--max-expansions",
        str(args.max_expansions),
        "--force",
    ]
    validation_gfas = trained_validation_gfas(model_path)
    if validation_gfas:
        command.extend(["--gfas", *validation_gfas])
    else:
        command.extend(["--config", str(args.config)])
    if args.smoke:
        command = [
            python_executable(),
            str(REPO_ROOT / "benchmark.py"),
            "--gfas",
            str(args.out_dir / "generated" / "test" / "test_0000.gfa"),
            "--out-csv",
            str(out_csv),
            "--solvers",
            args.qubo_baselines,
            "--time-limit",
            "1",
            "--jobs",
            "1",
            "--local-jobs",
            "1",
            "--n_ants",
            "8",
            "--H",
            "1",
            "--mini_H",
            "1",
            "--force",
        ]
    return command


def dynaco_test_command(args, model_path: Path, out_csv: Path) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "test.py"),
        "--model",
        str(model_path),
        "--out-csv",
        str(out_csv),
        "--time-limit",
        str(args.qubo_time_limit),
        "--jobs",
        str(args.qubo_jobs),
        "--n_ants",
        str(args.eval_ants),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
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
    ]
    validation_gfas = trained_validation_gfas(model_path)
    if validation_gfas:
        command.extend(["--gfas", *validation_gfas])
    else:
        command.extend(["--config", str(args.config)])
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if args.smoke:
        command = [
            python_executable(),
            str(REPO_ROOT / "test.py"),
            "--gfas",
            str(args.out_dir / "generated" / "test" / "test_0000.gfa"),
            "--model",
            str(model_path),
            "--out-csv",
            str(out_csv),
            "--time-limit",
            "1",
            "--jobs",
            "1",
            "--n_ants",
            "8",
            "--H",
            "1",
            "--mini_H",
            "1",
            "--device",
            "cpu",
            "--parallel-traced",
            "--force",
        ]
    return command


def validation_selected_checkpoint(model_path: Path) -> Path:
    candidate = model_path.with_name(f"{model_path.stem}_val_best{model_path.suffix}")
    return candidate if candidate.exists() else model_path


def candidate_checkpoints(model_path: Path) -> list[Path]:
    candidates = []
    candidates.extend(sorted(model_path.parent.glob(f"{model_path.stem}_epoch*{model_path.suffix}")))
    candidates.extend(
        [
            model_path.with_name(f"{model_path.stem}_val_best{model_path.suffix}"),
            model_path.with_name(f"{model_path.stem}_best{model_path.suffix}"),
            model_path,
        ]
    )
    seen = set()
    existing = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        existing.append(candidate)
    return existing


def full_pipeline_command(args, model_path: Path, out_dir: Path, solvers: str) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "scripts" / "reproduce_paper_results.py"),
        "--out-dir",
        str(out_dir),
        "--seeds",
        str(args.full_seeds),
        "--test-sequences",
        str(args.full_test_sequences),
        "--annotators",
        args.full_annotators,
        "--jobs",
        str(args.full_jobs),
        "--time-limits",
        args.full_time_limits,
        "--solvers",
        solvers,
        "--neural-model",
        str(model_path),
        "--device",
        args.device,
        "--n_ants",
        str(args.eval_ants),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--rho",
        str(args.aco_evaporation),
        "--max-expansions",
        str(args.max_expansions),
        "--shred-depth",
        str(args.shred_depth),
    ]
    command.append("--pathfinder-graph" if args.pathfinder_graph else "--no-pathfinder-graph")
    if args.run_full:
        command.append("--run")
    return command


def coverage_selection_command(args, model_path: Path, out_dir: Path) -> list[str]:
    command = [
        python_executable(),
        str(REPO_ROOT / "scripts" / "reproduce_paper_results.py"),
        "--out-dir",
        str(out_dir),
        "--seeds",
        str(args.coverage_selection_seeds),
        "--test-sequences",
        str(args.coverage_selection_test_sequences),
        "--annotators",
        args.full_annotators,
        "--jobs",
        str(args.coverage_selection_jobs),
        "--time-limits",
        args.coverage_selection_time_limits,
        "--solvers",
        "neural_aco",
        "--neural-model",
        str(model_path),
        "--device",
        args.device,
        "--n_ants",
        str(args.eval_ants),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
        "--alpha",
        str(args.aco_alpha),
        "--beta",
        str(args.aco_beta),
        "--rho",
        str(args.aco_evaporation),
        "--max-expansions",
        str(args.max_expansions),
        "--shred-depth",
        str(args.shred_depth),
    ]
    command.append("--pathfinder-graph" if args.pathfinder_graph else "--no-pathfinder-graph")
    command.append("--run")
    return command


def best_covered_metric(full_rows: list[dict[str, object]]) -> float | None:
    covered = [
        float(row["mean"])
        for row in full_rows
        if str(row.get("solver")) == "neural_aco"
        and str(row.get("metric")) == "covered"
    ]
    return max(covered) if covered else None


def coverage_select_checkpoint(args, model_path: Path, env: dict[str, str], log_path: Path, manifest: dict) -> Path:
    candidates = candidate_checkpoints(model_path)
    if not candidates:
        if not args.execute:
            candidates = [model_path]
        else:
            print(f"warning: no checkpoint candidates found for coverage selection; using {model_path}")
            return model_path
    rows = []
    best_path = model_path
    best_covered = float("-inf")
    selection_root = args.out_dir / "coverage_selection"
    for index, candidate in enumerate(candidates, start=1):
        candidate_dir = selection_root / f"{index:03d}_{candidate.stem}"
        code = run_command(
            coverage_selection_command(args, candidate, candidate_dir),
            cwd=REPO_ROOT,
            env=make_env(args, args.out_dir, candidate),
            log_path=log_path,
            execute=args.execute,
        )
        if code != 0:
            rows.append({"checkpoint": str(candidate), "covered": "", "status": f"exit_{code}", "out_dir": str(candidate_dir)})
            continue
        full_rows = summarize_full(candidate_dir, candidate_dir / "full_assembly_summary.csv")
        covered = best_covered_metric(full_rows)
        status = "ok" if covered is not None else "no_covered_metric"
        rows.append({"checkpoint": str(candidate), "covered": "" if covered is None else covered, "status": status, "out_dir": str(candidate_dir)})
        if covered is not None and covered > best_covered:
            best_covered = covered
            best_path = candidate
    selection_csv = args.out_dir / "analytics" / "coverage_selection.csv"
    selection_csv.parent.mkdir(parents=True, exist_ok=True)
    with selection_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "covered", "status", "out_dir"])
        writer.writeheader()
        writer.writerows(rows)
    if best_covered == float("-inf"):
        print(f"warning: coverage selection found no covered metric; using {model_path}")
        return model_path
    print(f"selected coverage checkpoint: {best_path}\tcovered={best_covered:.6g}")
    manifest["coverage_selected_model_path"] = str(best_path)
    manifest["coverage_selected_covered"] = best_covered
    write_json(args.out_dir / "manifest.json", manifest)
    return best_path


def uses_paper_pipeline_config(config_path: Path) -> bool:
    try:
        text = config_path.read_text()
    except OSError:
        return False
    return "paper_pipeline_train: true" in text or "paper_pipeline_validation: true" in text


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_qubo(baseline_csv: Path, dynaco_csv: Path, out_csv: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(baseline_csv) + read_csv_rows(dynaco_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("solver", "")].append(row)

    summary = []
    for solver, solver_rows in sorted(grouped.items()):
        ok = [row for row in solver_rows if row.get("status") == "ok" and row.get("energy")]
        energies = [float(row["energy"]) for row in ok]
        runtimes = [float(row["runtime_s"]) for row in ok if row.get("runtime_s")]
        gaps = [float(row["gap_to_local"]) for row in ok if row.get("gap_to_local") not in {"", "NA", None}]
        summary.append(
            {
                "solver": solver,
                "rows": len(solver_rows),
                "ok": len(ok),
                "fail": len(solver_rows) - len(ok),
                "mean_energy": mean(energies) if energies else "",
                "best_energy": min(energies) if energies else "",
                "mean_runtime_s": mean(runtimes) if runtimes else "",
                "mean_gap_to_local": mean(gaps) if gaps else "",
            }
        )

    by_instance: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("energy"):
            by_instance[row["gfa"]].append(row)
    wins = Counter()
    for instance_rows in by_instance.values():
        best = min(float(row["energy"]) for row in instance_rows)
        for row in instance_rows:
            if abs(float(row["energy"]) - best) <= 1e-9:
                wins[row["solver"]] += 1
    for item in summary:
        item["instance_wins_or_ties"] = wins[item["solver"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        fieldnames = [
            "solver",
            "rows",
            "ok",
            "fail",
            "mean_energy",
            "best_energy",
            "mean_runtime_s",
            "mean_gap_to_local",
            "instance_wins_or_ties",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return summary


AVG_HEADER_RE = re.compile(r"Average stats(?: for best runs with time limit (?P<time>\S+))?")


def parse_full_avg_file(path: Path) -> list[dict[str, object]]:
    rows = []
    parts = path.name.split(".")
    if len(parts) < 5:
        return rows
    solver, annotator = parts[0], parts[1]
    current_time = "0" if solver == "pathfinder" else ""
    for line in path.read_text(errors="replace").splitlines():
        match = AVG_HEADER_RE.match(line)
        if match:
            current_time = match.group("time") or ("0" if solver == "pathfinder" else "")
            continue
        fields = line.split()
        if len(fields) == 3 and fields[0] not in {"Column", "==============="}:
            try:
                metric, metric_mean, stddev = fields[0], float(fields[1]), float(fields[2])
            except ValueError:
                continue
            rows.append(
                {
                    "solver": solver,
                    "annotator": annotator,
                    "time_limit": current_time,
                    "metric": metric,
                    "mean": metric_mean,
                    "stddev": stddev,
                    "source": str(path),
                }
            )
    return rows


def summarize_full(full_dir: Path, out_csv: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(full_dir.glob("*.cons.avg.txt")):
        rows.extend(parse_full_avg_file(path))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        fieldnames = ["solver", "annotator", "time_limit", "metric", "mean", "stddev", "source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize_errors(search_dir: Path, out_csv: Path) -> list[dict[str, object]]:
    patterns = {
        "command_not_found": "command not found",
        "missing_file": "No such file",
        "pathfinder_missing": "pathfinder: No such file",
        "unknown_pathfinder_option": "unknown option",
        "candidate_stats_failed": "candidate_stats.pl",
        "samtools_failed": "samtools",
        "traceback": "Traceback",
        "end_failed": "END failed",
    }
    counts = Counter()
    for path in search_dir.rglob("*"):
        if path.is_file() and path.suffix in {".log", ".err", ".out"}:
            text = path.read_text(errors="ignore")
            for name, needle in patterns.items():
                if needle in text:
                    counts[name] += text.count(needle)
    rows = [{"error_class": key, "count": value} for key, value in sorted(counts.items())]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["error_class", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _fmt_cell(value: object) -> str:
    if value in {"", None}:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_full_paper_tables(full_rows: list[dict[str, object]], out_prefix: Path) -> None:
    """Write paper-ready full-assembly tables from parsed *.cons.avg.txt rows."""
    metrics_by_group: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in full_rows:
        key = (str(row["annotator"]).upper(), str(row["solver"]), str(row["time_limit"]))
        metrics_by_group[key][str(row["metric"])] = float(row["mean"])

    fieldnames = [
        "graph",
        "solver",
        "time",
        "covered",
        "used",
        "contigs",
        "breaks",
        "indels",
        "diffs",
        "identity",
    ]
    rows = []
    for (annotator, solver, time_limit), metrics in sorted(metrics_by_group.items()):
        rows.append(
            {
                "graph": annotator,
                "solver": solver,
                "time": time_limit,
                "covered": metrics.get("covered", ""),
                "used": metrics.get("used", ""),
                "contigs": metrics.get("ncontig", ""),
                "breaks": metrics.get("nbreaks", ""),
                "indels": metrics.get("nindels", ""),
                "diffs": metrics.get("ndiffs", ""),
                "identity": metrics.get("identity", ""),
            }
        )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| graph | solver | time | covered | used | contigs | breaks | indels | diffs | identity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(_fmt_cell(row[field]) for field in fieldnames)
            + " |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        "Graph & Solver & Time & Covered & Used & Contigs & Breaks & Indels & Diffs & Identity \\\\",
        "\\hline",
    ]
    current_graph = None
    for row in rows:
        if current_graph is not None and row["graph"] != current_graph:
            tex_lines.append("\\hline")
        current_graph = row["graph"]
        tex_lines.append(
            " & ".join(
                _latex_escape(_fmt_cell(row[field])) if field in {"graph", "solver"} else _fmt_cell(row[field])
                for field in fieldnames
            )
            + " \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    out_prefix.with_suffix(".tex").write_text("\n".join(tex_lines))


def write_qubo_paper_tables(qubo_summary: list[dict[str, object]], out_prefix: Path) -> None:
    fieldnames = ["solver", "ok", "fail", "mean_energy", "mean_runtime_s", "instance_wins_or_ties"]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in qubo_summary:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    md_lines = [
        "| solver | ok | fail | mean energy | mean runtime s | wins/ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in qubo_summary:
        md_lines.append(
            "| "
            + " | ".join(_fmt_cell(row.get(field, "")) for field in fieldnames)
            + " |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Solver & OK & Fail & Mean energy & Mean runtime (s) & Wins/ties \\\\",
        "\\hline",
    ]
    for row in qubo_summary:
        tex_lines.append(
            " & ".join(
                [_latex_escape(row.get("solver", ""))]
                + [_fmt_cell(row.get(field, "")) for field in fieldnames[1:]]
            )
            + " \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    out_prefix.with_suffix(".tex").write_text("\n".join(tex_lines))


def write_report(path: Path, manifest: dict, qubo_summary: list[dict[str, object]], full_rows: list[dict[str, object]], error_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Overnight DyNACO Paper Benchmark Report",
        "",
        f"- Created at: {manifest['created_at']}",
        f"- Output directory: `{manifest['out_dir']}`",
        f"- Full dataset: {manifest['args']['full_seeds']} seeds x {manifest['args']['full_test_sequences']} held-out sequences x {manifest['args']['full_annotators']} annotators",
        f"- Full solver time limits: `{manifest['args']['full_time_limits']}` seconds",
        "",
        "## QUBO Stage",
        "",
        "| solver | ok | fail | mean energy | mean runtime s | wins/ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in qubo_summary:
        lines.append(
            "| {solver} | {ok} | {fail} | {mean_energy} | {mean_runtime_s} | {instance_wins_or_ties} |".format(
                **{key: row.get(key, "") for key in row}
            )
        )

    lines.extend(["", "## Full Assembly Stage", ""])
    if not full_rows:
        lines.append("No full assembly metric rows were parsed. If this was a dry run, this is expected.")
    else:
        lines.extend(
            [
                "Paper-ready full assembly tables are written to:",
                "",
                "- `analytics/paper_full_assembly_table.csv`",
                "- `analytics/paper_full_assembly_table.md`",
                "- `analytics/paper_full_assembly_table.tex`",
                "",
            ]
        )
        by_group: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
        for row in full_rows:
            by_group[(str(row["solver"]), str(row["annotator"]), str(row["time_limit"]))][str(row["metric"])] = float(row["mean"])
        lines.extend(
            [
                "| solver | annotator | time | covered | used | ncontig | nbreaks | n50 | identity |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (solver, annotator, time_limit), metrics in sorted(by_group.items()):
            lines.append(
                f"| {solver} | {annotator} | {time_limit} | "
                f"{metrics.get('covered', '')} | {metrics.get('used', '')} | "
                f"{metrics.get('ncontig', '')} | {metrics.get('nbreaks', '')} | "
                f"{metrics.get('n50', '')} | {metrics.get('identity', '')} |"
            )

    lines.extend(["", "## Error Signals", ""])
    if not error_rows:
        lines.append("No known error patterns found.")
    else:
        for row in error_rows:
            lines.append(f"- {row['error_class']}: {row['count']}")
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "overnight_dynaco_paper" / timestamp)
    parser.add_argument("--config", type=Path, default=DEFAULT_DYNACO_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="Tiny end-to-end check with one-second budgets.")
    parser.add_argument("--execute", action="store_true", help="Run commands. Default is dry-run planning.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-qubo", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--run-full", action="store_true", help="Actually execute the expensive full assembly stage.")
    parser.add_argument("--model", type=Path, help="Existing DyNACO checkpoint to use instead of training output.")
    parser.add_argument("--device", default="cuda")
    traced_group = parser.add_mutually_exclusive_group()
    traced_group.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    traced_group.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for the C++ ACO backend.")
    parser.add_argument("--epochs", type=int, default=DYNACO_REFERENCE_EPOCHS)
    parser.add_argument(
        "--steps-per-epoch",
        "--instances-per-epoch",
        dest="steps_per_epoch",
        type=int,
        default=DYNACO_REFERENCE_STEPS_PER_EPOCH,
        help="Number of fresh/sampled training instances per epoch.",
    )
    parser.add_argument("--H", "--online-steps", dest="online_steps", type=int, default=DYNACO_REFERENCE_H)
    parser.add_argument("--mini_H", "--mini-h", dest="mini_h", type=int, default=DYNACO_REFERENCE_MINI_H)
    parser.add_argument("--n_ants", type=int, default=DYNACO_REFERENCE_ANTS)
    parser.add_argument("--train-ants", type=int, default=None)
    parser.add_argument("--eval-ants", type=int, default=None)
    parser.add_argument(
        "--eval-min-iterations",
        type=int,
        default=None,
        help="Minimum evaluation ACO iterations. Defaults to --H * --mini_H so evaluation uses the same rollout budget as training.",
    )
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", type=float, default=DYNACO_REFERENCE_ALPHA)
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", type=float, default=DYNACO_REFERENCE_BETA)
    parser.add_argument("--rho", "--aco-evaporation", dest="aco_evaporation", type=float, default=DYNACO_REFERENCE_RHO)
    parser.add_argument("--gamma", "--aco-gamma", dest="aco_gamma", type=float, default=DYNACO_REFERENCE_GAMMA)
    parser.add_argument("--qubo-time-limit", type=int, default=3)
    parser.add_argument("--qubo-jobs", type=int, default=3)
    parser.add_argument("--qubo-local-jobs", type=int, default=5)
    parser.add_argument("--qubo-baselines", default=DEFAULT_QUBO_BASELINES)
    parser.add_argument("--full-solvers", default=DEFAULT_FULL_SOLVERS)
    parser.add_argument("--full-annotators", default="mg", help="Comma-separated full-assembly annotation routes. Default is MG only.")
    parser.add_argument("--full-seeds", type=int, default=20)
    parser.add_argument("--full-test-sequences", type=int, default=5)
    parser.add_argument("--full-jobs", type=int, default=3)
    parser.add_argument("--full-time-limits", default=str(DEFAULT_FULL_TIME_LIMIT))
    parser.add_argument(
        "--coverage-select",
        action="store_true",
        help="Select the deployed checkpoint by maximum MG full-assembly covered on a small held-out validation run.",
    )
    parser.add_argument("--coverage-selection-seeds", type=int, default=1)
    parser.add_argument("--coverage-selection-test-sequences", type=int, default=1)
    parser.add_argument("--coverage-selection-jobs", type=int, default=1)
    parser.add_argument("--coverage-selection-time-limits", default="10")
    parser.add_argument(
        "--pathfinder-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Pathfinder preprocessing; requires compatible QPG-era pathfinder.",
    )
    parser.add_argument("--include-pathfinder", action="store_true", help="Add pathfinder to the full solver list.")
    parser.add_argument("--require-pathfinder", action="store_true", help="Fail if compatible pathfinder is unavailable.")
    parser.add_argument("--shred-depth", type=int, default=30)
    parser.add_argument("--shuf-random-source", default="/usr/bin/emacs")
    parser.add_argument("--max-expansions", type=int, default=100000)
    args = parser.parse_args()
    if args.train_ants is None:
        args.train_ants = args.n_ants
    if args.eval_ants is None:
        args.eval_ants = args.n_ants
    if args.smoke:
        args.execute = True
        args.device = "cpu"
        args.epochs = 1
        args.steps_per_epoch = 1
        args.online_steps = 1
        args.mini_h = 1
        args.train_ants = 4
        args.eval_ants = 8
        if args.eval_min_iterations is None:
            args.eval_min_iterations = 1
        args.aco_evaporation = 0.2
        args.qubo_time_limit = 1
        args.qubo_jobs = 1
        args.qubo_local_jobs = 1
        args.full_seeds = 1
        args.full_test_sequences = 1
        args.full_jobs = 1
        args.full_time_limits = "1"
        args.coverage_selection_seeds = 1
        args.coverage_selection_test_sequences = 1
        args.coverage_selection_jobs = 1
        args.coverage_selection_time_limits = "1"
        args.run_full = False
    if args.eval_min_iterations is None:
        args.eval_min_iterations = args.online_steps * args.mini_h
    return args


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run.log"
    model_path = args.model or (args.out_dir / "dynaco_overnight.pt")
    generated_dir = args.out_dir / "generated"
    baseline_csv = args.out_dir / "qubo_baselines.csv"
    dynaco_csv = args.out_dir / "qubo_dynaco.csv"
    full_dir = args.out_dir / "full_assembly"
    analytics_dir = args.out_dir / "analytics"
    env = make_env(args, args.out_dir, model_path)

    full_solvers = split_csv(args.full_solvers)
    if args.include_pathfinder and "pathfinder" not in full_solvers:
        full_solvers.insert(0, "pathfinder")
    pf_ok, pf_message = pathfinder_compatible(env)
    if (args.pathfinder_graph or "pathfinder" in full_solvers) and not pf_ok:
        message = f"compatible QPG-era pathfinder unavailable: {pf_message}"
        if args.require_pathfinder:
            print(message, file=sys.stderr)
            return 2
        print(f"warning: {message}; pathfinder-dependent execution may fail.")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "out_dir": str(args.out_dir),
        "model_path": str(model_path),
        "tool_status": tool_status(env),
        "pathfinder_compatible": {"ok": pf_ok, "message": pf_message},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.out_dir / "manifest.json", manifest)

    if not args.skip_train and args.model is None:
        code = run_command(train_command(args, model_path, generated_dir), cwd=REPO_ROOT, env=env, log_path=log_path, execute=args.execute)
        if code != 0:
            return code
        if args.coverage_select:
            model_path = coverage_select_checkpoint(args, model_path, env, log_path, manifest)
            env = make_env(args, args.out_dir, model_path)
        else:
            selected_model_path = validation_selected_checkpoint(model_path)
            if selected_model_path != model_path:
                print(f"selected validation checkpoint: {selected_model_path}")
                model_path = selected_model_path
                env = make_env(args, args.out_dir, model_path)
                manifest["selected_model_path"] = str(model_path)
                write_json(args.out_dir / "manifest.json", manifest)
    elif args.coverage_select:
        model_path = coverage_select_checkpoint(args, model_path, env, log_path, manifest)
        env = make_env(args, args.out_dir, model_path)

    if not args.skip_qubo:
        code = run_command(benchmark_command(args, model_path, baseline_csv), cwd=REPO_ROOT, env=env, log_path=log_path, execute=args.execute)
        if code != 0:
            return code
        code = run_command(dynaco_test_command(args, model_path, dynaco_csv), cwd=REPO_ROOT, env=env, log_path=log_path, execute=args.execute)
        if code != 0:
            return code

    if not args.skip_full:
        code = run_command(
            full_pipeline_command(args, model_path, full_dir, ",".join(full_solvers)),
            cwd=REPO_ROOT,
            env=env,
            log_path=log_path,
            execute=args.execute,
        )
        if code != 0:
            return code

    qubo_summary = summarize_qubo(baseline_csv, dynaco_csv, analytics_dir / "qubo_summary.csv")
    full_rows = summarize_full(full_dir, analytics_dir / "full_assembly_summary.csv")
    error_rows = summarize_errors(args.out_dir, analytics_dir / "error_summary.csv")
    write_qubo_paper_tables(qubo_summary, analytics_dir / "paper_qubo_table")
    write_full_paper_tables(full_rows, analytics_dir / "paper_full_assembly_table")
    write_report(args.out_dir / "REPORT.md", manifest, qubo_summary, full_rows, error_rows)
    print(f"wrote report: {args.out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
