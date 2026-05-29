#!/usr/bin/env python3
"""Shared train/test/benchmark helpers for QPG DyNACO workflows."""

from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "qubo"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from qubo_solvers.definitions import QuboDescription, Solver  # noqa: E402
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers  # noqa: E402
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    aco_sample_qubo,
    astar_sample_qubo,
    beam_search_sample_qubo,
    exact_sample_qubo,
    greedy_residual_sample_qubo,
    gurobi_sample_qubo,
    local_sample_qubo,
    mqlib_sample_qubo,
    neural_aco_sample_qubo,
    random_residual_sample_qubo,
)


DEFAULT_DATASET_GLOBS = [
    "examples/*.gfa",
    "data/*.gfa",
    "synthetic_quality_sweep/**/*.gfa",
]

BASELINE_SOLVER_FUNCS = {
    Solver.MQLIB: mqlib_sample_qubo,
    Solver.GUROBI: gurobi_sample_qubo,
    Solver.EXACT: exact_sample_qubo,
    Solver.LOCAL: local_sample_qubo,
    Solver.GREEDY_RESIDUAL: greedy_residual_sample_qubo,
    Solver.RANDOM_RESIDUAL: random_residual_sample_qubo,
    Solver.BEAM: beam_search_sample_qubo,
    Solver.ACO: aco_sample_qubo,
    Solver.ASTAR: astar_sample_qubo,
}

NEURAL_SOLVER_FUNCS = {
    Solver.NEURAL_ACO: neural_aco_sample_qubo,
}

BASELINE_SOLVERS = ",".join(solver.value for solver in BASELINE_SOLVER_FUNCS)


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return [parse_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(path.read_text())
        return {} if data is None else {str(key).replace("-", "_"): value for key, value in dict(data).items()}

    data: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key is not None:
            data.setdefault(current_list_key, [])
            data[current_list_key].append(parse_scalar(line[4:]))
            continue
        current_list_key = None
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if value:
            data[key] = parse_scalar(value)
        else:
            data[key] = []
            current_list_key = key
    return data


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def collect_gfas(values: Iterable[str] | str | None, glob_values: Iterable[str] | str | None) -> list[Path]:
    paths: list[Path] = []
    for value in as_list(values):
        matches = sorted(Path(match) for match in glob.glob(value, recursive=True))
        paths.extend(matches if matches else [Path(value)])
    for value in as_list(glob_values):
        paths.extend(sorted(Path(match) for match in glob.glob(value, recursive=True)))

    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(path)
        seen.add(resolved)
        unique.append(resolved)
    return unique


def count_segments(gfa: Path) -> int:
    with gfa.open() as handle:
        return sum(1 for line in handle if line.startswith("S\t"))


def parse_copy_numbers(value: str | None, gfa: Path) -> list[float] | None:
    if value in {None, "sc", "gfa", "SC"}:
        return None
    if value == "ones":
        return [1.0] * count_segments(gfa)
    if value in {"paper", "paper_int", "paper_float"}:
        mode = "f" if value == "paper_float" else "i"
        output = subprocess.check_output(
            [
                str(REPO_ROOT / "tag_gfa_copy_numbers.pl"),
                "-c",
                "0.45",
                f"--mode={mode}",
                "--offset=0.4",
                "-d=5.0",
                str(gfa),
            ],
            text=True,
        ).strip()
        return [float(item) for item in output.split(",") if item]
    return [float(item) for item in value.split(",") if item]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def build_description(
    gfa: Path,
    solver: Solver,
    *,
    copy_numbers: str,
    penalties: str,
    alpha: float,
    time_limit: int,
    jobs: int,
    data_dir: Path | None = None,
) -> tuple[QuboDescription, tuple[int, int], int]:
    graph = oriented_graph_with_copy_numbers(gfa, parse_copy_numbers(copy_numbers, gfa))
    q_matrix, offset, t_max, original_node_count = qubo_matrix_from_graph(
        graph,
        alpha=alpha,
        penalties=parse_csv_ints(penalties),
    )
    description = QuboDescription(
        filename=gfa.name,
        data_dir=str(data_dir or gfa.parent),
        graph=graph,
        time_limits=[time_limit],
        jobs=jobs,
        Q=q_matrix,
        offset=offset,
        T=t_max,
        V=original_node_count,
        solver=solver,
    )
    return description, q_matrix.shape, t_max


def write_mqlib_input(description: QuboDescription) -> None:
    output = Path(description.data_dir) / f"mqlib_input_{description.filename}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    upper = np.triu(description.Q)
    non_zero = np.nonzero(upper)
    with output.open("w") as handle:
        handle.write(f"{description.Q.shape[0]} {int(non_zero[0].shape[0])}\n")
        for row, col in zip(non_zero[0], non_zero[1], strict=False):
            handle.write(f"{int(row) + 1} {int(col) + 1} {-description.Q[row, col]} \n")


def best_result(paths: dict[int, list[tuple]]) -> tuple[float, list]:
    best_energy = np.inf
    best_path = []
    for runs in paths.values():
        for _solution, energy, path in runs:
            if float(energy) < best_energy:
                best_energy = float(energy)
                best_path = path
    return float(best_energy), best_path


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_completed(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as handle:
        return {
            (row.get("gfa", ""), row.get("solver", ""))
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def append_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_solver_row(
    gfa: Path,
    solver: Solver,
    func,
    *,
    copy_numbers: str,
    penalties: str,
    alpha: float,
    time_limit: int,
    jobs: int,
    env: dict[str, str | None] | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="qpg_mqlib_") as scratch:
            description, q_shape, horizon = build_description(
                gfa,
                solver,
                copy_numbers=copy_numbers,
                penalties=penalties,
                alpha=alpha,
                time_limit=time_limit,
                jobs=jobs,
                data_dir=Path(scratch) if solver == Solver.MQLIB else None,
            )
            if solver == Solver.MQLIB:
                write_mqlib_input(description)
            with temporary_env(env or {}):
                paths = func(description)
        energy, path = best_result(paths)
        return {
            "gfa": str(gfa),
            "solver": solver.value,
            "segments": description.V,
            "horizon": horizon,
            "qubo_variables": q_shape[0],
            "energy": energy,
            "runtime_s": time.perf_counter() - started,
            "status": "ok",
            "error": "",
            "path": " ".join(map(str, path)),
        }
    except Exception as exc:
        return {
            "gfa": str(gfa),
            "solver": solver.value,
            "segments": "",
            "horizon": "",
            "qubo_variables": "",
            "energy": "",
            "runtime_s": time.perf_counter() - started,
            "status": "error",
            "error": str(exc),
            "path": "",
        }


def add_gap_columns(rows: list[dict[str, object]]) -> None:
    by_gfa: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_gfa.setdefault(str(row["gfa"]), {})[str(row["solver"])] = float(row["energy"])
    for row in rows:
        if row.get("status") != "ok":
            row["gap_to_exact"] = ""
            row["gap_to_local"] = ""
            continue
        energies = by_gfa.get(str(row["gfa"]), {})
        energy = float(row["energy"])
        row["gap_to_exact"] = "" if "exact" not in energies else energy - energies["exact"]
        row["gap_to_local"] = "" if "local" not in energies else energy - energies["local"]
