#!/usr/bin/env python3
"""Score Pathfinder outputs on a selected QUBO-stage GFA slice."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qpg_dynaco_workflow import build_description, count_segments  # noqa: E402
from qubo_solvers.definitions import Solver  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import _energy_from_choices  # noqa: E402


DEFAULT_PATHFINDER_OPTS = ["-X50", "-c0", "--min-seq-cov", "1", "--neighbour-steps", "1", "-v", "3"]

FIELDNAMES = [
    "gfa",
    "solver",
    "segments",
    "horizon",
    "qubo_variables",
    "energy",
    "runtime_s",
    "status",
    "error",
    "path",
    "pathfinder_paths",
    "pathfinder_stdout",
    "pathfinder_stderr",
]


def read_selected_gfas(path: Path) -> list[Path]:
    with path.open(newline="") as handle:
        return [Path(row["gfa"]).resolve() for row in csv.DictReader(handle)]


def parse_pathfinder_paths(stdout: str) -> list[list[str]]:
    paths: list[list[str]] = []
    current: list[str] | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("PATH "):
            if current:
                paths.append(current)
            current = []
            continue
        if current is None:
            continue
        if line.startswith("SUBGRAPH "):
            if current:
                paths.append(current)
            current = None
            continue
        if line.startswith("["):
            parts = line.split()
            if len(parts) >= 3:
                current.append(parts[2])
    if current:
        paths.append(current)
    return paths


def path_name_to_state(name: str, node_to_index: dict[str, int]) -> int:
    if name.endswith("+"):
        key = f"{name[:-1]}_+"
    elif name.endswith("-"):
        key = f"{name[:-1]}_-"
    else:
        key = f"{name}_+"
    if key not in node_to_index:
        raise KeyError(f"Pathfinder node {name!r} is not in the oriented graph")
    return node_to_index[key]


def choices_from_pathfinder(paths: list[list[str]], description) -> tuple[int, ...]:
    node_to_index = {node: index for index, node in enumerate(description.graph.nodes)}
    end_index = description.V * 2
    choices: list[int] = []
    for fragment_index, fragment in enumerate(paths):
        if fragment_index > 0:
            choices.append(end_index)
        for node_name in fragment:
            choices.append(path_name_to_state(node_name, node_to_index))
    if len(choices) < description.T:
        choices.extend([end_index] * (description.T - len(choices)))
    return tuple(choices[: description.T])


def score_gfa(args: argparse.Namespace, gfa: Path) -> dict[str, object]:
    started = time.perf_counter()
    description = None
    try:
        description, q_shape, horizon = build_description(
            gfa,
            Solver.LOCAL,
            copy_numbers=args.copy_numbers,
            penalties=args.penalties,
            alpha=args.alpha_qubo,
            time_limit=1,
            jobs=1,
        )
        env = os.environ.copy()
        env["PATH"] = (
            os.pathsep.join(
                [
                    str(REPO_ROOT / ".tools" / "bin"),
                    str(REPO_ROOT / ".tools" / "htslib" / "build" / "bin"),
                    env.get("PATH", ""),
                ]
            )
        )
        htslib = REPO_ROOT / ".tools" / "htslib" / "build" / "lib"
        env["LD_LIBRARY_PATH"] = str(htslib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        completed = subprocess.run(
            [str(args.pathfinder), *args.pathfinder_opts, str(gfa)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Pathfinder exited with code {completed.returncode}: {completed.stderr.strip()}")
        paths = parse_pathfinder_paths(completed.stdout)
        if not paths:
            raise RuntimeError("Pathfinder produced no PATH blocks")
        choices = choices_from_pathfinder(paths, description)
        energy = _energy_from_choices(choices, description)
        return {
            "gfa": str(gfa),
            "solver": "pathfinder",
            "segments": description.V,
            "horizon": horizon,
            "qubo_variables": q_shape[0],
            "energy": energy,
            "runtime_s": time.perf_counter() - started,
            "status": "ok",
            "error": "",
            "path": " ".join(str(item) for item in choices),
            "pathfinder_paths": " | ".join(" ".join(fragment) for fragment in paths),
            "pathfinder_stdout": completed.stdout,
            "pathfinder_stderr": completed.stderr,
        }
    except Exception as exc:
        return {
            "gfa": str(gfa),
            "solver": "pathfinder",
            "segments": count_segments(gfa) if gfa.exists() else "",
            "horizon": "" if description is None else description.T,
            "qubo_variables": "" if description is None else len(description.Q),
            "energy": "",
            "runtime_s": time.perf_counter() - started,
            "status": "error",
            "error": str(exc),
            "path": "",
            "pathfinder_paths": "",
            "pathfinder_stdout": "",
            "pathfinder_stderr": "",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-gfas", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--pathfinder", type=Path, default=REPO_ROOT / ".tools" / "bin" / "pathfinder")
    parser.add_argument("--pathfinder-opts", nargs="*", default=DEFAULT_PATHFINDER_OPTS)
    parser.add_argument("--copy-numbers", default="ones")
    parser.add_argument("--penalties", default="200,50,1")
    parser.add_argument("--alpha-qubo", type=float, default=1.1)
    args = parser.parse_args()

    gfas = read_selected_gfas(args.selected_gfas)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gfa in gfas:
        row = score_gfa(args, gfa)
        rows.append(row)
        print(f"{row['status']}\t{gfa}\t{row['energy']}\t{float(row['runtime_s']):.3f}s")
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out_csv}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
