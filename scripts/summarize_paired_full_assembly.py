#!/usr/bin/env python3
"""Summarize paired full-assembly eval files into paper-ready rows.

This script is intentionally narrow: it reads the per-sequence
``*.eval_cons.<time>.<job>`` files emitted by ``run_gfa_sim.sh`` for the
paired minigraph full-assembly experiment and writes one provenance bundle:
per-sequence rows, solver means, a Markdown table, and a LaTeX table body.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from statistics import mean


EVAL_RE = re.compile(
    r"^(?P<seq>\S+)\s+\d+\s+\d+\s+(?P<covered>[\d.]+)%\s+(?P<used>[\d.]+)%\s+"
    r"(?P<contigs>\d+)\s+(?P<breaks>\d+)\s+(?P<indels>\d+)\s+"
    r"(?P<diffs>\d+)\s+(?P<identity>[\d.]+)%"
)
EVAL_NAME_RE = re.compile(r"(?P<seq>.+)\.eval_cons\.(?P<limit>[^.]+)\.(?P<job>\d+)$")
DIR_RE = re.compile(r"(?P<solver>.+)\.(?P<annotator>[^.]+)\.(?P<seed>\d+)$")


def parse_eval_file(path: Path) -> dict[str, object] | None:
    name_match = EVAL_NAME_RE.match(path.name)
    dir_match = DIR_RE.match(path.parent.name)
    if name_match is None or dir_match is None:
        return None
    for line in path.read_text(errors="replace").splitlines():
        match = EVAL_RE.match(line)
        if match is None:
            continue
        return {
            "solver": dir_match.group("solver"),
            "annotator": dir_match.group("annotator"),
            "seed": int(dir_match.group("seed")),
            "sequence": match.group("seq"),
            "configured_limit": name_match.group("limit"),
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


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["solver"],
            row["annotator"],
            row["seed"],
            row["sequence"],
            row["configured_limit"],
        )
        grouped[key].append(row)
    out = []
    for candidates in grouped.values():
        out.append(
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
    return sorted(out, key=lambda row: (str(row["solver"]), int(row["seed"]), str(row["sequence"])))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["solver"]), str(row["annotator"]), str(row["configured_limit"]))].append(row)

    out = []
    for (solver, annotator, configured_limit), group in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in group})
        out.append(
            {
                "graph": "Minigraph" if annotator == "mg" else annotator,
                "solver": solver,
                "configured_limit": configured_limit,
                "seqs": len(group),
                "seed_start": seeds[0] if seeds else "",
                "seed_end": seeds[-1] if seeds else "",
                "covered": mean(float(row["covered"]) for row in group),
                "used": mean(float(row["used"]) for row in group),
                "contigs": mean(float(row["contigs"]) for row in group),
                "breaks": mean(float(row["breaks"]) for row in group),
                "indels": mean(float(row["indels"]) for row in group),
                "diffs": mean(float(row["diffs"]) for row in group),
                "identity": mean(float(row["identity"]) for row in group),
            }
        )
    return out


def validate_paired(
    rows: list[dict[str, object]],
    solvers: set[str],
    expected_per_solver: int | None,
) -> None:
    by_solver: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_instance: dict[tuple[int, str], set[str]] = defaultdict(set)
    for row in rows:
        solver = str(row["solver"])
        by_solver[solver].append(row)
        by_instance[(int(row["seed"]), str(row["sequence"]))].add(solver)

    missing_solvers = sorted(solvers - set(by_solver))
    if missing_solvers:
        raise SystemExit(f"missing solver rows: {', '.join(missing_solvers)}")

    if expected_per_solver is not None:
        bad_counts = {
            solver: len(by_solver[solver])
            for solver in sorted(solvers)
            if len(by_solver[solver]) != expected_per_solver
        }
        if bad_counts:
            details = ", ".join(f"{solver}={count}" for solver, count in bad_counts.items())
            raise SystemExit(f"unexpected per-solver row counts: {details}")

    incomplete = {
        key: sorted(solvers - seen)
        for key, seen in by_instance.items()
        if seen != solvers
    }
    if incomplete:
        preview = "; ".join(
            f"seed={seed} sequence={sequence} missing={','.join(missing)}"
            for (seed, sequence), missing in list(sorted(incomplete.items()))[:5]
        )
        raise SystemExit(f"unpaired assemblies found: {preview}")


def solver_label(solver: str) -> str:
    return {
        "pathfinder": "Pathfinder",
        "aco": "FW-ACO",
        "neural_aco": "LP-ACO",
        "neural_aco_zero": "LP-ACO zero prior",
        "mqlib": "MQLib",
        "beam_search": "Beam search",
    }.get(solver, solver)


def limit_label(row: dict[str, object]) -> str:
    if row["solver"] == "pathfinder":
        return "--"
    return str(row["configured_limit"])


def format_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    order = {
        "pathfinder": 0,
        "mqlib": 1,
        "beam_search": 2,
        "aco": 3,
        "neural_aco_zero": 4,
        "neural_aco": 5,
    }
    formatted = []
    for row in sorted(rows, key=lambda item: (order.get(str(item["solver"]), 99), str(item["solver"]))):
        formatted.append(
            {
                "graph": row["graph"],
                "solver": solver_label(str(row["solver"])),
                "configured_limit": limit_label(row),
                "seqs": row["seqs"],
                "covered": f"{float(row['covered']):.2f}",
                "used": f"{float(row['used']):.2f}",
                "contigs": f"{float(row['contigs']):.2f}",
                "breaks": f"{float(row['breaks']):.2f}",
                "indels": f"{float(row['indels']):.2f}",
                "diffs": f"{float(row['diffs']):.2f}",
                "identity": f"{float(row['identity']):.2f}",
                "seed_start": row["seed_start"],
                "seed_end": row["seed_end"],
            }
        )
    return formatted


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Graph | Solver | Configured limit | Seqs | Covered | Used | Contigs | Breaks | Identity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['graph']} | {row['solver']} | {row['configured_limit']} | {row['seqs']} | "
            f"{row['covered']} | {row['used']} | {row['contigs']} | {row['breaks']} | {row['identity']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def paired_deltas(rows: list[dict[str, object]], reference_solver: str) -> list[dict[str, object]]:
    by_instance: dict[tuple[int, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_instance[(int(row["seed"]), str(row["sequence"]))][str(row["solver"])] = row

    out = []
    solvers = sorted({str(row["solver"]) for row in rows if str(row["solver"]) != reference_solver})
    for solver in solvers:
        groups = [
            group
            for group in by_instance.values()
            if reference_solver in group and solver in group
        ]
        if not groups:
            continue
        delta_rows = []
        for group in groups:
            row = group[solver]
            ref = group[reference_solver]
            delta_rows.append(
                {
                    "solver": solver_label(solver),
                    "reference": solver_label(reference_solver),
                    "seed": row["seed"],
                    "sequence": row["sequence"],
                    "covered_delta": float(row["covered"]) - float(ref["covered"]),
                    "used_delta": float(row["used"]) - float(ref["used"]),
                    "contigs_delta": float(row["contigs"]) - float(ref["contigs"]),
                    "breaks_delta": float(row["breaks"]) - float(ref["breaks"]),
                    "identity_delta": float(row["identity"]) - float(ref["identity"]),
                    "covered_win": int(float(row["covered"]) > float(ref["covered"])),
                    "covered_tie": int(float(row["covered"]) == float(ref["covered"])),
                    "covered_loss": int(float(row["covered"]) < float(ref["covered"])),
                }
            )
        out.extend(delta_rows)
    return out


def paired_delta_summary(delta_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(str(row["solver"]), str(row["reference"]))].append(row)

    out = []
    for (solver, reference), group in sorted(grouped.items()):
        covered = [float(row["covered_delta"]) for row in group]
        used = [float(row["used_delta"]) for row in group]
        breaks = [float(row["breaks_delta"]) for row in group]
        identity = [float(row["identity_delta"]) for row in group]
        out.append(
            {
                "solver": solver,
                "reference": reference,
                "pairs": len(group),
                "mean_covered_delta": mean(covered),
                "mean_used_delta": mean(used),
                "mean_breaks_delta": mean(breaks),
                "mean_identity_delta": mean(identity),
                "covered_wins": sum(int(row["covered_win"]) for row in group),
                "covered_ties": sum(int(row["covered_tie"]) for row in group),
                "covered_losses": sum(int(row["covered_loss"]) for row in group),
            }
        )
    return out


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = []
    for row in rows:
        lines.append(
            f"{row['graph']} & {row['solver']} & {row['configured_limit']} & {row['seqs']} & "
            f"{row['covered']} & {row['used']} & {row['contigs']} & {row['breaks']} & {row['identity']} \\\\"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--solvers", default="pathfinder,aco,neural_aco")
    parser.add_argument("--annotator", default="mg")
    parser.add_argument("--expected-per-solver", type=int, default=40)
    parser.add_argument("--reference-solver", help="Optional solver used for paired delta outputs.")
    args = parser.parse_args()

    solvers = {item.strip() for item in args.solvers.split(",") if item.strip()}
    rows = []
    for path in args.full_dir.rglob("*.eval_cons.*"):
        row = parse_eval_file(path)
        if row is None:
            continue
        if row["solver"] not in solvers or row["annotator"] != args.annotator:
            continue
        rows.append(row)

    selected = best_rows(rows)
    validate_paired(selected, solvers, args.expected_per_solver)
    summary = summarize(selected)
    formatted = format_summary(summary)

    row_fields = [
        "solver",
        "annotator",
        "seed",
        "sequence",
        "configured_limit",
        "job",
        "covered",
        "used",
        "contigs",
        "breaks",
        "indels",
        "diffs",
        "identity",
        "source",
    ]
    summary_fields = [
        "graph",
        "solver",
        "configured_limit",
        "seqs",
        "seed_start",
        "seed_end",
        "covered",
        "used",
        "contigs",
        "breaks",
        "indels",
        "diffs",
        "identity",
    ]
    formatted_fields = [
        "graph",
        "solver",
        "configured_limit",
        "seqs",
        "covered",
        "used",
        "contigs",
        "breaks",
        "indels",
        "diffs",
        "identity",
        "seed_start",
        "seed_end",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "paired_full_assembly_rows.csv", selected, row_fields)
    write_csv(args.out_dir / "paired_full_assembly_summary.csv", summary, summary_fields)
    write_csv(args.out_dir / "paired_full_assembly_paper_table.csv", formatted, formatted_fields)
    write_markdown(args.out_dir / "paired_full_assembly_paper_table.md", formatted)
    write_latex(args.out_dir / "paired_full_assembly_paper_table.tex", formatted)
    if args.reference_solver:
        delta_rows = paired_deltas(selected, args.reference_solver)
        delta_fields = [
            "solver",
            "reference",
            "seed",
            "sequence",
            "covered_delta",
            "used_delta",
            "contigs_delta",
            "breaks_delta",
            "identity_delta",
            "covered_win",
            "covered_tie",
            "covered_loss",
        ]
        delta_summary = paired_delta_summary(delta_rows)
        delta_summary_fields = [
            "solver",
            "reference",
            "pairs",
            "mean_covered_delta",
            "mean_used_delta",
            "mean_breaks_delta",
            "mean_identity_delta",
            "covered_wins",
            "covered_ties",
            "covered_losses",
        ]
        write_csv(args.out_dir / "paired_full_assembly_deltas.csv", delta_rows, delta_fields)
        write_csv(args.out_dir / "paired_full_assembly_delta_summary.csv", delta_summary, delta_summary_fields)
    print(f"wrote {args.out_dir / 'paired_full_assembly_paper_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
