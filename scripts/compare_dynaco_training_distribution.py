#!/usr/bin/env python3
"""Compare DyNACO synthetic training GFAs with paper-pipeline GFA inputs."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median


@dataclass
class GfaStats:
    group: str
    path: str
    segments: int
    links: int
    total_length: int
    mean_length: float
    median_length: float
    links_per_segment: float
    avg_undirected_degree: float
    branch_fraction: float
    component_count: int
    minus_oriented_link_fraction: float
    sc_tag_fraction: float
    sc_mean: float
    sc_positive_fraction: float
    copy_number_mean: float
    copy_number_nonunit_fraction: float


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def components(self) -> int:
        return len({self.find(item) for item in self.parent})


def tag_value(fields: list[str], key: str) -> float | None:
    prefix = f"{key}:"
    for field in fields:
        if field.startswith(prefix):
            try:
                return float(field.rsplit(":", 1)[-1])
            except ValueError:
                return None
    return None


def parse_gfa(path: Path, group: str) -> GfaStats:
    lengths: list[int] = []
    sc_values: list[float] = []
    copy_numbers: list[int] = []
    degree: Counter[str] = Counter()
    links = 0
    minus_links = 0
    uf = UnionFind()

    with path.open() as handle:
        for line in handle:
            if not line or line.startswith("H"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            if fields[0] == "S" and len(fields) >= 3:
                name = fields[1]
                uf.add(name)
                ln = tag_value(fields[3:], "LN")
                if ln is None:
                    ln = len(fields[2]) if fields[2] != "*" else 0
                lengths.append(int(ln))
                sc = tag_value(fields[3:], "SC")
                if sc is not None:
                    sc_values.append(sc)
                    copy_numbers.append(int(sc / 30.0 + 0.8))
                else:
                    copy_numbers.append(1)
            elif fields[0] == "L" and len(fields) >= 5:
                left = fields[1]
                right = fields[3]
                links += 1
                degree[left] += 1
                degree[right] += 1
                uf.union(left, right)
                if fields[2] == "-" or fields[4] == "-":
                    minus_links += 1

    segments = len(lengths)
    total_length = sum(lengths)
    branch_nodes = sum(1 for value in degree.values() if value > 2)
    nonunit = sum(1 for value in copy_numbers if value != 1)

    return GfaStats(
        group=group,
        path=str(path),
        segments=segments,
        links=links,
        total_length=total_length,
        mean_length=mean(lengths) if lengths else 0.0,
        median_length=median(lengths) if lengths else 0.0,
        links_per_segment=links / segments if segments else 0.0,
        avg_undirected_degree=(2 * links) / segments if segments else 0.0,
        branch_fraction=branch_nodes / segments if segments else 0.0,
        component_count=uf.components(),
        minus_oriented_link_fraction=minus_links / links if links else 0.0,
        sc_tag_fraction=len(sc_values) / segments if segments else 0.0,
        sc_mean=mean(sc_values) if sc_values else 0.0,
        sc_positive_fraction=sum(1 for value in sc_values if value > 0) / len(sc_values) if sc_values else 0.0,
        copy_number_mean=mean(copy_numbers) if copy_numbers else 0.0,
        copy_number_nonunit_fraction=nonunit / len(copy_numbers) if copy_numbers else 0.0,
    )


def collect(patterns: list[str], group: str, max_files: int, seed: int) -> list[tuple[str, Path]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if max_files > 0 and len(unique) > max_files:
        rng = random.Random(seed)
        unique = sorted(rng.sample(unique, max_files))
    return [(group, path) for path in unique]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(rows: list[GfaStats]) -> list[dict[str, object]]:
    fields = [
        "segments",
        "links",
        "total_length",
        "links_per_segment",
        "avg_undirected_degree",
        "branch_fraction",
        "component_count",
        "minus_oriented_link_fraction",
        "sc_tag_fraction",
        "copy_number_mean",
        "copy_number_nonunit_fraction",
    ]
    by_group: dict[str, list[GfaStats]] = defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)

    summary: list[dict[str, object]] = []
    for group, group_rows in sorted(by_group.items()):
        nonempty_rows = [row for row in group_rows if row.segments > 0]
        out: dict[str, object] = {
            "group": group,
            "n": len(group_rows),
            "n_nonempty": len(nonempty_rows),
            "n_empty": len(group_rows) - len(nonempty_rows),
        }
        for field in fields:
            values = [float(getattr(row, field)) for row in nonempty_rows]
            out[f"{field}_mean"] = mean(values) if values else 0.0
            out[f"{field}_p10"] = percentile(values, 0.10)
            out[f"{field}_p50"] = percentile(values, 0.50)
            out[f"{field}_p90"] = percentile(values, 0.90)
        summary.append(out)
    return summary


def write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    compact_fields = [
        "group",
        "n",
        "n_nonempty",
        "n_empty",
        "segments_p50",
        "segments_p90",
        "links_per_segment_mean",
        "branch_fraction_mean",
        "minus_oriented_link_fraction_mean",
        "sc_tag_fraction_mean",
        "copy_number_mean_mean",
        "copy_number_nonunit_fraction_mean",
    ]
    with path.open("w") as handle:
        handle.write("# DyNACO Training Distribution Audit\n\n")
        handle.write("| " + " | ".join(compact_fields) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(compact_fields)) + " |\n")
        for row in summary_rows:
            values = []
            for field in compact_fields:
                value = row[field]
                if isinstance(value, float):
                    values.append(f"{value:.4g}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-files", type=int, default=250, help="Maximum files sampled per group; 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/dynaco_distribution_audit"))
    parser.add_argument(
        "--train-glob",
        action="append",
        default=None,
        help="Glob for synthetic training GFAs. Can be repeated.",
    )
    parser.add_argument(
        "--pipeline-glob",
        action="append",
        default=None,
        help="Glob for full annotated pipeline query GFAs. Can be repeated.",
    )
    parser.add_argument(
        "--pathfinder-glob",
        action="append",
        default=None,
        help="Glob for Pathfinder-extracted pipeline subgraph GFAs. Can be repeated.",
    )
    args = parser.parse_args()
    train_globs = args.train_glob or [
        "results/dynaco_online/generated_overnight/train/*.gfa",
        "results/overnight_dynaco_paper/*/generated/train/*.gfa",
    ]
    pipeline_globs = args.pipeline_glob or ["results/paper_repro/**/seq_*.gfa"]
    pathfinder_globs = args.pathfinder_glob or ["results/paper_repro/**/*.subgraph.*.gfa"]

    groups = []
    groups.extend(collect(train_globs, "synthetic_train", args.max_files, args.seed))
    groups.extend(collect(pipeline_globs, "pipeline_full_seq", args.max_files, args.seed + 1))
    groups.extend(collect(pathfinder_globs, "pipeline_pathfinder_subgraph", args.max_files, args.seed + 2))
    if not groups:
        raise SystemExit("No GFA files found for the configured globs.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [parse_gfa(path, group) for group, path in groups]
    detail_csv = args.out_dir / "gfa_distribution_detail.csv"
    summary_csv = args.out_dir / "gfa_distribution_summary.csv"
    summary_md = args.out_dir / "gfa_distribution_summary.md"

    with detail_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary_rows = summarize(rows)
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown(summary_md, summary_rows)

    print(f"Wrote {detail_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")
    print()
    print(summary_md.read_text())


if __name__ == "__main__":
    main()
