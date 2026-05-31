#!/usr/bin/env python3
"""Generate structural-walk evidence files from GFA/GAF artifacts.

The trainer consumes three optional TSVs:
  - edge support: source,target,support,gfa
  - link support: source,target,support,gfa
  - haplotype labels: node,haplotype,gfa

This script derives edge/link evidence from GAF path strings. For the current
synthetic haploid pipeline, haplotype labels are conservative: every
read-supported node is assigned to one haploid thread unless a separate phasing
source is supplied later.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re


PATH_TOKEN_RE = re.compile(r"([<>])([^<>]+)")


def parse_gfa_nodes(path: Path) -> set[str]:
    nodes: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.startswith("S\t"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    nodes.add(fields[1])
    return nodes


def parse_gfa_edge_support(path: Path) -> Counter[tuple[str, str]]:
    support: Counter[tuple[str, str]] = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.startswith("L\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                continue
            src = f"{fields[1]}_{fields[2]}"
            dst = f"{fields[3]}_{fields[4]}"
            value = 1.0
            for field in fields[5:]:
                if field.startswith(("EC:i:", "EC:f:", "RC:i:", "RC:f:", "KC:i:", "KC:f:")):
                    try:
                        value = float(field.rsplit(":", 1)[1])
                    except ValueError:
                        value = 1.0
                    break
            support[(src, dst)] += value
            support[(flip_oriented(dst), flip_oriented(src))] += value
    return support


def nodes_from_pair_support(rows: Counter[tuple[str, str]]) -> Counter[str]:
    nodes: Counter[str] = Counter()
    for (source, target), support in rows.items():
        if support <= 0:
            continue
        nodes[source[:-2] if source.endswith(("_+", "_-")) else source] += 1
        nodes[target[:-2] if target.endswith(("_+", "_-")) else target] += 1
    return nodes


def flip_oriented(node: str) -> str:
    if node.endswith("_+"):
        return f"{node[:-2]}_-"
    if node.endswith("_-"):
        return f"{node[:-2]}_+"
    if node.endswith("+"):
        return f"{node[:-1]}-"
    if node.endswith("-"):
        return f"{node[:-1]}+"
    return node


def normalize_token(orient: str, node: str) -> str:
    return f"{node}_{'+' if orient == '>' else '-'}"


def gaf_path_nodes(path_field: str, gfa_nodes: set[str]) -> list[str]:
    nodes = [normalize_token(orient, node) for orient, node in PATH_TOKEN_RE.findall(path_field)]
    return [node for node in nodes if node[:-2] in gfa_nodes]


def iter_gaf_paths(path: Path, min_mapq: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            try:
                mapq = int(fields[11])
            except ValueError:
                mapq = 0
            if mapq < min_mapq:
                continue
            rows.append(fields)
    return rows


def collect_from_gaf(gaf: Path, gfa_nodes: set[str], link_window: int, min_mapq: int):
    edge_support: Counter[tuple[str, str]] = Counter()
    link_support: Counter[tuple[str, str]] = Counter()
    supported_nodes: Counter[str] = Counter()

    for fields in iter_gaf_paths(gaf, min_mapq):
        path_nodes = gaf_path_nodes(fields[5], gfa_nodes)
        if not path_nodes:
            continue
        for node in path_nodes:
            supported_nodes[node[:-2]] += 1
        for src, dst in zip(path_nodes, path_nodes[1:]):
            edge_support[(src, dst)] += 1
        for left_index, src in enumerate(path_nodes):
            for dst in path_nodes[left_index + 1 : left_index + 1 + link_window]:
                if src != dst:
                    link_support[(src, dst)] += 1

    return edge_support, link_support, supported_nodes


def write_pair_support(path: Path, rows: Counter[tuple[str, str]], gfa: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gfa", "source", "target", "support"])
        for (source, target), support in sorted(rows.items()):
            writer.writerow([str(gfa), source, target, f"{float(support):.12g}"])


def write_haplotypes(path: Path, nodes: Counter[str], gfa: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gfa", "node", "haplotype"])
        for node, count in sorted(nodes.items()):
            if count > 0:
                writer.writerow([str(gfa), node, label])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gfa", required=True, type=Path)
    parser.add_argument("--gaf", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--link-window", type=int, default=8)
    parser.add_argument("--min-mapq", type=int, default=0)
    parser.add_argument("--include-gfa-link-tags", action="store_true")
    parser.add_argument("--haplotype-label", default="hap1")
    args = parser.parse_args()

    gfa_nodes = parse_gfa_nodes(args.gfa)
    edge_support: Counter[tuple[str, str]] = Counter()
    link_support: Counter[tuple[str, str]] = Counter()
    supported_nodes: Counter[str] = Counter()

    if args.include_gfa_link_tags:
        gfa_edges = parse_gfa_edge_support(args.gfa)
        edge_support.update(gfa_edges)
        link_support.update(gfa_edges)
        supported_nodes.update(nodes_from_pair_support(gfa_edges))

    for gaf in args.gaf:
        gaf_edges, gaf_links, gaf_nodes = collect_from_gaf(gaf, gfa_nodes, args.link_window, args.min_mapq)
        edge_support.update(gaf_edges)
        link_support.update(gaf_links)
        supported_nodes.update(gaf_nodes)

    prefix = args.prefix or args.gfa.stem
    edge_path = args.out_dir / f"{prefix}.edge_support.tsv"
    link_path = args.out_dir / f"{prefix}.link_support.tsv"
    hap_path = args.out_dir / f"{prefix}.haplotypes.tsv"
    write_pair_support(edge_path, edge_support, args.gfa)
    write_pair_support(link_path, link_support, args.gfa)
    write_haplotypes(hap_path, supported_nodes, args.gfa, args.haplotype_label)

    print(f"edge_support: {edge_path} rows={len(edge_support)}")
    print(f"link_support: {link_path} rows={len(link_support)}")
    print(f"haplotypes: {hap_path} rows={sum(1 for value in supported_nodes.values() if value > 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
