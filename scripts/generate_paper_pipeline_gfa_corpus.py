#!/usr/bin/env python3
"""Generate DyNACO train/test GFAs from the original paper data pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def make_env(args) -> dict[str, str]:
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
    htslib = REPO_ROOT / ".tools" / "htslib" / "build" / "lib"
    env["LD_LIBRARY_PATH"] = str(htslib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, execute: bool) -> int:
    printable = shell_join(command)
    print(printable, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        if not execute:
            return 0
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
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def config_for(annotator: str) -> Path:
    path = REPO_ROOT / f"config_illumina_{annotator}.sh"
    if not path.exists():
        raise SystemExit(f"Missing annotator config: {path}")
    return path


def collect_rows(out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        parts = run_dir.name.split(".")
        if len(parts) < 2:
            continue
        annotator = parts[0]
        seed = parts[-1]
        for gfa in sorted(run_dir.glob("seq_*.gfa")):
            if ".subgraph." in gfa.name:
                continue
            rows.append(
                {
                    "split": "",
                    "annotator": annotator,
                    "seed": seed,
                    "gfa": str(gfa.resolve()),
                    "segments": str(count_lines(gfa, "S")),
                    "links": str(count_lines(gfa, "L")),
                    "bytes": str(gfa.stat().st_size),
                }
            )
    return rows


def count_lines(path: Path, record_type: str) -> int:
    prefix = f"{record_type}\t"
    count = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(prefix):
                count += 1
    return count


def assign_splits(rows: list[dict[str, str]], train_frac: float, val_frac: float, split_seed: int) -> None:
    rows.sort(key=lambda row: (row["seed"], row["annotator"], row["gfa"]))
    random.Random(split_seed).shuffle(rows)
    n = len(rows)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    for index, row in enumerate(rows):
        if index < train_end:
            row["split"] = "train"
        elif index < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"


def write_outputs(out_dir: Path, rows: list[dict[str, str]], manifest: dict[str, object], small_max_segments: int) -> None:
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "annotator", "seed", "gfa", "segments", "links", "bytes"]
    with (metadata_dir / "gfas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for split in ["train", "val", "test"]:
        paths = [row["gfa"] for row in rows if row["split"] == split and int(row["segments"]) > 0]
        (metadata_dir / f"{split}.txt").write_text("\n".join(paths) + ("\n" if paths else ""))
        small_paths = [
            row["gfa"]
            for row in rows
            if row["split"] == split and 0 < int(row["segments"]) <= small_max_segments
        ]
        (metadata_dir / f"{split}.le{small_max_segments}.txt").write_text(
            "\n".join(small_paths) + ("\n" if small_paths else "")
        )
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute commands. Default is dry-run.")
    parser.add_argument("--index-only", action="store_true", help="Only rebuild metadata for an existing corpus directory.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to results/paper_pipeline_gfa_corpus/<timestamp>.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--test-sequences", type=int, default=5)
    parser.add_argument("--annotators", default="mg,km,ga", help="Comma-separated annotators to generate.")
    parser.add_argument("--shred-depth", type=int, default=30)
    parser.add_argument("--shuf-random-source", default="/usr/bin/emacs")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--small-max-segments", type=int, default=30)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or (REPO_ROOT / "results" / "paper_pipeline_gfa_corpus" / timestamp)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    env = make_env(args)
    annotators = split_csv(args.annotators)

    if not args.index_only:
        for annotator in annotators:
            config = config_for(annotator)
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                command = [
                    str(REPO_ROOT / "run_gfa_sim.sh"),
                    "-s",
                    str(seed),
                    "-c",
                    str(config),
                    "-a",
                    annotator,
                    "--solver",
                    "aco",
                    "-p",
                    f"{annotator}.",
                    "-n",
                    str(args.test_sequences),
                    "--data-only",
                ]
                code = run_command(command, cwd=out_dir, env=env, log_path=log_path, execute=args.run)
                if code != 0:
                    print(f"failed with exit code {code}: {shell_join(command)}", file=sys.stderr)
                    return code

    rows = collect_rows(out_dir) if args.run or args.index_only else []
    assign_splits(rows, args.train_frac, args.val_frac, args.split_seed)
    manifest = {
        "created_at": timestamp,
        "repo_root": str(REPO_ROOT),
        "command": sys.argv,
        "executed": args.run,
        "index_only": args.index_only,
        "paper_pipeline": "genome_create -> minigraph pop.gfa -> shred held-out genomes -> mg/km/ga annotation -> seq_*.gfa",
        "args": vars(args) | {"out_dir": str(out_dir)},
        "counts": {
            "total_gfas": len(rows),
            "nonempty_gfas": sum(1 for row in rows if int(row["segments"]) > 0),
            "train": sum(1 for row in rows if row["split"] == "train"),
            "val": sum(1 for row in rows if row["split"] == "val"),
            "test": sum(1 for row in rows if row["split"] == "test"),
            f"small_le{args.small_max_segments}": sum(
                1 for row in rows if 0 < int(row["segments"]) <= args.small_max_segments
            ),
        },
        "tool_paths": {
            "genome_create": shutil.which("genome_create", path=env["PATH"]),
            "minigraph": shutil.which("minigraph", path=env["PATH"]),
            "GraphAligner": shutil.which("GraphAligner", path=env["PATH"]),
            "kmer2node4": shutil.which("kmer2node4", path=env["PATH"]),
        },
    }
    write_outputs(out_dir, rows, manifest, args.small_max_segments)
    print(f"wrote: {out_dir / 'metadata' / 'manifest.json'}")
    print(f"wrote: {out_dir / 'metadata' / 'gfas.csv'}")
    print(f"wrote: {out_dir / 'metadata' / 'train.txt'}")
    print(f"wrote: {out_dir / 'metadata' / 'val.txt'}")
    print(f"wrote: {out_dir / 'metadata' / 'test.txt'}")
    print(f"wrote: {out_dir / 'metadata' / f'train.le{args.small_max_segments}.txt'}")
    print(f"wrote: {out_dir / 'metadata' / f'val.le{args.small_max_segments}.txt'}")
    print(f"wrote: {out_dir / 'metadata' / f'test.le{args.small_max_segments}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
