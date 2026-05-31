#!/usr/bin/env python3
"""One-command runner for evidence-aware structural-walk training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dynaco_paper_pipeline_mg_walk_coverage.yaml"
SMOKE_GFA = REPO_ROOT / "results" / "dynaco_online" / "paper_pipeline_online_smoke_cache" / "train" / "km.00001" / "seq_1068-0021-#1#1.gfa"
SMOKE_GAF = REPO_ROOT / "results" / "dynaco_online" / "paper_pipeline_online_smoke_cache" / "train" / "km.00001" / "pop.gaf"


def python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def run_command(command: list[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_gfas_csv(path: Path, max_segments: int = 0) -> list[Path]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if max_segments > 0:
                try:
                    if int(row.get("segments") or 0) > max_segments:
                        continue
                except ValueError:
                    pass
            if row.get("gfa"):
                rows.append(Path(row["gfa"]))
    return rows


def usable_gfa_count(path: Path, max_segments: int) -> int:
    return len(read_gfas_csv(path, max_segments))


def next_seed_start(path: Path, fallback: int) -> int:
    if not path.exists():
        return fallback
    max_seed = fallback - 1
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                max_seed = max(max_seed, int(row.get("seed") or fallback - 1))
            except ValueError:
                continue
    return max_seed + 1


def generate_data(args) -> Path:
    out_dir = args.data_cache_dir.resolve()
    metadata_csv = out_dir / "metadata" / "gfas.csv"
    target_instances = args.target_train_instances
    if target_instances is None:
        target_instances = args.steps_per_epoch
    current = 0 if args.force_data else usable_gfa_count(metadata_csv, args.paper_pipeline_max_segments)
    if current >= target_instances:
        print(f"reuse data: {metadata_csv} usable_gfas={current}", flush=True)
        return metadata_csv

    seed_start = args.data_seed_start if args.force_data else next_seed_start(metadata_csv, args.data_seed_start)
    stagnant_rounds = 0
    while current < target_instances:
        seeds_this_round = max(args.data_seeds, 1)
        command = [
            python_executable(),
            str(REPO_ROOT / "scripts" / "generate_paper_pipeline_gfa_corpus.py"),
            "--run",
            "--out-dir",
            str(out_dir),
            "--seeds",
            str(seeds_this_round),
            "--seed-start",
            str(seed_start),
            "--test-sequences",
            str(args.paper_pipeline_test_sequences),
            "--annotators",
            args.paper_pipeline_annotators,
            "--shred-depth",
            str(args.paper_pipeline_shred_depth),
            "--train-frac",
            "1.0",
            "--val-frac",
            "0.0",
            "--small-max-segments",
            str(args.paper_pipeline_max_segments),
        ]
        run_command(command, dry_run=args.dry_run)
        if args.dry_run:
            return metadata_csv
        previous = current
        current = usable_gfa_count(metadata_csv, args.paper_pipeline_max_segments)
        print(f"data_status: usable_gfas={current}/{target_instances}", flush=True)
        seed_start += seeds_this_round
        stagnant_rounds = stagnant_rounds + 1 if current <= previous else 0
        if stagnant_rounds >= args.data_generation_attempt_limit:
            raise RuntimeError(
                f"Could not collect {target_instances} usable GFAs after {stagnant_rounds} stagnant rounds. "
                f"Current usable={current}; max_segments={args.paper_pipeline_max_segments}. "
                "Raise --paper-pipeline-max-segments, increase --data-seeds, or lower --target-train-instances."
            )
    return metadata_csv


def collect_gfas(args) -> list[Path]:
    gfas: list[Path] = []
    for value in args.gfa or []:
        path = Path(value)
        matches = sorted(Path(match) for match in REPO_ROOT.glob(value)) if any(ch in value for ch in "*?[]") else []
        gfas.extend(matches or [path])
    for csv_path in args.gfas_csv or []:
        gfas.extend(read_gfas_csv(Path(csv_path), args.paper_pipeline_max_segments))
    if args.smoke:
        gfas.append(SMOKE_GFA)
    unique = []
    seen = set()
    for gfa in gfas:
        resolved = gfa if gfa.is_absolute() else (REPO_ROOT / gfa)
        resolved = resolved.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def collect_gafs(args) -> list[Path]:
    gafs: list[Path] = []
    for value in args.gaf or []:
        path = Path(value)
        matches = sorted(Path(match) for match in REPO_ROOT.glob(value)) if any(ch in value for ch in "*?[]") else []
        gafs.extend(matches or [path])
    if args.smoke and SMOKE_GAF.exists():
        gafs.append(SMOKE_GAF)
    unique = []
    seen = set()
    for gaf in gafs:
        resolved = gaf if gaf.is_absolute() else (REPO_ROOT / gaf)
        resolved = resolved.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def discover_gafs_for_gfa(gfa: Path) -> list[Path]:
    run_dir = gfa.parent
    candidates = []
    stem = gfa.stem
    candidates.extend(sorted(run_dir.glob(f"{stem}.shred.fa.gaf")))
    candidates.extend(sorted(run_dir.glob("*.shred.fa.gaf")))
    candidates.extend(sorted(run_dir.glob("pop.gaf")))
    candidates.extend(sorted(run_dir.glob("*.gaf")))
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def evidence_prefix(gfas: list[Path]) -> str:
    if len(gfas) == 1:
        return gfas[0].stem.replace("#", "_")
    return "combined"


def generate_evidence(args, gfas: list[Path], gafs: list[Path]) -> tuple[Path, Path, Path]:
    if not gfas:
        raise ValueError("No GFA inputs. Use --gfa, --gfas-csv, or --smoke.")

    out_dir = args.evidence_dir.resolve()
    generated_edges: list[Path] = []
    generated_links: list[Path] = []
    generated_haps: list[Path] = []
    for gfa in gfas:
        gfa_gafs = gafs or discover_gafs_for_gfa(gfa)
        if not gfa_gafs and not args.include_gfa_link_tags:
            raise ValueError(
                f"No GAF found for {gfa}. Pass --gaf, use a paper-pipeline cache with *.gaf, "
                "or set --include-gfa-link-tags for GFA-link-only edge evidence."
            )
        gfa_stem = gfa.stem.replace("#", "_")
        prefix = f"{args.prefix}_{gfa_stem}" if args.prefix and len(gfas) > 1 else (args.prefix or gfa_stem)
        command = [
            python_executable(),
            str(REPO_ROOT / "scripts" / "generate_walk_evidence.py"),
            "--gfa",
            str(gfa),
            "--out-dir",
            str(out_dir),
            "--prefix",
            prefix,
            "--link-window",
            str(args.link_window),
            "--min-mapq",
            str(args.min_mapq),
            "--haplotype-label",
            args.haplotype_label,
        ]
        for gaf in gfa_gafs:
            command.extend(["--gaf", str(gaf)])
        if args.include_gfa_link_tags:
            command.append("--include-gfa-link-tags")
        run_command(command, dry_run=args.dry_run)
        generated_edges.append(out_dir / f"{prefix}.edge_support.tsv")
        generated_links.append(out_dir / f"{prefix}.link_support.tsv")
        generated_haps.append(out_dir / f"{prefix}.haplotypes.tsv")

    if len(generated_edges) == 1:
        return generated_edges[0], generated_links[0], generated_haps[0]
    return (
        concatenate_tsv(out_dir / f"{args.prefix or 'combined'}.edge_support.tsv", generated_edges, args.dry_run),
        concatenate_tsv(out_dir / f"{args.prefix or 'combined'}.link_support.tsv", generated_links, args.dry_run),
        concatenate_tsv(out_dir / f"{args.prefix or 'combined'}.haplotypes.tsv", generated_haps, args.dry_run),
    )


def concatenate_tsv(out_path: Path, paths: list[Path], dry_run: bool) -> Path:
    print(f"+ concatenate {len(paths)} TSVs -> {out_path}", flush=True)
    if dry_run:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with out_path.open("w") as out:
        for path in paths:
            with path.open() as handle:
                for line_index, line in enumerate(handle):
                    if line_index == 0:
                        if wrote_header:
                            continue
                        wrote_header = True
                    out.write(line)
    return out_path


def train(args, gfas: list[Path], edge_file: Path | None, link_file: Path | None, hap_file: Path | None) -> None:
    command = [
        python_executable(),
        str(REPO_ROOT / "train_walk_coverage_prior.py"),
        "--copy-numbers",
        args.copy_numbers,
        "--out",
        str(args.out),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--H",
        str(args.online_steps),
        "--mini_H",
        str(args.mini_h),
        "--n_ants",
        str(args.ants),
        "--units",
        str(args.units),
        "--depth",
        str(args.depth),
        "--device",
        args.device,
        "--eval-time-limit",
        str(args.eval_time_limit),
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if gfas:
        command.append("--train-gfas")
        command.extend(str(gfa) for gfa in gfas)
        command.append("--test-gfas")
        command.extend(str(gfa) for gfa in gfas[: max(1, min(len(gfas), args.test_limit))])
    if edge_file is not None:
        command.extend(["--edge-support-file", str(edge_file)])
    if link_file is not None:
        command.extend(["--link-support-file", str(link_file)])
    if hap_file is not None:
        command.extend(["--haplotype-file", str(hap_file)])
    command.extend(["--edge-loss-weight", str(args.edge_loss_weight)])
    command.extend(["--link-loss-weight", str(args.link_loss_weight)])
    command.extend(["--haplotype-switch-weight", str(args.haplotype_switch_weight)])
    command.extend(["--link-window", str(args.link_window)])
    if args.no_wandb:
        command.append("--no-wandb")
    if args.skip_final_eval:
        command.append("--skip-final-eval")
    if args.validate_every is not None:
        command.extend(["--validate-every", str(args.validate_every)])
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    run_command(command, dry_run=args.dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "evidence", "train"), default="all")
    parser.add_argument("--smoke", action="store_true", help="Use the cached paper-pipeline smoke GFA/GAF.")
    parser.add_argument(
        "--generate-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When no --gfa/--gfas-csv is supplied, generate paper-pipeline GFAs first.",
    )
    parser.add_argument("--force-data", action="store_true", help="Regenerate the paper-pipeline cache even if metadata exists.")
    parser.add_argument("--data-cache-dir", type=Path, default=Path("results/structural_walk/data"))
    parser.add_argument("--data-seeds", type=int, default=1)
    parser.add_argument("--data-seed-start", type=int, default=1)
    parser.add_argument(
        "--target-train-instances",
        type=int,
        default=None,
        help="Generate/reuse paper-pipeline data until this many usable GFAs exist. Defaults to --steps-per-epoch.",
    )
    parser.add_argument("--data-generation-attempt-limit", type=int, default=8)
    parser.add_argument("--paper-pipeline-annotators", default="mg")
    parser.add_argument("--paper-pipeline-test-sequences", type=int, default=5)
    parser.add_argument("--paper-pipeline-shred-depth", type=int, default=30)
    parser.add_argument("--paper-pipeline-max-segments", type=int, default=80)
    parser.add_argument("--gfa", action="append", help="Training GFA path or glob. Repeatable.")
    parser.add_argument("--gaf", action="append", help="Read-to-graph GAF path or glob. Repeatable.")
    parser.add_argument("--gfas-csv", action="append", type=Path, help="metadata/gfas.csv file from a paper-pipeline cache.")
    parser.add_argument("--evidence-dir", type=Path, default=Path("results/evidence"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--include-gfa-link-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EC/RC/KC tags on GFA links as fallback edge/link support.",
    )
    parser.add_argument("--link-window", type=int, default=8)
    parser.add_argument("--min-mapq", type=int, default=0)
    parser.add_argument("--haplotype-label", default="hap1")
    parser.add_argument("--edge-support-file", type=Path)
    parser.add_argument("--link-support-file", type=Path)
    parser.add_argument("--haplotype-file", type=Path)

    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/structural_walk/run.pt"))
    parser.add_argument("--copy-numbers", default="paper")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--H", "--online-steps", dest="online_steps", type=int, default=1)
    parser.add_argument("--mini_H", "--mini-h", dest="mini_h", type=int, default=1)
    parser.add_argument("--n_ants", "--ants", dest="ants", type=int, default=8)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-time-limit", type=int, default=1)
    parser.add_argument("--test-limit", type=int, default=4)
    parser.add_argument("--edge-loss-weight", type=float, default=0.5)
    parser.add_argument("--link-loss-weight", type=float, default=0.5)
    parser.add_argument("--haplotype-switch-weight", type=float, default=0.5)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-wandb", action="store_true", default=True)
    parser.add_argument("--skip-final-eval", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.smoke and not args.gfa and not args.gfas_csv and args.generate_data:
        args.gfas_csv = [generate_data(args)]
    gfas = collect_gfas(args)
    gafs = collect_gafs(args)
    if args.dry_run and not gfas:
        print("dry-run stopped after data-generation command because generated metadata does not exist yet.", flush=True)
        return 0
    edge_file = args.edge_support_file
    link_file = args.link_support_file
    hap_file = args.haplotype_file

    if args.mode in {"all", "evidence"}:
        edge_file, link_file, hap_file = generate_evidence(args, gfas, gafs)
    if args.mode in {"all", "train"}:
        train(args, gfas, edge_file, link_file, hap_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
