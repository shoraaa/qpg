#!/usr/bin/env python3
"""Multi-instance online DyNACO-style RL training for QPG neural ACO.

This trainer does not precompute labels.  It samples/builds QPG instances,
generates ant trajectories online with the C++ ACO sampler, replays the sampled
traces through the current GNN prior, and saves a general model checkpoint.
Instance pheromone is local to each online rollout and is not saved.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    class _NullProgress:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _NullProgress(iterable, *args, **kwargs)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from train_qpg_dynaco_cpp import (  # noqa: E402
    build_prior_tensor,
    parse_copy_numbers,
    parse_csv_ints,
    replay_log_probs,
    update_pheromone_from_batch,
)
from qubo_solvers.definitions import QuboDescription, Solver  # noqa: E402
from qubo_solvers.oriented_tangle import qpg_aco_cpp  # noqa: E402
from qubo_solvers.oriented_tangle.neural_gnn import QPGSeeAGNN, require_torch, torch  # noqa: E402
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers  # noqa: E402
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    _aco_static_edge_arrays,
    _node_weights_and_lengths,
    _path_result_from_choices,
    _states_per_time,
    aco_sample_qubo,
)


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
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
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


def load_config(path: Path) -> dict:
    """Load a simple YAML config, using PyYAML when available."""
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(path.read_text())
        return {} if data is None else dict(data)

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


def config_defaults(config: dict) -> dict:
    defaults = {key.replace("-", "_"): value for key, value in config.items()}
    if "instances_per_epoch" in defaults and "steps_per_epoch" not in defaults:
        defaults["steps_per_epoch"] = defaults["instances_per_epoch"]
    if "H" in defaults and "online_steps" not in defaults:
        defaults["online_steps"] = defaults["H"]
    if "mini_H" in defaults and "mini_h" not in defaults:
        defaults["mini_h"] = defaults["mini_H"]
    if "n_ants" in defaults and "ants" not in defaults:
        defaults["ants"] = defaults["n_ants"]
    if "rho" in defaults and "evaporation" not in defaults:
        defaults["evaporation"] = defaults["rho"]
    if "alpha" in defaults and "aco_alpha" not in defaults:
        defaults["aco_alpha"] = defaults["alpha"]
    if "beta" in defaults and "aco_beta" not in defaults:
        defaults["aco_beta"] = defaults["beta"]
    return defaults


def checkpoint_payload(
    model,
    optimizer,
    args,
    train_gfas: list[Path],
    *,
    epoch: int,
    best_energy: float,
    training_seconds: float,
    val_energy: float | None = None,
    val_objective: float | None = None,
    validation_objective: str | None = None,
) -> dict:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "units": args.units,
            "depth": args.depth,
            "source": "dynaco_online",
            "train_instances": [str(path) for path in train_gfas],
        },
        "epoch": epoch,
        "best_energy": best_energy,
        "training_seconds": training_seconds,
    }
    if val_energy is not None:
        checkpoint["val_energy"] = val_energy
    if val_objective is not None:
        checkpoint["val_objective"] = val_objective
    if validation_objective is not None:
        checkpoint["validation_objective"] = validation_objective
    return checkpoint


def checkpoint_path(out_path: Path, suffix: str) -> Path:
    return out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}")


def wandb_config(args) -> dict[str, object]:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        else:
            config[key] = str(value)
    return config


def init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("wandb logging requested, but wandb is not installed. Run `uv sync`.") from exc
    tags = split_csv(args.wandb_tags) if args.wandb_tags else None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.out.parent.name,
        mode=args.wandb_mode,
        tags=tags,
        config=wandb_config(args),
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/*", step_metric="train/global_step")
    wandb.define_metric("checkpoint/*", step_metric="train/global_step")
    return run


@dataclass
class OnlineInstance:
    gfa: Path
    graph: object
    description: QuboDescription
    offsets: np.ndarray
    targets: np.ndarray
    heuristic: np.ndarray
    weights: list[float]
    lengths: list[float]
    weights_array: np.ndarray
    lengths_array: np.ndarray
    q_float: np.ndarray
    start_source: int
    end_index: int
    states_per_time: int


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def collect_gfas(values: list[str] | str | None, glob_values: list[str] | str | None, required: bool = True) -> list[Path]:
    paths: list[Path] = []
    for value in _as_list(values):
        matches = sorted(Path(match) for match in glob.glob(value))
        paths.extend(matches if matches else [Path(value)])
    for value in _as_list(glob_values):
        paths.extend(sorted(Path(match) for match in glob.glob(value)))
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
    if not unique and required:
        raise ValueError("No GFA files were provided.")
    return unique


def collect_gfa_lists(values: list[str] | str | None) -> list[Path]:
    paths: list[Path] = []
    for value in _as_list(values):
        list_path = Path(value)
        if not list_path.exists():
            raise FileNotFoundError(list_path)
        for raw_line in list_path.read_text().splitlines():
            line = raw_line.strip()
            if line:
                paths.append(Path(line).resolve())
    return paths


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def estimate_gfa_horizon(path: Path, alpha: float, copy_numbers_mode: str) -> int:
    copy_numbers = parse_copy_numbers(copy_numbers_mode, path)
    if copy_numbers is None:
        copy_numbers = parse_copy_numbers("paper", path)
    return int(np.floor(2.0 * sum(copy_numbers) * alpha)) if copy_numbers else 0


def horizon_cache_path(metadata_csv: Path, copy_numbers_mode: str, alpha: float) -> Path:
    safe_mode = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in copy_numbers_mode)
    safe_alpha = str(alpha).replace(".", "p")
    return metadata_csv.with_name(f"horizon_cache.{safe_mode}.{safe_alpha}.json")


def load_horizon_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_horizon_cache(path: Path, cache: dict[str, dict[str, object]]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def corpus_rows(
    metadata_csv: Path,
    max_segments: int,
    max_horizon: int,
    alpha_qubo: float,
    copy_numbers_mode: str,
    index_horizon: bool,
) -> list[dict[str, str]]:
    if not metadata_csv.exists():
        return []
    with metadata_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    cache_file = horizon_cache_path(metadata_csv, copy_numbers_mode, alpha_qubo)
    horizon_cache = load_horizon_cache(cache_file)
    cache_dirty = False
    filtered = []
    for row in tqdm(rows, desc=f"index {metadata_csv.parent.parent.name}", unit="gfa", leave=False):
        segments = int(row.get("segments") or 0)
        if segments <= 0:
            continue
        if max_segments > 0 and segments > max_segments:
            continue
        path = Path(row["gfa"])
        if not path.exists():
            continue
        if index_horizon:
            key = str(path.resolve())
            stat = path.stat()
            cached = horizon_cache.get(key)
            horizon = None
            if (
                isinstance(cached, dict)
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and cached.get("size") == stat.st_size
            ):
                try:
                    horizon = int(cached["horizon"])
                except (KeyError, TypeError, ValueError):
                    horizon = None
            if horizon is None:
                try:
                    horizon = estimate_gfa_horizon(path, alpha_qubo, copy_numbers_mode)
                except subprocess.TimeoutExpired:
                    print(f"skip_timeout\t{path}\tcopy_number_timeout={os.environ.get('QPG_COPY_NUMBER_TIMEOUT', '30')}", flush=True)
                    continue
                horizon_cache[key] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "horizon": horizon,
                }
                cache_dirty = True
        else:
            horizon = int(np.floor(2.0 * segments * alpha_qubo))
        if max_horizon > 0 and horizon > max_horizon:
            continue
        row["estimated_horizon"] = str(horizon)
        filtered.append(row)
    if cache_dirty:
        write_horizon_cache(cache_file, horizon_cache)
    return filtered


class PaperPipelineGfaSource:
    def __init__(self, args, split: str, *, start_seed: int, target_instances: int = 0) -> None:
        self.args = args
        self.split = split
        self.seed = start_seed
        self.target_instances = target_instances
        self.annotators = split_csv(args.paper_pipeline_annotators)
        self.cache_dir = Path(args.paper_pipeline_cache_dir) / split
        self.max_segments = args.paper_pipeline_max_segments
        self.max_horizon = args.paper_pipeline_max_horizon
        self.generated_seeds: set[int] = set()
        self._paths: list[Path] = []
        self.refresh()

    @property
    def metadata_csv(self) -> Path:
        return self.cache_dir / "metadata" / "gfas.csv"

    def refresh(self) -> None:
        rows = corpus_rows(
            self.metadata_csv,
            self.max_segments,
            self.max_horizon,
            self.args.alpha_qubo,
            self.args.copy_numbers,
            self.args.paper_pipeline_index_horizon,
        )
        self._paths = [Path(row["gfa"]).resolve() for row in rows]

    def paths(self) -> list[Path]:
        self.refresh()
        return list(self._paths)

    def ensure(self, min_instances: int) -> list[Path]:
        attempts = 0
        current = len(self.paths())
        with tqdm(
            total=min_instances,
            initial=min(current, min_instances),
            desc=f"paper {self.split} GFAs",
            unit="gfa",
            leave=False,
        ) as progress:
            while current < min_instances:
                before = len(self._paths)
                self.generate_next_seed()
                after = len(self.paths())
                progress.update(max(0, min(after, min_instances) - min(current, min_instances)))
                progress.set_postfix(accepted=after, seed=self.seed - 1, stagnant=attempts)
                current = after
                attempts = attempts + 1 if after <= before else 0
                if attempts >= self.args.paper_pipeline_generation_attempt_limit:
                    raise RuntimeError(
                        f"Paper-pipeline {self.split} generation could not collect {min_instances} "
                        f"GFAs after {attempts} consecutive seeds. Current count={after}; "
                        f"max_segments={self.max_segments}, max_horizon={self.max_horizon}. "
                        "Increase --paper-pipeline-max-segments/--paper-pipeline-max-horizon, "
                        "reduce the requested validation/pool size, or move QUBO construction to a sparse path."
                    )
        return self.paths()

    def ensure_new(self, min_new_instances: int) -> list[Path]:
        before = set(self.paths())
        attempts = 0
        new_paths: list[Path] = []
        with tqdm(
            total=min_new_instances,
            desc=f"paper {self.split} fresh GFAs",
            unit="gfa",
            leave=False,
        ) as progress:
            while len(new_paths) < min_new_instances:
                previous_count = len(self._paths)
                self.generate_next_seed()
                current_paths = self.paths()
                new_paths = [path for path in current_paths if path not in before]
                progress.n = min(len(new_paths), min_new_instances)
                progress.set_postfix(accepted=len(current_paths), seed=self.seed - 1, stagnant=attempts)
                progress.refresh()
                attempts = attempts + 1 if len(current_paths) <= previous_count else 0
                if attempts >= self.args.paper_pipeline_generation_attempt_limit:
                    raise RuntimeError(
                        f"Paper-pipeline {self.split} generation could not collect "
                        f"{min_new_instances} fresh GFAs after {attempts} consecutive seeds. "
                        f"Fresh count={len(new_paths)}; max_segments={self.max_segments}, "
                        f"max_horizon={self.max_horizon}."
                    )
        return new_paths[:min_new_instances]

    def generate_next_seed(self) -> None:
        seed = self.seed
        self.seed += 1
        if seed in self.generated_seeds:
            return
        self.generated_seeds.add(seed)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_paper_pipeline_gfa_corpus.py"),
            "--run",
            "--out-dir",
            str(self.cache_dir),
            "--seeds",
            "1",
            "--seed-start",
            str(seed),
            "--test-sequences",
            str(self.args.paper_pipeline_test_sequences),
            "--annotators",
            self.args.paper_pipeline_annotators,
            "--shred-depth",
            str(self.args.paper_pipeline_shred_depth),
            "--train-frac",
            "1.0",
            "--val-frac",
            "0.0",
            "--small-max-segments",
            str(self.max_segments),
        ]
        print(f"paper_pipeline_generate\t{self.split}\t{shell_join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        self.refresh()


def random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(length))


def write_synthetic_gfa(path: Path, rng: random.Random, args) -> None:
    """Generate a small QUBO-stage tangle GFA for online RL training."""
    n_segments = rng.randint(args.synthetic_min_segments, args.synthetic_max_segments)
    path.parent.mkdir(parents=True, exist_ok=True)
    links: set[tuple[str, str, str, str]] = set()
    for index in range(1, n_segments):
        links.add((f"s{index}", "+", f"s{index + 1}", "+"))
    for index in range(1, n_segments - 1):
        if rng.random() < args.synthetic_bubble_rate:
            links.add((f"s{index}", "+", f"s{index + 2}", "+"))
        if rng.random() < args.synthetic_orientation_rate:
            links.add((f"s{index}", "+", f"s{index + 1}", "-"))
    for _ in range(args.synthetic_shortcuts):
        source = rng.randint(1, max(1, n_segments - 2))
        target = rng.randint(source + 1, n_segments)
        links.add((f"s{source}", "+", f"s{target}", "+"))

    with path.open("w") as handle:
        handle.write("H\tVN:Z:1.0\n")
        for index in range(1, n_segments + 1):
            length = rng.randint(args.synthetic_min_length, args.synthetic_max_length)
            handle.write(f"S\ts{index}\t{random_dna(rng, length)}\tLN:i:{length}\n")
        for source, source_orient, target, target_orient in sorted(links):
            handle.write(f"L\t{source}\t{source_orient}\t{target}\t{target_orient}\t0M\n")


def generate_synthetic_gfas(args, split: str, count: int) -> list[Path]:
    if count <= 0:
        return []
    rng = random.Random(args.synthetic_seed + (0 if split == "train" else 1_000_000))
    out_dir = args.synthetic_dir / split
    paths = []
    for index in range(count):
        path = out_dir / f"{split}_{index:04d}.gfa"
        write_synthetic_gfa(path, rng, args)
        paths.append(path.resolve())
    return paths


def build_instance(gfa: Path, args) -> OnlineInstance:
    graph = oriented_graph_with_copy_numbers(gfa, parse_copy_numbers(args.copy_numbers, gfa))
    q_matrix, offset, horizon, biological_nodes = qubo_matrix_from_graph(
        graph,
        alpha=args.alpha_qubo,
        penalties=parse_csv_ints(args.penalties),
    )
    description = QuboDescription(
        filename=gfa.name,
        data_dir=str(gfa.parent),
        graph=graph,
        time_limits=[args.eval_time_limit],
        jobs=1,
        Q=q_matrix,
        offset=offset,
        T=horizon,
        V=biological_nodes,
        solver=Solver.NEURAL_ACO,
    )
    offsets, targets, heuristic, _prior0, start_source = _aco_static_edge_arrays(description)
    weights, lengths = _node_weights_and_lengths(description)
    return OnlineInstance(
        gfa=gfa,
        graph=graph,
        description=description,
        offsets=offsets,
        targets=targets,
        heuristic=heuristic,
        weights=weights,
        lengths=lengths,
        weights_array=np.asarray(weights, dtype=np.float32),
        lengths_array=np.asarray(lengths, dtype=np.float32),
        q_float=np.asarray(q_matrix, dtype=np.float32),
        start_source=start_source,
        end_index=biological_nodes * 2,
        states_per_time=_states_per_time(description),
    )


def try_build_instance(gfa: Path, args, *, split: str) -> OnlineInstance | None:
    try:
        return build_instance(gfa, args)
    except subprocess.TimeoutExpired as exc:
        print(
            f"skip_timeout\t{split}\t{gfa}\tcopy_number_timeout={exc.timeout}",
            flush=True,
        )
        return None


def train_on_instance(model, optimizer, instance: OnlineInstance, args, device, seed: int) -> dict[str, float]:
    pheromone = np.ones_like(instance.heuristic, dtype=np.float32)
    best_energy = float("inf")
    mean_energies = []
    losses = []

    for step in range(args.online_steps):
        with torch.no_grad():
            prior_old = (
                build_prior_tensor(
                    model,
                    instance.graph,
                    instance.description,
                    instance.offsets,
                    instance.targets,
                    device,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        batches = []
        pheromones = []
        for inner in range(args.mini_h):
            pheromones.append(pheromone.copy())
            batch = qpg_aco_cpp.sample_batch(
                instance.offsets,
                instance.targets,
                pheromone,
                instance.heuristic,
                prior_old,
                instance.weights_array,
                instance.lengths_array,
                instance.q_float,
                float(instance.description.offset),
                instance.states_per_time,
                instance.description.T,
                args.ants,
                instance.start_source,
                instance.end_index,
                args.aco_alpha,
                args.aco_beta,
                args.gamma,
                seed + step * 1000 + inner,
                args.parallel_traced,
            )
            batches.append(batch)
            energies = np.asarray(batch["energies"], dtype=np.float32)
            mean_energies.append(float(np.mean(energies)))
            best_energy = min(best_energy, float(np.min(energies)))
            update_pheromone_from_batch(pheromone, batch, args.evaporation, args.elite_frac)

        optimizer.zero_grad(set_to_none=True)
        prior_new = build_prior_tensor(
            model,
            instance.graph,
            instance.description,
            instance.offsets,
            instance.targets,
            device,
        )
        step_losses = []
        for batch, tau in zip(batches, pheromones):
            costs_t = torch.as_tensor(
                np.asarray(batch["energies"], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            logp, _ndec = replay_log_probs(
                np.asarray(batch["trace_starts"], dtype=np.int32),
                np.asarray(batch["trace_chosen_edges"], dtype=np.int32),
                instance.offsets,
                instance.targets,
                instance.weights,
                instance.lengths,
                instance.end_index,
                tau,
                prior_new,
                args.aco_alpha,
                args.aco_beta,
                args.gamma,
                device,
            )
            advantage = (costs_t - costs_t.mean()).detach()
            step_losses.append((logp * advantage).mean())
        loss = torch.stack(step_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return {
        "best_energy": best_energy,
        "mean_energy": float(np.mean(mean_energies)),
        "loss": float(np.mean(losses)),
    }


def run_neural_aco_eval(model, instance: OnlineInstance, args, device, seed: int) -> tuple[float, float]:
    started = time.perf_counter()
    pheromone = np.ones_like(instance.heuristic, dtype=np.float32)
    best = None
    iterations = 0
    with torch.no_grad():
        prior = (
            build_prior_tensor(
                model,
                instance.graph,
                instance.description,
                instance.offsets,
                instance.targets,
                device,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    deadline = max(args.eval_time_limit, 1)
    while iterations < args.eval_min_iterations or time.perf_counter() - started < deadline:
        batch = qpg_aco_cpp.sample_batch(
            instance.offsets,
            instance.targets,
            pheromone,
            instance.heuristic,
            prior,
            instance.weights_array,
            instance.lengths_array,
            instance.q_float,
            float(instance.description.offset),
            instance.states_per_time,
            instance.description.T,
            args.eval_ants,
            instance.start_source,
            instance.end_index,
            args.aco_alpha,
            args.aco_beta,
            args.gamma,
            seed + iterations,
            args.parallel_traced,
        )
        choices_batch = np.asarray(batch["choices"], dtype=np.int32)
        energies = np.asarray(batch["energies"], dtype=np.float32)
        for ant_index in range(args.eval_ants):
            choices = tuple(int(x) for x in choices_batch[ant_index])
            result = _path_result_from_choices(choices, instance.description)
            if best is None or result[1] < best:
                best = float(result[1])

        pheromone *= 1.0 - args.evaporation
        np.maximum(pheromone, 1e-6, out=pheromone)
        worst_energy = float(energies.max())
        elite = np.argsort(energies)[: max(1, args.eval_ants // 4)]
        trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
        trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
        for ant_index in elite:
            deposit = (worst_energy - float(energies[ant_index]) + 1.0) / (abs(worst_energy) + 1.0)
            begin = int(trace_starts[ant_index])
            end = int(trace_starts[ant_index + 1])
            for edge_id in trace_edges[begin:end]:
                pheromone[int(edge_id)] += deposit
        iterations += 1
        if time.perf_counter() - started >= deadline and iterations >= args.eval_min_iterations:
            break
    return float(best), time.perf_counter() - started


def evaluate(model, test_gfas: list[Path], args, device) -> list[dict[str, object]]:
    rows = []
    for index, gfa in enumerate(tqdm(test_gfas, desc="final eval", unit="gfa")):
        instance = try_build_instance(gfa, args, split="eval")
        if instance is None:
            continue
        neural_energy, neural_runtime = run_neural_aco_eval(
            model,
            instance,
            args,
            device,
            args.seed + 900000 + index * 10000,
        )
        baseline_energy = None
        baseline_runtime = None
        if not args.no_aco_eval:
            started = time.perf_counter()
            previous_env = {
                "QPG_ACO_ANTS": os.environ.get("QPG_ACO_ANTS"),
                "QPG_ACO_MIN_ITERATIONS": os.environ.get("QPG_ACO_MIN_ITERATIONS"),
                "QPG_ACO_ALPHA": os.environ.get("QPG_ACO_ALPHA"),
                "QPG_ACO_BETA": os.environ.get("QPG_ACO_BETA"),
                "QPG_ACO_EVAPORATION": os.environ.get("QPG_ACO_EVAPORATION"),
            }
            os.environ["QPG_ACO_ANTS"] = str(args.eval_ants)
            os.environ["QPG_ACO_MIN_ITERATIONS"] = str(args.eval_min_iterations)
            os.environ["QPG_ACO_ALPHA"] = str(args.aco_alpha)
            os.environ["QPG_ACO_BETA"] = str(args.aco_beta)
            os.environ["QPG_ACO_EVAPORATION"] = str(args.evaporation)
            try:
                baseline_paths = aco_sample_qubo(instance.description)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            baseline_runtime = time.perf_counter() - started
            baseline_energy = min(
                float(energy)
                for runs in baseline_paths.values()
                for _solution, energy, _path in runs
            )
        row = {
            "gfa": str(gfa),
            "aco_energy": baseline_energy,
            "aco_runtime": baseline_runtime,
            "neural_aco_energy": neural_energy,
            "neural_aco_runtime": neural_runtime,
        }
        rows.append(row)
        if baseline_energy is None:
            print(f"eval\t{gfa}\tneural_aco={neural_energy:.12g}\t{neural_runtime:.3f}s")
        else:
            print(
                f"eval\t{gfa}\taco={baseline_energy:.12g}\t"
                f"neural_aco={neural_energy:.12g}\t{neural_runtime:.3f}s"
            )
    return rows


def validation_score(model, test_gfas: list[Path], args, device, *, epoch: int) -> dict[str, float] | None:
    if not test_gfas:
        return None
    val_gfas = test_gfas[: args.validate_limit] if args.validate_limit > 0 else test_gfas
    energies = []
    gaps_to_aco = []
    for index, gfa in enumerate(tqdm(val_gfas, desc=f"validation epoch {epoch}", unit="gfa", leave=False)):
        instance = try_build_instance(gfa, args, split="val")
        if instance is None:
            continue
        energy, runtime = run_neural_aco_eval(
            model,
            instance,
            args,
            device,
            args.seed + 700000 + epoch * 10000 + index,
        )
        energies.append(energy)
        baseline_energy = None
        if args.validation_objective == "gap_to_aco":
            previous_env = {
                "QPG_ACO_ANTS": os.environ.get("QPG_ACO_ANTS"),
                "QPG_ACO_MIN_ITERATIONS": os.environ.get("QPG_ACO_MIN_ITERATIONS"),
                "QPG_ACO_ALPHA": os.environ.get("QPG_ACO_ALPHA"),
                "QPG_ACO_BETA": os.environ.get("QPG_ACO_BETA"),
                "QPG_ACO_EVAPORATION": os.environ.get("QPG_ACO_EVAPORATION"),
            }
            os.environ["QPG_ACO_ANTS"] = str(args.eval_ants)
            os.environ["QPG_ACO_MIN_ITERATIONS"] = str(args.eval_min_iterations)
            os.environ["QPG_ACO_ALPHA"] = str(args.aco_alpha)
            os.environ["QPG_ACO_BETA"] = str(args.aco_beta)
            os.environ["QPG_ACO_EVAPORATION"] = str(args.evaporation)
            try:
                baseline_paths = aco_sample_qubo(instance.description)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            baseline_energy = min(
                float(score)
                for runs in baseline_paths.values()
                for _solution, score, _path in runs
            )
            gaps_to_aco.append(energy - baseline_energy)
        if baseline_energy is None:
            print(f"val\t{epoch}\t{Path(gfa).name}\tneural_aco={energy:.12g}\t{runtime:.3f}s")
        else:
            print(
                f"val\t{epoch}\t{Path(gfa).name}\taco={baseline_energy:.12g}\t"
                f"neural_aco={energy:.12g}\tgap_to_aco={energy - baseline_energy:.12g}\t{runtime:.3f}s"
            )
    if not energies:
        return None
    mean_energy = float(np.mean(energies))
    result = {"mean_energy": mean_energy}
    if gaps_to_aco:
        result["mean_gap_to_aco"] = float(np.mean(gaps_to_aco))
    objective = result["mean_gap_to_aco"] if args.validation_objective == "gap_to_aco" else mean_energy
    result["objective"] = objective
    print(
        f"val_summary\t{epoch}\tinstances={len(val_gfas)}\t"
        f"mean_energy={mean_energy:.12g}\tobjective={objective:.12g}\t"
        f"objective_name={args.validation_objective}"
    )
    return result


def main() -> int:
    require_torch()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, help="YAML config file. CLI flags override YAML values.")
    pre_args, _remaining = pre_parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--train-gfas", nargs="+", help="Training GFA files or shell-style patterns.")
    parser.add_argument("--train-list", action="append", help="Text file containing one training GFA path per line.")
    parser.add_argument("--train-glob", action="append", help="Additional glob for training GFAs.")
    parser.add_argument("--test-gfas", nargs="+", help="Held-out test GFA files or shell-style patterns.")
    parser.add_argument("--test-list", action="append", help="Text file containing one held-out GFA path per line.")
    parser.add_argument("--test-glob", action="append", help="Additional glob for held-out GFAs.")
    parser.add_argument("--generate-synthetic-train", default=0, type=int, help="Generate this many training GFAs.")
    parser.add_argument("--generate-synthetic-test", default=0, type=int, help="Generate this many held-out test GFAs.")
    parser.add_argument("--synthetic-dir", default=Path("results/dynaco_online/generated"), type=Path)
    parser.add_argument("--synthetic-seed", default=1000, type=int)
    parser.add_argument("--synthetic-min-segments", default=4, type=int)
    parser.add_argument("--synthetic-max-segments", default=10, type=int)
    parser.add_argument("--synthetic-min-length", default=40, type=int)
    parser.add_argument("--synthetic-max-length", default=160, type=int)
    parser.add_argument("--synthetic-bubble-rate", default=0.35, type=float)
    parser.add_argument("--synthetic-orientation-rate", default=0.15, type=float)
    parser.add_argument("--synthetic-shortcuts", default=2, type=int)
    parser.add_argument("--paper-pipeline-train", action="store_true", help="Generate/cache training GFAs from the paper data pipeline during training.")
    parser.add_argument("--paper-pipeline-validation", action="store_true", help="Generate a fixed validation set from the paper data pipeline.")
    parser.add_argument("--paper-pipeline-cache-dir", default=Path("results/dynaco_online/paper_pipeline_cache"), type=Path)
    parser.add_argument("--paper-pipeline-annotators", default="mg,km,ga")
    parser.add_argument("--paper-pipeline-test-sequences", default=5, type=int)
    parser.add_argument("--paper-pipeline-shred-depth", default=30, type=int)
    parser.add_argument("--paper-pipeline-train-start-seed", default=1, type=int)
    parser.add_argument("--paper-pipeline-val-start-seed", default=100001, type=int)
    parser.add_argument("--paper-pipeline-min-train-instances", default=16, type=int)
    parser.add_argument("--paper-pipeline-val-instances", default=16, type=int)
    parser.add_argument("--paper-pipeline-generation-attempt-limit", default=25, type=int)
    parser.add_argument(
        "--paper-pipeline-max-segments",
        default=30,
        type=int,
        help="Keep only generated GFAs with at most this many segments; 0 disables filtering.",
    )
    parser.add_argument(
        "--paper-pipeline-max-horizon",
        default=300,
        type=int,
        help="Keep only generated GFAs with estimated QUBO horizon at most this value; 0 disables filtering.",
    )
    parser.add_argument(
        "--paper-pipeline-index-horizon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use exact paper-copy-number horizon filtering while indexing generated GFAs. "
            "Disable to use a cheap segment-count horizon proxy and defer exact copy numbers "
            "to selected training instances."
        ),
    )
    parser.add_argument(
        "--paper-pipeline-fresh-epoch-instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In paper-pipeline training, train each epoch on fresh generated GFAs instead of resampling a growing pool.",
    )
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument(
        "--edge-support-file",
        type=Path,
        help="Optional CSV/TSV with source,target,support[,gfa] edge read-support targets.",
    )
    parser.add_argument(
        "--link-support-file",
        type=Path,
        help="Optional CSV/TSV with source,target,support[,gfa] long-range read/link support.",
    )
    parser.add_argument(
        "--haplotype-file",
        type=Path,
        help="Optional CSV/TSV with node,haplotype[,gfa] labels for haplotype switch penalties.",
    )
    parser.add_argument("--edge-loss-weight", default=0.5, type=float)
    parser.add_argument("--link-loss-weight", default=0.5, type=float)
    parser.add_argument("--haplotype-switch-weight", default=0.5, type=float)
    parser.add_argument("--link-window", default=8, type=int, help="Maximum oriented-walk span for long-range pair support scoring.")
    parser.add_argument("-p", "--penalties", default="200,50,1")
    parser.add_argument("--alpha-qubo", default=1.1, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument(
        "--steps-per-epoch",
        "--instances-per-epoch",
        dest="steps_per_epoch",
        default=1,
        type=int,
        help="Number of fresh/sampled training instances per epoch.",
    )
    parser.add_argument(
        "--checkpoint-every",
        default=1,
        type=int,
        help="Save a recoverable *_latest.pt snapshot every N epochs; 0 disables latest snapshots. Best checkpoints are always saved on improvement.",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist *_epochNNNN.pt checkpoints so downstream coverage selection can choose by full assembly quality.",
    )
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state from a training checkpoint.")
    traced_group = parser.add_mutually_exclusive_group()
    traced_group.add_argument(
        "--parallel-traced",
        dest="parallel_traced",
        action="store_true",
        help="Parallelize traced C++ ant sampling across OpenMP threads.",
    )
    traced_group.add_argument(
        "--no-parallel-traced",
        dest="parallel_traced",
        action="store_false",
        help="Force single-thread traced C++ sampling.",
    )
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for the C++ ACO backend.")
    parser.add_argument(
        "--validate-every",
        default=1,
        type=int,
        help="Run held-out validation every N epochs and save *_val_best.pt by mean validation energy; 0 disables in-loop validation.",
    )
    parser.add_argument(
        "--validate-limit",
        default=16,
        type=int,
        help="Maximum held-out GFAs to use for in-loop validation; 0 means all held-out GFAs.",
    )
    parser.add_argument(
        "--validation-objective",
        choices=("energy", "gap_to_aco"),
        default="energy",
        help=(
            "Checkpoint-selection objective for held-out validation. "
            "'energy' selects the lowest feasible-walk QUBO proxy energy; "
            "'gap_to_aco' selects the model with the best mean improvement over plain ACO."
        ),
    )
    parser.add_argument("--H", "--online-steps", dest="online_steps", default=10, type=int)
    parser.add_argument("--mini_H", "--mini-h", dest="mini_h", default=10, type=int)
    parser.add_argument("--n_ants", "--ants", dest="ants", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--alpha", "--aco-alpha", dest="aco_alpha", default=1.0, type=float)
    parser.add_argument("--beta", "--aco-beta", dest="aco_beta", default=1.0, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--rho", "--evaporation", dest="evaporation", default=0.1, type=float)
    parser.add_argument("--elite-frac", default=0.25, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--units", default=16, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-time-limit", default=1, type=int)
    parser.add_argument("--eval-min-iterations", default=None, type=int)
    parser.add_argument("--eval-ants", default=None, type=int)
    parser.add_argument("--no-aco-eval", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true", help="Save checkpoint/history without running final held-out evaluation.")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Log training and validation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="qpg-dynaco")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--out", type=Path)
    if pre_args.config is not None:
        parser.set_defaults(**config_defaults(load_config(pre_args.config)))
    args = parser.parse_args()
    if args.eval_min_iterations is None:
        args.eval_min_iterations = args.online_steps * args.mini_h
    if args.eval_ants is None:
        args.eval_ants = args.ants
    if args.out is None:
        raise ValueError("Provide --out or set out in the YAML config.")
    args.synthetic_dir = Path(args.synthetic_dir)
    args.paper_pipeline_cache_dir = Path(args.paper_pipeline_cache_dir)
    args.out = Path(args.out)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_gfas = collect_gfas(args.train_gfas, args.train_glob, required=False)
    test_gfas = collect_gfas(args.test_gfas, args.test_glob, required=False)
    train_gfas.extend(collect_gfa_lists(args.train_list))
    test_gfas.extend(collect_gfa_lists(args.test_list))
    train_gfas.extend(generate_synthetic_gfas(args, "train", args.generate_synthetic_train))
    test_gfas.extend(generate_synthetic_gfas(args, "test", args.generate_synthetic_test))
    train_source = None
    if args.paper_pipeline_train:
        train_source = PaperPipelineGfaSource(
            args,
            "train",
            start_seed=args.paper_pipeline_train_start_seed,
        )
        if not args.paper_pipeline_fresh_epoch_instances:
            train_gfas.extend(train_source.ensure(args.paper_pipeline_min_train_instances))
    if args.paper_pipeline_validation:
        val_source = PaperPipelineGfaSource(
            args,
            "val",
            start_seed=args.paper_pipeline_val_start_seed,
            target_instances=args.paper_pipeline_val_instances,
        )
        test_gfas.extend(val_source.ensure(args.paper_pipeline_val_instances))
    if not train_gfas and train_source is None:
        raise ValueError("Provide --train-gfas/--train-glob, --generate-synthetic-train, or --paper-pipeline-train.")

    device = torch.device(args.device)
    if args.threads is not None:
        qpg_aco_cpp.set_num_threads(args.threads)
    print(f"cpp_threads: {qpg_aco_cpp.get_max_threads()}")
    print(f"parallel_traced: {args.parallel_traced}")
    model = QPGSeeAGNN(units=args.units, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed_checkpoint: {args.resume}\tstart_epoch={start_epoch}")
    started = time.perf_counter()
    print(f"train_instances: {len(train_gfas)}")
    if args.paper_pipeline_fresh_epoch_instances:
        print(f"paper_pipeline_fresh_epoch_instances: {args.steps_per_epoch}")
    if test_gfas:
        print(f"test_instances: {len(test_gfas)}")
    print("epoch\tinstance\tmean_energy\tbest_energy\tloss")
    wandb_run = init_wandb(args)

    history = []
    best_checkpoint_energy = float("inf")
    best_val_energy = float("inf")
    best_checkpoint = checkpoint_path(args.out, "best")
    best_val_checkpoint = checkpoint_path(args.out, "val_best")
    latest_checkpoint = checkpoint_path(args.out, "latest")
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="train epochs", unit="epoch"):
        if train_source is not None:
            if args.paper_pipeline_fresh_epoch_instances:
                train_gfas = train_source.ensure_new(args.steps_per_epoch)
                print(f"paper_pipeline_epoch_fresh\tepoch={epoch}\ttrain_instances={len(train_gfas)}", flush=True)
            else:
                target_pool = args.paper_pipeline_min_train_instances + (epoch - start_epoch + 1) * args.steps_per_epoch
                train_gfas = train_source.ensure(target_pool)
                print(f"paper_pipeline_pool\tepoch={epoch}\ttrain_instances={len(train_gfas)}", flush=True)
        epoch_rows = []
        local_index = 0
        skipped_instances = 0
        for local_index in tqdm(
            range(args.steps_per_epoch),
            desc=f"epoch {epoch} instances",
            unit="inst",
            leave=False,
        ):
            while True:
                if args.paper_pipeline_fresh_epoch_instances:
                    if local_index >= len(train_gfas):
                        train_gfas.extend(train_source.ensure_new(args.steps_per_epoch - len(epoch_rows)))
                    gfa = train_gfas[local_index]
                else:
                    gfa = random.choice(train_gfas)
                instance = try_build_instance(gfa, args, split="train")
                if instance is not None:
                    break
                skipped_instances += 1
                if args.paper_pipeline_fresh_epoch_instances:
                    replacement = train_source.ensure_new(1)
                    train_gfas.extend(replacement)
                    local_index += 1
                    continue
                if skipped_instances >= args.paper_pipeline_generation_attempt_limit:
                    raise RuntimeError(
                        f"Could not build a train instance after {skipped_instances} skipped GFAs. "
                        "Check copy-number timeouts or lower the paper-pipeline size guard."
                    )
            stats = train_on_instance(
                model,
                optimizer,
                instance,
                args,
                device,
                args.seed + epoch * 100000 + local_index * 1000,
            )
            row = {"epoch": epoch, "gfa": str(gfa), **stats}
            history.append(row)
            epoch_rows.append(row)
            global_step = (epoch - 1) * args.steps_per_epoch + local_index + 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/global_step": global_step,
                        "train/instance_mean_energy": stats["mean_energy"],
                        "train/instance_best_energy": stats["best_energy"],
                        "train/instance_loss": stats["loss"],
                        "train/epoch": epoch,
                        "train/local_index": local_index,
                        "train/instance_segments": instance.description.V,
                        "train/instance_horizon": instance.description.T,
                        "train/instance_qubo_variables": int(instance.q_float.shape[0]),
                    },
                    step=global_step,
                )
        if not epoch_rows:
            raise RuntimeError(f"Epoch {epoch} produced no trainable instances.")
        mean_energy = float(np.mean([row["mean_energy"] for row in epoch_rows]))
        best_energy = float(np.min([row["best_energy"] for row in epoch_rows]))
        loss = float(np.mean([row["loss"] for row in epoch_rows]))
        global_step = epoch * args.steps_per_epoch
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/global_step": global_step,
                    "train/epoch_mean_energy": mean_energy,
                    "train/epoch_best_energy": best_energy,
                    "train/epoch_loss": loss,
                    "train/epoch_instances": len(epoch_rows),
                    "train/current_train_gfas": len(train_gfas),
                    "time/training_seconds": time.perf_counter() - started,
                },
                step=global_step,
            )
        checkpoint = checkpoint_payload(
            model,
            optimizer,
            args,
            train_gfas,
            epoch=epoch,
            best_energy=best_energy,
            training_seconds=time.perf_counter() - started,
        )
        if best_energy < best_checkpoint_energy:
            best_checkpoint_energy = best_energy
            best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_checkpoint)
            print(f"best_checkpoint: {best_checkpoint}\tepoch={epoch}\tbest_energy={best_energy:.12g}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/global_step": global_step,
                        "checkpoint/best_energy": best_energy,
                        "checkpoint/best_epoch": epoch,
                    },
                    step=global_step,
                )
        if args.checkpoint_every > 0 and (epoch == start_epoch or epoch % args.checkpoint_every == 0):
            latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, latest_checkpoint)
            if args.save_epoch_checkpoints:
                epoch_checkpoint = checkpoint_path(args.out, f"epoch{epoch:04d}")
                torch.save(checkpoint, epoch_checkpoint)
                print(f"epoch_checkpoint: {epoch_checkpoint}\tepoch={epoch}")
        if args.validate_every > 0 and test_gfas and (epoch == start_epoch or epoch % args.validate_every == 0):
            val_scores = validation_score(model, test_gfas, args, device, epoch=epoch)
            if wandb_run is not None and val_scores is not None:
                val_payload = {
                    "train/global_step": global_step,
                    "val/mean_energy": val_scores["mean_energy"],
                    "val/objective": val_scores["objective"],
                    "val/instances": min(len(test_gfas), args.validate_limit) if args.validate_limit > 0 else len(test_gfas),
                    "val/epoch": epoch,
                }
                if "mean_gap_to_aco" in val_scores:
                    val_payload["val/mean_gap_to_aco"] = val_scores["mean_gap_to_aco"]
                wandb_run.log(val_payload, step=global_step)
            if val_scores is not None and val_scores["objective"] < best_val_energy:
                best_val_energy = val_scores["objective"]
                val_checkpoint = checkpoint_payload(
                    model,
                    optimizer,
                    args,
                    train_gfas,
                    epoch=epoch,
                    best_energy=best_energy,
                    training_seconds=time.perf_counter() - started,
                    val_energy=val_scores["mean_energy"],
                    val_objective=val_scores["objective"],
                    validation_objective=args.validation_objective,
                )
                best_val_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(val_checkpoint, best_val_checkpoint)
                print(
                    f"val_best_checkpoint: {best_val_checkpoint}\tepoch={epoch}\t"
                    f"val_objective={val_scores['objective']:.12g}\t"
                    f"objective_name={args.validation_objective}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/global_step": global_step,
                            "checkpoint/val_best_objective": val_scores["objective"],
                            "checkpoint/val_best_epoch": epoch,
                        },
                        step=global_step,
                    )
        if epoch == start_epoch or epoch % max(1, args.epochs // 10) == 0:
            shown = Path(epoch_rows[-1]["gfa"]).name
            print(f"{epoch}\t{shown}\t{mean_energy:.12g}\t{best_energy:.12g}\t{loss:.6g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_payload(
        model,
        optimizer,
        args,
        train_gfas,
        epoch=args.epochs,
        best_energy=float(min(row["best_energy"] for row in history)) if history else float("inf"),
        training_seconds=time.perf_counter() - started,
    )
    torch.save(checkpoint, args.out)

    eval_rows = [] if args.skip_final_eval else (evaluate(model, test_gfas, args, device) if test_gfas else [])
    with args.out.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "train_gfas": [str(path) for path in train_gfas],
                "test_gfas": [str(path) for path in test_gfas],
                "history": history,
                "eval": eval_rows,
                "training_seconds": time.perf_counter() - started,
                "checkpoint": str(args.out),
                "best_checkpoint": str(best_checkpoint),
                "latest_checkpoint": str(latest_checkpoint),
                "best_checkpoint_energy": best_checkpoint_energy,
            },
            handle,
            indent=2,
        )
    print(f"checkpoint: {args.out}")
    print(f"best_checkpoint: {best_checkpoint}")
    if args.checkpoint_every > 0:
        print(f"latest_checkpoint: {latest_checkpoint}")
    print(f"training_seconds: {time.perf_counter() - started:.3f}")
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/global_step": args.epochs * args.steps_per_epoch,
                "time/training_seconds_total": time.perf_counter() - started,
                "checkpoint/final": str(args.out),
            },
            step=args.epochs * args.steps_per_epoch,
        )
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
