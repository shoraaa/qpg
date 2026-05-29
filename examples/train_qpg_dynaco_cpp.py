#!/usr/bin/env python3
"""C++-backed DyNACO-style training for QPG neural ACO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qubo_solvers.definitions import QuboDescription, Solver  # noqa: E402
from qubo_solvers.oriented_tangle import qpg_aco_cpp  # noqa: E402
from qubo_solvers.oriented_tangle.neural_gnn import QPGSeeAGNN, build_qpg_graph_tensor, require_torch, torch  # noqa: E402
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers  # noqa: E402
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    _aco_static_edge_arrays,
    _initial_indices,
    _node_weights_and_lengths,
    _one_hot_solution,
    _residual_successor_score,
    _path_result_from_choices,
    _states_per_time,
    aco_sample_qubo,
    sample_list_to_path,
)


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


def build_prior_tensor(model, graph, description: QuboDescription, offsets, targets, device):
    start_index = description.V * 2 + 1
    tensor = build_qpg_graph_tensor(
        graph,
        counts=[0] * description.V,
        current_index=start_index,
        depth=0,
        horizon=description.T,
        device=device,
    )
    edge_logits = model(tensor)["edge_logits"]
    pair_to_logit = {pair: edge_logits[idx] for idx, pair in enumerate(tensor.edge_pairs)}
    values = []
    for source in range(len(offsets) - 1):
        for edge_pos in range(int(offsets[source]), int(offsets[source + 1])):
            target = int(targets[edge_pos])
            values.append(pair_to_logit.get((source, target), edge_logits.new_zeros(())))
    return torch.stack(values)


def replay_log_probs(trace_starts, trace_edges, offsets, targets, weights, lengths, end_index, pheromone, prior, alpha, beta, gamma, device):
    offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=device)
    trace_starts_t = torch.as_tensor(trace_starts, dtype=torch.long, device=device)
    trace_edges_t = torch.as_tensor(trace_edges, dtype=torch.long, device=device)
    pheromone_t = torch.as_tensor(pheromone, dtype=torch.float32, device=device)
    targets_np = np.asarray(targets, dtype=np.int32)
    ant_logps = []
    ant_ndec = []
    for ant in range(trace_starts_t.numel() - 1):
        begin = int(trace_starts_t[ant])
        end = int(trace_starts_t[ant + 1])
        total = prior.new_zeros(())
        ndec = 0
        counts = [0] * len(weights)
        for idx in range(begin, end):
            chosen_edge = int(trace_edges_t[idx])
            source = int(np.searchsorted(offsets, chosen_edge, side="right") - 1)
            row_begin = int(offsets_t[source])
            row_end = int(offsets_t[source + 1])
            row_heuristic = torch.as_tensor(
                [
                    _residual_successor_score(int(targets_np[edge]), counts, weights, lengths, end_index)
                    for edge in range(row_begin, row_end)
                ],
                dtype=torch.float32,
                device=device,
            )
            row_static = (
                alpha * torch.log(pheromone_t[row_begin:row_end].clamp_min(1e-12))
                + beta * torch.log(row_heuristic.clamp_min(1e-12))
            )
            row_logits = row_static + gamma * prior[row_begin:row_end]
            local_edge = chosen_edge - row_begin
            total = total + row_logits[local_edge] - torch.logsumexp(row_logits, dim=0)
            ndec += 1
            target = int(targets_np[chosen_edge])
            if target != end_index:
                counts[target // 2] += 1
        ant_logps.append(total / max(ndec, 1))
        ant_ndec.append(ndec)
    return torch.stack(ant_logps), torch.tensor(ant_ndec, dtype=torch.float32, device=device)


def update_pheromone_from_batch(pheromone, batch, evaporation: float, elite_frac: float):
    energies = np.asarray(batch["energies"], dtype=np.float32)
    trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
    trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
    pheromone *= (1.0 - evaporation)
    np.maximum(pheromone, 1e-6, out=pheromone)
    elite_n = max(1, int(np.ceil(len(energies) * elite_frac)))
    elite = np.argsort(energies)[:elite_n]
    worst = float(energies.max())
    for ant in elite:
        deposit = (worst - float(energies[ant]) + 1.0) / (abs(worst) + 1.0)
        for edge_id in trace_edges[int(trace_starts[ant]) : int(trace_starts[ant + 1])]:
            pheromone[int(edge_id)] += deposit


def main() -> int:
    require_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--gfa", required=True, type=Path)
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument("-p", "--penalties", default="200,50,1")
    parser.add_argument("--alpha-qubo", default=1.1, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--outer", default=4, type=int)
    parser.add_argument("--mini-h", default=4, type=int)
    parser.add_argument("--ants", default=64, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--aco-alpha", default=1.0, type=float)
    parser.add_argument("--aco-beta", default=2.0, type=float)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--evaporation", default=0.2, type=float)
    parser.add_argument("--elite-frac", default=0.25, type=float)
    parser.add_argument("--units", default=16, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    traced_group = parser.add_mutually_exclusive_group()
    traced_group.add_argument("--parallel-traced", dest="parallel_traced", action="store_true")
    traced_group.add_argument("--no-parallel-traced", dest="parallel_traced", action="store_false")
    parser.set_defaults(parallel_traced=True)
    parser.add_argument("--threads", type=int, help="OpenMP thread count for the C++ ACO backend.")
    parser.add_argument("--stop-on-beat", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads is not None:
        qpg_aco_cpp.set_num_threads(args.threads)
    graph = oriented_graph_with_copy_numbers(args.gfa, parse_copy_numbers(args.copy_numbers, args.gfa))
    q_matrix, offset, horizon, biological_nodes = qubo_matrix_from_graph(
        graph,
        alpha=args.alpha_qubo,
        penalties=parse_csv_ints(args.penalties),
    )
    description = QuboDescription(
        filename=args.gfa.name,
        data_dir=str(args.gfa.parent),
        graph=graph,
        time_limits=[1],
        jobs=1,
        Q=q_matrix,
        offset=offset,
        T=horizon,
        V=biological_nodes,
        solver=Solver.NEURAL_ACO,
    )
    offsets, targets, heuristic, _prior0, start_source = _aco_static_edge_arrays(description)
    weights, lengths = _node_weights_and_lengths(description)
    weights_array = np.asarray(weights, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)
    end_index = biological_nodes * 2
    pheromone = np.ones_like(heuristic, dtype=np.float32)
    best_pheromone = pheromone.copy()
    states_per_time = _states_per_time(description)
    device = torch.device(args.device)
    model = QPGSeeAGNN(units=args.units, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    baseline_paths = aco_sample_qubo(description)
    baseline_energy = min(float(energy) for runs in baseline_paths.values() for _sol, energy, _path in runs)
    best_energy = float("inf")
    best_choices = None
    started = time.perf_counter()

    print(f"GFA: {args.gfa}")
    print(f"QUBO shape: {q_matrix.shape}, T={horizon}, V={biological_nodes}, edges={len(targets)}")
    print(f"aco_baseline_energy: {baseline_energy:.12g}")
    print("epoch\tmean_energy\tbest_energy\tloss")

    q_float = np.asarray(q_matrix, dtype=np.float32)
    for epoch in range(1, args.epochs + 1):
        epoch_energies = []
        epoch_losses = []
        for outer in range(args.outer):
            with torch.no_grad():
                prior_old = build_prior_tensor(model, graph, description, offsets, targets, device).detach().cpu().numpy().astype(np.float32)
            batches = []
            pheromones = []
            for inner in range(args.mini_h):
                pheromones.append(pheromone.copy())
                batch = qpg_aco_cpp.sample_batch(
                    offsets,
                    targets,
                    pheromone,
                    heuristic,
                    prior_old,
                    weights_array,
                    lengths_array,
                    q_float,
                    float(offset),
                    states_per_time,
                    horizon,
                    args.ants,
                    start_source,
                    end_index,
                    args.aco_alpha,
                    args.aco_beta,
                    args.gamma,
                    args.seed + epoch * 100000 + outer * 1000 + inner,
                    args.parallel_traced,
                )
                batches.append(batch)
                energies = np.asarray(batch["energies"], dtype=np.float32)
                epoch_energies.extend(float(x) for x in energies)
                best_idx = int(np.argmin(energies))
                if float(energies[best_idx]) < best_energy:
                    best_energy = float(energies[best_idx])
                    best_choices = tuple(int(x) for x in np.asarray(batch["choices"], dtype=np.int32)[best_idx])
                    best_pheromone = pheromone.copy()
                update_pheromone_from_batch(pheromone, batch, args.evaporation, args.elite_frac)

            optimizer.zero_grad(set_to_none=True)
            prior_new = build_prior_tensor(model, graph, description, offsets, targets, device)
            losses = []
            for batch, tau in zip(batches, pheromones):
                costs_t = torch.as_tensor(np.asarray(batch["energies"], dtype=np.float32), dtype=torch.float32, device=device)
                logp, _ndec = replay_log_probs(
                    np.asarray(batch["trace_starts"], dtype=np.int32),
                    np.asarray(batch["trace_chosen_edges"], dtype=np.int32),
                    offsets,
                    targets,
                    weights,
                    lengths,
                    end_index,
                    tau,
                    prior_new,
                    args.aco_alpha,
                    args.aco_beta,
                    args.gamma,
                    device,
                )
                adv = (costs_t - costs_t.mean()).detach()
                losses.append((logp * adv).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"{epoch}\t{float(np.mean(epoch_energies)):.12g}\t{best_energy:.12g}\t{float(np.mean(epoch_losses)):.6g}")
        if args.stop_on_beat and best_energy < baseline_energy:
            print(f"beat_aco_at_epoch: {epoch}")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "units": args.units,
                "depth": args.depth,
                "source": "dynaco_cpp",
                "gfa": str(args.gfa),
            },
            "best_energy": best_energy,
            "baseline_energy": baseline_energy,
            "horizon": horizon,
            "biological_nodes": biological_nodes,
            "pheromone": best_pheromone,
            "offsets": offsets,
            "targets": targets,
            "heuristic": heuristic,
            "weights": weights_array,
            "lengths": lengths_array,
        },
        args.out,
    )
    with args.out.with_suffix(".json").open("w") as handle:
        json.dump(
            {
                "gfa": str(args.gfa),
                "best_energy": best_energy,
                "baseline_energy": baseline_energy,
                "horizon": horizon,
                "biological_nodes": biological_nodes,
                "training_seconds": time.perf_counter() - started,
            },
            handle,
            indent=2,
        )

    print(f"training_seconds: {time.perf_counter() - started:.3f}")
    print(f"best_sampled_energy: {best_energy:.12g}")
    if best_choices is not None:
        solution = _one_hot_solution(best_choices, len(q_matrix) // horizon)
        print("best_path:", sample_list_to_path(solution, graph, horizon, biological_nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
