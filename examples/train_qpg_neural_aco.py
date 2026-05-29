#!/usr/bin/env python3
"""DyNACO-style neural ACO training for QPG.

The loop mirrors the useful DyNACO pattern in this repo's setting:
  1. keep an ACO pheromone table;
  2. build a neural edge prior with a QPG GNN;
  3. sample ant walks from pheromone + residual heuristic + neural prior;
  4. score full walks by QUBO energy;
  5. train the prior with REINFORCE advantages and update pheromone from elites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qubo"))

from qubo_solvers.definitions import QuboDescription, Solver  # noqa: E402
from qubo_solvers.oriented_tangle.neural_gnn import (  # noqa: E402
    QPGSeeAGNN,
    build_qpg_graph_tensor,
    require_torch,
    torch,
)
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers  # noqa: E402
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    _energy_from_choices,
    _initial_indices,
    _node_weights_and_lengths,
    _one_hot_solution,
    _path_result_from_choices,
    _prefix_counts,
    _residual_successor_score,
    _successor_indices,
    aco_sample_qubo,
    sample_list_to_path,
)


def count_segments(gfa: Path) -> int:
    with gfa.open() as handle:
        return sum(1 for line in handle if line.startswith("S\t"))


def parse_copy_numbers(value: str, gfa: Path) -> list[float]:
    if value == "ones":
        return [1.0] * count_segments(gfa)
    return [float(item) for item in value.split(",") if item]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def edge_ids_for_source(tensor, source: int, legal: list[int]) -> list[int]:
    legal_set = set(legal)
    return [
        idx
        for idx, (edge_source, edge_target) in enumerate(tensor.edge_pairs)
        if edge_source == source and edge_target in legal_set
    ]


def sample_neural_ant(
    model: QPGSeeAGNN,
    description: QuboDescription,
    successors: dict[int, list[int]],
    pheromone: dict[tuple[int, int], float],
    device,
    alpha: float,
    beta: float,
    gamma: float,
):
    end_index = description.V * 2
    weights, lengths = _node_weights_and_lengths(description)
    counts = [0] * description.V
    choices: list[int] = []
    log_probs = []
    entropies = []
    current = None

    while len(choices) < description.T:
        legal = _initial_indices(description) if current is None else successors[current]
        current_index = end_index + 1 if current is None else current
        tensor = build_qpg_graph_tensor(
            description.graph,
            counts=counts,
            current_index=current_index,
            depth=len(choices),
            horizon=description.T,
            device=device,
        )
        output = model(tensor)
        source_index = tensor.start_index if current is None else current
        edge_ids = edge_ids_for_source(tensor, source_index, legal)
        if not edge_ids:
            next_state = end_index
            choices.append(next_state)
            current = next_state
            continue

        edge_id_tensor = torch.tensor(edge_ids, dtype=torch.long, device=device)
        neural_logits = output["edge_logits"].index_select(0, edge_id_tensor)
        log_scores = []
        source_key = end_index if current is None else current
        for edge_id in edge_ids:
            option = tensor.edge_pairs[edge_id][1]
            heuristic = _residual_successor_score(option, counts, weights, lengths, end_index)
            trail = pheromone.get((source_key, option), 1.0)
            log_scores.append(
                alpha * np.log(max(trail, 1e-12)) + beta * np.log(max(heuristic, 1e-12))
            )
        static_logits = torch.tensor(log_scores, dtype=torch.float32, device=device)
        logits = static_logits + gamma * neural_logits
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        selected_edge = edge_ids[int(action.detach().cpu())]
        next_state = tensor.edge_pairs[selected_edge][1]
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        choices.append(next_state)
        if next_state != end_index:
            counts[next_state // 2] += 1
        current = next_state

    energy = _energy_from_choices(choices, description)
    return tuple(choices), float(energy), log_probs, entropies


def update_pheromone(
    pheromone: dict[tuple[int, int], float],
    elite: list[tuple[tuple[int, ...], float]],
    evaporation: float,
    start_source: int,
) -> None:
    for edge in list(pheromone):
        pheromone[edge] *= 1.0 - evaporation
        if pheromone[edge] < 1e-6:
            pheromone[edge] = 1e-6
    if not elite:
        return
    worst = max(energy for _choices, energy in elite)
    for choices, energy in elite:
        deposit = (worst - energy + 1.0) / (abs(worst) + 1.0)
        source = start_source
        for target in choices:
            pheromone[(source, target)] = pheromone.get((source, target), 1.0) + deposit
            source = target


def main() -> int:
    require_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--gfa", required=True, type=Path)
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument("-p", "--penalties", default="200,50,1")
    parser.add_argument("--alpha-qubo", default=1.1, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--ants", default=32, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--aco-alpha", default=1.0, type=float)
    parser.add_argument("--aco-beta", default=2.0, type=float)
    parser.add_argument("--neural-gamma", default=1.0, type=float)
    parser.add_argument("--evaporation", default=0.2, type=float)
    parser.add_argument("--entropy", default=0.01, type=float)
    parser.add_argument("--units", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init", type=Path, help="Optional checkpoint to initialize the GNN.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    device = torch.device(args.device)
    model = QPGSeeAGNN(units=args.units, depth=args.depth).to(device)
    if args.init is not None:
        checkpoint = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    successors = _successor_indices(description)
    start_source = description.V * 2
    pheromone = {
        (source, target): 1.0
        for source, targets in successors.items()
        for target in targets
    }
    for start in _initial_indices(description):
        pheromone[(start_source, start)] = 1.0

    baseline_paths = aco_sample_qubo(description)
    baseline_energy = min(float(energy) for runs in baseline_paths.values() for _sol, energy, _path in runs)
    moving_baseline = baseline_energy
    best_choices = None
    best_energy = float("inf")
    started = time.perf_counter()

    print(f"GFA: {args.gfa}")
    print(f"QUBO shape: {q_matrix.shape}, T={horizon}, V={biological_nodes}")
    print(f"aco_baseline_energy: {baseline_energy:.12g}")
    print("epoch\tmean_energy\tbest_energy\tmoving_baseline\tloss")

    for epoch in range(1, args.epochs + 1):
        batch_terms = []
        sampled = []
        energies = []
        optimizer.zero_grad()
        for _ant in range(args.ants):
            choices, energy, log_probs, entropies = sample_neural_ant(
                model,
                description,
                successors,
                pheromone,
                device,
                alpha=args.aco_alpha,
                beta=args.aco_beta,
                gamma=args.neural_gamma,
            )
            sampled.append((choices, energy))
            energies.append(energy)
            if energy < best_energy:
                best_energy = energy
                best_choices = choices
            reward = (moving_baseline - energy) / (abs(moving_baseline) + 1.0)
            if log_probs:
                logp_sum = torch.stack(log_probs).sum()
                entropy_sum = torch.stack(entropies).sum()
            else:
                logp_sum = torch.zeros((), dtype=torch.float32, device=device)
                entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
            advantage = torch.tensor(reward, dtype=torch.float32, device=device)
            batch_terms.append(-logp_sum * advantage - args.entropy * entropy_sum)

        loss = torch.stack(batch_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        moving_baseline = 0.9 * moving_baseline + 0.1 * float(np.mean(energies))
        elite = sorted(sampled, key=lambda item: item[1])[: max(1, args.ants // 4)]
        update_pheromone(pheromone, elite, args.evaporation, start_source)

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(
                f"{epoch}\t{float(np.mean(energies)):.12g}\t{best_energy:.12g}\t"
                f"{moving_baseline:.12g}\t{float(loss.detach()):.6g}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "units": args.units,
                "depth": args.depth,
                "source": "neural_aco",
                "gfa": str(args.gfa),
            },
            "best_energy": best_energy,
            "baseline_energy": baseline_energy,
            "horizon": horizon,
            "biological_nodes": biological_nodes,
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
