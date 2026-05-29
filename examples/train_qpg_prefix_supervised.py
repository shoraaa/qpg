#!/usr/bin/env python3
"""Supervised prefix training for neural SeeA*.

This is closer to method.tex than pure REINFORCE: collect complete legal walks,
turn every prefix into a value target y(n)=J(W)-g(n), and train the GNN value
head plus an auxiliary next-action policy head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
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
    _greedy_complete_prefix,
    _initial_indices,
    _one_hot_solution,
    _prefix_counts,
    _prefix_overcopy_cost,
    _successor_indices,
    _terminal_node_objective,
    astar_sample_qubo,
    greedy_residual_sample_qubo,
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


def choices_from_solution(solution: np.ndarray, horizon: int) -> tuple[int, ...]:
    states_per_time = len(solution) // horizon
    return tuple(int(np.argmax(solution[t * states_per_time : (t + 1) * states_per_time])) for t in range(horizon))


def random_residual_choices(description: QuboDescription, rng: random.Random) -> tuple[int, ...]:
    successors = _successor_indices(description)
    weights = [
        float(description.graph.nodes[list(description.graph.nodes)[2 * i]]["weight"])
        for i in range(description.V)
    ]
    current = rng.choice(_initial_indices(description))
    choices = [current]
    counts = list(_prefix_counts(tuple(choices), description.V))
    end_index = description.V * 2
    while len(choices) < description.T:
        legal = successors[current]
        weighted = []
        for successor in legal:
            if successor == end_index:
                remaining = sum(max(0.0, weights[i] - counts[i]) for i in range(description.V))
                score = 0.05 if remaining > 0 else 1.0
            else:
                score = 0.05 + max(0.0, weights[successor // 2] - counts[successor // 2])
            weighted.append(score)
        total = sum(weighted)
        pick = rng.random() * total
        acc = 0.0
        next_state = legal[-1]
        for successor, score in zip(legal, weighted):
            acc += score
            if acc >= pick:
                next_state = successor
                break
        choices.append(next_state)
        if next_state != end_index:
            counts[next_state // 2] += 1
        current = next_state
    return tuple(choices)


def collect_trajectories(description: QuboDescription, random_walks: int, seed: int) -> list[tuple[int, ...]]:
    trajectories: list[tuple[int, ...]] = []
    for sampler in (greedy_residual_sample_qubo, astar_sample_qubo):
        paths = sampler(description)
        for runs in paths.values():
            for solution, _energy, _path in runs:
                trajectories.append(choices_from_solution(solution, description.T))

    successors = _successor_indices(description)
    for start in _initial_indices(description):
        trajectories.append(_greedy_complete_prefix((start,), description, successors))

    rng = random.Random(seed)
    for _ in range(random_walks):
        trajectories.append(random_residual_choices(description, rng))

    unique = []
    seen = set()
    for choices in trajectories:
        if choices not in seen:
            unique.append(choices)
            seen.add(choices)
    return unique


def edge_index_for_action(edge_pairs: list[tuple[int, int]], current: int, next_state: int) -> int | None:
    for idx, pair in enumerate(edge_pairs):
        if pair == (current, next_state):
            return idx
    return None


def main() -> int:
    require_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--gfa", required=True, type=Path)
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument("-p", "--penalties", default="200,50,1")
    parser.add_argument("--alpha", default=1.1, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--random-walks", default=64, type=int)
    parser.add_argument(
        "--max-prefixes",
        default=512,
        type=int,
        help="Maximum sampled prefix training items per epoch. Use <=0 for all prefixes.",
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--policy-weight", default=0.1, type=float)
    parser.add_argument(
        "--target",
        choices=["energy-gap", "cost-to-go"],
        default="energy-gap",
        help="Value target: QUBO energy gap to best collected path, or method-style node cost-to-go.",
    )
    parser.add_argument("--units", default=32, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    graph = oriented_graph_with_copy_numbers(args.gfa, parse_copy_numbers(args.copy_numbers, args.gfa))
    q_matrix, offset, horizon, biological_nodes = qubo_matrix_from_graph(
        graph,
        alpha=args.alpha,
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
        solver=Solver.SEEA,
    )
    trajectories = collect_trajectories(description, args.random_walks, args.seed)
    device = torch.device(args.device)
    model = QPGSeeAGNN(units=args.units, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    started = time.perf_counter()

    print(f"GFA: {args.gfa}")
    print(f"QUBO shape: {q_matrix.shape}, T={horizon}, V={biological_nodes}")
    print(f"trajectories: {len(trajectories)}")
    print("epoch\tmean_loss\tbest_energy")

    trajectory_energies = {
        choices: _energy_from_choices(choices, description)
        for choices in trajectories
    }
    best_energy = min(trajectory_energies.values())
    training_items = []
    for choices in trajectories:
        terminal_target = _terminal_node_objective(choices, description)
        terminal_energy = trajectory_energies[choices]
        for prefix_len in range(1, len(choices) + 1):
            prefix = choices[:prefix_len]
            if args.target == "energy-gap":
                target = max(0.0, terminal_energy - best_energy)
            else:
                target = max(0.0, terminal_target - _prefix_overcopy_cost(prefix, description))
            next_state = choices[prefix_len] if prefix_len < len(choices) else None
            training_items.append((prefix, target, next_state, terminal_energy))

    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training_items)
        if args.max_prefixes > 0:
            epoch_items = training_items[: min(args.max_prefixes, len(training_items))]
        else:
            epoch_items = training_items
        losses = []
        for prefix, target, next_state, _terminal_energy in epoch_items:
            counts = _prefix_counts(prefix, biological_nodes)
            tensor = build_qpg_graph_tensor(
                graph,
                counts=counts,
                current_index=prefix[-1],
                depth=len(prefix),
                horizon=horizon,
                device=device,
            )
            output = model(tensor)
            target_t = torch.tensor(float(target), dtype=torch.float32, device=device)
            value_loss = torch.nn.functional.smooth_l1_loss(
                torch.log1p(output["value"]),
                torch.log1p(target_t),
            )
            policy_loss = torch.zeros((), dtype=torch.float32, device=device)
            if next_state is not None:
                edge_id = edge_index_for_action(tensor.edge_pairs, prefix[-1], next_state)
                if edge_id is not None:
                    legal = [idx for idx, (source, _target) in enumerate(tensor.edge_pairs) if source == prefix[-1]]
                    legal_t = torch.tensor(legal, dtype=torch.long, device=device)
                    logits = output["edge_logits"].index_select(0, legal_t)
                    local_target = legal.index(edge_id)
                    policy_loss = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([local_target], dtype=torch.long, device=device),
                    )
            loss = value_loss + args.policy_weight * policy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"{epoch}\t{np.mean(losses):.6g}\t{best_energy:.12g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "units": args.units,
                "depth": args.depth,
                "source": "prefix_supervised",
                "gfa": str(args.gfa),
                "target": args.target,
            },
            "best_energy": best_energy,
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
                "horizon": horizon,
                "biological_nodes": biological_nodes,
                "trajectories": len(trajectories),
                "training_items": len(training_items),
                "training_seconds": time.perf_counter() - started,
            },
            handle,
            indent=2,
        )

    best_choices = min(trajectories, key=lambda choices: _energy_from_choices(choices, description))
    solution = _one_hot_solution(best_choices, len(q_matrix) // horizon)
    print(f"training_seconds: {time.perf_counter() - started:.3f}")
    print(f"best_training_energy: {best_energy:.12g}")
    print("best_training_path:", sample_list_to_path(solution, graph, horizon, biological_nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
