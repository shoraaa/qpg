#!/usr/bin/env python3
"""Simple REINFORCE training for the QPG GNN policy.

This is intentionally minimal and DyNACO-like: sample legal walks from neural
edge logits, score terminal QUBO energy, subtract a moving/best-known baseline,
and update policy/value heads from the sampled trajectory.
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
from qubo_solvers.oriented_tangle.utils.graph_utils import (  # noqa: E402
    oriented_graph_with_copy_numbers,
)
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph  # noqa: E402
from qubo_solvers.oriented_tangle.utils.sampling_utils import (  # noqa: E402
    _energy_from_choices,
    greedy_residual_sample_qubo,
    sample_list_to_path,
    _one_hot_solution,
    _prefix_overcopy_cost,
    _terminal_node_objective,
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


def greedy_baseline_energy(description: QuboDescription) -> float:
    paths = greedy_residual_sample_qubo(description)
    return float(paths[description.time_limits[0]][0][1])


def edge_indices_from_source(edge_pairs: list[tuple[int, int]], source: int) -> list[int]:
    return [idx for idx, (src, _) in enumerate(edge_pairs) if src == source]


def sample_episode(model, graph, q_matrix, offset, horizon: int, biological_nodes: int, device, greedy: bool = False):
    counts = [0] * biological_nodes
    choices: list[int] = []
    log_probs = []
    entropies = []
    values = []
    prefix_costs = []
    current = None

    description = QuboDescription(
        filename="episode",
        data_dir="",
        graph=graph,
        time_limits=[1],
        jobs=1,
        Q=q_matrix,
        offset=offset,
        T=horizon,
        V=biological_nodes,
        solver=Solver.EXACT,
    )

    for depth in range(horizon):
        if current is None:
            tensor = build_qpg_graph_tensor(
                graph,
                counts=counts,
                current_index=len(graph.nodes) + 1,
                depth=depth,
                horizon=horizon,
                device=device,
            )
            current = tensor.start_index
        else:
            tensor = build_qpg_graph_tensor(
                graph,
                counts=counts,
                current_index=current,
                depth=depth,
                horizon=horizon,
                device=device,
            )
        output = model(tensor)
        values.append(output["value"])
        prefix_costs.append(_prefix_overcopy_cost(choices, description))
        edge_ids = edge_indices_from_source(tensor.edge_pairs, current)
        if not edge_ids:
            next_state = tensor.end_index
            choices.append(next_state)
            current = next_state
            continue

        edge_id_tensor = torch.tensor(edge_ids, dtype=torch.long, device=device)
        logits = output["edge_logits"].index_select(0, edge_id_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            local_action = torch.argmax(logits)
        else:
            local_action = dist.sample()
        selected_edge_id = edge_ids[int(local_action.detach().cpu())]
        next_state = tensor.edge_pairs[selected_edge_id][1]
        log_probs.append(dist.log_prob(local_action))
        entropies.append(dist.entropy())

        choices.append(next_state if next_state < tensor.end_index else tensor.end_index)
        if next_state < tensor.end_index:
            counts[next_state // 2] += 1
        current = next_state

    energy = _energy_from_choices(choices, description)
    terminal_cost = _terminal_node_objective(choices, description)
    cost_to_go_targets = [max(0.0, terminal_cost - prefix_cost) for prefix_cost in prefix_costs]
    return choices, float(energy), float(terminal_cost), log_probs, entropies, values, cost_to_go_targets


def main() -> int:
    require_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--gfa", default=REPO_ROOT / "examples" / "tiny_line.gfa", type=Path)
    parser.add_argument("-c", "--copy-numbers", default="ones")
    parser.add_argument("-p", "--penalties", default="200,50,1")
    parser.add_argument("--alpha", default=1.1, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--episodes", default=16, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--entropy", default=0.01, type=float)
    parser.add_argument("--value-weight", default=0.5, type=float)
    parser.add_argument("--units", default=64, type=int)
    parser.add_argument("--depth", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
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
        solver=Solver.GREEDY_RESIDUAL,
    )
    baseline_energy = greedy_baseline_energy(description)
    device = torch.device(args.device)
    model = QPGSeeAGNN(units=args.units, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    moving_baseline = baseline_energy
    best_energy = float("inf")
    best_choices: list[int] | None = None
    started = time.perf_counter()

    print(f"GFA: {args.gfa}")
    print(f"QUBO shape: {q_matrix.shape}, T={horizon}, V={biological_nodes}")
    print(f"greedy_baseline_energy: {baseline_energy:.12g}")
    print("epoch\tmean_energy\tbest_energy\tmoving_baseline\tloss")

    for epoch in range(1, args.epochs + 1):
        batch_terms = []
        energies = []
        optimizer.zero_grad()
        for _episode in range(args.episodes):
            choices, energy, terminal_cost, log_probs, entropies, values, cost_to_go_targets = sample_episode(
                model,
                graph,
                q_matrix,
                offset,
                horizon,
                biological_nodes,
                device,
            )
            energies.append(energy)
            if energy < best_energy:
                best_energy = energy
                best_choices = list(choices)
            reward = (moving_baseline - energy) / (abs(moving_baseline) + 1.0)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            if log_probs:
                logp_sum = torch.stack(log_probs).sum()
                entropy_sum = torch.stack(entropies).sum()
            else:
                logp_sum = torch.zeros((), dtype=torch.float32, device=device)
                entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
            advantage = reward_t
            policy_loss = -logp_sum * advantage
            target_values = torch.tensor(cost_to_go_targets, dtype=torch.float32, device=device)
            predicted_values = torch.stack(values)
            value_loss = torch.nn.functional.smooth_l1_loss(
                torch.log1p(predicted_values),
                torch.log1p(target_values),
            )
            entropy_loss = -args.entropy * entropy_sum
            batch_terms.append(policy_loss + args.value_weight * value_loss + entropy_loss)

        loss = torch.stack(batch_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mean_energy = float(np.mean(energies))
        moving_baseline = 0.9 * moving_baseline + 0.1 * mean_energy
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"{epoch}\t{mean_energy:.12g}\t{best_energy:.12g}\t{moving_baseline:.12g}\t{float(loss.detach()):.6g}")

    greedy_choices, greedy_energy, *_ = sample_episode(
        model,
        graph,
        q_matrix,
        offset,
        horizon,
        biological_nodes,
        device,
        greedy=True,
    )
    if greedy_energy < best_energy:
        best_energy = greedy_energy
        best_choices = list(greedy_choices)

    print(f"training_seconds: {time.perf_counter() - started:.3f}")
    print(f"best_sampled_energy: {best_energy:.12g}")
    print(f"greedy_policy_energy: {greedy_energy:.12g}")
    if best_choices is not None:
        solution = _one_hot_solution(best_choices, len(q_matrix) // horizon)
        path = sample_list_to_path(solution, graph, horizon, biological_nodes)
        print("best_path:", path)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "best_energy": best_energy,
                "best_choices": best_choices,
                "baseline_energy": baseline_energy,
            },
            args.out,
        )
        with args.out.with_suffix(".json").open("w") as handle:
            json.dump(
                {
                    "gfa": str(args.gfa),
                    "best_energy": best_energy,
                    "greedy_policy_energy": greedy_energy,
                    "baseline_energy": baseline_energy,
                    "horizon": horizon,
                    "biological_nodes": biological_nodes,
                },
                handle,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
