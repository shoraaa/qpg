#!/usr/bin/env python3
"""Train a QPG neural ACO prior with a walk-coverage surrogate reward.

This entrypoint intentionally leaves train.py unchanged.  It reuses the same
paper-pipeline data loader, model, C++ feasible-walk sampler, validation code,
and checkpoint format as the QUBO-energy trainer, but replaces the REINFORCE
training cost with a cheap walk-level surrogate shaped toward sequence
coverage and copy-number satisfaction.
"""

from __future__ import annotations

import numpy as np
import torch

from examples import train_qpg_dynaco_online as base
from train_qpg_dynaco_cpp import build_prior_tensor, parse_copy_numbers, replay_log_probs
from qubo_solvers.definitions import QuboDescription, Solver
from qubo_solvers.oriented_tangle import qpg_aco_cpp
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers
from qubo_solvers.oriented_tangle.utils.sampling_utils import (
    _aco_static_edge_arrays,
    _node_weights_and_lengths,
    _states_per_time,
)


BASE_CHECKPOINT_PAYLOAD = base.checkpoint_payload
BASE_BUILD_INSTANCE = base.build_instance
BASE_VALIDATION_SCORE = base.validation_score
BASE_EVALUATE = base.evaluate


def build_instance_no_qubo(gfa, args) -> base.OnlineInstance:
    graph = oriented_graph_with_copy_numbers(gfa, parse_copy_numbers(args.copy_numbers, gfa))
    nodes = list(graph.nodes)
    biological_nodes = int(len(nodes) / 2)
    total_weight = sum(float(graph.nodes[node]["weight"]) for node in nodes) / 2.0
    horizon = max(int(np.floor(total_weight * args.alpha_qubo)), 1)
    description = QuboDescription(
        filename=gfa.name,
        data_dir=str(gfa.parent),
        graph=graph,
        time_limits=[args.eval_time_limit],
        jobs=1,
        Q=np.zeros((1, 1), dtype=np.float32),
        offset=0.0,
        T=horizon,
        V=biological_nodes,
        solver=Solver.NEURAL_ACO,
    )
    offsets, targets, heuristic, _prior0, start_source = _aco_static_edge_arrays(description)
    weights, lengths = _node_weights_and_lengths(description)
    return base.OnlineInstance(
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
        q_float=np.zeros((1, 1), dtype=np.float32),
        start_source=start_source,
        end_index=biological_nodes * 2,
        states_per_time=_states_per_time(description),
    )


def walk_coverage_proxy_costs(choices_batch: np.ndarray, instance: base.OnlineInstance) -> np.ndarray:
    """Return costs where lower is better for coverage-shaped feasible walks."""
    weights = np.asarray(instance.weights, dtype=np.float32)
    lengths = np.asarray(instance.lengths, dtype=np.float32)
    target_copy = np.maximum(weights, 0.0)
    target_bases = float(np.sum(np.minimum(target_copy, 1.0) * lengths))
    if not np.isfinite(target_bases) or target_bases <= 0.0:
        target_bases = float(np.sum(lengths)) if float(np.sum(lengths)) > 0.0 else 1.0

    costs = np.zeros(choices_batch.shape[0], dtype=np.float32)
    for ant_index, choices in enumerate(choices_batch):
        counts = np.zeros_like(weights, dtype=np.float32)
        first_end = None
        for step, target in enumerate(choices):
            target = int(target)
            if target == instance.end_index:
                if first_end is None:
                    first_end = step
                continue
            node_index = target // 2
            if 0 <= node_index < len(counts):
                counts[node_index] += 1.0

        covered_bases = float(np.sum(np.minimum(counts, target_copy) * lengths))
        overcopy_bases = float(np.sum(np.maximum(0.0, counts - target_copy) * lengths))
        undercopy_bases = float(np.sum(np.maximum(0.0, target_copy - counts) * lengths))
        used_bases = float(np.sum((counts > 0.0).astype(np.float32) * lengths))
        end_step = choices_batch.shape[1] if first_end is None else first_end
        early_stop = max(0.0, 1.0 - float(end_step) / max(1.0, float(choices_batch.shape[1])))

        covered_frac = covered_bases / target_bases
        used_frac = used_bases / target_bases
        overcopy_frac = overcopy_bases / target_bases
        undercopy_frac = undercopy_bases / target_bases

        reward = (
            covered_frac
            + 0.10 * used_frac
            - 0.75 * overcopy_frac
            - 0.25 * undercopy_frac
            - 0.10 * early_stop
        )
        costs[ant_index] = -float(reward)
    return costs


def update_pheromone_from_proxy(
    pheromone: np.ndarray,
    batch,
    costs: np.ndarray,
    evaporation: float,
    elite_frac: float,
) -> None:
    pheromone *= 1.0 - evaporation
    np.maximum(pheromone, 1e-6, out=pheromone)
    elite_count = max(1, int(len(costs) * elite_frac))
    elite_indices = np.argsort(costs)[:elite_count]
    worst_cost = float(np.max(costs))
    trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
    trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
    for ant_index in elite_indices:
        deposit = (worst_cost - float(costs[ant_index]) + 1.0) / (abs(worst_cost) + 1.0)
        begin = int(trace_starts[ant_index])
        end = int(trace_starts[ant_index + 1])
        for edge_id in trace_edges[begin:end]:
            pheromone[int(edge_id)] += deposit


def train_on_instance_walk_proxy(model, optimizer, instance: base.OnlineInstance, args, device, seed: int) -> dict[str, float]:
    pheromone = np.ones_like(instance.heuristic, dtype=np.float32)
    best_cost = float("inf")
    mean_costs = []
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
        batch_costs = []
        for inner in range(args.mini_h):
            pheromones.append(pheromone.copy())
            batch = qpg_aco_cpp.sample_batch_traces(
                instance.offsets,
                instance.targets,
                pheromone,
                instance.heuristic,
                prior_old,
                instance.weights_array,
                instance.lengths_array,
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
            costs = walk_coverage_proxy_costs(np.asarray(batch["choices"], dtype=np.int32), instance)
            batches.append(batch)
            batch_costs.append(costs)
            mean_costs.append(float(np.mean(costs)))
            best_cost = min(best_cost, float(np.min(costs)))
            update_pheromone_from_proxy(pheromone, batch, costs, args.evaporation, args.elite_frac)

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
        for batch, tau, costs in zip(batches, pheromones, batch_costs):
            costs_t = torch.as_tensor(costs, dtype=torch.float32, device=device)
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
        "best_energy": best_cost,
        "mean_energy": float(np.mean(mean_costs)),
        "loss": float(np.mean(losses)),
    }


def checkpoint_payload_walk_proxy(*args, **kwargs):
    payload = BASE_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload["training_objective"] = "walk_coverage_proxy"
    payload.setdefault("config", {})["source"] = "dynaco_walk_coverage_proxy"
    payload["config"]["training_objective"] = "walk_coverage_proxy"
    return payload


def validation_score_qubo(model, test_gfas, args, device, *, epoch: int):
    previous = base.build_instance
    base.build_instance = BASE_BUILD_INSTANCE
    try:
        return BASE_VALIDATION_SCORE(model, test_gfas, args, device, epoch=epoch)
    finally:
        base.build_instance = previous


def evaluate_qubo(model, test_gfas, args, device):
    previous = base.build_instance
    base.build_instance = BASE_BUILD_INSTANCE
    try:
        return BASE_EVALUATE(model, test_gfas, args, device)
    finally:
        base.build_instance = previous


def main() -> int:
    if not hasattr(qpg_aco_cpp, "sample_batch_traces"):
        raise RuntimeError(
            "qpg_aco_cpp.sample_batch_traces is missing. Rebuild the C++ extension "
            "from qubo/qubo_solvers/oriented_tangle/cpp/qpg_aco_cpp.cpp."
        )
    base.build_instance = build_instance_no_qubo
    base.train_on_instance = train_on_instance_walk_proxy
    base.checkpoint_payload = checkpoint_payload_walk_proxy
    base.validation_score = validation_score_qubo
    base.evaluate = evaluate_qubo
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
