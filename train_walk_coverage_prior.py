#!/usr/bin/env python3
"""Train a QPG neural ACO prior with a structural walk surrogate reward.

This entrypoint intentionally leaves train.py unchanged.  It reuses the same
paper-pipeline data loader, model, C++ feasible-walk sampler, validation code,
and checkpoint format as the QUBO-energy trainer, but replaces the REINFORCE
training cost with a cheap walk-level surrogate shaped toward copy-number
coverage, parsimonious traversal, and structural edge reuse.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

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


def _open_evidence_rows(path: Path):
    sample = path.read_text().splitlines()
    sample = [line for line in sample if line.strip() and not line.lstrip().startswith("#")]
    if not sample:
        return []
    delimiter = "\t" if "\t" in sample[0] else ","
    first = [cell.strip() for cell in sample[0].split(delimiter)]
    has_header = any(cell.lower() in {"gfa", "graph", "source", "src", "from", "target", "dst", "to", "node", "support", "weight", "haplotype", "hap"} for cell in first)
    rows = []
    if has_header:
        reader = csv.DictReader(sample, delimiter=delimiter)
        for row in reader:
            rows.append({str(key).strip().lower(): str(value).strip() for key, value in row.items() if key is not None})
    else:
        for line in sample:
            cells = [cell.strip() for cell in line.split(delimiter)]
            rows.append({str(index): value for index, value in enumerate(cells)})
    return rows


def _row_matches_gfa(row: dict[str, str], gfa: Path) -> bool:
    value = row.get("gfa") or row.get("graph") or row.get("file") or row.get("filename")
    if not value:
        return True
    path = Path(value)
    return value == str(gfa) or path.name == gfa.name or path.stem == gfa.stem


def _row_value(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return default


def _parse_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_id(token: str, node_to_index: dict[str, int], bio_to_indices: dict[str, list[int]]) -> int | None:
    token = str(token).strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    if token in node_to_index:
        return node_to_index[token]
    if token.endswith("+"):
        return node_to_index.get(f"{token[:-1]}_+")
    if token.endswith("-"):
        return node_to_index.get(f"{token[:-1]}_-")
    matches = bio_to_indices.get(token)
    if matches and len(matches) == 1:
        return matches[0]
    return None


def _state_pair_candidates(
    source_token: str,
    target_token: str,
    node_to_index: dict[str, int],
    bio_to_indices: dict[str, list[int]],
) -> list[tuple[int, int]]:
    source = _state_id(source_token, node_to_index, bio_to_indices)
    target = _state_id(target_token, node_to_index, bio_to_indices)
    if source is not None and target is not None:
        return [(source, target)]
    source_states = bio_to_indices.get(source_token, [])
    target_states = bio_to_indices.get(target_token, [])
    return [(src, dst) for src in source_states for dst in target_states]


def _evidence_maps(graph, gfa: Path, edge_pairs: list[tuple[int, int]], args) -> tuple[np.ndarray | None, dict[tuple[int, int], float], np.ndarray | None]:
    node_names = list(graph.nodes)
    biological_nodes = len(node_names) // 2
    node_to_index = {node: index for index, node in enumerate(node_names)}
    bio_to_indices: dict[str, list[int]] = {}
    for index, node in enumerate(node_names):
        base = node[:-2] if node.endswith(("_+", "_-")) else node
        bio_to_indices.setdefault(base, []).append(index)

    pair_to_edge = {pair: index for index, pair in enumerate(edge_pairs)}
    edge_support = None
    if getattr(args, "edge_support_file", None):
        edge_support = np.full(len(edge_pairs), np.nan, dtype=np.float32)
        for row in _open_evidence_rows(Path(args.edge_support_file)):
            if not _row_matches_gfa(row, gfa):
                continue
            source = _row_value(row, "source", "src", "from", "0")
            target = _row_value(row, "target", "dst", "to", "1")
            support = _parse_float(_row_value(row, "support", "weight", "copy", "2"))
            if not math.isfinite(support):
                continue
            for pair in _state_pair_candidates(source, target, node_to_index, bio_to_indices):
                edge_id = pair_to_edge.get(pair)
                if edge_id is not None:
                    edge_support[edge_id] = support

    link_support: dict[tuple[int, int], float] = {}
    if getattr(args, "link_support_file", None):
        for row in _open_evidence_rows(Path(args.link_support_file)):
            if not _row_matches_gfa(row, gfa):
                continue
            source = _row_value(row, "source", "src", "from", "left", "0")
            target = _row_value(row, "target", "dst", "to", "right", "1")
            support = _parse_float(_row_value(row, "support", "weight", "count", "2"), default=1.0)
            if not math.isfinite(support) or support <= 0.0:
                continue
            for pair in _state_pair_candidates(source, target, node_to_index, bio_to_indices):
                link_support[pair] = max(link_support.get(pair, 0.0), support)

    node_haplotype = None
    if getattr(args, "haplotype_file", None):
        label_to_value: dict[str, float] = {}
        node_haplotype = np.full(biological_nodes, np.nan, dtype=np.float32)
        for row in _open_evidence_rows(Path(args.haplotype_file)):
            if not _row_matches_gfa(row, gfa):
                continue
            node = _row_value(row, "node", "segment", "source", "0")
            label = _row_value(row, "haplotype", "hap", "label", "1")
            if not node or not label:
                continue
            value = _parse_float(label)
            if not math.isfinite(value):
                if label not in label_to_value:
                    label_to_value[label] = float(len(label_to_value) + 1)
                value = label_to_value[label]
            states = bio_to_indices.get(node, [])
            state = _state_id(node, node_to_index, bio_to_indices)
            if state is not None:
                states = [state]
            for state_index in states:
                if 0 <= state_index < biological_nodes * 2:
                    node_haplotype[state_index // 2] = value

    return edge_support, link_support, node_haplotype


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
    edge_pairs = [
        (source, int(targets[edge_pos]))
        for source in range(len(offsets) - 1)
        for edge_pos in range(int(offsets[source]), int(offsets[source + 1]))
    ]
    edge_support, link_support, node_haplotype = _evidence_maps(graph, gfa, edge_pairs, args)
    instance = base.OnlineInstance(
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
    instance.edge_support = edge_support
    instance.link_support = link_support
    instance.node_haplotype = node_haplotype
    instance.link_window = int(getattr(args, "link_window", 8))
    instance.edge_loss_weight = float(getattr(args, "edge_loss_weight", 0.5))
    instance.link_loss_weight = float(getattr(args, "link_loss_weight", 0.5))
    instance.haplotype_switch_weight = float(getattr(args, "haplotype_switch_weight", 0.5))
    return instance


def structural_walk_proxy_costs(
    choices_batch: np.ndarray,
    instance: base.OnlineInstance,
    trace_starts: np.ndarray | None = None,
    trace_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Return costs for parsimonious, structurally supported feasible walks.

    The current paper-pipeline GFAs do not carry read-pair or haplotype support,
    so the edge term is a conservative graph-structural surrogate: repeated
    traversal of the same oriented edge is penalized once it exceeds the copy
    support implied by its incident node-copy estimates.
    """
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
        high_conf_target = np.minimum(target_copy, 1.0)
        high_conf_covered_bases = float(np.sum(np.minimum(counts, high_conf_target) * lengths))
        overcopy_bases = float(np.sum(np.maximum(0.0, counts - target_copy) * lengths))
        undercopy_bases = float(np.sum(np.maximum(0.0, target_copy - counts) * lengths))
        low_conf_used_bases = float(np.sum(((counts > 0.0) & (target_copy < 0.5)).astype(np.float32) * lengths))
        used_bases = float(np.sum((counts > 0.0).astype(np.float32) * lengths))
        redundant_bases = max(0.0, used_bases - high_conf_covered_bases)
        end_step = choices_batch.shape[1] if first_end is None else first_end
        active_step_frac = float(end_step) / max(1.0, float(choices_batch.shape[1]))

        covered_frac = covered_bases / target_bases
        high_conf_covered_frac = high_conf_covered_bases / target_bases
        overcopy_frac = overcopy_bases / target_bases
        undercopy_frac = undercopy_bases / target_bases
        low_conf_used_frac = low_conf_used_bases / target_bases
        redundant_frac = redundant_bases / target_bases

        edge_overuse_frac = 0.0
        link_penalty_frac = 0.0
        link_reward_frac = 0.0
        haplotype_switch_frac = 0.0
        if trace_starts is not None and trace_edges is not None:
            begin = int(trace_starts[ant_index])
            end = int(trace_starts[ant_index + 1])
            edge_counts: dict[int, int] = {}
            edge_supports: dict[int, float] = {}
            active_edges = 0
            oriented_path: list[int] = []
            link_support = getattr(instance, "link_support", {}) or {}
            node_haplotype = getattr(instance, "node_haplotype", None)
            link_evidence_nodes = {node for pair in link_support for node in pair}
            hap_switches = 0.0
            hap_edges = 0
            for edge_id in np.asarray(trace_edges[begin:end], dtype=np.int32):
                edge_id = int(edge_id)
                target = int(instance.targets[edge_id])
                if target == instance.end_index:
                    continue
                source = int(np.searchsorted(instance.offsets, edge_id, side="right") - 1)
                if source == instance.end_index:
                    continue
                if not oriented_path or oriented_path[-1] != source:
                    oriented_path.append(source)
                oriented_path.append(target)
                source_node = source // 2
                target_node = target // 2
                if not (0 <= source_node < len(target_copy) and 0 <= target_node < len(target_copy)):
                    continue
                edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
                explicit_edge_support = getattr(instance, "edge_support", None)
                edge_support = math.nan
                if explicit_edge_support is not None and 0 <= edge_id < len(explicit_edge_support):
                    edge_support = float(explicit_edge_support[edge_id])
                if not math.isfinite(edge_support):
                    edge_support = max(0.0, min(float(target_copy[source_node]), float(target_copy[target_node])))
                edge_supports[edge_id] = edge_support
                if node_haplotype is not None:
                    source_hap = float(node_haplotype[source_node])
                    target_hap = float(node_haplotype[target_node])
                    if math.isfinite(source_hap) and math.isfinite(target_hap):
                        hap_edges += 1
                        link_value = float(link_support.get((source, target), 0.0))
                        if source_hap != target_hap and edge_support <= 0.0 and link_value <= 0.0:
                            hap_switches += 1.0
                active_edges += 1
            if active_edges:
                edge_excess = 0.0
                for edge_id, count in edge_counts.items():
                    edge_support = float(edge_supports.get(edge_id, 0.0))
                    edge_excess += max(0.0, float(count) - edge_support)
                edge_overuse_frac = edge_excess / float(active_edges)
            if link_support and oriented_path:
                supported_links = 0.0
                unsupported_pairs = 0.0
                checked_pairs = 0
                max_span = int(getattr(instance, "link_window", 8))
                for left_index, left in enumerate(oriented_path):
                    for right in oriented_path[left_index + 1 : left_index + 1 + max_span]:
                        if left == instance.end_index or right == instance.end_index:
                            continue
                        pair_support = float(link_support.get((left, right), 0.0))
                        reverse_support = float(link_support.get((right, left), 0.0))
                        support = max(pair_support, reverse_support)
                        if support > 0.0:
                            supported_links += min(support, 1.0)
                            checked_pairs += 1
                        elif left in link_evidence_nodes and right in link_evidence_nodes:
                            unsupported_pairs += 1.0
                            checked_pairs += 1
                if checked_pairs:
                    link_reward_frac = supported_links / float(checked_pairs)
                    link_penalty_frac = unsupported_pairs / float(checked_pairs)
            if hap_edges:
                haplotype_switch_frac = hap_switches / float(hap_edges)

        edge_weight = float(getattr(instance, "edge_loss_weight", 0.50))
        link_weight = float(getattr(instance, "link_loss_weight", 0.50))
        hap_weight = float(getattr(instance, "haplotype_switch_weight", 0.50))
        reward = (
            0.70 * covered_frac
            + 0.30 * high_conf_covered_frac
            - 0.85 * overcopy_frac
            - 0.30 * undercopy_frac
            - 0.35 * redundant_frac
            - 0.25 * low_conf_used_frac
            - 0.20 * active_step_frac
            - edge_weight * edge_overuse_frac
            + 0.25 * link_weight * link_reward_frac
            - link_weight * link_penalty_frac
            - hap_weight * haplotype_switch_frac
        )
        costs[ant_index] = -float(reward)
    return costs


walk_coverage_proxy_costs = structural_walk_proxy_costs


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
                    node_haplotype=getattr(instance, "node_haplotype", None),
                    edge_support=getattr(instance, "edge_support", None),
                    link_support=getattr(instance, "link_support", None),
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
            costs = structural_walk_proxy_costs(
                np.asarray(batch["choices"], dtype=np.int32),
                instance,
                np.asarray(batch["trace_starts"], dtype=np.int32),
                np.asarray(batch["trace_chosen_edges"], dtype=np.int32),
            )
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
            node_haplotype=getattr(instance, "node_haplotype", None),
            edge_support=getattr(instance, "edge_support", None),
            link_support=getattr(instance, "link_support", None),
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
    payload["training_objective"] = "structural_walk_proxy"
    payload.setdefault("config", {})["source"] = "dynaco_structural_walk_proxy"
    payload["config"]["training_objective"] = "structural_walk_proxy"
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
