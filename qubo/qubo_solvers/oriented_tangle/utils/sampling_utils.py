import numpy as np
import networkx as nx
import re
import subprocess
import time
import heapq
import os
from pathlib import Path
from qubo_solvers.definitions import QuboDescription
from qubo_solvers.logging import get_logger
from math import floor

logger = get_logger(__name__)

rng = np.random.default_rng()


def _states_per_time(qubo_description: QuboDescription) -> int:
    return len(qubo_description.Q) // qubo_description.T


def _one_hot_solution(choices: list[int] | np.ndarray, states_per_time: int) -> np.ndarray:
    solution = np.zeros(len(choices) * states_per_time, dtype=int)
    for t, choice in enumerate(choices):
        solution[t * states_per_time + int(choice)] = 1
    return solution


def _energy(solution: np.ndarray, qubo_description: QuboDescription) -> float:
    Q = np.asarray(qubo_description.Q)
    return float(solution @ Q @ solution + qubo_description.offset)


def _energy_from_choices(
    choices: tuple[int, ...] | list[int] | np.ndarray,
    qubo_description: QuboDescription,
) -> float:
    """QUBO energy for one-hot path choices without materializing x @ Q @ x."""
    states_per_time = _states_per_time(qubo_description)
    Q = np.asarray(qubo_description.Q)
    active = [t * states_per_time + int(choice) for t, choice in enumerate(choices)]
    energy = qubo_description.offset
    for row in active:
        for col in active:
            energy += Q[row, col]
    return float(energy)


def exact_sample_qubo(qubo_description: QuboDescription):
    """Enumerate one-hot path encodings exactly.

    This is only intended for tiny examples. It searches the path-shaped
    subspace used by this formulation: one active state per time step.
    """
    states_per_time = _states_per_time(qubo_description)
    search_size = states_per_time ** qubo_description.T
    max_search_size = 5_000_000
    if search_size > max_search_size:
        raise ValueError(
            f"Exact search would evaluate {search_size} candidates. "
            f"Use the local solver or reduce the graph/T."
        )

    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        for _ in range(qubo_description.jobs):
            best_solution = None
            best_energy = np.inf
            best_path = []
            choices = np.zeros(qubo_description.T, dtype=int)
            while True:
                energy = _energy_from_choices(choices, qubo_description)
                if energy < best_energy:
                    solution = _one_hot_solution(choices, states_per_time)
                    best_solution = solution
                    best_energy = energy
                    best_path = sample_list_to_path(
                        solution,
                        qubo_description.graph,
                        qubo_description.T,
                        qubo_description.V,
                    )

                for idx in range(qubo_description.T - 1, -1, -1):
                    choices[idx] += 1
                    if choices[idx] < states_per_time:
                        break
                    choices[idx] = 0
                else:
                    break

            paths[time_limit].append((best_solution, best_energy, best_path))
    return paths


def local_sample_qubo(qubo_description: QuboDescription):
    """Random-restart one-hot local search over the QUBO energy.

    This is the intended scaffold for custom/data-driven solvers: replace the
    proposal policy that selects `t` and `new_choice`, while preserving the
    returned `(solution, energy, path)` contract.
    """
    states_per_time = _states_per_time(qubo_description)
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)
        for _ in range(qubo_description.jobs):
            started_at = time.monotonic()
            best_solution = None
            best_energy = np.inf

            while time.monotonic() - started_at < deadline:
                choices = rng.integers(0, states_per_time, size=qubo_description.T)
                solution = _one_hot_solution(choices, states_per_time)
                current_energy = _energy_from_choices(choices, qubo_description)

                improved = True
                while improved and time.monotonic() - started_at < deadline:
                    improved = False
                    for t in rng.permutation(qubo_description.T):
                        old_choice = choices[t]
                        candidate_choices = rng.permutation(states_per_time)
                        for new_choice in candidate_choices:
                            if new_choice == old_choice:
                                continue
                            choices[t] = new_choice
                            candidate_energy = _energy_from_choices(choices, qubo_description)
                            if candidate_energy < current_energy:
                                solution = _one_hot_solution(choices, states_per_time)
                                current_energy = candidate_energy
                                improved = True
                                break
                            choices[t] = old_choice

                if current_energy < best_energy:
                    best_solution = solution
                    best_energy = current_energy

            best_path = sample_list_to_path(
                best_solution,
                qubo_description.graph,
                qubo_description.T,
                qubo_description.V,
            )
            paths[time_limit].append((best_solution, best_energy, best_path))
    return paths


def _successor_indices(qubo_description: QuboDescription) -> dict[int, list[int]]:
    """Legal path successors in the same state order used by the QUBO vector."""
    nodes = list(qubo_description.graph.nodes)
    end_index = qubo_description.V * 2
    node_to_index = {node: index for index, node in enumerate(nodes)}
    successors = {end_index: [end_index]}

    for index, node in enumerate(nodes):
        legal = [
            node_to_index[successor]
            for successor in qubo_description.graph.successors(node)
            if successor in node_to_index
        ]
        legal.append(end_index)
        successors[index] = sorted(set(legal))
    return successors


def _prefix_counts(prefix: tuple[int, ...], V: int) -> tuple[int, ...]:
    counts = [0] * V
    for state_index in prefix:
        if state_index < 2 * V:
            counts[state_index // 2] += 1
    return tuple(counts)


def _prefix_overcopy_cost(prefix: tuple[int, ...] | list[int], qubo_description: QuboDescription) -> float:
    """Irreversible prefix cost g(n): only over-copy residual can no longer be fixed."""
    weights, _ = _node_weights_and_lengths(qubo_description)
    counts = _prefix_counts(tuple(prefix), qubo_description.V)
    return float(sum(max(0.0, counts[i] - float(weights[i])) ** 2 for i in range(qubo_description.V)))


def _terminal_node_objective(choices: tuple[int, ...] | list[int], qubo_description: QuboDescription) -> float:
    """Node-count terminal objective used for neural cost-to-go targets."""
    weights, _ = _node_weights_and_lengths(qubo_description)
    counts = _prefix_counts(tuple(choices), qubo_description.V)
    return float(sum((counts[i] - float(weights[i])) ** 2 for i in range(qubo_description.V)))


def _initial_indices(qubo_description: QuboDescription) -> list[int]:
    """The current QUBO has no fixed start node, so any oriented node may start."""
    end_index = qubo_description.V * 2
    return list(range(end_index)) + [end_index]


def _node_weights_and_lengths(qubo_description: QuboDescription) -> tuple[list[float], list[float]]:
    nodes = list(qubo_description.graph.nodes)
    weights = [qubo_description.graph.nodes[nodes[2 * i]]["weight"] for i in range(qubo_description.V)]
    lengths = [qubo_description.graph.nodes[nodes[2 * i]]["length"] for i in range(qubo_description.V)]
    return weights, lengths


def _residual_successor_score(
    successor: int,
    counts: list[int],
    weights: list[float],
    lengths: list[float],
    end_index: int,
) -> float:
    if successor == end_index:
        remaining = sum(max(0.0, weights[i] - counts[i]) for i in range(len(weights)))
        return 1.0 if remaining <= 0 else 0.01
    node_index = successor // 2
    residual = max(0.0, weights[node_index] - counts[node_index])
    return 0.01 + residual * (1.0 + np.sqrt(1.0 + lengths[node_index]) / 10.0)


def _path_result_from_choices(
    choices: tuple[int, ...] | list[int] | np.ndarray,
    qubo_description: QuboDescription,
) -> tuple[np.ndarray, float, list]:
    states_per_time = _states_per_time(qubo_description)
    solution = _one_hot_solution(choices, states_per_time)
    energy = _energy_from_choices(choices, qubo_description)
    path = sample_list_to_path(
        solution,
        qubo_description.graph,
        qubo_description.T,
        qubo_description.V,
    )
    return solution, energy, path


def _path_result_from_choices_with_cost(
    choices: tuple[int, ...] | list[int] | np.ndarray,
    cost: float,
    qubo_description: QuboDescription,
) -> tuple[np.ndarray, float, list]:
    states_per_time = _states_per_time(qubo_description)
    solution = _one_hot_solution(choices, states_per_time)
    path = sample_list_to_path(
        solution,
        qubo_description.graph,
        qubo_description.T,
        qubo_description.V,
    )
    return solution, float(cost), path


def _walk_coverage_proxy_costs(
    choices_batch: np.ndarray,
    weights: list[float] | np.ndarray,
    lengths: list[float] | np.ndarray,
    end_index: int,
    *,
    trace_starts: np.ndarray | None = None,
    trace_edges: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> np.ndarray:
    """Parsimonious structural feasible-walk cost used by ACO/prior training.

    The available GFA annotations provide node-copy targets but not read-pair
    or haplotype labels. The edge term therefore acts as a conservative
    structural surrogate by discouraging repeated traversal of an oriented edge
    beyond the copy support implied by its incident nodes.
    """
    weights = np.asarray(weights, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
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
            if target == end_index:
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
        if trace_starts is not None and trace_edges is not None and offsets is not None and targets is not None:
            offsets_np = np.asarray(offsets, dtype=np.int32)
            targets_np = np.asarray(targets, dtype=np.int32)
            begin = int(trace_starts[ant_index])
            end = int(trace_starts[ant_index + 1])
            edge_counts: dict[int, int] = {}
            edge_supports: dict[int, float] = {}
            active_edges = 0
            for edge_id in np.asarray(trace_edges[begin:end], dtype=np.int32):
                edge_id = int(edge_id)
                target = int(targets_np[edge_id])
                if target == end_index:
                    continue
                source = int(np.searchsorted(offsets_np, edge_id, side="right") - 1)
                if source == end_index:
                    continue
                source_node = source // 2
                target_node = target // 2
                if not (0 <= source_node < len(target_copy) and 0 <= target_node < len(target_copy)):
                    continue
                edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
                edge_supports[edge_id] = max(0.0, min(float(target_copy[source_node]), float(target_copy[target_node])))
                active_edges += 1
            if active_edges:
                edge_excess = 0.0
                for edge_id, count in edge_counts.items():
                    edge_support = float(edge_supports.get(edge_id, 0.0))
                    edge_excess += max(0.0, float(count) - edge_support)
                edge_overuse_frac = edge_excess / float(active_edges)

        reward = (
            0.70 * covered_frac
            + 0.30 * high_conf_covered_frac
            - 0.85 * overcopy_frac
            - 0.30 * undercopy_frac
            - 0.35 * redundant_frac
            - 0.25 * low_conf_used_frac
            - 0.20 * active_step_frac
            - 0.50 * edge_overuse_frac
        )
        costs[ant_index] = -float(reward)
    return costs


def _greedy_complete_prefix(
    prefix: tuple[int, ...],
    qubo_description: QuboDescription,
    successors: dict[int, list[int]],
) -> tuple[int, ...]:
    """Fast QUBO-compatible completion used only as a search ranking heuristic."""
    if len(prefix) >= qubo_description.T:
        return prefix[:qubo_description.T]

    end_index = qubo_description.V * 2
    weights, lengths = _node_weights_and_lengths(qubo_description)
    counts = list(_prefix_counts(prefix, qubo_description.V))
    completed = list(prefix)

    while len(completed) < qubo_description.T:
        last = completed[-1]
        if last == end_index:
            completed.append(end_index)
            continue

        best_successor = end_index
        best_score = 0.0
        for successor in successors[last]:
            if successor == end_index:
                continue
            score = _residual_successor_score(successor, counts, weights, lengths, end_index)
            if score > best_score:
                best_successor = successor
                best_score = score

        completed.append(best_successor)
        if best_successor != end_index:
            counts[best_successor // 2] += 1

    return tuple(completed)


def _path_search_priority(
    prefix: tuple[int, ...],
    qubo_description: QuboDescription,
    successors: dict[int, list[int]],
) -> float:
    completed = _greedy_complete_prefix(prefix, qubo_description, successors)
    return _energy_from_choices(completed, qubo_description)


def _load_neural_seea_model():
    model_path = os.environ.get("QPG_SEEA_MODEL")
    if not model_path:
        raise ValueError(
            "SeeA* is configured as the neural solver and requires QPG_SEEA_MODEL. "
            "Train a checkpoint with examples/train_qpg_reinforce.py --out <path>, "
            "or use solver=astar for the non-neural baseline."
        )

    from qubo_solvers.oriented_tangle.neural_gnn import QPGSeeAGNN, require_torch, torch

    require_torch()
    device = torch.device(os.environ.get("QPG_SEEA_DEVICE", "cpu"))
    try:
        checkpoint = torch.load(Path(model_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(model_path), map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    units = int(config.get("units", os.environ.get("QPG_SEEA_UNITS", 64)))
    depth = int(config.get("depth", os.environ.get("QPG_SEEA_DEPTH", 4)))
    model = QPGSeeAGNN(units=units, depth=depth).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, torch, device, model_path


def _load_neural_aco_model():
    model_path = os.environ.get("QPG_ACO_MODEL") or os.environ.get("QPG_SEEA_MODEL")
    if not model_path:
        raise ValueError(
            "neural_aco requires QPG_ACO_MODEL or --neural-model. "
            "Train a checkpoint with examples/train_qpg_neural_aco.py --out <path>, "
            "or use solver=aco for the non-neural baseline."
        )

    from qubo_solvers.oriented_tangle.neural_gnn import QPGSeeAGNN, require_torch, torch

    require_torch()
    device = torch.device(os.environ.get("QPG_ACO_DEVICE", os.environ.get("QPG_SEEA_DEVICE", "cpu")))
    try:
        checkpoint = torch.load(Path(model_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(model_path), map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    units = int(config.get("units", os.environ.get("QPG_ACO_UNITS", 64)))
    depth = int(config.get("depth", os.environ.get("QPG_ACO_DEPTH", 4)))
    model = QPGSeeAGNN(units=units, depth=depth).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, torch, device, model_path, checkpoint if isinstance(checkpoint, dict) else {}


def _neural_seea_priority(
    prefix: tuple[int, ...],
    qubo_description: QuboDescription,
    neural_context,
) -> float:
    """Neural SeeA* rank: f_theta(n)=g(n)+beta*h_theta(n)."""
    model, torch, device, _path, _checkpoint = neural_context
    from qubo_solvers.oriented_tangle.neural_gnn import build_qpg_graph_tensor

    beta = float(os.environ.get("QPG_SEEA_BETA", "1.0"))
    current_index = int(prefix[-1]) if prefix else qubo_description.V * 2 + 1
    counts = _prefix_counts(prefix, qubo_description.V)
    with torch.no_grad():
        tensor = build_qpg_graph_tensor(
            qubo_description.graph,
            counts=counts,
            current_index=current_index,
            depth=len(prefix),
            horizon=qubo_description.T,
            device=device,
        )
        h_value = float(model(tensor)["value"].detach().cpu())
    return _prefix_overcopy_cost(prefix, qubo_description) + beta * h_value


def _neural_child_priorities(
    prefix: tuple[int, ...],
    qubo_description: QuboDescription,
    successors: dict[int, list[int]],
    neural_context,
) -> list[tuple[int, float]]:
    """Score all legal children with one GNN call for the current prefix."""
    model, torch, device, _path, _checkpoint = neural_context
    from qubo_solvers.oriented_tangle.neural_gnn import build_qpg_graph_tensor

    beta = float(os.environ.get("QPG_SEEA_BETA", "1.0"))
    policy_weight = float(os.environ.get("QPG_SEEA_POLICY_WEIGHT", "1.0"))
    start_index = qubo_description.V * 2 + 1
    current_index = int(prefix[-1]) if prefix else start_index
    counts = _prefix_counts(prefix, qubo_description.V)
    legal_successors = _initial_indices(qubo_description) if not prefix else successors[prefix[-1]]
    with torch.no_grad():
        tensor = build_qpg_graph_tensor(
            qubo_description.graph,
            counts=counts,
            current_index=current_index,
            depth=len(prefix),
            horizon=qubo_description.T,
            device=device,
        )
        output = model(tensor)
        source_index = tensor.start_index if not prefix else current_index
        edge_ids = [
            idx
            for idx, (source, target) in enumerate(tensor.edge_pairs)
            if source == source_index and target in legal_successors
        ]
        if edge_ids:
            edge_id_tensor = torch.tensor(edge_ids, dtype=torch.long, device=device)
            logits = output["edge_logits"].index_select(0, edge_id_tensor)
            log_probs = torch.nn.functional.log_softmax(logits, dim=0)
            edge_to_log_prob = {
                tensor.edge_pairs[edge_ids[i]][1]: float(log_probs[i].detach().cpu())
                for i in range(len(edge_ids))
            }
        else:
            edge_to_log_prob = {}
        value = float(output["value"].detach().cpu())

    child_scores = []
    for successor in legal_successors:
        child = prefix + (successor,)
        policy_bonus = edge_to_log_prob.get(successor, 0.0)
        priority = (
            _prefix_overcopy_cost(child, qubo_description)
            + beta * value
            - policy_weight * policy_bonus
        )
        child_scores.append((successor, float(priority)))
    return child_scores


def greedy_residual_sample_qubo(qubo_description: QuboDescription):
    """Try every start, then greedily follow unmet copy-number residual."""
    successors = _successor_indices(qubo_description)
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        for _ in range(qubo_description.jobs):
            best = None
            for start in _initial_indices(qubo_description):
                choices = _greedy_complete_prefix((start,), qubo_description, successors)
                result = _path_result_from_choices(choices, qubo_description)
                if best is None or result[1] < best[1]:
                    best = result
            paths[time_limit].append(best)
    return paths


def _random_residual_choices(
    qubo_description: QuboDescription,
    successors: dict[int, list[int]],
    pheromone: dict[tuple[int, int], float] | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> tuple[int, ...]:
    end_index = qubo_description.V * 2
    weights, lengths = _node_weights_and_lengths(qubo_description)
    counts = [0] * qubo_description.V

    start_options = _initial_indices(qubo_description)
    start_scores = np.array([
        _residual_successor_score(option, counts, weights, lengths, end_index)
        for option in start_options
    ], dtype=float)
    start_probs = start_scores / start_scores.sum()
    current = int(rng.choice(start_options, p=start_probs))
    choices = [current]
    if current != end_index:
        counts[current // 2] += 1

    while len(choices) < qubo_description.T:
        options = successors[current]
        scores = []
        for option in options:
            heuristic = _residual_successor_score(option, counts, weights, lengths, end_index)
            trail = 1.0 if pheromone is None else pheromone.get((current, option), 1.0)
            scores.append((trail ** alpha) * (heuristic ** beta))
        scores_array = np.array(scores, dtype=float)
        if not np.isfinite(scores_array).all() or scores_array.sum() <= 0:
            probs = np.full(len(options), 1.0 / len(options))
        else:
            probs = scores_array / scores_array.sum()
        current = int(rng.choice(options, p=probs))
        choices.append(current)
        if current != end_index:
            counts[current // 2] += 1
    return tuple(choices)


def _neural_aco_choices(
    qubo_description: QuboDescription,
    successors: dict[int, list[int]],
    pheromone: dict[tuple[int, int], float],
    neural_context,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> tuple[int, ...]:
    """DyNACO-style ant construction with pheromone, heuristic, and neural prior."""
    model, torch, device, _path, _checkpoint = neural_context
    from qubo_solvers.oriented_tangle.neural_gnn import build_qpg_graph_tensor

    end_index = qubo_description.V * 2
    weights, lengths = _node_weights_and_lengths(qubo_description)
    counts = [0] * qubo_description.V
    choices: list[int] = []
    current = None

    while len(choices) < qubo_description.T:
        legal = _initial_indices(qubo_description) if current is None else successors[current]
        current_index = qubo_description.V * 2 + 1 if current is None else current
        with torch.no_grad():
            tensor = build_qpg_graph_tensor(
                qubo_description.graph,
                counts=counts,
                current_index=current_index,
                depth=len(choices),
                horizon=qubo_description.T,
                device=device,
            )
            output = model(tensor)
            source_index = tensor.start_index if current is None else current
            edge_ids = [
                idx
                for idx, (source, target) in enumerate(tensor.edge_pairs)
                if source == source_index and target in legal
            ]
            neural_logits = {}
            if edge_ids:
                logits = output["edge_logits"].index_select(
                    0,
                    torch.tensor(edge_ids, dtype=torch.long, device=device),
                )
                for local_idx, edge_id in enumerate(edge_ids):
                    neural_logits[tensor.edge_pairs[edge_id][1]] = float(logits[local_idx].detach().cpu())

        scores = []
        source_key = end_index if current is None else current
        for option in legal:
            heuristic = _residual_successor_score(option, counts, weights, lengths, end_index)
            trail = pheromone.get((source_key, option), 1.0)
            log_score = (
                alpha * np.log(max(trail, 1e-12))
                + beta * np.log(max(heuristic, 1e-12))
                + gamma * neural_logits.get(option, 0.0)
            )
            scores.append(log_score)
        score_array = np.array(scores, dtype=float)
        score_array -= score_array.max()
        probs = np.exp(score_array)
        probs = probs / probs.sum()
        current = int(rng.choice(legal, p=probs))
        choices.append(current)
        if current != end_index:
            counts[current // 2] += 1
    return tuple(choices)


def _aco_static_edge_arrays(qubo_description: QuboDescription, prior_by_edge: dict[tuple[int, int], float] | None = None):
    end_index = qubo_description.V * 2
    start_index = end_index + 1
    successors = _successor_indices(qubo_description)
    weights, lengths = _node_weights_and_lengths(qubo_description)
    counts = [0] * qubo_description.V
    offsets = [0]
    targets: list[int] = []
    heuristic: list[float] = []
    prior: list[float] = []
    sources = list(range(start_index + 1))
    for source in sources:
        if source == start_index:
            legal = _initial_indices(qubo_description)
        elif source == end_index:
            legal = [end_index]
        else:
            legal = successors[source]
        for target in legal:
            targets.append(target)
            heuristic.append(_residual_successor_score(target, counts, weights, lengths, end_index))
            prior.append(0.0 if prior_by_edge is None else prior_by_edge.get((source, target), 0.0))
        offsets.append(len(targets))
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        np.asarray(heuristic, dtype=np.float32),
        np.asarray(prior, dtype=np.float32),
        start_index,
    )


def _neural_static_prior_by_edge(qubo_description: QuboDescription, neural_context) -> dict[tuple[int, int], float]:
    model, torch, device, _path, _checkpoint = neural_context
    from qubo_solvers.oriented_tangle.neural_gnn import build_qpg_graph_tensor

    counts = [0] * qubo_description.V
    start_index = qubo_description.V * 2 + 1
    with torch.no_grad():
        tensor = build_qpg_graph_tensor(
            qubo_description.graph,
            counts=counts,
            current_index=start_index,
            depth=0,
            horizon=qubo_description.T,
            device=device,
        )
        logits = model(tensor)["edge_logits"].detach().cpu().numpy()
    return {
        pair: float(logits[idx])
        for idx, pair in enumerate(tensor.edge_pairs)
    }


def random_residual_sample_qubo(qubo_description: QuboDescription):
    """Random legal walks biased toward unmet copy-number residual."""
    successors = _successor_indices(qubo_description)
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)
        for _ in range(qubo_description.jobs):
            started_at = time.monotonic()
            best = None
            trials = 0
            min_trials = int(os.environ.get("QPG_RANDOM_RESIDUAL_MIN_TRIALS", "64"))
            while trials < min_trials or time.monotonic() - started_at < deadline:
                choices = _random_residual_choices(qubo_description, successors)
                result = _path_result_from_choices(choices, qubo_description)
                if best is None or result[1] < best[1]:
                    best = result
                trials += 1
                if time.monotonic() - started_at >= deadline and trials >= min_trials:
                    break
            logger.info(f'random_residual_walk evaluated {trials} walks, best energy {best[1]}')
            paths[time_limit].append(best)
    return paths


def beam_search_sample_qubo(qubo_description: QuboDescription):
    """Beam search over legal path prefixes using greedy-completion QUBO score."""
    successors = _successor_indices(qubo_description)
    beam_width = int(os.environ.get("QPG_BEAM_WIDTH", "100"))
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)
        for _ in range(qubo_description.jobs):
            started_at = time.monotonic()
            beam = []
            for start in _initial_indices(qubo_description):
                prefix = (start,)
                beam.append((_path_search_priority(prefix, qubo_description, successors), prefix))

            for _depth in range(1, qubo_description.T):
                if time.monotonic() - started_at >= deadline:
                    break
                candidates = {}
                for _, prefix in beam:
                    if time.monotonic() - started_at >= deadline:
                        break
                    for successor in successors[prefix[-1]]:
                        child = prefix + (successor,)
                        key = (child[-1], _prefix_counts(child, qubo_description.V))
                        priority = _path_search_priority(child, qubo_description, successors)
                        if key not in candidates or priority < candidates[key][0]:
                            candidates[key] = (priority, child)
                if not candidates:
                    break
                beam = heapq.nsmallest(beam_width, candidates.values(), key=lambda item: item[0])

            best_prefix = min(beam, key=lambda item: item[0])[1]
            best_choices = _greedy_complete_prefix(best_prefix, qubo_description, successors)
            result = _path_result_from_choices(best_choices, qubo_description)
            logger.info(f'beam_search width {beam_width}, best energy {result[1]}')
            paths[time_limit].append(result)
    return paths


def aco_sample_qubo(qubo_description: QuboDescription):
    """Ant colony construction over legal graph walks."""
    from qubo_solvers.oriented_tangle import qpg_aco_cpp

    ant_count = int(os.environ.get("QPG_ACO_ANTS", "64"))
    evaporation = float(os.environ.get("QPG_ACO_EVAPORATION", "0.2"))
    alpha = float(os.environ.get("QPG_ACO_ALPHA", "1.0"))
    beta = float(os.environ.get("QPG_ACO_BETA", "2.0"))
    min_iterations = int(os.environ.get("QPG_ACO_MIN_ITERATIONS", "1"))
    parallel_traced = os.environ.get("QPG_ACO_PARALLEL_TRACED", "1").lower() not in {"0", "false", "no"}
    threads = os.environ.get("QPG_ACO_THREADS")
    if threads:
        qpg_aco_cpp.set_num_threads(int(threads))
    offsets, targets, heuristic, prior, start_source = _aco_static_edge_arrays(qubo_description)
    weights, lengths = _node_weights_and_lengths(qubo_description)
    weights_array = np.asarray(weights, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)
    initial_pheromone = np.ones_like(heuristic, dtype=np.float32)
    states_per_time = _states_per_time(qubo_description)
    end_index = qubo_description.V * 2
    seed = int(os.environ.get("QPG_ACO_SEED", "1"))

    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)
        for job_index in range(qubo_description.jobs):
            pheromone = initial_pheromone.copy()
            started_at = time.monotonic()
            best = None
            iterations = 0
            while iterations < min_iterations or time.monotonic() - started_at < deadline:
                batch = qpg_aco_cpp.sample_batch_traces(
                    offsets,
                    targets,
                    pheromone,
                    heuristic,
                    prior,
                    weights_array,
                    lengths_array,
                    qubo_description.T,
                    ant_count,
                    start_source,
                    end_index,
                    alpha,
                    beta,
                    0.0,
                    seed + job_index * 1000003 + iterations,
                    parallel_traced,
                )
                choices_batch = np.asarray(batch["choices"], dtype=np.int32)
                trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
                trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
                costs = _walk_coverage_proxy_costs(
                    choices_batch,
                    weights,
                    lengths,
                    end_index,
                    trace_starts=trace_starts,
                    trace_edges=trace_edges,
                    offsets=offsets,
                    targets=targets,
                )
                best_ant = int(np.argmin(costs))
                if best is None or float(costs[best_ant]) < best[1]:
                    choices = tuple(int(x) for x in choices_batch[best_ant])
                    best = _path_result_from_choices_with_cost(
                        choices,
                        float(costs[best_ant]),
                        qubo_description,
                    )

                pheromone *= (1.0 - evaporation)
                np.maximum(pheromone, 1e-6, out=pheromone)
                worst_cost = float(costs.max())
                elite_indices = np.argsort(costs)[: max(1, ant_count // 4)]
                for ant_index in elite_indices:
                    deposit = (worst_cost - float(costs[ant_index]) + 1.0) / (abs(worst_cost) + 1.0)
                    begin = int(trace_starts[ant_index])
                    end = int(trace_starts[ant_index + 1])
                    for edge_id in trace_edges[begin:end]:
                        pheromone[int(edge_id)] += deposit

                iterations += 1
                if time.monotonic() - started_at >= deadline and iterations >= min_iterations:
                    break

            logger.info(
                f'aco job {job_index}, iterations {iterations}, ants {ant_count}, best energy {best[1]}'
            )
            paths[time_limit].append(best)
    return paths


def neural_aco_sample_qubo(qubo_description: QuboDescription):
    """ACO with a DyNACO-style neural edge prior."""
    from qubo_solvers.oriented_tangle import qpg_aco_cpp

    neural_context = _load_neural_aco_model()
    logger.info(
        f'Loaded neural ACO model from {neural_context[3]} '
        f'with gamma={os.environ.get("QPG_ACO_GAMMA", "1.0")}'
    )
    ant_count = int(os.environ.get("QPG_ACO_ANTS", "64"))
    evaporation = float(os.environ.get("QPG_ACO_EVAPORATION", "0.2"))
    alpha = float(os.environ.get("QPG_ACO_ALPHA", "1.0"))
    beta = float(os.environ.get("QPG_ACO_BETA", "2.0"))
    gamma = float(os.environ.get("QPG_ACO_GAMMA", "1.0"))
    min_iterations = int(os.environ.get("QPG_ACO_MIN_ITERATIONS", "1"))
    parallel_traced = os.environ.get("QPG_ACO_PARALLEL_TRACED", "1").lower() not in {"0", "false", "no"}
    threads = os.environ.get("QPG_ACO_THREADS")
    if threads:
        qpg_aco_cpp.set_num_threads(int(threads))
    prior_by_edge = _neural_static_prior_by_edge(qubo_description, neural_context)
    offsets, targets, heuristic, prior, start_source = _aco_static_edge_arrays(
        qubo_description,
        prior_by_edge=prior_by_edge,
    )
    prior_mode = os.environ.get("QPG_ACO_PRIOR_MODE", "learned").strip().lower()
    seed = int(os.environ.get("QPG_ACO_SEED", "1"))
    if prior_mode in {"zero", "none"}:
        prior = np.zeros_like(prior, dtype=np.float32)
    elif prior_mode in {"shuffle", "shuffled"}:
        prior = np.random.default_rng(seed).permutation(prior).astype(np.float32, copy=False)
    elif prior_mode == "random":
        scale = float(np.std(prior)) if prior.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        prior = np.random.default_rng(seed).normal(0.0, scale, size=prior.shape).astype(np.float32)
    elif prior_mode != "learned":
        raise ValueError(
            "QPG_ACO_PRIOR_MODE must be one of learned, zero, shuffle, or random; "
            f"got {prior_mode!r}"
        )
    weights, lengths = _node_weights_and_lengths(qubo_description)
    weights_array = np.asarray(weights, dtype=np.float32)
    lengths_array = np.asarray(lengths, dtype=np.float32)
    end_index = qubo_description.V * 2
    checkpoint = neural_context[4]
    checkpoint_pheromone = checkpoint.get("pheromone") if isinstance(checkpoint, dict) else None
    if checkpoint_pheromone is not None and np.asarray(checkpoint_pheromone).shape == heuristic.shape:
        initial_pheromone = np.asarray(checkpoint_pheromone, dtype=np.float32).copy()
        logger.info("Loaded neural ACO pheromone state from checkpoint")
    else:
        initial_pheromone = np.ones_like(heuristic, dtype=np.float32)
    states_per_time = _states_per_time(qubo_description)
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)
        for job_index in range(qubo_description.jobs):
            pheromone = initial_pheromone.copy()
            started_at = time.monotonic()
            best = None
            iterations = 0
            while iterations < min_iterations or time.monotonic() - started_at < deadline:
                batch = qpg_aco_cpp.sample_batch_traces(
                    offsets,
                    targets,
                    pheromone,
                    heuristic,
                    prior,
                    weights_array,
                    lengths_array,
                    qubo_description.T,
                    ant_count,
                    start_source,
                    end_index,
                    alpha,
                    beta,
                    gamma,
                    seed + job_index * 1000003 + iterations,
                    parallel_traced,
                )
                choices_batch = np.asarray(batch["choices"], dtype=np.int32)
                trace_starts = np.asarray(batch["trace_starts"], dtype=np.int32)
                trace_edges = np.asarray(batch["trace_chosen_edges"], dtype=np.int32)
                costs = _walk_coverage_proxy_costs(
                    choices_batch,
                    weights,
                    lengths,
                    end_index,
                    trace_starts=trace_starts,
                    trace_edges=trace_edges,
                    offsets=offsets,
                    targets=targets,
                )
                best_ant = int(np.argmin(costs))
                if best is None or float(costs[best_ant]) < best[1]:
                    choices = tuple(int(x) for x in choices_batch[best_ant])
                    best = _path_result_from_choices_with_cost(
                        choices,
                        float(costs[best_ant]),
                        qubo_description,
                    )

                pheromone *= (1.0 - evaporation)
                np.maximum(pheromone, 1e-6, out=pheromone)
                worst_cost = float(costs.max())
                elite_indices = np.argsort(costs)[: max(1, ant_count // 4)]
                for ant_index in elite_indices:
                    deposit = (worst_cost - float(costs[ant_index]) + 1.0) / (abs(worst_cost) + 1.0)
                    begin = int(trace_starts[ant_index])
                    end = int(trace_starts[ant_index + 1])
                    for edge_id in trace_edges[begin:end]:
                        pheromone[int(edge_id)] += deposit

                iterations += 1
                if time.monotonic() - started_at >= deadline and iterations >= min_iterations:
                    break

            logger.info(
                f'neural_aco job {job_index}, iterations {iterations}, ants {ant_count}, best energy {best[1]}'
            )
            paths[time_limit].append(best)
    return paths


def _astar_like_sample_qubo(qubo_description: QuboDescription, use_seea: bool):
    """Best-first legal-prefix search.

    This searches only graph-valid paths, then scores every completed candidate
    with the exact QUBO energy. Plain A* expands the globally best ranked OPEN
    node; SeeA* samples a candidate subset of OPEN and expands the best node in
    that subset.
    """
    states_per_time = _states_per_time(qubo_description)
    successors = _successor_indices(qubo_description)
    max_expansions = int(os.environ.get("QPG_ASTAR_MAX_EXPANSIONS", "100000"))
    seea_k = int(os.environ.get("QPG_SEEA_K", "50"))
    solver_name = "seea" if use_seea else "astar"
    neural_context = _load_neural_seea_model() if use_seea else None
    if neural_context is not None:
        logger.info(
            f'Loaded neural SeeA* model from {neural_context[3]} '
            f'with beta={os.environ.get("QPG_SEEA_BETA", "1.0")}'
        )
    paths = {}

    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        deadline = max(time_limit, 1)

        for _ in range(qubo_description.jobs):
            started_at = time.monotonic()
            best_solution = None
            best_energy = np.inf
            best_path = []
            expansions = 0
            counter = 0
            seen: dict[tuple[int, int, tuple[int, ...]], float] = {}

            if use_seea:
                open_nodes: list[tuple[float, int, tuple[int, ...]]] = []
            else:
                open_heap: list[tuple[float, int, tuple[int, ...]]] = []

            if use_seea:
                initial_items = _neural_child_priorities((), qubo_description, successors, neural_context)
                for state_index, priority in initial_items:
                    prefix = (state_index,)
                    item = (priority, counter, prefix)
                    counter += 1
                    key = (len(prefix), state_index, _prefix_counts(prefix, qubo_description.V))
                    seen[key] = priority
                    open_nodes.append(item)
            else:
                for state_index in _initial_indices(qubo_description):
                    prefix = (state_index,)
                    priority = _path_search_priority(prefix, qubo_description, successors)
                    item = (priority, counter, prefix)
                    counter += 1
                    key = (len(prefix), state_index, _prefix_counts(prefix, qubo_description.V))
                    seen[key] = priority
                    heapq.heappush(open_heap, item)

            while expansions < max_expansions and time.monotonic() - started_at < deadline:
                if use_seea:
                    if not open_nodes:
                        break
                    if len(open_nodes) <= seea_k:
                        candidate_indices = np.arange(len(open_nodes))
                    else:
                        candidate_indices = rng.choice(len(open_nodes), size=seea_k, replace=False)
                    selected_index = min(candidate_indices, key=lambda idx: open_nodes[int(idx)][0])
                    priority, _, prefix = open_nodes.pop(int(selected_index))
                else:
                    if not open_heap:
                        break
                    priority, _, prefix = heapq.heappop(open_heap)

                expansions += 1
                completed = _greedy_complete_prefix(prefix, qubo_description, successors)
                completed_energy = _energy_from_choices(completed, qubo_description)
                if completed_energy < best_energy:
                    best_solution = _one_hot_solution(completed, states_per_time)
                    best_energy = completed_energy
                    best_path = sample_list_to_path(
                        best_solution,
                        qubo_description.graph,
                        qubo_description.T,
                        qubo_description.V,
                    )

                if len(prefix) == qubo_description.T:
                    solution = _one_hot_solution(prefix, states_per_time)
                    energy = _energy_from_choices(prefix, qubo_description)
                    if energy < best_energy:
                        best_solution = solution
                        best_energy = energy
                        best_path = sample_list_to_path(
                            solution,
                            qubo_description.graph,
                            qubo_description.T,
                            qubo_description.V,
                        )
                    continue

                if use_seea:
                    child_priorities = _neural_child_priorities(
                        prefix,
                        qubo_description,
                        successors,
                        neural_context,
                    )
                else:
                    child_priorities = [
                        (
                            successor,
                            _path_search_priority(prefix + (successor,), qubo_description, successors),
                        )
                        for successor in successors[prefix[-1]]
                    ]

                for successor, child_priority in child_priorities:
                    child = prefix + (successor,)
                    if not use_seea:
                        child_priority = _path_search_priority(child, qubo_description, successors)
                    key = (len(child), successor, _prefix_counts(child, qubo_description.V))
                    if key in seen and seen[key] <= child_priority:
                        continue
                    seen[key] = child_priority
                    item = (child_priority, counter, child)
                    counter += 1
                    if use_seea:
                        open_nodes.append(item)
                    else:
                        heapq.heappush(open_heap, item)

            if best_solution is None:
                logger.info(f'{solver_name} did not reach a terminal path; forcing greedy completion')
                if use_seea and open_nodes:
                    _, _, prefix = min(open_nodes, key=lambda item: item[0])
                elif (not use_seea) and open_heap:
                    _, _, prefix = min(open_heap, key=lambda item: item[0])
                else:
                    prefix = (qubo_description.V * 2,)
                completed = _greedy_complete_prefix(prefix, qubo_description, successors)
                best_solution = _one_hot_solution(completed, states_per_time)
                best_energy = _energy_from_choices(completed, qubo_description)
                best_path = sample_list_to_path(
                    best_solution,
                    qubo_description.graph,
                    qubo_description.T,
                    qubo_description.V,
                )

            logger.info(
                f'{solver_name} expanded {expansions} nodes, best energy {best_energy}'
            )
            paths[time_limit].append((best_solution, best_energy, best_path))

    return paths


def astar_sample_qubo(qubo_description: QuboDescription):
    return _astar_like_sample_qubo(qubo_description, use_seea=False)


def seea_sample_qubo(qubo_description: QuboDescription):
    return _astar_like_sample_qubo(qubo_description, use_seea=True)

def mqlib_sample_qubo(qubo_description: QuboDescription):
    input_filepath = f"{qubo_description.data_dir}/mqlib_input_{qubo_description.filename}.txt"

    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        
        for _ in range(qubo_description.jobs):
            if qubo_description.T < 5:
                logger.info(f'Small problem, T = {qubo_description.T}. Setting time limit to <=5')
                actual_time_limit = min(time_limit, 5)
            elif qubo_description.T < 10:
                logger.info(f'Small problem, T = {qubo_description.T}. Setting time limit to <=10')
                actual_time_limit = min(time_limit, 10)
            else:
                actual_time_limit = time_limit
            # Run the MQLib solver and capture output
            process = subprocess.run(["MQLib", "-fQ", input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(actual_time_limit), "-s", str(rng.integers(0, 65535)), "-ps"], capture_output=True)

            out = process.stdout.decode("utf-8")

            try:
                # First line of output includes run data. 3rd line contains the solution.
                out_data = [x for x in out.split('\n') if len(x) > 0]
                solution = out_data[2].split()
                solution = [int(x) for x in solution]
                logger.info(out_data[0].split(','))
                logger.info(out_data[0].split(',')[-1])
                solution_energy = float(out_data[0].split(',')[3])
            except (ValueError, IndexError):
                logger.error('Could not parse mqlib data')
                logger.error(out)
                paths[time_limit].append(([], np.inf, []))
                continue
            energy = qubo_description.offset - solution_energy
            path = sample_list_to_path(solution, qubo_description.graph, qubo_description.T, qubo_description.V)
            paths[time_limit].append((solution, energy, path))
            
    return paths


def dwave_sample_qubo(qubo_description: QuboDescription) -> dict[int, tuple]:
    """Perform a batch of annealing on a given Binary Quadratic Model.

    Args:
        qubo_description (QuboDescription): a description of the problem
        
    Returns:
        (dict): Returns the best sample, energy and path for each job run.
    """
    
    from dimod import BQM
    from dwave.system import LeapHybridSampler
    bqm = BQM(qubo_description.Q, 'BINARY')
    bqm.offset = qubo_description.offset
    sampler = LeapHybridSampler()
    
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        for _ in range(qubo_description.jobs):
            sampleset = sampler.sample(bqm, time_limit, label=f'oriented_{qubo_description.filename}')
            try:
                logger.info(f"D-Wave access time: {round(sampleset.info['run_time'] / 10 ** 6)}")
            except KeyError:
                pass
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            path = sample_list_to_path(np.array(list(best_sample.values())), qubo_description.graph, qubo_description.T, qubo_description.V)
            paths[time_limit].append((best_sample, best_energy, path))
            
    return paths


def gurobi_sample_qubo(qubo_description: QuboDescription):
    import gurobipy as gp
    from gurobipy import GRB

    total_weight = int(sum(qubo_description.graph.nodes[node]["weight"] for node in list(qubo_description.graph.nodes)) / 2)
    
    logger.info(f'Offset: {qubo_description.offset}')
    logger.info(f'Total weight: {total_weight}')
    logger.info(f'T_max: {qubo_description.T}')
    
    paths = {}
    Q = np.array(qubo_description.Q)
    with gp.Env() as env, gp.Model(env=env) as model:
        model_vars = model.addMVar(shape=Q.shape[0], vtype=GRB.BINARY, name="x")
        model.setObjective(model_vars @ Q @ model_vars, GRB.MINIMIZE)
        model.Params.BestObjStop = - qubo_description.offset
        
        for time_limit in qubo_description.time_limits:
            paths[time_limit] = []
            model.Params.TimeLimit = time_limit
            for _ in range(qubo_description.jobs):
                model.Params.Seed = rng.integers(0, 100000)
                model.optimize()
                energy = model.ObjVal + qubo_description.offset
                path = sample_list_to_path(model_vars.X, qubo_description.graph, qubo_description.T, qubo_description.V)
                paths[time_limit].append((model_vars.X, energy, path))
                model.reset()
    
    return paths


def sample_array_to_path(sample_array: np.ndarray, nodes: list, V: int):
    nz = np.nonzero(sample_array == 1)
    return [
        (
            int(nz[0][i]), 
            nodes[nz[1][i] * 2 + nz[2][i]] if nz[1][i] in range(V) else 'end'
        ) for i in range(nz[0].shape[0])
    ]


def sample_list_to_path(sample_list: np.ndarray, graph: nx.Graph, T_max: int, V: int):
    for idx in [t * (V + 1) * 2 + V * 2 + 1 for t in range(T_max)]:
        sample_list = np.insert(sample_list, idx, 0)
    sample_array = sample_list.reshape((T_max, V + 1, 2))
    return sample_array_to_path(sample_array, list(graph.nodes), V)
    

def print_path(path: list):
    """Pretty print a path"""
    num_per_line = 6
    if len(path) < num_per_line:
        print(path)
        return
    
    for i in range(floor(len(path) / num_per_line)):
        print(path[i * num_per_line: (i + 1) * num_per_line])
    if not (i + 1) * num_per_line == len(path):
        print(path[(i + 1)*num_per_line:])
        
        
def get_original_vertex_name(vertex_name):
    pattern = r'(.+)_([\+\-])+'
    match = re.search(pattern, vertex_name)
    if match is None:
        raise Exception('Could not retrieve vertex name')
    else:
        return match.group(1)
        

def validate_path(path: list, graph: nx.Graph):
    """Checks the constraints for a path on a graph.
    
    In particular:
     - does the path go along graph edges at each time step
     - is each node visited the correct number of times
     - is exactly one node visited per time step

    Args:
        path (list): _description_
        graph (nx.Graph): _description_
    """
    logger.info("Best path:")
    print_path(path)
    if not len(path):
        return
    
    end_nodes = set()
    start_nodes = set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            end_nodes.add(node)
        elif val == 'start':
            start_nodes.add(node)
    if len(end_nodes) > 0:
        end_nodes.add('end')
    
    if len(start_nodes) > 0 and path[0][1] not in start_nodes:
        logger.info('Did not start at start')
    
    time_offset = 0
    i = 0
    while i < len(path):
        if i + time_offset == path[i][0]:
            i += 1
            continue
        if path[i][0] < i + time_offset:
            logger.info(f'Extra node at time {path[i][0]}')
            time_offset -= 1
            i += 1
            continue
        if path[i][0] > i + time_offset:
            logger.info(f'Skipped time {path[i][0] - 1}')
            time_offset += 1
            i += 1
            continue
    
    node_dict = {node: 0 for node in graph.nodes}
    node_dict['end'] = 0
    
    for x in range(len(path) - 1):
        v1 = path[x][1]
        node_dict[v1] += 1
        v2 = path[x + 1][1]            
        if v1 == 'end' and not v2 == 'end':
            logger.info(f'Left the end node at path entry {x}')
        elif (not v1 == 'end') and (not v2 == 'end') and ((v1, v2) not in graph.edges):
            logger.info(f'Broke graph edge at path entry {x}')
        elif len(end_nodes) > 0 and (v2 == 'end') and (v1 not in end_nodes):
            logger.info(f'Went to end node illegally at path entry {x}')
    if len(path) > 1:
        node_dict[v2] += 1
    if len(path) == 1:
        node_dict[path[0][1]] += 1
    
    nodes = list(graph.nodes)
    for i in range(int(len(nodes) / 2)):
        visits = node_dict[nodes[2 * i]] + node_dict[nodes[2 * i + 1]]
        missing_visits = graph.nodes[nodes[2 * i]]["weight"] - visits
        if  missing_visits != 0:
            logger.info(f'Did not meet node weight for node: {get_original_vertex_name(nodes[2 * i])}. Missing visits: {missing_visits}')
            
            
            
def validate_edge2node_path(path: list, graph: nx.Graph):
    """Checks the constraints for a path on a graph.
    
    In particular:
     - does the path go along graph edges at each time step
     - is each node visited the correct number of times
     - is exactly one node visited per time step

    Args:
        path (list): _description_
        graph (nx.Graph): _description_
    """
    logger.info("Best path:")
    print_path(path)
    if not len(path):
        return
    
    end_nodes = set()
    start_nodes = set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            end_nodes.add(node)
        elif val == 'start':
            start_nodes.add(node)
    if len(end_nodes) > 0:
        end_nodes.add('end')
    
    if len(start_nodes) > 0 and path[0][1] not in start_nodes:
        logger.info('Did not start at start')
    
    time_offset = 0
    i = 0
    while i < len(path):
        if i + time_offset == path[i][0]:
            i += 1
            continue
        if path[i][0] < i + time_offset:
            logger.info(f'Extra node at time {path[i][0]}')
            time_offset -= 1
            i += 1
            continue
        if path[i][0] > i + time_offset:
            logger.info(f'Skipped time {path[i][0] - 1}')
            time_offset += 1
            i += 1
            continue
    
    node_dict = {node: 0 for node in graph.nodes}
    node_dict['end'] = 0
    
    for x in range(len(path) - 1):
        v1 = path[x][1]
        node_dict[v1] += 1
        v2 = path[x + 1][1]            
        if v1 == 'end' and not v2 == 'end':
            logger.info(f'Left the end node at path entry {x}')
        elif (not v1 == 'end') and (not v2 == 'end') and ((v1, v2) not in graph.edges):
            logger.info(f'Broke graph edge at path entry {x}')
        elif len(end_nodes) > 0 and (v2 == 'end') and (v1 not in end_nodes):
            logger.info(f'Went to end node illegally at path entry {x}')
    if len(path) > 1:
        node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(int(len(nodes))):
        visits = node_dict[nodes[i]]
        missing_visits = graph.nodes[nodes[i]]["weight"] - visits
        if  missing_visits != 0:
            logger.info(f'Did not meet node weight for node: {nodes[i]}. Missing visits: {missing_visits}')
