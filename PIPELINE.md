# QUBO Solver Development Pipeline

This repo implements the pangenome-guided assembly formulation from
`paper.md`. For solver development, start from the graph-level problem:

```text
GFA graph + node copy numbers -> QUBO -> path
```

The full DNA simulation path is still important, but it depends on external
bioinformatics tools. New solver work should first prove itself on the QUBO
stage, then move back into the full pipeline.

## Minimal Environment

From the repo root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python gfapy networkx numpy
uv pip install --python .venv/bin/python -e qubo
```

Use `PYTHONPATH=qubo` when running the package CLIs.

## Solver Contract

The active QUBO solver entrypoint is:

```text
qubo/qubo_solvers/oriented_tangle/oriented_max_path.py
```

Registered solver names include:

```text
exact
local
greedy_residual
random_residual_walk
beam_search
aco
neural_aco
astar
seea
mqlib
gurobi
dwave
```

A solver implementation lives in:

```text
qubo/qubo_solvers/oriented_tangle/utils/sampling_utils.py
```

and must return:

```python
{
    time_limit: [
        (binary_solution_vector, energy, decoded_path),
        ...
    ]
}
```

where:

```text
binary_solution_vector: one-dimensional binary vector for the QUBO variables
energy:                 QUBO energy, equivalent to x @ Q @ x + offset
decoded_path:           output of sample_list_to_path(...)
```

`astar` is the non-neural legal-prefix search baseline. `seea` is the neural
SeeA* solver: it searches legal prefixes, ranks OPEN nodes with
`g(n)+beta*h_theta(n)`, requires a neural checkpoint, pads completed paths with
`end`, and scores terminal paths with the same QUBO energy used by `exact` and
`local`. This is important: benchmark SeeA* in the existing QUBO energy space
before introducing a new biological or edge-aware objective.

The lightweight graph-walk baselines are:

```text
greedy_residual       deterministic residual-copy greedy construction
random_residual_walk  stochastic residual-copy legal walks
beam_search           bounded legal-prefix beam search
aco                   ant-colony construction over legal oriented edges
neural_aco            DyNACO-style neural-prior ACO
```

## Benchmark Rule

Use three levels of comparison:

1. On tiny cases, `exact` is the reference. A new solver should have
   `gap_to_exact = 0`.
2. Once `exact` refuses the search space, `local` is the simple heuristic
   baseline. A* and neural SeeA* should be faster and/or lower energy than
   `local`.
3. Biological reconstruction quality is a separate second-stage check because
   multiple decoded paths can have the same QUBO energy.

The benchmark helper is:

```text
examples/benchmark_search_solvers.py
```

It reports `energy`, `gap_to_exact`, `gap_to_local`, runtime, and status.

## Active Paper Run Policy

The stopped full-assembly run under
`results/overnight_dynaco_paper/20260522_013319/full_assembly` is now the
baseline cache for the retained original-paper solver target, MQLib. Do not
rerun MQLib by default. Future paper-run compute should go to DyNACO variants
only, using the cached MQLib rows below as the comparison target.

The active picture uses only the retained `mg` annotation route. The excluded
annotation routes are not optimization targets for the current DyNACO story.

Source artifact:

```text
results/overnight_dynaco_paper/20260522_013319/analytics/partial_full_assembly_best_consensus.csv
```

The values below are parsed from available `*.eval_cons.*` files after the run
was stopped. For repeated stochastic jobs, the parser chooses the best row per
solver/graph/seed/sequence by covered, used, fewer breaks, then identity. This
is a reusable partial baseline cache, not the official completed benchmark
parser output. Pathfinder rows from the stopped run are intentionally not part
of the active baseline target; Pathfinder remains preprocessing/protocol
context, while the paper target is to beat MQLib on the retained minigraph
annotator.

| solver | graph | seqs | seeds | covered | used | contigs | breaks | indels | diffs | identity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mqlib | mg | 44 | 00001-00009 | 56.23 | 94.87 | 3.91 | 1.43 | 0.27 | 0.00 | 99.17 |

Pure ACO and current DyNACO rows from the same stopped run are useful as
ablation/context, but they are not the target baseline because future runs
should replace DyNACO settings and only MQLib represents the retained original
paper solver target:

| solver | graph | seqs | seeds | covered | used | contigs | breaks | indels | diffs | identity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aco | mg | 35 | 00001-00007 | 52.74 | 94.15 | 4.46 | 1.49 | 0.31 | 0.06 | 99.23 |
| neural_aco | mg | 40 | 00001-00008 | 68.39 | 93.11 | 4.12 | 2.48 | 0.40 | 0.03 | 99.22 |

For follow-up runs, keep the cached MQLib baseline rows fixed and run only
DyNACO. The immediate target is to preserve the MG win over MQLib
(`68.39 > 56.23` covered in the stopped snapshot). Example:

```bash
python scripts/run_dynaco_claim_experiments.py \
  --execute \
  --out-dir results/dynaco_claims/<new-mg-run> \
  --suite full-budget \
  --model <checkpoint> \
  --annotators mg \
  --seeds 8 \
  --test-sequences 5 \
  --budgets 10,100 \
  --full-jobs 1 \
  --pathfinder-graph
```

## Handcrafted Smoke Test

The smallest wiring test is:

```text
examples/tiny_line.gfa
```

It is a five-node line:

```text
A -> B -> C -> D -> E
```

with copy numbers:

```text
1,1,1,1,1
```

Expected optimum:

```text
A_+ -> B_+ -> C_+ -> D_+ -> E_+
```

Run the non-neural QUBO-stage solvers:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  -t 1 \
  -j 1 \
  --solvers exact,local,greedy_residual,random_residual_walk,beam_search,aco,astar
```

Validated result from this checkout:

```text
QUBO shape: (55, 55), T=5, states_per_time=11
exact: -0.059016994375 in 0.5936s
local: -0.059016994375 in 1.0003s
greedy_residual:      -0.059016994375 in 0.0005s
random_residual_walk: -0.059016994375 in 1.0002s
beam_search:          -0.059016994375 in 0.0007s
aco:                  -0.059016994375 in 1.0026s
astar: -0.059016994375 in 0.0008s
```

This file is only a smoke test for solver wiring. It is not a paper-style
benchmark.

## Generated Exact-Check Case

The first non-handcrafted solver-quality check is:

```text
synthetic_quality_sweep/case_s3_l3000/seq_0039-0006-#1#1.gfa
```

It was generated with `genome_create` and `minigraph`, then held-out reads were
annotated back onto the synthetic pangenome. The generated held-out GFA has
three segment nodes and four links.

Run:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f 'synthetic_quality_sweep/case_s3_l3000/seq_0039-0006-#1#1.gfa' \
  -c 1,1,1 \
  -t 1 \
  -j 1 \
  --solvers exact,local,greedy_residual,random_residual_walk,beam_search,aco,astar
```

Validated result from this checkout:

```text
QUBO shape: (21, 21), T=3, states_per_time=7
exact: -0.00369190060474
local: -0.00369190060474
greedy_residual:      -0.00369190060474
random_residual_walk: -0.00369190060474
beam_search:          -0.00369190060474
aco:                  -0.00369190060474
astar: -0.00369190060474
```

Representative biological reconstruction scores for exact/Gurobi/forward-local
on this case were:

```text
len-t 2995, len-q 3006, covered 92.35%, used 88.66%, identity 95.64%
```

Some reverse-complement or orientation-flipped paths have the same QUBO energy
but different reconstruction metrics. Treat that as objective degeneracy, not a
solver failure.

## Harder Search Benchmarks

### 7-node Generated Pangenome Graph

This case is generated, larger than the exact-check case, and already too large
for `exact` enumeration:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f synthetic_quality_sweep/case_s2_l3000/pop.gfa \
  -t 1 \
  -j 1 \
  --solvers exact,local,astar \
  --no-paths
```

`--copy-numbers` defaults to `ones`, so this command uses clean all-one copy
numbers.

Validated result from this checkout:

```text
QUBO shape: (105, 105), T=7, states_per_time=15
exact: refuses 170,859,375 candidates
local: 1.76548544057 in 1.0004s
greedy_residual:      2.41533288721 in 0.0009s
random_residual_walk: 1.76548544057 in 1.0002s
beam_search:          1.76548544057 in 0.0021s
aco:                  1.76548544057 in 1.0070s
astar: 1.76548544057 in 0.0023s
```

This verifies that the non-neural A* baseline can match the local heuristic
energy much faster once exact enumeration is unavailable.

### 52-node Built-in Tangle Graph

This is the current larger stress test for the non-neural search:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f data/drb1+tangle1.gfa \
  -t 1 \
  -j 1 \
  --max-expansions 1000 \
  --solvers exact,local,astar \
  --no-paths
```

Validated result from this checkout:

```text
QUBO shape: (5985, 5985), T=57, states_per_time=105
exact: refuses an astronomically large search space
local: 1169.90197482 in 1.8050s
greedy_residual:      30.4725528509 in 0.0860s
random_residual_walk: 13.5785922629 in 1.0003s
beam_search:          20.4480896787 in 1.0017s
aco:                  14.9673123420 in 1.0638s
astar:                14.0154352609 in 0.4048s
```

This is not a biological benchmark because copy numbers are clean all-ones, but
it does test the intended solver property: much lower QUBO energy than the
simple local baseline while exact is unavailable.

## Direct Solver CLI

To build a QUBO manually:

```bash
PYTHONPATH=qubo .venv/bin/python qubo/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  -p 200,50,1 \
  -d examples/tiny_out
```

To run a single solver through the package entrypoint:

```bash
PYTHONPATH=qubo .venv/bin/python qubo/qubo_solvers/oriented_tangle/oriented_max_path.py \
  -f examples/tiny_line.gfa \
  -s astar \
  -t 1 \
  -j 1 \
  -d examples/tiny_out \
  -o examples/tiny_out/astar.path
```

Use `exact` only for tiny graphs. It intentionally refuses large search spaces.

For A* and neural SeeA*, the expansion cap can be controlled with:

```bash
QPG_ASTAR_MAX_EXPANSIONS=1000
```

For SeeA*, the candidate subset size can be controlled with:

```bash
QPG_SEEA_K=50
```

## Neural SeeA* GNN

The first neural component is implemented in:

```text
qubo/qubo_solvers/oriented_tangle/neural_gnn.py
```

It adapts the DyNACO GNN pattern to QPG rather than importing the TSP/CVRP model
directly. The reusable DyNACO part is the architecture style: node and edge
embeddings, repeated edge-aware message passing, an edge/action decoder, and a
state-value head. The QPG-specific part is the input representation: oriented
GFA nodes, merged copy-number residual counts, dynamic current-node features,
legal GFA edges, an artificial `start` action, and an artificial `end` action.

Smoke-test the untrained model:

```bash
PYTHONPATH=qubo .venv/bin/python examples/smoke_qpg_gnn.py \
  -f examples/tiny_line.gfa
```

Validated result from this checkout:

```text
nodes: 12, edges: 30
value_shape: (), edge_logits_shape: (30,), embedding_shape: (102,)
```

Larger graph smoke test:

```bash
PYTHONPATH=qubo .venv/bin/python examples/smoke_qpg_gnn.py \
  -f data/drb1+tangle1.gfa \
  --units 32 \
  --depth 3
```

Validated result from this checkout:

```text
nodes: 106, edges: 374
value_shape: (), edge_logits_shape: (374,), embedding_shape: (102,)
```

## Simple REINFORCE Training

The current learned component is a simple DyNACO-style REINFORCE trainer:

```text
examples/train_qpg_reinforce.py
```

It samples legal graph walks from the GNN edge logits, scores each completed
walk with the same terminal QUBO energy as the search solvers, trains the value
head against the method's prefix cost-to-go target, and keeps the best sampled
terminal path. The policy reward is `-energy`, so lower QUBO energy is better;
the value head is the neural `h_theta` used by SeeA*.

Tiny-line mechanics smoke:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_reinforce.py \
  -f examples/tiny_line.gfa \
  --epochs 5 \
  --episodes 4 \
  --units 16 \
  --depth 2 \
  --device cpu
```

Validated result from this checkout:

```text
greedy_baseline_energy: -0.0590169943751
best_sampled_energy:   -0.0590169943751
greedy_policy_energy:   1.16458980337
best_path: E_- -> D_- -> C_- -> B_- -> A_-
```

Three-node generated smoke:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_reinforce.py \
  -f 'synthetic_quality_sweep/case_s3_l3000/seq_0039-0006-#1#1.gfa' \
  -c 1,1,1 \
  --epochs 20 \
  --episodes 8 \
  --units 16 \
  --depth 2 \
  --device cpu
```

Validated result from this checkout:

```text
greedy_baseline_energy: -0.00369190060476
best_sampled_energy:   -0.00369190060476
greedy_policy_energy:   2.75938861678
best_path: s1_+ -> s2_+ -> s3_+
```

To save a checkpoint:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_reinforce.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  --epochs 20 \
  --episodes 8 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --out results/neural_seea/tiny_reinforce.pt
```

## Neural SeeA* Solver

`seea` is now the neural SeeA* solver. It requires a checkpoint and fails
closed if no checkpoint is provided. Use `astar` for the non-neural baseline.
The SeeA* priority is:

```text
f_theta(n) = g(n) + beta * h_theta(n)
```

where `g(n)` is the irreversible over-copy prefix cost from `method.tex`, and
`h_theta(n)` is the GNN value head loaded from the checkpoint. Operationally,
the implementation evaluates the GNN once per expanded prefix and uses both the
value head and policy logits to score children:

```text
priority(child) = g(child) + beta * value(prefix) - policy_weight * log pi(child | prefix)
```

Every expanded prefix is greedily completed and scored as a terminal incumbent,
so neural SeeA* remains an anytime optimizer even when the learned policy does
not reach depth `T` within the wall-clock budget. Terminal benchmark scoring
still uses exact QUBO energy so neural SeeA* remains comparable to exact, local,
ACO, beam, and A*.

For larger graphs, supervised prefix training is more useful than pure
REINFORCE:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_prefix_supervised.py \
  -f data/drb1+tangle1.gfa \
  --epochs 15 \
  --random-walks 16 \
  --max-prefixes 384 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --target energy-gap \
  --out results/neural_seea/drb1_prefix_gap.pt
```

Validated training result from this checkout:

```text
trajectories: 122
best_training_energy: 14.0154352609
training_seconds: 21.238
```

Run neural SeeA* on the tiny exact-check case:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  -t 1 \
  -j 1 \
  --solvers exact,astar,seea \
  --neural-model results/neural_seea/tiny_reinforce.pt \
  --device cpu \
  --no-paths
```

Validated result from this checkout:

```text
exact: -0.0590169943751
astar: -0.0590169943751
seea:  -0.0590169943751
```

Run a larger neural smoke benchmark:

```bash
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f data/drb1+tangle1.gfa \
  -t 2 \
  -j 1 \
  --max-expansions 1000 \
  --solvers local,greedy_residual,astar,seea \
  --neural-model results/neural_seea/drb1_prefix_gap.pt \
  --device cpu \
  --no-paths
```

Validated result from this checkout:

```text
local:           921.197547821
greedy_residual: 30.4725528509
astar:           14.0154352609
seea:            22.3147114933
```

This confirms that SeeA* is now truly neural and can beat both the simple local
heuristic and greedy residual on the larger graph. It is still worse and slower
than non-neural A* on this instance, so the next research step is not to claim a
win over A*. The next step is to train `h_theta` on a graph split that matches
the stress/test distribution, then compare neural SeeA* against A*, ACO, beam,
and local under equal expansion and wall-clock budgets.

## DyNACO-Style Neural ACO

`neural_aco` is a QPG-native adaptation of the original DyNACO training pattern.
It keeps the ACO pheromone table and residual-copy heuristic, then adds the GNN
edge prior to ant transition scores:

```text
log_score(edge) =
  alpha * log pheromone(edge)
  + beta * log residual_heuristic(edge)
  + gamma * neural_logit(edge)
```

The deployed solver is now backed by a C++/pybind ant sampler:

```text
qubo/qubo_solvers/oriented_tangle/cpp/qpg_aco_cpp.cpp
qubo/qubo_solvers/oriented_tangle/qpg_aco_cpp*.so
```

The sampler builds complete ant batches, returns compact edge traces for
probability replay, and uses the same dynamic residual-copy heuristic as the
plain Python ACO baseline. Rebuild it after editing the C++ source:

```bash
c++ -O3 -shared -std=c++17 -fPIC -fopenmp $(python -m pybind11 --includes) \
  qubo/qubo_solvers/oriented_tangle/cpp/qpg_aco_cpp.cpp \
  -o qubo/qubo_solvers/oriented_tangle/qpg_aco_cpp$(python-config --extension-suffix)
```

The DyNACO-style trainer is:

```text
examples/train_qpg_dynaco_cpp.py
```

Its loop follows the original DyNACO structure more closely than the older
Python-only prototype: compute a neural prior, sample C++ ACO batches, update
pheromone from elite ants, recompute the prior, replay C++ traces through a
differentiable log-probability calculation, and apply REINFORCE with
batch-centered costs. The checkpoint saves both the GNN state and the best
pheromone state so the deployed solver uses the trained ACO state.

Tiny smoke training:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_dynaco_cpp.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  --epochs 2 \
  --outer 1 \
  --mini-h 1 \
  --ants 8 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --out results/dynaco_cpp/tiny_dynamic.pt
```

Validated result from this checkout:

```text
aco_baseline_energy: -0.0590169943751
best_sampled_energy: -0.0590515136719
```

The tiny C++ training score is reported from float32 QUBO arithmetic; benchmark
commands below recompute the returned path with the normal Python QUBO energy.

Stress-graph training:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_dynaco_cpp.py \
  -f data/drb1+tangle1.gfa \
  --epochs 80 \
  --outer 4 \
  --mini-h 4 \
  --ants 64 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --out results/dynaco_cpp/drb1_dynamic.pt
```

Validated result from this checkout:

```text
aco_baseline_energy: 13.8360867049
best_sampled_energy: 11.6552734375
training_seconds: 492.909
```

Benchmark the resulting checkpoint against plain ACO on the same QUBO:

```bash
env QPG_ACO_ANTS=64 QPG_ACO_MIN_ITERATIONS=4 QPG_ACO_GAMMA=1.0 \
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f data/drb1+tangle1.gfa \
  -t 1 \
  -j 3 \
  --solvers aco,neural_aco \
  --neural-model results/dynaco_cpp/drb1_dynamic.pt \
  --device cpu \
  --no-paths
```

Validated result from this checkout:

```text
aco:        12.9127610234 in 3.1725s
neural_aco: 12.3224328280 in 1.7186s
```

This is the current passing stress result for the intended middle ground:
`neural_aco` is much faster than exact, lower-energy than simple heuristics, and
lower-energy than plain ACO on this repeated benchmark. Because ACO is
stochastic, keep reporting repeated-job results rather than a single draw when
claiming a solver improvement.

### Multi-Instance Online DyNACO Training

For a train-then-test ML experiment, use the online trainer:

```text
examples/train_qpg_dynaco_online.py
```

This is different from the single-instance adaptation run above. During
training it samples a training GFA, builds its QUBO, initializes fresh
pheromone, generates C++ ACO trajectories online, replays those sampled traces
through the current GNN prior, updates the shared model with REINFORCE, and then
moves to another sampled instance. The saved checkpoint contains the GNN only:
instance pheromone is deliberately not saved because pheromone is local ACO
memory, not a general learned model.

Small generated-data smoke run:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_dynaco_online.py \
  --generate-synthetic-train 3 \
  --generate-synthetic-test 1 \
  --synthetic-dir results/dynaco_online/generated_smoke \
  --synthetic-min-segments 4 \
  --synthetic-max-segments 6 \
  --epochs 1 \
  --instances-per-epoch 2 \
  --online-steps 1 \
  --mini-h 1 \
  --ants 4 \
  --eval-ants 8 \
  --eval-min-iterations 1 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --out results/dynaco_online/smoke.pt
```

Validated result from this checkout:

```text
train_instances: 3
test_instances: 1
generated held-out eval:
  aco        -0.0144540549247
  neural_aco -0.0144540549246
checkpoint keys: config, model_state_dict, training_seconds
checkpoint has no pheromone state
```

The resulting model-only checkpoint can be tested through the normal benchmark
entrypoint:

```bash
env QPG_ACO_ANTS=8 QPG_ACO_MIN_ITERATIONS=1 QPG_ACO_GAMMA=1.0 \
PYTHONPATH=qubo .venv/bin/python examples/benchmark_search_solvers.py \
  -f results/dynaco_online/generated_smoke/test/test_0000.gfa \
  -t 1 \
  -j 1 \
  --solvers aco,neural_aco \
  --neural-model results/dynaco_online/generated_smoke.pt \
  --device cpu \
  --no-paths
```

For the real generalization experiment, split generated GFAs into train and
held-out sets. The reusable overnight CUDA config is:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_dynaco_online.py \
  --config configs/dynaco_online_hard.yaml
```

This config generates `1000` training GFAs and `100` held-out generated GFAs
with `48-96` segments, uses the Torch `cuda` device, and also includes
`data/drb1+tangle1.gfa` as a fixed held-out stress graph.

Use this setup when claiming a learned checkpoint: online RL trajectory
generation happens only during training, and held-out testing loads the saved
model with fresh pheromone.

You can also mix generated data with existing GFAs by adding `--train-gfas`,
`--train-glob`, `--test-gfas`, or `--test-glob`. Keep those splits disjoint.

Larger generated split benchmark:

```bash
PYTHONPATH=qubo .venv/bin/python examples/train_qpg_dynaco_online.py \
  --generate-synthetic-train 40 \
  --generate-synthetic-test 10 \
  --synthetic-dir results/dynaco_online/generated_larger \
  --synthetic-min-segments 8 \
  --synthetic-max-segments 16 \
  --synthetic-bubble-rate 0.4 \
  --synthetic-orientation-rate 0.2 \
  --synthetic-shortcuts 4 \
  --epochs 15 \
  --instances-per-epoch 2 \
  --online-steps 2 \
  --mini-h 2 \
  --ants 32 \
  --eval-ants 64 \
  --eval-min-iterations 4 \
  --units 16 \
  --depth 2 \
  --device cpu \
  --out results/dynaco_online/larger.pt
```

Validated result from this checkout:

```text
train_instances: 40
test_instances: 10
training_seconds: 26.807
held-out comparison: neural_aco 2 wins, 8 ties, 0 losses vs aco
mean aco energy:        0.133797687626
mean neural_aco energy: -0.032147035529
```

The saved checkpoint was also checked through `benchmark_search_solvers.py` on
held-out `test_0000.gfa`:

```text
aco:        0.789471904664
neural_aco: -0.0345651302566
```

## Full Synthetic Pipeline

The paper-style synthetic path is:

1. Generate a synthetic haploid population with fixed random seeds.
2. Include shared complex structure: STRs, CNVs, repeats, translocations,
   inversions, and point mutations.
3. Build a pangenome graph from training genomes using `minigraph`.
4. Hold out synthetic genomes as test genomes.
5. Simulate whole-genome shotgun reads from held-out genomes.
6. Align/map reads back to the pangenome graph.
7. Convert graph annotations to node copy numbers.
8. Solve the weighted graph path problem.

The repo implementation starts at:

```text
genome_create.c
run_sim_create_gfa.sh
run_gfa_sim.sh
run_sim_add_gfa_weights_*.sh
run_sim_solver_qubo.sh
```

The smoke config is:

```text
examples/config_synthetic_tiny.sh
```

It validates the machinery but may collapse to a one-node graph. For solver
quality, increase `TRAIN_COUNT`, `genome_opts -l`, and/or mutation rates until
`pop.gfa` has a nontrivial number of `S` lines.

If `minigraph` is missing:

```bash
mkdir -p .tools
git clone --depth 1 https://github.com/lh3/minigraph .tools/minigraph
make -C .tools/minigraph
```

Build the local generator:

```bash
gcc -O2 genome_create.c -o genome_create -lm
```

Create one synthetic pangenome:

```bash
mkdir -p synthetic_mg_004
cd synthetic_mg_004

env QDIR=/home/shora/Research/qpg \
  CONFIG=/home/shora/Research/qpg/examples/config_synthetic_tiny.sh \
  PATH=/home/shora/Research/qpg:/home/shora/Research/qpg/.tools/minigraph:$PATH \
  ../run_sim_create_gfa.sh 1 1
```

Validated smoke result from this checkout:

```text
Node count: 1
fofn.train: 3 sequences
fofn.test:  1 held-out sequence
```

The full local evaluation path also requires `minimap2`, `bwa`, and `samtools`.
Once those are available, run:

```bash
env QDIR=/home/shora/Research/qpg \
  PYTHON=/home/shora/Research/qpg/.venv/bin/python \
  PYTHONPATH=/home/shora/Research/qpg/qubo \
  PATH=/home/shora/Research/qpg:/home/shora/Research/qpg/.tools/minigraph:/home/shora/Research/qpg/.tools/minimap2:/home/shora/Research/qpg/.tools/bwa:/home/shora/Research/qpg/.tools/samtools/build/bin:/home/shora/Research/qpg/.tools/htslib/build/bin:$PATH \
  LD_LIBRARY_PATH=/home/shora/Research/qpg/.tools/htslib/build/lib:$LD_LIBRARY_PATH \
  BWA=/home/shora/Research/qpg/.tools/bwa/bwa \
  ./run_gfa_sim.sh \
  -c examples/config_synthetic_tiny.sh \
  --solver local \
  -t 1 \
  -j 1 \
  -n 1 \
  -p synthetic_full_ \
  -s 1
```

Validated full smoke output from this checkout:

```text
Node count: 1
COPY_NUMBERS=1
Best path: [(0, 's1_-')]
Energy of path: -0.0005597872361136069
eval seq:  99.75% covered, 99.75% identity
eval cons: 100.00% covered, 100.00% identity
```

This validates plumbing only. It is not a solver-quality benchmark.

## Biological Path Scoring

To score a representative path biologically, convert it to FASTA and run
`candidate_stats.sh` serially:

```bash
cd /home/shora/Research/qpg/synthetic_quality_sweep/case_s3_l3000

{ printf '>contig_1\n'; /home/shora/Research/qpg/path2seq.pl \
  seq_0039-0006-#1#1.gfa qout_clean/exact.path.1.0; } \
  > qout_clean/exact.path_seq.1.0

env PATH=/home/shora/Research/qpg:/home/shora/Research/qpg/.tools/bwa:/home/shora/Research/qpg/.tools/samtools/build/bin:/home/shora/Research/qpg/.tools/htslib/build/bin:$PATH \
  LD_LIBRARY_PATH=/home/shora/Research/qpg/.tools/htslib/build/lib:$LD_LIBRARY_PATH \
  BWA=/home/shora/Research/qpg/.tools/bwa/bwa \
  /home/shora/Research/qpg/candidate_stats.sh \
  seq_0039-0006-#1#1 qout_clean/exact.path_seq.1.0
```

Do not run `candidate_stats.sh` in parallel in the same directory. It writes
fixed scratch files such as `log.txt`, `_.sam`, `bwa_idx.err`, and
`bwa_mem.err`.

## Correctness Checklist

Before moving to larger or DNA-derived examples, verify:

1. `exact` returns the expected path on `tiny_line.gfa`.
2. The new solver returns the same energy on `tiny_line.gfa`.
3. On the generated 3-node case, the new solver has `gap_to_exact = 0`.
4. On larger cases where exact refuses, the new solver improves or matches
   `local` under the same or lower runtime.
5. The decoded path has one active state per time step.
6. The decoded path does not break graph edges, unless the QUBO penalty tradeoff
   is explicit and intentional.
7. Biological reconstruction metrics are reported separately from QUBO energy.

`oriented_max_path.py` calls `validate_path` after each solver run.

## Next Test Cases To Add

Do not grow the benchmark set with more handcrafted graphs. Keep
`examples/tiny_line.gfa` only as a wiring smoke test. The real examples should
come from the paper-style synthetic haploid pangenome pipeline:

1. Sweep `genome_create` seed and length.
2. Build `pop.gfa` with `minigraph`.
3. Keep cases with more than one `S` node.
4. Keep a small subset where exact still runs, for correctness.
5. Keep a larger subset where exact refuses, for speed/quality stress testing.
6. Annotate held-out genomes with `run_sim_add_gfa_weights_mg.sh`.
7. Compare solvers on QUBO energy first.
8. Then compare representative candidate reconstruction metrics with
   `candidate_stats.sh`.

For each retained generated case, record:

```text
generation seed and genome length
held-out query name
number of S and L records
copy-number vector
QUBO shape, T, states_per_time
exact/gurobi energy if available
local energy and runtime
astar/seea energy, runtime, expansion cap, and hit rate across repeated jobs
candidate_stats.sh metrics for representative degenerate optima
```
