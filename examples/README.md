# Tiny QUBO Solver Harness

This directory contains a minimal graph-level test case for developing QUBO
solvers without running the full DNA simulation pipeline.

The data unit is:

```text
GFA graph + node copy numbers -> QUBO -> path
```

Create the QUBO data:

```bash
PYTHONPATH=qubo .venv/bin/python qubo/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py \
  -f examples/tiny_line.gfa \
  -c 1,1,1,1,1 \
  -p 200,50,1 \
  -d examples/tiny_out
```

Run the exact baseline on this tiny instance:

```bash
PYTHONPATH=qubo .venv/bin/python qubo/qubo_solvers/oriented_tangle/oriented_max_path.py \
  -f examples/tiny_line.gfa \
  -s exact \
  -t 1 \
  -j 1 \
  -d examples/tiny_out \
  -o examples/tiny_out/exact.path
```

Run the built-in local-search heuristic:

```bash
PYTHONPATH=qubo .venv/bin/python qubo/qubo_solvers/oriented_tangle/oriented_max_path.py \
  -f examples/tiny_line.gfa \
  -s local \
  -t 1 \
  -j 3 \
  -d examples/tiny_out \
  -o examples/tiny_out/local.path
```

To test a custom solver, add a function with the same return contract as
`local_sample_qubo` in `qubo/qubo_solvers/oriented_tangle/utils/sampling_utils.py`:

```python
{
    time_limit: [
        (binary_solution_vector, energy, decoded_path),
        ...
    ]
}
```

Then add the solver name to `Solver` in `qubo/qubo_solvers/definitions.py` and
dispatch to it from `oriented_max_path.py`.
