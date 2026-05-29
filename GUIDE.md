# First-Time Setup Guide

This guide is for a new collaborator who wants to run the QUBO solver and the
synthetic pangenome smoke pipeline locally.

The repo has two kinds of requirements:

1. Python packages for the QUBO code.
2. Native bioinformatics command-line tools for synthetic graph construction
   and evaluation.

`uv` handles the Python side. It does not install native tools such as
`minigraph`, `minimap2`, `bwa`, or `samtools`.

## Repo Root

All commands below assume:

```bash
cd /home/shora/Research/qpg
```

## Python Environment

Create the local environment and install the QUBO package:

```bash
uv venv .venv
uv pip install --python .venv/bin/python gfapy networkx numpy
uv pip install --python .venv/bin/python -e qubo
```

Use this Python when running solver CLIs:

```bash
export PYTHON=/home/shora/Research/qpg/.venv/bin/python
export PYTHONPATH=/home/shora/Research/qpg/qubo
```

## Build Local Repo Binaries

The synthetic genome generator is in this repo:

```bash
gcc -O2 genome_create.c -o genome_create -lm
```

`kmer2node4` requires `htslib` headers. It is not needed for the validated
`mg` smoke pipeline, so skip it unless working on kmer2node annotation.

## Build Native Tools

Place external tools under `.tools/` so the repo stays self-contained.

### minigraph

Used to build the synthetic pangenome and to annotate reads in the `mg` path.

```bash
mkdir -p .tools
git clone --depth 1 https://github.com/lh3/minigraph .tools/minigraph
make -C .tools/minigraph
```

### minimap2

Used during consensus/evaluation remapping.

```bash
git clone --depth 1 https://github.com/lh3/minimap2 .tools/minimap2
make -C .tools/minimap2
```

### BWA

Used by `candidate_stats.pl` for candidate/truth alignment.

```bash
git clone --depth 1 https://github.com/lh3/bwa .tools/bwa
make -C .tools/bwa
```

### htslib and samtools

Used by `run_sim_evaluate_path.sh` and `candidate_stats.pl`.

```bash
git clone --depth 1 https://github.com/samtools/htslib .tools/htslib
cd .tools/htslib
git submodule update --init --recursive
autoreconf -i
./configure --disable-libcurl --prefix=/home/shora/Research/qpg/.tools/htslib/build
make -j2
make install
cd /home/shora/Research/qpg

git clone --depth 1 https://github.com/samtools/samtools .tools/samtools
cd .tools/samtools
autoheader
autoconf
./configure --with-htslib=/home/shora/Research/qpg/.tools/htslib/build \
  --prefix=/home/shora/Research/qpg/.tools/samtools/build
make -j2
make install
cd /home/shora/Research/qpg
```

## Environment For Runs

Use these variables for local runs:

```bash
export QDIR=/home/shora/Research/qpg
export PYTHON=/home/shora/Research/qpg/.venv/bin/python
export PYTHONPATH=/home/shora/Research/qpg/qubo
export BWA=/home/shora/Research/qpg/.tools/bwa/bwa
export LD_LIBRARY_PATH=/home/shora/Research/qpg/.tools/htslib/build/lib:$LD_LIBRARY_PATH
export PATH=/home/shora/Research/qpg:/home/shora/Research/qpg/.tools/minigraph:/home/shora/Research/qpg/.tools/minimap2:/home/shora/Research/qpg/.tools/bwa:/home/shora/Research/qpg/.tools/samtools/build/bin:/home/shora/Research/qpg/.tools/htslib/build/bin:$PATH
```

Quick check:

```bash
which minigraph
which minimap2
which bwa
which samtools
$PYTHON -c "import gfapy, networkx, numpy; print('python ok')"
```

## Smoke Tests

### QUBO-Only Tiny Test

This checks the solver interface without the synthetic pangenome pipeline:

```bash
PYTHONPATH=qubo .venv/bin/python examples/run_tiny_oriented_qubo.py
```

Expected path:

```text
A_+ -> B_+ -> C_+ -> D_+ -> E_+
```

### Full Synthetic Smoke Test

This runs synthetic generation, minigraph pangenome construction, read
annotation, QUBO solving, candidate evaluation, and consensus evaluation:

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

Validated result in this checkout:

```text
Node count: 1
COPY_NUMBERS=1
eval seq:  99.75% covered, 99.75% identity
eval cons: 100.00% covered, 100.00% identity
```

This validates the end-to-end plumbing. It is not a solver-quality benchmark
because the smoke graph has one node.

### Solver-Quality Synthetic Test

After the one-node smoke path works, use the generated multi-node case:

```text
synthetic_quality_sweep/case_s3_l3000
```

This case was generated from `genome_create` and `minigraph`, not handcrafted.
It has a three-node held-out GFA and is small enough to compare `local` or a new
solver against `exact` and `gurobi`.

Validated result in this checkout:

```text
exact:  s1 -> s2 -> s3, energy -0.003691900604735565
gurobi: s1 -> s2 -> s3, energy -0.003691900604735565
local:  20/20 runs reached energy -0.003691900604735565
```

Representative reconstruction scores on the held-out genome:

```text
forward optimum: covered 92.35%, used 88.66%, identity 95.64%
reverse optimum: covered 88.98%, used 88.66%, identity 97.19%
```

So the first solver-quality bar is objective quality: match `exact`/`gurobi`
energy on this synthetic case. Biological reconstruction quality is a separate
secondary check because multiple decoded paths can have the same QUBO energy.
Run `candidate_stats.sh` serially, not in parallel in one directory, because it
uses fixed scratch files.

## Where To Work Next

For QUBO solver development:

```text
qubo/qubo_solvers/oriented_tangle/utils/sampling_utils.py
qubo/qubo_solvers/definitions.py
qubo/qubo_solvers/oriented_tangle/oriented_max_path.py
```

Use `local_sample_qubo` as the first custom-solver scaffold.

For synthetic benchmark tuning:

```text
examples/config_synthetic_tiny.sh
run_sim_create_gfa.sh
run_gfa_sim.sh
```

Increase graph difficulty by changing `TRAIN_COUNT`, `genome_opts -l`, and
mutation rates in `examples/config_synthetic_tiny.sh`. The next milestone is a
small generated `pop.gfa` with more than one `S` node while still small enough
for the `exact` solver to provide a reference energy.

More detailed solver-pipeline notes are in:

```text
PIPELINE.md
```
