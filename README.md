QPG DyNACO MG Replication Package
=================================

This repository contains the QUBO pangenome graph code and the DyNACO
experiments used for the current MG-focused paper draft. The tracked
replication package is intentionally compact: it includes the pretrained
DyNACO checkpoint and the summary CSV files needed to reproduce the reported
tables, while ignoring bulky generated graphs, alignments, and solver
intermediates.

The paper-facing experiments focus on minigraph (`mg`) annotation. MQLib is the
main paper-method baseline; ACO is kept as an internal carrier/ablation rather
than as the external contribution target.

Tracked paper artifacts
-----------------------

The key tracked artifacts are:

- `results/overnight_dynaco_paper/20260522_013319/dynaco_overnight_best.pt`:
  pretrained DyNACO checkpoint used by the benchmark scripts.
- `results/dynaco_claims/20260524_013526/qubo_scale/scale_summary.csv`:
  QUBO-stage 80-segment MG slice.
- `results/dynaco_claims/20260524_125230/qubo_scale/scale_summary.csv`:
  QUBO-stage 100-segment MG slice.
- `results/dynaco_claims/20260524_013755/full_budget/best_consensus_summary.csv`:
  main DyNACO full-assembly budget run.
- `results/dynaco_claims/20260524_125236/full_budget/best_consensus_summary.csv`:
  DyNACO 10-second full-assembly repeat.
- `results/paper_reproduction/matched_full_assembly_summary.csv`:
  matched MG full-assembly table rows, including the MQLib row restricted to
  the same 40 sequence/seed pairs as the DyNACO runs.

To inspect the tracked table sources:

```bash
column -s, -t results/paper_reproduction/matched_full_assembly_summary.csv
column -s, -t results/dynaco_claims/20260524_013526/qubo_scale/scale_summary.csv
column -s, -t results/dynaco_claims/20260524_125230/qubo_scale/scale_summary.csv
```

Environment
-----------

The current local environment was built with `uv`, Python 3.14, and ROCm
PyTorch. The `pyproject.toml` and `uv.lock` are tracked, but GPU users should
install the PyTorch build that matches their machine if they are not using the
same ROCm stack.

```bash
uv sync
uv pip install numpy networkx pybind11 pyyaml
```

Build the C++ ACO extension locally after installing `pybind11`:

```bash
c++ -O3 -shared -std=c++17 -fPIC -fopenmp $(python -m pybind11 --includes) \
  qubo/qubo_solvers/oriented_tangle/cpp/qpg_aco_cpp.cpp \
  -o qubo/qubo_solvers/oriented_tangle/qpg_aco_cpp$(python-config --extension-suffix)
```

Full assembly experiments also require external command-line tools on `PATH`:
`minigraph`, `GraphAligner`, `minimap2`, `samtools`, `bwa`, GNU `parallel`,
`pathfinder`, and `MQLib`. Local installs under `.tools/` are supported by the
runner but are ignored by git.

Re-running the MG claims
------------------------

The compact CSV files above reproduce the current paper tables. Re-running the
raw experiments requires the external toolchain and generated GFA cache. The
claim runner looks for GFAs under the cached overnight directories, generated
online directories, and `examples/*.gfa`; if a fresh clone does not contain a
matching generated cache, first regenerate the MG pipeline data with the
simulation scripts below or run the full pipeline to create new GFAs.

QUBO-stage 80-segment MG run:

```bash
./scripts/run_dynaco_claim_experiments.py \
  --suite qubo-scale \
  --budgets 1,3,5,10 \
  --max-gfas 24 \
  --min-segments 30 \
  --max-segments 80 \
  --qubo-jobs 1 \
  --n_ants 8 \
  --H 3 \
  --mini_H 5 \
  --aco-min-iterations 15 \
  --qubo-baselines aco,beam_search,local \
  --run-mqlib \
  --device cuda \
  --execute
```

QUBO-stage 100-segment MG run:

```bash
./scripts/run_dynaco_claim_experiments.py \
  --suite qubo-scale \
  --budgets 1,3,5,10 \
  --max-gfas 12 \
  --min-segments 80 \
  --max-segments 100 \
  --qubo-jobs 1 \
  --n_ants 8 \
  --H 3 \
  --mini_H 5 \
  --aco-min-iterations 15 \
  --qubo-baselines aco,beam_search,local \
  --run-mqlib \
  --device cuda \
  --execute
```

DyNACO full-assembly MG budget run:

```bash
./scripts/run_dynaco_claim_experiments.py \
  --suite full-budget \
  --annotators mg \
  --budgets 5,10,30 \
  --seeds 8 \
  --test-sequences 5 \
  --full-jobs 1 \
  --n_ants 8 \
  --H 3 \
  --mini_H 5 \
  --aco-min-iterations 15 \
  --device cuda \
  --execute
```

The 10-second repeat uses the same command with `--budgets 10`.

MQLib comparison row
--------------------

The matched MQLib row in
`results/paper_reproduction/matched_full_assembly_summary.csv` is computed from
the cached original-paper-method run under
`results/overnight_dynaco_paper/20260522_013319/full_assembly/`, restricted to
MG seeds `00001` through `00008` and five test sequences per seed. This avoids
mixing the 40-sequence DyNACO runs with the older partial ninth seed in the raw
overnight cache.

Repository hygiene
------------------

The `.gitignore` keeps local environments, external tools, compiled binaries,
LaTeX auxiliary files, synthetic scratch data, and most of `results/` out of
the published repository. Only the compact paper artifacts and the pretrained
DyNACO checkpoint are whitelisted.

Instructions
============

This is very much a work in progress and a personal exploration into
how pangenome alignment could work.  It is not to be considered a
robust and production ready suite of tools.

The main entry to running the simulations is the run_gfa_sim.sh
script.  The usage is

    Usage: run_gfa_sim.sh [options] [seed solver [out_prefix]]
    Options:
        -c,--config    FILE    Use FILE as configuration
        -s,--seed      INT     Specify random number seed [1]
        -p,--prefix    STR     Use STR as the output dir prefix [sim_]
           --solver    STR     Specify the solver [pathfinder]
        -a,--annotate  STR     GFA node weight algorithm [km]
           --shred_len INT     Shotgun read length
           --shred_err FLOAT   Shotgun read error rate (fraction)
        -t,--times     INT_LIST   Time limits provided to QUBO solvers
        -j,--jobs      INT     Number of runs of QUBO solvers
        -n,--training  INT     Number of strings to use as training set [10]
           --edge2node         Use edge2node version
           --trim-edges        Use trim_edges.pl
           --pathfinder        Use pathfinder to get subgraphs

Solver can be mqlib, gurobi, dwave, pathfinder, exact, local,
greedy_residual, random_residual_walk, beam_search, aco, neural_aco, astar,
or seea.

It uses a configuration file to control parameters such as the graph
complexity, but also which algorithms to use.  Use the CONFIG environment
variable or run_gfa_sim -c FILE to point to one of these configurations.
Premade configuration files are:

    config_base.sh                      Included by all other config files
    config_hifi.sh                      2kbp  long reads at 0.001 error
    config_illumina.sh                  200bp long reads at 0.001 error    
    config_hifi_{km,mg,ga,vg}.sh        Hifi using a specific --annotate option
    config_illumina_{km,mg,ga,vg}.sh    Ilumina using a specific --annotate opt

We recommend using Pathfinder to partition the graph and assign copy numbers based on the annotated graph. This can be enabled with the --pathfinder option.

For example:

    run_gfa_sim.sh -c config_hifi_mg.sh -s 1 --solver pathfinder -p pf_hifi_mg_ --pathfinder


Multiple runs can be launch via xargs:

    ( p=illumina; m=ga; seq 100 150 | xargs -I % -P 4 ./run_gfa_sim.sh -c config_${p}_$m.sh --pathfinder --solver mqlib --trim-edges -t 30 -j 1 -n 5 -p ~/lustre/tmp/mqlib-tmq10_${p}_${m}_ % )

Summaries from multiple runs with different solvers and annotations can be
produced with:

    ( for x in pf2-base sa-base ma-base mqlib-base mqlib-trim1.3;do for m in sa ma km mg ga vg;do p=illumina; if [ ! -e ~/lustre/tmp/${x}_${p}_${m}_00100 ];then continue;fi; printf "%-13s %3s " $x $m; awk '!/contig/ {for (i=2;i<11;i++) {a[i]+=$i}n++} END {for (i=2; i<11; i++) {printf("%7.1f ", a[i]/n)} print n}' ~/lustre/tmp/${x}_${p}_${m}_001*/*eval_cons* 2>/dev/null || echo;done;echo;done)


Other selected programs
=======================

run_syncasm_sim.sh
------------------

De novo assembly via SyncAsm.

run_miniasm_sim.sh
------------------

De novo assembly via Miniasm.


run_sim_create_gfa.sh
---------------------

Creates a synthetic population (via genome_create) and trains a GFA on
a subset of it (fofn.test).  Also produces a fofn.test list of sequences to
evaluate.

run_sim_add_gfa_weights_${annotate}.sh
--------------------------------------

A series of scripts to annotate the node weights in the population GFA by
aligning the fofn.test sequences.

Selected via the run_gfa_sim --annotate option.
This produces a series of seq*.gfa files with the annotated test sequence
graphs.

run_sim_solver_${solver}.sh
---------------------------

A series of scripts to solve the path.


gaf2nodeseq.pl
--------------

Merges a GFA graph, a GAF alignment file and the input FASTA used in
generating the GAF to identify the parts of the sequences that were aligned
against each GFA file (using the path and CIGAR fields).

These sequences are then written to a "nodeseq" file, which constitutes an
@node name and a series of sequences.  The first sequence is from the graph
while all subsequent sequences are from the input fasta (with KMER-1 prior
bases of context).

This is called by the run_sim_create_gfa.sh script if the solver is kmer2node.


kmer2node4
----------

A program which builds a kmer index from a nodeseq file and then compares
kmers in a set of query sequences to identify the depth of coverage for all
nodes in the graph.  This forms a rudimentary alignment.

The program produces a lot of debugging output, but piping the output to "grep
Node" will return the final answers.

See also merge_kmer2node.pl to allow running kmer2node4 with multiple kmers
and merge the results to form a single set of node weights.


pathfinder2seq.pl
-----------------

With an input GFA and pathfinder PATH information this creates a sequence by
concatenating (and complementing) nodes together to produce a candidate
assembly sequence.


run_sim_evaluate_path.sh
------------------------

Compares a true sequence to a candidate assembly sequence and reports how well
they match.  It compares A to B and B to A to get symmetric data on coverage
(how much of A is in B and vice versa).


QUBO
============

Code used to map annotated graphs into a QUBO problem and for sampling solutions from said QUBO problem are provided in `qubo/`.

`build_oriented_qubo_matrix.py` takes 4 arguments:
    -f : the path to a .gfa file
    -c : a list of copy numbers for the graph (comma separated)
    -p : a list of 3 Lagrange multipliers for the 1-node-per-time, graph steps and node weight constraints respectively (comma separated)
    -d : a directory to write the results to

`oriented_max_path.py` takes arguments:
    -f : the path to a .gfa file
    -t : a list of time limits to provide to the solver
    -j : the number of jobs to run per time limit
    -s : the solver [mqlib, gurobi, dwave]
    -d : the directory to read QUBO input data from
    -o : a directory in which to write the located paths

To use MQLib, the binary must be installed and added to the path. Instructions for installing MQLib can be found at [https://github.com/MQLib/MQLib](https://github.com/MQLib/MQLib).

To use Gurobi, a license must be obtained. Free academic licenses are available. Instructions are available at [https://www.gurobi.com/solutions/licensing/](https://www.gurobi.com/solutions/licensing/).

To use D-Wave, an API key must be obtained; free trials are available. Instructions can be found at [https://www.dwavequantum.com/quantum-launchpad/](https://www.dwavequantum.com/quantum-launchpad/).
