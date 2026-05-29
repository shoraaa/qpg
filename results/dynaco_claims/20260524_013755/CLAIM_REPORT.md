# DyNACO Claim Experiments

This run is scoped to DyNACO-first evidence for two possible claims:

- Budgeted inference: full assembly quality as the neural_aco budget changes on minigraph-annotated graphs.
- Runtime/scale: QUBO-stage runtime and energy as graph size grows, compared with configured baseline solvers.

- Executed commands: `True`
- Model: `/home/shora/Research/qpg/results/overnight_dynaco_paper/20260522_013319/dynaco_overnight_best.pt`
- Annotators: `mg`
- Budgets: `5,10,30` seconds
- MQLib cache: `/home/shora/Research/qpg/results/overnight_dynaco_paper/20260522_013319/analytics/partial_full_assembly_best_consensus.csv`

## Cached MQLib Target

| graph | seqs | seeds | covered | used | contigs | breaks | identity |
|---|---:|---|---:|---:|---:|---:|---:|
| mg | 44 | 00001-00009 | 56.23 | 94.87 | 3.91 | 1.43 | 99.17 |

## Full-Budget Summary

    solver,graph,budget_s,seqs,seeds,covered,used,contigs,breaks,indels,diffs,identity
    neural_aco,mg,5,40,00001-00008,58.8815,93.2065,4.85,1.4,0.4,0.0,99.376
    neural_aco,mg,10,40,00001-00008,62.673,93.44275,4.55,1.025,0.25,0.025,99.149
    neural_aco,mg,30,40,00001-00008,59.3895,91.73225,4.55,2.325,0.275,0.025,99.265

