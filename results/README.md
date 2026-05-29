# Result Artifacts

This directory is ignored by default because full benchmark runs produce large
intermediate graph, alignment, and solver files. The repository keeps only the
compact artifacts needed to reproduce the tables in `method.tex`:

- `paper_reproduction/`: paper-facing summary tables with matched settings.
- `overnight_dynaco_paper/20260522_013319/dynaco_overnight_best.pt`: pretrained
  DyNACO checkpoint used by the replication commands.
- Selected `dynaco_claims/*` reports and summary CSV files for the MG QUBO and
  full-assembly experiments.

Raw run directories can be regenerated with `scripts/run_dynaco_claim_experiments.py`
after installing the external graph and assembly tools listed in the repository
README.
