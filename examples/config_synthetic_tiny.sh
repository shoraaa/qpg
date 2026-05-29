. "$QDIR/config_base.sh"

# Small synthetic haploid population for solver development smoke tests.
# This keeps the paper's generator/mechanism but makes minigraph fast enough
# for local validation.
genome_opts="-l 2000 -S 0.004 -C 0.0005 -N 0.01 -n 0.01 -A 0.0005 -L 0.0001 -T 0.0001 -I 0.0002 -P 60 -G 1"
shred_len=200
shred_err=0.001
shred_depth=30
annotate=mg
MINIGRAPH_BUILD_OPTS="-l 100 -d 1000 -n 2,5"
TRAIN_COUNT=3
