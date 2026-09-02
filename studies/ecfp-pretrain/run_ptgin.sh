#!/bin/bash
# The whole PT-GIN sweep on apollo: embed, select, predict, for both data sets.
#
#     nohup ./run_ptgin.sh > logs/ptgin.log 2>&1 &
#
# Embedding wants the GPU and takes about two minutes for all ten checkpoints.
# Everything after it is LightGBM on the CPU, so --jobs is the number of cores to
# spend. Each phase skips work that is already done, so this can be re-run.
set -u
cd "$(dirname "$0")"

export PTGIN_HOME=${PTGIN_HOME:-$HOME/software/topological-pretraining}
JOBS=${JOBS:-24}

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ptgin

for ds in expansion biogen; do
    echo "=== $ds embed $(date) ==="
    ADME_DATASET=$ds python 01_run_ptgin.py --embed --workers "$JOBS" || exit 1

    echo "=== $ds select $(date) ==="
    ADME_DATASET=$ds python 01_run_ptgin.py --select --jobs "$JOBS" || exit 1

    echo "=== $ds predict $(date) ==="
    ADME_DATASET=$ds python 01_run_ptgin.py --jobs "$JOBS" || exit 1
done

echo "=== ptgin done $(date) ==="
