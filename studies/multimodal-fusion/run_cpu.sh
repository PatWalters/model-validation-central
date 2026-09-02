#!/bin/bash
# The CPU half of the sweep, both data sets in sequence. Run on apollo alongside
# run_gpu.sh -- these two never contend, one is bound by the card and the other by
# the 32 cores.
#
#   nohup ./run_cpu.sh > logs/cpu.log 2>&1 &
#
# Phase 1 is the hyperparameter search: 60 sampled settings scored over a
# cluster-grouped 3-fold split, once per endpoint per tuned configuration, reused
# by all 25 folds. It needs each endpoint's repeat 0 fold 0 encoder blocks, which
# run_gpu.sh writes first.
#
# Phase 2 fits the twenty-five tabular configurations across every fold, then the
# eight leak-free late-fusion controls.
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-~/miniforge3/envs/mmfusion/bin/python}
JOBS=24
mkdir -p logs

# OpenBLAS reads its thread count when numpy first imports, so setting these
# inside a worker is too late: every one of 24 processes would quietly run a
# 32-thread BLAS underneath the pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== cpu sweep started $(date) ==="

# The searches need the tuning fold's encoder blocks. Wait for the whole encoding
# phase rather than polling per endpoint: it is under three hours and racing it
# would only complicate the resume logic.
while [ ! -f results/.encoded ]; do
    if ! pgrep -f "[r]un_gpu.sh" > /dev/null; then
        echo "ABORT: run_gpu.sh is not running and encoding never finished"
        exit 1
    fi
    sleep 60
done

for ds in expansion biogen; do
    echo "--- tuning $ds $(date) ---"
    ADME_DATASET=$ds $PY 04_tune.py --jobs 32 || exit 1
    touch "results/$ds/.tuned"
done
echo "=== tuning done $(date) ==="

for ds in expansion biogen; do
    echo "--- tabular grid, $ds $(date) ---"
    ADME_DATASET=$ds $PY 05_run_grid.py --learner lgbm rf --jobs $JOBS || exit 1

    echo "--- leak-free late-fusion control, $ds $(date) ---"
    ADME_DATASET=$ds $PY 05_run_grid.py --control --jobs $JOBS || exit 1

    echo "--- released GNN block control, $ds $(date) ---"
    ADME_DATASET=$ds $PY 05_run_grid.py --paper-gnn-block --learner lgbm \
        --jobs $JOBS || exit 1

    echo "--- modality ablation and grouped SHAP, $ds $(date) ---"
    ADME_DATASET=$ds $PY 08_modality_contribution.py --jobs $JOBS || exit 1
done

touch results/.cpu_done
echo "=== cpu sweep done $(date) ==="
