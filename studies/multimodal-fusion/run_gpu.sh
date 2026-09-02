#!/bin/bash
# The GPU half of the sweep, both data sets in sequence. Run on apollo.
#
#   nohup ./run_gpu.sh > logs/gpu.log 2>&1 &
#
# Phase 1 encodes every fold: one AttentiveFP and one BiGRU per endpoint and
# fold, whose cached blocks all thirty-three configurations then read. Nothing
# else in the pipeline can start until an endpoint's repeat 0 fold 0 exists, so
# this runs first and alone.
#
# Phase 2 fits the eight graph meta-learner configurations, which is the long
# one: an AttentiveFP per configuration per fold, 1,800 networks on ExpansionRx
# and 1,200 on Biogen. It waits for the tuned base learners the late-fusion
# configurations need, which run_cpu.sh writes.
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-~/miniforge3/envs/mmfusion/bin/python}
mkdir -p logs

echo "=== gpu sweep started $(date) ==="

for ds in expansion biogen; do
    echo "--- encoding $ds $(date) ---"
    ADME_DATASET=$ds $PY 03_encode_folds.py --gpu 0 || exit 1
done
touch results/.encoded
echo "=== encoding done $(date) ==="

# The graph meta-learners need `uni_R_lgbm` and `uni_M_lgbm` tuned, because late
# fusion stacks those base learners' predictions. Wait for the CPU side rather
# than racing it.
for ds in expansion biogen; do
    while [ ! -f "results/$ds/.tuned" ]; do
        if ! pgrep -f "[r]un_cpu.sh" > /dev/null; then
            echo "ABORT: run_cpu.sh is not running and $ds is not tuned"
            exit 1
        fi
        sleep 60
    done
done

for ds in expansion biogen; do
    echo "--- graph meta-learners, $ds $(date) ---"
    ADME_DATASET=$ds $PY 05_run_grid.py --learner attfp --gpu 0 || exit 1
done

touch results/.gpu_done
echo "=== gpu sweep done $(date) ==="
