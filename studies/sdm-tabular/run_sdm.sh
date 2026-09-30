#!/bin/bash
# Every structured-data-models head over the Monroe embeddings, both data sets.
#
# Runs in the `sdm` environment (see the README). Each fold is a resume point,
# so this can be stopped and restarted. TabFM runs only when its licence has
# been accepted, by setting ACCEPT_TABFM_LICENSE=1.
#
#     nohup ./run_sdm.sh >> logs/sdm.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
P=${SDM_PYTHON:-~/miniforge3/envs/sdm/bin/python}
# Without this, fragmentation runs TabFM out of memory on the largest folds even
# when 01_run_sdm.py falls back to chunking the test rows. It changes where
# memory is mapped, not what is computed.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS="tabiclv2 kumo_small kumo_medium kumo_large"
EXTRA=""
if [ "${ACCEPT_TABFM_LICENSE:-0}" = "1" ]; then
  MODELS="$MODELS tabfm"
  EXTRA="--accept-tabfm-license"
fi

for model in $MODELS; do
  for ds in biogen expansion; do
    echo "=== $model $ds $(date "+%H:%M:%S")"
    ADME_DATASET=$ds $P 01_run_sdm.py --model "$model" $EXTRA || exit 1
  done
done
echo "=== sdm done $(date "+%H:%M:%S")"
