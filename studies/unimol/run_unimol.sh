#!/bin/bash
# A queue of Uni-Mol fine-tuning jobs for one machine, each "arm:dataset:repeat".
#
# Every repeat is a resume point -- 01_run_unimol.py skips one whose 25
# prediction files are on disk -- so the sweep is split between apollo and
# spark by handing each its own queue, and rebalanced by restarting with a
# different one. The tqdm bars are dropped from the log.
#
#     nohup ./run_unimol.sh unimol:biogen:1 unimol2:expansion:4 >> logs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
PY=${UNIMOL_PYTHON:-~/miniforge3/envs/unimol/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for job in "$@"; do
  IFS=: read -r arm ds repeat <<< "$job"
  echo "=== $job start $(date "+%F %T")"
  ADME_DATASET=$ds $PY 01_run_unimol.py --arm "$arm" --repeat "$repeat" 2>&1 \
    | tr "\r" "\n" | grep --line-buffered -vE "it/s|s/it|^\s*$" > "logs/${arm}_${ds}_r${repeat}.log"
  echo "=== $job exit ${PIPESTATUS[0]} $(date "+%F %T")"
done
echo "=== queue done $(date "+%F %T")"
