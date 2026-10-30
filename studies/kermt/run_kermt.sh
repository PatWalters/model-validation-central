#!/bin/bash
# A queue of KERMT fine-tuning jobs for one machine, each "arm:dataset" or
# "arm:dataset:r,r,..." for some of its repeats only.
#
# Every fold is a resume point -- 02_run_kermt.py skips one whose prediction
# files are on disk -- so the sweep is split between apollo and spark by
# handing each its own queue (spark runs two, which is ~20% more throughput
# on the GB10 and none on the RTX 5070 Ti), and rebalanced by restarting with
# a different one. Two machines on one arm and data set take disjoint
# repeats, so they never pick up the same fold.
#
#     nohup ./run_kermt.sh kermt_ft:expansion grover_large_ft:biogen >> logs/queue_a.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
PY=${KERMT_PYTHON:-~/miniforge3/envs/kermt/bin/python}
mkdir -p logs
for job in "$@"; do
  IFS=: read -r arm ds repeats <<< "$job"
  echo "=== $job start $(date "+%F %T")"
  # shellcheck disable=SC2086
  ADME_DATASET=$ds $PY 02_run_kermt.py --arm "$arm" ${repeats:+--repeat ${repeats//,/ }}
  echo "=== $job exit $? $(date "+%F %T")"
done
echo "=== queue done $(date "+%F %T")"
