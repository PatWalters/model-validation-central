#!/bin/bash
# The SCM sweep for one data set. Every series-fold is a resume point, and the
# runner picks up encoders as 03_train_encoder.py finishes them.
#
#     nohup ./run_causal.sh expansion 22 0 >> logs/causal_expansion.log 2>&1 &   # repeat 0
#     nohup ./run_causal.sh biogen 4 >> logs/causal_biogen.log 2>&1 &            # all repeats
set -u
cd "$(dirname "$0")" || exit 1
ds=$1; workers=$2; shift 2
repeats=${*:-0 1 2 3 4}
ADME_DATASET=$ds exec ${CAUSALCP_PYTHON:-~/miniforge3/envs/causalcp/bin/python} \
  04_run_causal.py --workers "$workers" --repeat $repeats
