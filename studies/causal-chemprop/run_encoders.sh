#!/bin/bash
# The encoder sweep, three processes sharing the GPU: repeat 0 first, so the
# SCM sweep (04_run_causal.py) can start on it while the rest train.
#
#     nohup ./run_encoders.sh >> logs/encoders.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
PY=${CAUSALCP_PYTHON:-~/miniforge3/envs/causalcp/bin/python}
export OMP_NUM_THREADS=2
lane() {  # one GPU lane: a data set and its endpoints, repeat 0 then the rest
  local ds=$1; shift
  ADME_DATASET=$ds $PY 03_train_encoder.py --repeat 0 "$@" && \
  ADME_DATASET=$ds $PY 03_train_encoder.py "$@"
}
lane expansion --endpoint LogD LogS LOG_HLM LOG_MLM > logs/encoder_exp_a.log 2>&1 &
lane expansion --endpoint LOG_Caco_AB LOG_Caco_Efflux LOG_MPPB LOG_MBPB > logs/encoder_exp_b.log 2>&1 &
lane biogen > logs/encoder_biogen.log 2>&1 &
wait
echo "=== encoders done $(date "+%F %T")"
