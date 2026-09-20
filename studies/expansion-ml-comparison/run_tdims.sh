#!/bin/bash
# The TDiMS arm, both data sets, all three phases.
#
# Everything runs in the monroe35 environment: that is the one where the
# installed tabpfn loads TabPFN 3.5, which is the head this arm shares with the
# `monroe35` arm, and it also has the RDKit the descriptor needs. The descriptor
# itself comes from the authors' release at TDIMS_HOME.
#
#     nohup ./run_tdims.sh >> logs/tdims.log 2>&1 &
set -u
cd ~/EXPANSION_ML_COMPARISON || exit 1
export TDIMS_HOME=~/software/tdims
export MONROE_HOME=~/software/monroe
export TABPFN_TOKEN=$(cat ~/.cache/tabpfn/auth_token | tr -d "\n")
# Feature selection is the cost of this arm, it is memory-bandwidth bound per
# fold and does not go faster in parallel, so instead it runs one fold ahead of
# the GPU. Keep BLAS off the remaining cores so it does not fight the selector.
export TDIMS_THREADS=${TDIMS_THREADS:-2}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P=~/miniforge3/envs/monroe35/bin/python

for ds in biogen expansion; do
  echo "=== tdims features $ds $(date "+%H:%M:%S")"
  ADME_DATASET=$ds $P 15_run_tdims.py --features || exit 1
done

for ds in biogen expansion; do
  echo "=== tdims config selection $ds $(date "+%H:%M:%S")"
  ADME_DATASET=$ds $P 15_run_tdims.py --select || exit 1
done

for ds in biogen expansion; do
  echo "=== tdims folds $ds $(date "+%H:%M:%S")"
  ADME_DATASET=$ds $P 15_run_tdims.py || exit 1
done
echo "=== tdims done $(date "+%H:%M:%S")"
