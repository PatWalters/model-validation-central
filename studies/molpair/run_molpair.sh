#!/bin/bash
# MolPAIR, both data sets, featurisation then every fold.
#
# Featurisation runs in the `chemprop` environment, which has chemprop 2.3 for
# the CheMeleon fingerprint. The folds run in `monroe`, where the installed
# tabpfn (8.x) loads TabPFN 3, the backbone the paper uses. Each fold is a
# resume point, so this can be stopped and restarted.
#
#     nohup ./run_molpair.sh >> logs/molpair.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
export TABPFN_TOKEN=$(cat ~/.cache/tabpfn/auth_token | tr -d "\n")
FEAT=${FEATURIZE_PYTHON:-~/miniforge3/envs/chemprop/bin/python}
P=${MOLPAIR_PYTHON:-~/miniforge3/envs/monroe/bin/python}

for ds in biogen expansion; do
  if [ ! -f data/$ds/molpair_features.npz ]; then
    echo "=== featurize $ds $(date "+%H:%M:%S")"
    ADME_DATASET=$ds $FEAT 01_featurize.py || exit 1
  fi
done

for ds in biogen expansion; do
  echo "=== molpair $ds $(date "+%H:%M:%S")"
  ADME_DATASET=$ds $P 02_run_molpair.py || exit 1
done
echo "=== molpair done $(date "+%H:%M:%S")"
