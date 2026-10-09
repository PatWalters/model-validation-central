#!/bin/bash
# Atom-JEPA, both data sets: embeddings, the frozen TabPFN arm, then fine-tuning.
#
# Conformers (01) and embeddings (02) run in `atomjepa`. The TabPFN arm (03)
# runs in `monroe`, where tabpfn 8.x loads TabPFN 3, the head `monroe` used.
# Fine-tuning (04) runs in `atomjepa` and is nearly all of the GPU time. Every
# fold is a resume point, so this can be stopped and restarted.
#
# The env's bin goes first on PATH because torch.compile shells out to
# `openssl`, and the one in /usr/local/bin on apollo does not load.
#
#     nohup ./run_atomjepa.sh >> logs/atomjepa.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
AJ_ENV=${ATOMJEPA_ENV:-~/miniforge3/envs/atomjepa}
export PATH=$AJ_ENV/bin:$PATH WANDB_MODE=disabled
export TABPFN_TOKEN=$(cat ~/.cache/tabpfn/auth_token | tr -d "\n")
AJ=$AJ_ENV/bin/python
TP=${TABPFN_PYTHON:-~/miniforge3/envs/monroe/bin/python}

for ds in biogen expansion; do
  echo "=== conformers + embeddings $ds $(date "+%F %T")"
  ADME_DATASET=$ds $AJ 01_conformers.py || exit 1
  [ -f data/$ds/atomjepa_embeddings.npz ] || ADME_DATASET=$ds $AJ 02_embed.py || exit 1
  echo "=== tabpfn $ds $(date "+%F %T")"
  ADME_DATASET=$ds $TP 03_run_tabpfn.py || exit 1
done

for ds in biogen expansion; do
  echo "=== fine-tune $ds $(date "+%F %T")"
  ADME_DATASET=$ds $AJ 04_run_finetune.py || exit 1
done
echo "=== atomjepa done $(date "+%F %T")"
