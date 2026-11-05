#!/bin/bash
# Atom-JEPA-Bio, both data sets, on one CUDA machine (mercury, an RTX PRO 5000
# Blackwell with 48 GB). Every fold is a resume point, so this can be stopped
# and restarted.
#
#   conformers (01) and embeddings (02)  `atomjepa` env, both encoders
#   frozen arms (03)                      `monroe` env, tabpfn 8.3 -> TabPFN 3,
#                                         in the background, sharing the GPU
#   fine-tuning (04)                      `atomjepa` env: bio_ft on Biogen, the
#                                         five-fold stage-1 re-run check on
#                                         Biogen repeat 0, then bio_ft on ExpansionRx
#
# The env's bin goes first on PATH because torch.compile shells out to tools
# that it expects to find there.
#
#     nohup ./run_atomjepa_bio.sh >> logs/atomjepa_bio.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
AJ_ENV=${ATOMJEPA_ENV:-~/anaconda3/envs/atomjepa}
export PATH=$AJ_ENV/bin:$PATH WANDB_MODE=disabled
# TabPFN 3 weights are read from ~/.cache/tabpfn (copied from apollo, md5 8ff62694...), no token needed
AJ=$AJ_ENV/bin/python
TP=${TABPFN_PYTHON:-~/anaconda3/envs/monroe/bin/python}
mkdir -p logs

for ds in biogen expansion; do
  echo "=== conformers + embeddings $ds $(date "+%F %T")"
  ADME_DATASET=$ds $AJ 01_conformers.py || exit 1
  ADME_DATASET=$ds $AJ 02_embed.py || exit 1
done

( for ds in biogen expansion; do
    echo "=== tabpfn $ds $(date "+%F %T")"
    ADME_DATASET=$ds $TP 03_run_tabpfn.py || exit 1
  done
  echo "=== tabpfn done $(date "+%F %T")" ) >> logs/tabpfn.log 2>&1 &

echo "=== fine-tune bio biogen $(date "+%F %T")"
ADME_DATASET=biogen $AJ 04_run_finetune.py --method bio_ft || exit 1
echo "=== re-run check biogen r0 $(date "+%F %T")"
ADME_DATASET=biogen $AJ 04_run_finetune.py --method atomjepa_ft_rerun --repeat 0 || exit 1
echo "=== fine-tune bio expansion $(date "+%F %T")"
ADME_DATASET=expansion $AJ 04_run_finetune.py --method bio_ft || exit 1
wait
echo "=== atomjepa-bio done $(date "+%F %T")"
