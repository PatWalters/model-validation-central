#!/bin/bash
# The second machine: apollo (RTX 5070 Ti, 16 GB) fine-tunes Bio on ExpansionRx
# from the last fold backwards while mercury works forwards; folds the other
# machine has finished or is running are listed in folds/expansion/claimed_folds.txt
# and skipped. Uses a clone of apollo's `atomjepa` env brought to mercury's
# versions (torch 2.14.1+cu130, e3nn 0.6.0, cuEquivariance 0.12.0) and a
# checkout of the authors' code at 64d539b; the env that ran the atom-jepa
# study has since been downgraded by another project and is left alone.
#
#     nohup setsid ./run_apollo.sh >> logs/apollo.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")" || exit 1
AJ_ENV=${ATOMJEPA_ENV:-~/miniforge3/envs/atomjepa_bio}
export PATH=$AJ_ENV/bin:$PATH WANDB_MODE=disabled
export ATOMJEPA_HOME=${ATOMJEPA_HOME:-~/software/atom-jepa-bio}
AJ=$AJ_ENV/bin/python
mkdir -p logs
ADME_DATASET=expansion $AJ 01_conformers.py || exit 1
echo "=== fine-tune bio expansion, reverse $(date "+%F %T")"
ADME_DATASET=expansion $AJ 04_run_finetune.py --method bio_ft --reverse || exit 1
echo "=== apollo done $(date "+%F %T")"
