#!/bin/bash
# Fine-tuned Atom-JEPA on the paper tasks, in the atomjepa environment.
#     nohup ./run_paper_atomjepa.sh >> logs/paper_atomjepa.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
AJ_ENV=${ATOMJEPA_ENV:-~/miniforge3/envs/atomjepa}
export PATH=$AJ_ENV/bin:$PATH WANDB_MODE=disabled
exec $AJ_ENV/bin/python 07_paper_atomjepa.py "$@"
