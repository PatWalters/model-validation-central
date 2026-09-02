#!/bin/bash
# The graph meta-learner phase, sharded across several GPU processes.
#
#     nohup ./run_attfp.sh 3 > logs/attfp.log 2>&1 &
#
# This is the long pole: eight configurations per fold, each training an
# AttentiveFP, so 1,800 networks on ExpansionRx and 1,200 on Biogen. The models
# are small enough that one process leaves the card about half idle -- it is
# launch-latency bound, not compute bound -- so the endpoints are dealt out
# round-robin across a few processes instead.
#
# Sharding is by endpoint, not by fold, so two processes never contend for the
# same output file. Each is resumable on its own, and a fold whose prediction
# file exists is skipped, so this can be stopped and restarted at any point.
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-~/miniforge3/envs/mmfusion/bin/python}
JOBS=${1:-3}
mkdir -p logs

echo "=== attfp sweep started $(date), $JOBS processes ==="

for ds in expansion biogen; do
    # The graph meta-learners are not themselves tuned. What they wait on is the
    # two unimodal LightGBM base learners per endpoint, which late fusion stacks
    # -- not the full search, which takes hours longer and would leave the card
    # idle for all of them. 04_tune.py does those first for every endpoint.
    echo "--- waiting for $ds base learners $(date) ---"
    until ADME_DATASET=$ds $PY - <<'EOF'
import json, sys
import config as cfg
try:
    store = json.loads(cfg.HPARAM_JSON.read_text())
except (OSError, ValueError):
    sys.exit(1)
need = [cfg.unimodal_method(m, "lgbm") for m in ("rdkit", "mol2vec")]
missing = [f"{e}/{k}" for e in cfg.TARGET_COLS for k in need
           if k not in store.get(e, {})]
if missing:
    print(f"  still missing {len(missing)}: {', '.join(missing[:4])}", flush=True)
    sys.exit(1)
EOF
    do
        if ! pgrep -f "[r]un_cpu.sh" > /dev/null; then
            echo "ABORT: run_cpu.sh is not running and $ds base learners are missing"
            exit 1
        fi
        sleep 60
    done
    echo "--- $ds base learners ready $(date) ---"

    # The endpoint list comes from config so this never drifts from the data set.
    mapfile -t ENDPOINTS < <(ADME_DATASET=$ds $PY -c \
        'import config as cfg; print("\n".join(cfg.TARGET_COLS))')

    echo "--- graph meta-learners, $ds: ${#ENDPOINTS[@]} endpoints $(date) ---"
    pids=()
    for ((shard = 0; shard < JOBS; shard++)); do
        mine=()
        for ((i = shard; i < ${#ENDPOINTS[@]}; i += JOBS)); do
            mine+=("${ENDPOINTS[$i]}")
        done
        [ ${#mine[@]} -eq 0 ] && continue
        echo "  shard $shard: ${mine[*]}"
        ADME_DATASET=$ds $PY 05_run_grid.py --learner attfp --gpu 0 \
            --endpoint "${mine[@]}" > "logs/attfp_${ds}_$shard.log" 2>&1 &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
        echo "ABORT: a $ds shard exited non-zero; see logs/attfp_${ds}_*.log"
        exit 1
    fi
    echo "--- $ds done $(date) ---"
done

touch results/.gpu_done
echo "=== attfp sweep done $(date) ==="
