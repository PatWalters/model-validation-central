#!/bin/bash
# Hand the GPU over from the single-process driver to the sharded one.
#
#     nohup ./swap_to_shards.sh 3 > logs/swap.log 2>&1 &
#
# run_gpu.sh does encoding and then the graph meta-learners in one process. The
# encoding phase is the right shape for that -- it is one model at a time and the
# card is busy. The graph phase is not: the networks are small enough to be
# launch-latency bound, so one process leaves the card about half idle across
# 3,000 fits.
#
# This waits for encoding to finish, stops run_gpu.sh before it can start the
# second phase, and starts run_attfp.sh in its place. If it is never run,
# run_gpu.sh simply does the graph phase itself, single-process and slower, so
# the fallback is correct rather than broken.
set -u
cd "$(dirname "$0")" || exit 1
JOBS=${1:-3}

echo "=== swap watcher started $(date), will shard into $JOBS processes ==="

while [ ! -f results/.encoded ]; do
    if ! pgrep -f "[r]un_gpu.sh" > /dev/null; then
        echo "ABORT: run_gpu.sh stopped before encoding finished"
        exit 1
    fi
    sleep 60
done
echo "encoding finished $(date)"

# Stop the driver and any encode still winding down, then make sure the card is
# actually free before starting three processes on it.
pkill -f "[r]un_gpu.sh"
pkill -f "[0]3_encode_folds"
sleep 10
if pgrep -f "[0]5_run_grid.py" > /dev/null; then
    echo "ABORT: run_gpu.sh had already started the graph phase; leaving it alone"
    exit 1
fi

nohup ./run_attfp.sh "$JOBS" > logs/attfp.log 2>&1 &
echo "=== started run_attfp.sh with $JOBS shards $(date) ==="
