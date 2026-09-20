#!/usr/bin/env bash
# Step 3 as a work queue over N GPU workers.
#
# 300 CheMeleon fine-tunings is the long pole of this study: about 270 s apiece
# under four-way contention for the four schemes that train on ~950 molecules,
# and considerably less for the two that train on far fewer. One process takes
# some eight hours and leaves most of a 16 GB card idle -- a fold is 15 batches
# an epoch, so the GPU spends much of its time waiting for the next one. Four
# processes measure 1.6x the throughput of two, so the card is not saturated.
#
# Partitioning by scheme, which is the obvious thing, balances badly: the six
# schemes are not the same size, and a worker holding both of the cheap ones
# finishes hours before one holding two expensive ones. So the unit of work here
# is a (scheme, target) chunk -- 60 chunks of five folds each -- and workers pull
# from a shared queue instead of being assigned a share up front. Chunks are
# queued longest-first, so the cheap ones are left to fill the gaps at the end.
#
# The queue is the only thing the workers share, and `flock` makes a pop atomic,
# so no two workers can ever take the same chunk. That matters for more than
# efficiency: two processes training the same fold would write into the same
# scratch directory and corrupt each other.
#
# Two details that cost an hour the first time round, both worth keeping.
#
# The chunk count uses a glob under `nullglob`, not `ls`. With `set -e` in force,
# `ls` on a pattern matching nothing exits 2 and kills the script mid-loop --
# which truncated the queue to whichever chunks had been written before the first
# scheme with no folds on disk, and did it silently, because the line that
# reports the queue length comes afterwards.
#
# The worker loop is passed to bash inline rather than written to a file. An
# earlier version kept it in logs/_worker.sh; relaunching would rewrite that file
# underneath workers already executing it, and bash reads a script incrementally.
# Nothing a running worker depends on is written twice now.
#
# 03_run_chemprop.py skips any fold whose prediction file already exists, so a
# worker that dies costs one chunk, and rerunning this rebuilds the queue from
# what is actually on disk and picks up the rest.
#
#     ./run_chemprop.sh            # four workers, the default
#     ./run_chemprop.sh 5          # five, if there is memory for it
#
# Progress:  tail -f logs/chemprop_q*.log
#            ls predictions/chemeleon | wc -l

set -euo pipefail
shopt -s nullglob   # a scheme with no folds yet must expand to nothing, not to a
                    # literal pattern that `ls` then fails on -- see the queue below
cd "$(dirname "$0")"
mkdir -p logs predictions/chemeleon

WORKERS="${1:-4}"
QUEUE="$PWD/logs/work_queue.txt"
LOCK="$PWD/logs/work_queue.lock"

if pgrep -f "03_run_chemprop.py --split" > /dev/null; then
  echo "workers are already running -- stop them first:" >&2
  echo "  pkill -f 03_run_chemprop; pkill -f 'chemprop train'" >&2
  exit 1
fi

# Longest first: the four schemes that train on ~72% of a target, then the time
# split at ~50%, then the diverse split, which trains on a quarter of the data.
# The queue is rebuilt from disk every launch, so a chunk half-finished by a
# worker that died comes back with only its missing folds left to do.
: > "$QUEUE"
for s in random scaffold butina umap time diverse_25; do
  for t in 220 230 260 262 279 284 1865 2409 4005 4822; do
    have=(predictions/chemeleon/tid_${t}_${s}_f*.csv)
    [[ "${#have[@]}" -ge 5 ]] || echo "$s $t" >> "$QUEUE"
  done
done
touch "$LOCK"
echo "queued $(wc -l < "$QUEUE") chunks of up to five folds"

WORKER_LOOP='
set -uo pipefail
QUEUE="'"$QUEUE"'"; LOCK="'"$LOCK"'"
while true; do
  chunk=$(flock "$LOCK" bash -c "
    read -r line < \"$QUEUE\" || exit 1
    [ -n \"\$line\" ] || exit 1
    sed -i 1d \"$QUEUE\"
    printf %s \"\$line\"
  ") || break
  [ -n "$chunk" ] || break
  set -- $chunk
  echo "=== chunk: $1 tid_$2 ($(wc -l < "$QUEUE") left) ==="
  python -u 03_run_chemprop.py --split "$1" --target "$2" || echo "!!! chunk $1 $2 failed"
done
echo "queue empty, worker done"
'

for w in $(seq 1 "$WORKERS"); do
  nohup bash -c "$WORKER_LOOP" > "logs/chemprop_q${w}.log" 2>&1 &
  echo "worker $w started (pid $!, logs/chemprop_q${w}.log)"
done
