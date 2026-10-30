#!/bin/bash
# Merge the SCM sweep's per-series parts and claims between two machines.
#
# apollo and spark cannot reach each other directly, so a third machine that
# can reach both relays: pull both sides into a staging directory, push the
# union back to both. Parts and claims are written once and never edited, so a
# plain additive rsync is enough; nothing is ever deleted.
#
#     ./sync_parts.sh expansion /tmp/stage        # loops every two minutes
set -u
ds=${1:-expansion}
stage=${2:-./scratch/sync_$ds}
hosts=${SYNC_HOSTS:-"apollo spark"}
remote="CAUSAL_CHEMPROP/work/causal_parts/$ds/"
mkdir -p "$stage"
while true; do
  for h in $hosts; do
    rsync -a --ignore-existing --exclude="*.tmp" "$h:$remote" "$stage/" 2>/dev/null
  done
  for h in $hosts; do
    rsync -a --ignore-existing "$stage/" "$h:$remote" 2>/dev/null
  done
  echo "$(date '+%F %T') $(ls "$stage" | grep -c '\.csv$') parts"
  sleep "${SYNC_EVERY:-120}"
done
