#!/usr/bin/env bash
# Run a list of llama.cpp configurations, one GPU lock per configuration (so other agents can
# run between them). Each line of the config file:
#   <kind: tg|srv> <tag> <model path> [extra args...]
# env LLAMA selects the llama.cpp build (bin dir); CTX is passed through for srv.
# A configuration whose wait_idle fails (GPU busy) is retried up to 3 times, then marked FAILED.
# Usage: sweep.sh <outdir> <config file>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$1"; CFG="$2"
mkdir -p "$OUT"
while read -r kind tag model extra; do
  [ -z "${kind:-}" ] || [ "${kind:0:1}" = "#" ] && continue
  model="${model/#\~/$HOME}"
  for attempt in 1 2 3; do
    echo "=== $(date -Is) $kind $tag (attempt $attempt)"
    if [ "$kind" = tg ]; then
      MODEL="$model" "$ROOT/scripts/gpu_lock.sh" "$HERE/bench_tg.sh" "$OUT" "$tag" $extra < /dev/null && break
    else
      MODEL="$model" "$ROOT/scripts/gpu_lock.sh" "$HERE/run_server.sh" "$OUT" "$tag" $extra < /dev/null && break
    fi
    echo "!!! $tag attempt $attempt failed"; sleep 30
    [ "$attempt" = 3 ] && echo "FAILED $tag" >> "$OUT/FAILED.txt"
  done
done < "$CFG"
echo "=== sweep done $(date -Is)"
