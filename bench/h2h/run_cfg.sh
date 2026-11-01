#!/usr/bin/env bash
# One head-to-head configuration. Called *inside* scripts/gpu_lock.sh:
#   wait_idle.sh (retry) -> 200 ms nvidia-smi sampler (clocks, power, power-cap reason, VRAM) -> command.
# Usage: run_cfg.sh <outdir> <tag> <cmd...>
set -uo pipefail
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"; TAG="$2"; shift 2
mkdir -p "$OUT"
ok=0
for a in 1 2 3; do
  if "$Q/bench/baselines/llamacpp/wait_idle.sh" > "$OUT/$TAG.h2h_idle.txt" 2>&1; then ok=1; break; fi
  sleep 20
done
[ $ok = 1 ] || { echo "INVALID: GPU never idle" >> "$OUT/$TAG.h2h_idle.txt"; exit 3; }
{ echo "start $(date -Is)"; echo "cmd: $*"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$OUT/$TAG.h2h_meta.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.h2h_smi.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
"$@" > "$OUT/$TAG.h2h_log.txt" 2>&1
rc=$?
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
echo "end $(date -Is) rc=$rc" >> "$OUT/$TAG.h2h_meta.txt"
exit $rc
