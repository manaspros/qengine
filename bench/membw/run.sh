#!/usr/bin/env bash
# n=5 runs of the bandwidth microbenchmark with nvidia-smi clocks sampled every 200 ms.
# Usage: [BPS=8] [CHUNK=32] bench/membw/run.sh [outdir]   (default: build/membw-runs)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/build/membw-runs}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu --format=csv > "$OUT/before.csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv,noheader -lms 200 > "$OUT/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
# BPS (blocks/SM, default 8) and CHUNK (readblk chunk KiB, default 32) select the launch shape.
for r in 1 2 3 4 5; do "$ROOT/build/membw" "${BPS:-8}" 4096 "${CHUNK:-32}" > "$OUT/run$r.csv"; done
# decode-footprint sweeps (13.96 GB buffer, read as consecutive windows, one kernel each)
for r in 1 2 3 4 5; do for W in 64 1024; do "$ROOT/build/membw" sweep 13 "$W" 10; done; done > "$OUT/sweep.csv"
kill $SMI; wait $SMI 2>/dev/null || true
python3 "$ROOT/bench/membw/summarize.py" "$OUT"
