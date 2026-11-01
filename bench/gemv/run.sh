#!/usr/bin/env bash
# n=5 process runs of the W4 GEMV benchmark (per-shape + full-token sequence), each under the
# exclusive GPU lock and after wait_idle.sh; clocks sampled by nvidia-smi during every run.
# Usage: bench/gemv/run.sh <outdir> "<cfg>" "<N list>"   e.g. build/gemv/final "mode=1,pdl=1" "1,2,4,8"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?outdir}"; CFG="${2:-}"; NS="${3:-1}"
PY="${PY:-python3}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader > "$OUT/driver.txt"
git -C "$ROOT" rev-parse --short HEAD > "$OUT/commit.txt"
git -C "$ROOT" status --short csrc python bench/gemv >> "$OUT/commit.txt" || true
for r in 1 2 3 4 5; do
  for N in ${NS//,/ }; do
    "${GPU_LOCK:-$ROOT/scripts/gpu_lock.sh}" bash -c "
      '$ROOT/bench/baselines/llamacpp/wait_idle.sh' >> '$OUT/idle.txt' || exit 1
      nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
        --format=csv,noheader -lms 500 >> '$OUT/clocks.csv' & SMI=\$!
      '$PY' '$ROOT/bench/gemv/bench_gemv.py' shapes --n $N --cfg '$CFG' --json '$OUT/run$r.jsonl'
      '$PY' '$ROOT/bench/gemv/bench_gemv.py' sequence --n $N --cfg '$CFG' --json '$OUT/run$r.jsonl'
      kill \$SMI" > "$OUT/log_r${r}_n${N}.txt" 2>&1
  done
done
"$PY" "$ROOT/bench/gemv/summarize.py" "$OUT"
