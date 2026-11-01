#!/usr/bin/env bash
# llama.cpp llama-bench decode (tg) baseline. 5 repetitions inside llama-bench (-r 5), JSON output
# keeps each repetition's tok/s so we can take the median. Clocks sampled every 200 ms.
# Usage: bench_tg.sh <outdir> <tag> [extra llama-bench args...]
set -euo pipefail
LLAMA="${LLAMA:-$HOME/Codes/baselines/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}"
OUT="$1"; TAG="$2"; shift 2
mkdir -p "$OUT"
"$(dirname "${BASH_SOURCE[0]}")/wait_idle.sh" > "$OUT/$TAG.idle.txt"
{ echo "date: $(date -Is)"; echo "llama.cpp: $(git -C "$LLAMA/../.." rev-parse HEAD 2>/dev/null)"; echo "model: $MODEL"; echo "args: $*"; echo "compute apps before:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$OUT/$TAG.meta.txt"
nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu --format=csv,noheader > "$OUT/$TAG.before.csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
# 1 s+ warm-up: a short tg run, discarded.
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -p 0 -n 128 -r 1 "$@" -o json > /dev/null 2>&1
"$LLAMA/llama-bench" -m "$MODEL" -ngl 99 -r 5 "$@" -o json > "$OUT/$TAG.json" 2> "$OUT/$TAG.err"
kill $SMI; wait $SMI 2>/dev/null || true
python3 "$(dirname "${BASH_SOURCE[0]}")/summarize.py" "$OUT/$TAG.json" "$OUT/$TAG.clocks.csv"
