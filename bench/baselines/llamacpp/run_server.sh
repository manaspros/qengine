#!/usr/bin/env bash
# Start llama-server (full offload, flash-attn on, 1 slot), run bench_server.py, stop the server
# and wait until the process has fully exited (VRAM freed) before returning.
# Waits for an idle GPU first (wait_idle.sh); records commit, driver, compute apps and clocks
# (every 200 ms). Run it under scripts/gpu_lock.sh (one lock per configuration).
# Usage: run_server.sh <outdir> <tag> [extra llama-server args...]
#   env: LLAMA (bin dir), MODEL (gguf), CTX (default 4096)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LLAMA="${LLAMA:-$HOME/Codes/baselines/llama.cpp/build/bin}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf}"
CTX="${CTX:-4096}"
OUT="$1"; TAG="$2"; shift 2; PORT=18089
mkdir -p "$OUT"
"$HERE/wait_idle.sh" > "$OUT/$TAG.idle.txt"
{
  echo "date: $(date -Is)"
  echo "llama.cpp: $(git -C "$LLAMA/../.." rev-parse HEAD 2>/dev/null || echo '?')"
  echo "model: $MODEL"
  echo "bench: ${BENCH:-prose}"; echo "cmd: $LLAMA/llama-server -m $MODEL -ngl 99 -fa on -c $CTX -np 1 --port $PORT $*"
  nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv,noheader
  echo "compute apps before:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
} > "$OUT/$TAG.meta.txt"
"$LLAMA/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -np 1 --port $PORT "$@" > "$OUT/$TAG.server.log" 2>&1 &
SRV=$!
SMI=""
stop() {
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null || true
  kill "$SRV" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$SRV" 2>/dev/null || true
  wait "$SRV" 2>/dev/null || true
}
trap stop EXIT
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do sleep 1; kill -0 $SRV; done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
if [ "${BENCH:-prose}" = probe ]; then  # sudoingX probe workload (3 short prompts, thinking off)
  python3 "$HERE/bench_probe.py" $PORT "$OUT/$TAG.json" 5
else
  python3 "$HERE/bench_server.py" $PORT "${PROMPT:-$ROOT/bench/prompts/prose.txt}" "$OUT/$TAG.json"
fi
{ echo "compute apps after:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$OUT/$TAG.meta.txt"
kill $SMI; SMI=""
grep -E 'draft acceptance' "$OUT/$TAG.server.log" | tail -1 || true
python3 -c "import sys; sys.path.insert(0, '$HERE'); import summarize; print('clocks:', summarize.clocks('$OUT/$TAG.clocks.csv'))" | tee -a "$OUT/$TAG.meta.txt"
