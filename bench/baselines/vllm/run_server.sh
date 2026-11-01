#!/usr/bin/env bash
# One configuration of an OpenAI-compatible engine (vLLM or SparkInfer): wait for an idle GPU,
# start the server in its own process group, wait for /health, sample clocks every 200 ms, run
# bench_openai.py (1 warm-up + 5 timed, fixed prompt, greedy, 1,024 tokens), stop the server and
# wait until its processes have exited and VRAM is back to desktop level before returning.
# Run it under scripts/gpu_lock.sh (one lock per configuration, never across a sweep).
# Usage: run_server.sh <outdir> <tag> <engine: vllm|sparkinfer> <port> <server cmd...>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
IDLE="$ROOT/bench/baselines/llamacpp/wait_idle.sh"
OUT="$(realpath -m "$1")"; TAG="$2"; ENGINE="$3"; PORT="$4"; shift 4
mkdir -p "$OUT"
"$IDLE" > "$OUT/$TAG.idle.txt"
{
  echo "date: $(date -Is)"
  echo "engine: $ENGINE"; echo "cmd: $*"; echo "env: ${BENCH_ENV_NOTE:-}"
  nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv,noheader
  echo "compute apps before:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
} > "$OUT/$TAG.meta.txt"
setsid "$@" > "$OUT/$TAG.server.log" 2>&1 &
SRV=$!
SMI=""
stop() {
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null || true
  kill -TERM -- -"$SRV" 2>/dev/null || true
  for _ in $(seq 120); do pgrep -g "$SRV" > /dev/null || break; sleep 0.5; done
  kill -KILL -- -"$SRV" 2>/dev/null || true
  wait "$SRV" 2>/dev/null || true
  # VRAM must be back to desktop level before the lock is released
  for _ in $(seq 120); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$m" -lt 2200 ] && break; sleep 0.5
  done
  echo "VRAM after exit: ${m:-?} MiB" >> "$OUT/$TAG.meta.txt"
}
trap stop EXIT
for _ in $(seq 1800); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "server died, see $OUT/$TAG.server.log" >&2; tail -n 30 "$OUT/$TAG.server.log" >&2; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$OUT/$TAG.meta.txt"  # loaded, before load test
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
python3 "$HERE/bench_openai.py" --port "$PORT" --engine "$ENGINE" --prompt "${PROMPT:-$ROOT/bench/prompts/prose.txt}" \
  --out "$OUT/$TAG.json" 2>&1 | tee "$OUT/$TAG.log"
{ echo "compute apps after:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$OUT/$TAG.meta.txt"
kill $SMI; SMI=""
python3 -c "import sys; sys.path.insert(0, '$ROOT/bench/baselines/llamacpp'); import summarize; print('clocks:', summarize.clocks('$OUT/$TAG.clocks.csv'))" | tee -a "$OUT/$TAG.meta.txt"
