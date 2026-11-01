#!/usr/bin/env bash
# SparkInfer forward-only decode (its own bench binary, qwen3_gguf_bench = the harness behind the
# README's "decode @ ctx 128 / 4k / 16k" table; comparable to llama-bench tg).
# 5 process runs. Each: load model, the binary's own 64-token warm-up, then a sweep over contexts
# "0,<CTX>": the ctx-0 pass (N tokens, ~10 s) is extra warm-up and is discarded; the <CTX> pass is
# the timed value. Table = median of the 5 process values, min/max across them.
# Run under scripts/gpu_lock.sh. Usage: run_fwd.sh <outdir> <tag> [N_TOKENS=1024] [CTX=128]
#   env: SI (sparkinfer checkout), MODEL (weights dir), PROMPT_IDS (optional token-id file)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SI="${SI:-$HOME/Codes/baselines/sparkinfer}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor}"
OUT="$(realpath -m "$1")"; TAG="$2"; N="${3:-1024}"; CTX="${4:-128}"
mkdir -p "$OUT"
"$ROOT/bench/baselines/llamacpp/wait_idle.sh" > "$OUT/$TAG.idle.txt"
{
  echo "date: $(date -Is)"; echo "sparkinfer: $(git -C "$SI" rev-parse HEAD)"; echo "model: $MODEL"
  echo "cmd: SPARKINFER_BENCH_SWEEP_CTXS=0,$CTX SPARKINFER_BENCH_SWEEP_REPS=1 ${PROMPT_IDS:+SPARKINFER_BENCH_PROMPT_FILE=$PROMPT_IDS }$SI/build/runtime/qwen3_gguf_bench $MODEL $N sweep  (x5 processes)"
  nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv,noheader
  echo "compute apps before:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
} > "$OUT/$TAG.meta.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
export SPARKINFER_BENCH_SWEEP_CTXS="0,$CTX" SPARKINFER_BENCH_SWEEP_REPS=1
[ -n "${PROMPT_IDS:-}" ] && export SPARKINFER_BENCH_PROMPT_FILE="$PROMPT_IDS"
: > "$OUT/$TAG.log"
for i in 1 2 3 4 5; do
  echo "=== process run $i" >> "$OUT/$TAG.log"
  "$SI/build/runtime/qwen3_gguf_bench" "$MODEL" "$N" sweep >> "$OUT/$TAG.log" 2>&1
done
kill $SMI
python3 - "$OUT/$TAG.log" "$CTX" <<'PY' | tee -a "$OUT/$TAG.meta.txt"
import json, re, statistics, sys
vals = [json.loads(m.group(1))[sys.argv[2]]["decode_tps"]
        for m in re.finditer(r"^SWEEP_JSON (.*)$", open(sys.argv[1]).read(), re.M)]
print(f"decode @ctx {sys.argv[2]} per process: {[round(v, 2) for v in vals]}")
print(f"median {statistics.median(vals):.2f} (min {min(vals):.2f}, max {max(vals):.2f}) = "
      f"{100 * statistics.median(vals) / 124.5:.1f}% of 124.5 roofline, n={len(vals)}")
PY
python3 -c "import sys; sys.path.insert(0, '$ROOT/bench/baselines/llamacpp'); import summarize; print('clocks:', summarize.clocks('$OUT/$TAG.clocks.csv'))" | tee -a "$OUT/$TAG.meta.txt"
