#!/usr/bin/env bash
# SparkInfer's own speculative harness (runtime/examples/dspark_tau_check.cpp): AR reference, then
# speculative decode with a drafter (DFlash2 or DSpark) in the SAME process and model load; reports
# tau (mean tokens accepted per target verify), losslessness vs AR (token equality) and tok/s.
# Used here for the acceptance number the server does not expose. Prompt = the fixed prose prompt
# through the chat template (156 ids, bench/.../prose_chat_ids.txt). Deterministic mode on so AR
# reps and spec reps are comparable token for token. Run under scripts/gpu_lock.sh.
# Usage: run_tau.sh <outdir> <tag> <draft dir> [max_new=1024]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SI="${SI:-$HOME/Codes/baselines/sparkinfer}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor}"
OUT="$(realpath -m "$1")"; TAG="$2"; DRAFT="$3"; N="${4:-1024}"
IDS="$OUT/prose_chat_ids.txt"
mkdir -p "$OUT"
"$ROOT/bench/baselines/llamacpp/wait_idle.sh" > "$OUT/$TAG.idle.txt"
CMD=(env SPARKINFER_DETERMINISTIC=1 SPARKINFER_DSPARK_AR_REPS=5 SPARKINFER_DSPARK_SPEC_REPS=5
     "$SI/build/runtime/dspark_tau_check" "$MODEL" "$DRAFT" "$N" "@$IDS")
{
  echo "date: $(date -Is)"; echo "sparkinfer: $(git -C "$SI" rev-parse HEAD)"; echo "cmd: ${CMD[*]}"
  nvidia-smi --query-gpu=driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv,noheader
} > "$OUT/$TAG.meta.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
"${CMD[@]}" > "$OUT/$TAG.log" 2>&1
kill $SMI
grep -E "TAU|tau|TPS|lossless|REPS|speedup" "$OUT/$TAG.log" | grep -v "^\s*\[" | tail -n 20
python3 -c "import sys; sys.path.insert(0, '$ROOT/bench/baselines/llamacpp'); import summarize; print('clocks:', summarize.clocks('$OUT/$TAG.clocks.csv'))" | tee -a "$OUT/$TAG.meta.txt"
