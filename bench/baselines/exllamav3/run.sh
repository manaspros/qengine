#!/usr/bin/env bash
# Run bench_gen.py (exllamav3 Generator, fixed prompt, 1,024 tokens, warm-up + 5) from the exllamav3
# checkout with its own venv, sampling clocks every 200 ms. GPU state is logged before the run.
# Usage: run.sh <outdir> <tag> [extra bench_gen.py args, e.g. --mtp -ndt 3]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
EXL="${EXL:-$HOME/Codes/baselines/exllamav3}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-exl3-4.00bpw}"
OUT="$(realpath -m "$1")"; TAG="$2"; shift 2
mkdir -p "$OUT"
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv > "$OUT/$TAG.before.csv"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$OUT/$TAG.before.csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 > "$OUT/$TAG.clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
cd "$EXL"
.venv/bin/python "$HERE/bench_gen.py" -m "$MODEL" -prompt "${PROMPT:-$ROOT/bench/prompts/prose.txt}" \
  -out "$OUT/$TAG.json" "$@" 2>&1 | tee "$OUT/$TAG.log"
kill $SMI
python3 -c "import sys; sys.path.insert(0, '$ROOT/bench/baselines/llamacpp'); import summarize; print('clocks:', summarize.clocks('$OUT/$TAG.clocks.csv'))"
python3 -c "
rows=[l.split(', ') for l in open('$OUT/$TAG.clocks.csv') if l.strip()]
print('peak VRAM used (MiB, whole GPU):', max(int(r[6].split()[0]) for r in rows))"
