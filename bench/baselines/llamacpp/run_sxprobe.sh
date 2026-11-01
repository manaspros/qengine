#!/usr/bin/env bash
# Cross-check: run sudoingX/qwen38-mtp's own unchanged probe.py (client-side streaming clock,
# 3 prompts x 3 runs, thinking off, 400 tokens) against a llama-server started like run_server.sh.
# probe.py lives outside qengine (third-party): pass its path. Run under scripts/gpu_lock.sh.
# Usage: run_sxprobe.sh <outdir> <tag> <probe.py> [extra llama-server args...]   env: LLAMA, MODEL, CTX
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="${LLAMA:-$HOME/Codes/baselines/llama.cpp/build/bin}"
MODEL="${MODEL:?set MODEL}"; CTX="${CTX:-4096}"
OUT="$1"; TAG="$2"; PROBE="$3"; shift 3; PORT=18089
mkdir -p "$OUT"
"$HERE/wait_idle.sh" > "$OUT/$TAG.idle.txt"
echo "cmd: llama-server -m $MODEL -ngl 99 -fa on -c $CTX -np 1 $* ; probe: $PROBE" > "$OUT/$TAG.sx.meta.txt"
"$LLAMA/llama-server" -m "$MODEL" -ngl 99 -fa on -c "$CTX" -np 1 --port $PORT "$@" > "$OUT/$TAG.server.log" 2>&1 &
SRV=$!
stop() { kill "$SRV" 2>/dev/null || true; for _ in $(seq 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 0.5; done; kill -9 "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true; }
trap stop EXIT
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do sleep 1; kill -0 $SRV; done
python3 -I "$PROBE" http://127.0.0.1:$PORT | tee "$OUT/$TAG.sxprobe.txt"
grep -E 'draft acceptance' "$OUT/$TAG.server.log" | tail -3 || true
