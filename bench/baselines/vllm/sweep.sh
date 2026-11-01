#!/usr/bin/env bash
# vLLM batch-1 decode configurations, one GPU lock per configuration (never across the sweep).
# Usage: sweep.sh <outdir> [config names...]   (default: all)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
V="${V:-$HOME/Codes/baselines/vllm/.venv/bin}"
NV="$HOME/models/Qwen3.8-27B-NVFP4-nvidia"
GT="$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor"
OUT="$1"; shift
BASE=(--served-model-name qwen38 --port 18100 --max-model-len 4096 --max-num-seqs 1
      --gpu-memory-utilization 0.85 --language-model-only --generation-config vllm)
run() {  # run <tag> <model> [extra vllm args]
  local tag="$1" model="$2"; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; return; }
  echo "=== $tag"
  "$ROOT/scripts/gpu_lock.sh" "$HERE/run_server.sh" "$OUT" "$tag" vllm 18100 \
    "$V/vllm" serve "$model" "${BASE[@]}" "$@" 2>&1 | tail -n 6
}
mtp() { echo "{\"method\":\"mtp\",\"num_speculative_tokens\":$1}"; }
cfg() {
  case "$1" in
    mtp[1-9]_marlin) k="${1#mtp}"; k="${k%_marlin}"; run "$1" "$NV" --linear-backend marlin --speculative-config "$(mtp "$k")" ;;
    gt_marlin) run gt_plain_marlin "$GT" --linear-backend marlin ;;
    mtp[1-9]) run "$1" "$NV" --speculative-config "$(mtp "${1#mtp}")" ;;
    marlin)   run plain_marlin "$NV" --linear-backend marlin ;;
    cutlass)  run plain_cutlass "$NV" --linear-backend cutlass ;;
    gt)       run gt_plain "$GT" ;;
    *) echo "unknown config $1" ;;
  esac
}
ORDER=(mtp1 mtp2 mtp3 mtp4 marlin cutlass gt)
[ $# -gt 0 ] && ORDER=("$@")
for c in "${ORDER[@]}"; do cfg "$c"; done
