#!/usr/bin/env bash
# SparkInfer batch-1 decode configurations, one GPU lock per configuration (never across the sweep).
#   fwd_*   : forward-only bench binary (run_fwd.sh, 5 processes)
#   srv_*   : OpenAI server + bench_openai.py (vllm/run_server.sh, 1 warm-up + 5 timed requests)
# Usage: sweep.sh <outdir> [config names...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SI="${SI:-$HOME/Codes/baselines/sparkinfer}"
GT="$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor"
NV="$HOME/models/Qwen3.8-27B-NVFP4-nvidia"
DF="$HOME/models/Qwen3.8-27B-DFlash2-zlab"
OUT="$1"; shift
LOCK="$ROOT/scripts/gpu_lock.sh"
srv() {  # srv <tag> <model> [extra server args]
  local tag="$1" model="$2"; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; return; }
  echo "=== $tag"
  "$LOCK" "$ROOT/bench/baselines/vllm/run_server.sh" "$OUT" "$tag" sparkinfer 18101 \
    ${SRV_ENV:+env $SRV_ENV} "$SI/build/server/sparkinfer_server" -m "$model" --tokenizer "$model/tokenizer.json" \
    --host 127.0.0.1 --port 18101 "$@" 2>&1 | tail -n 7
}
fwd() {  # fwd <tag> <model> <n> <ctx>
  [ -f "$OUT/$1.log" ] && grep -q "^median" "$OUT/$1.meta.txt" 2>/dev/null && { echo "skip $1 (done)"; return; }
  echo "=== $1"
  MODEL="$2" "$LOCK" "$HERE/run_fwd.sh" "$OUT" "$1" "$3" "$4" 2>&1 | tail -n 3
}
cfg() {
  case "$1" in
    fwd1024)    fwd fwd_n1024_ctx128 "$GT" 1024 128 ;;
    fwd128)     fwd fwd_n128_ctx128 "$GT" 128 128 ;;      # the README table's configuration
    srv4k)      srv srv_plain_ctx4096 "$GT" --ctx 4096 ;;
    srv2k)      srv srv_plain_ctx2048 "$GT" --ctx 2048 ;;
    srvdf)      srv srv_dflash2_ctx4096 "$GT" --ctx 4096 --draft-model "$DF" ;;
    srv4k_det)  SRV_ENV=SPARKINFER_DETERMINISTIC=1 srv srv_plain_ctx4096_det "$GT" --ctx 4096 ;;
    srvdf_det)  SRV_ENV=SPARKINFER_DETERMINISTIC=1 srv srv_dflash2_ctx4096_det "$GT" --ctx 4096 --draft-model "$DF" ;;
    tau_df)     [ -f "$OUT/tau_dflash2.meta.txt" ] && grep -q clocks "$OUT/tau_dflash2.meta.txt" && { echo "skip tau_df"; return; }
                echo "=== tau_dflash2"; "$LOCK" "$HERE/run_tau.sh" "$OUT" tau_dflash2 "$DF" 1024 ;;
    srv_nv)     srv srv_nvidia_plain_ctx4096 "$NV" --ctx 4096 ;;
    *) echo "unknown config $1" ;;
  esac
}
ORDER=(fwd1024 fwd128 srv4k srv2k srvdf)
[ $# -gt 0 ] && ORDER=("$@")
for c in "${ORDER[@]}"; do cfg "$c"; done
