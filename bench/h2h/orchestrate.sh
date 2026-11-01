#!/usr/bin/env bash
# Same-session head-to-head: round 1 and round 2 interleaved over engines (prose), then a code round.
# One gpu_lock per configuration (released between configurations).
set -uo pipefail
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$Q"
H=$Q/build/h2h
LOCK=$Q/scripts/gpu_lock.sh
PY=${PY:-python3}
LL=$HOME/Codes/baselines/llama.cpp/build/bin
Q40=$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
MTPG=$HOME/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
SI=$HOME/Codes/baselines/sparkinfer
GT=$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor
NV=$HOME/models/Qwen3.8-27B-NVFP4-nvidia
V=$HOME/Codes/baselines/vllm/.venv/bin
VBASE=(--served-model-name qwen38 --port 18100 --max-model-len 4096 --max-num-seqs 1
       --gpu-memory-utilization 0.85 --language-model-only --generation-config vllm)
PROSE=$Q/bench/prompts/prose.txt
CODE=$Q/bench/prompts/code.txt

cfg() {  # cfg <round> <tag> <prompt>
  local R="$1" tag="$2" P="$3" O="$H/$1"
  mkdir -p "$O"
  [ -f "$O/$tag.h2h_meta.txt" ] && grep -q "rc=0" "$O/$tag.h2h_meta.txt" && { echo "skip $R/$tag"; return; }
  echo "=== $(date -Is) $R/$tag"
  case "$tag" in
    qe_raw)    $LOCK $H/run_cfg.sh "$O" $tag $PY bench/engine/bench_decode.py --model $HOME/models/Qwen3.8-27B --w4 --spec 2,3,4 --out "$O/$tag.json" ;;
    qe_chat)   $LOCK $H/run_cfg.sh "$O" $tag $PY $H/h2h_qengine.py --prompt "$P" --chat --spec 2,3,4 --out "$O/$tag.json" ;;
    qe_rawp)   $LOCK $H/run_cfg.sh "$O" $tag $PY $H/h2h_qengine.py --prompt "$P" --spec 2,3,4 --out "$O/$tag.json" ;;
    si_plain)  $LOCK $H/run_cfg.sh "$O" $tag env PROMPT="$P" $Q/bench/baselines/vllm/run_server.sh "$O" $tag sparkinfer 18101 \
                 env SPARKINFER_DETERMINISTIC=1 $SI/build/server/sparkinfer_server -m $GT --tokenizer $GT/tokenizer.json --host 127.0.0.1 --port 18101 --ctx 4096 ;;
    lc_tg)     $LOCK $H/run_cfg.sh "$O" $tag env MODEL=$Q40 $Q/bench/baselines/llamacpp/bench_tg.sh "$O" $tag -fa 1 -p 0 -n 1024 ;;
    lc_srv)    $LOCK $H/run_cfg.sh "$O" $tag env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag ;;
    lc_mtp2)   $LOCK $H/run_cfg.sh "$O" $tag env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag \
                 -md $MTPG -ngld 99 --spec-type draft-mtp --spec-draft-n-max 2 ;;
    lc_mtp4)   $LOCK $H/run_cfg.sh "$O" $tag env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag \
                 -md $MTPG -ngld 99 --spec-type draft-mtp --spec-draft-n-max 4 ;;
    ex_plain)  $LOCK $H/run_cfg.sh "$O" $tag env PROMPT="$P" $Q/bench/baselines/exllamav3/run.sh "$O" $tag ;;
    ex_mtp4)   $LOCK $H/run_cfg.sh "$O" $tag env PROMPT="$P" $Q/bench/baselines/exllamav3/run.sh "$O" $tag --mtp ;;
    vl_plain)  $LOCK $H/run_cfg.sh "$O" $tag env PROMPT="$P" $Q/bench/baselines/vllm/run_server.sh "$O" $tag vllm 18100 \
                 $V/vllm serve $GT "${VBASE[@]}" --linear-backend marlin ;;
    vl_mtp5)   $LOCK $H/run_cfg.sh "$O" $tag env PROMPT="$P" $Q/bench/baselines/vllm/run_server.sh "$O" $tag vllm 18100 \
                 $V/vllm serve $NV "${VBASE[@]}" --linear-backend marlin --speculative-config '{"method":"mtp","num_speculative_tokens":5}' ;;
  esac
  echo "    rc=$? $(date -Is)"
}
ROUND=(qe_raw qe_chat si_plain lc_srv lc_tg lc_mtp2 ex_plain ex_mtp4 vl_plain vl_mtp5)
for t in "${ROUND[@]}"; do cfg r1 $t $PROSE; done
for t in "${ROUND[@]}"; do cfg r2 $t $PROSE; done
for t in qe_chat qe_rawp lc_srv lc_mtp2 lc_mtp4 ex_plain ex_mtp4 vl_mtp5; do cfg code $t $CODE; done
echo "=== ALL DONE $(date -Is)"
