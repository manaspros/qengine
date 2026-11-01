#!/usr/bin/env bash
# Run the other engines on the 8-prompt MTP suite (bench/prompts/suite/*.txt), same commands as
# bench/h2h/orchestrate.sh with PROMPT=<suite file>; one gpu_lock per (prompt, engine config).
# Results: build/h2h_suite/<prompt stem>/<tag>.{json,h2h_meta.txt,h2h_smi.csv,...}; finished
# configs (meta has rc=0) are skipped, so the script can be re-run to resume.
#   TAGS="lc_srv lc_mtp2 ex_plain ex_mtp4 si_plain"  (default: all five)
#   PROMPTS="1 5"   prompt-file prefixes (default: all)
#   DRY=1           print each command and check that its binaries / models / prompt exist; run nothing
# Then: python bench/h2h/suite_aggregate.py   (per-prompt medians + geomean per config)
# qengine itself: bench/engine/bench_suite.py (one JSON per prompt in build/suite/).
set -uo pipefail
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$Q"
H=$Q/bench/h2h
OUTROOT=$Q/build/h2h_suite
LOCK=$Q/scripts/gpu_lock.sh
LL=$HOME/Codes/baselines/llama.cpp/build/bin
Q40=$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
MTPG=$HOME/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
SI=$HOME/Codes/baselines/sparkinfer
GT=$HOME/models/Qwen3.8-27B-NVFP4-RTX5090-gittensor
TAGS=${TAGS:-"lc_srv lc_mtp2 ex_plain ex_mtp4 si_plain"}
DRY=${DRY:-0}

cmd_for() {  # cmd_for <tag> <outdir> <prompt file>  -> prints the command line (bash array via declare)
  local tag="$1" O="$2" P="$3"
  case "$tag" in
    lc_srv)   CMD=(env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag) ;;
    lc_mtp2)  CMD=(env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag
                   -md $MTPG -ngld 99 --spec-type draft-mtp --spec-draft-n-max 2) ;;
    lc_mtp3)  CMD=(env MODEL=$Q40 PROMPT="$P" $Q/bench/baselines/llamacpp/run_server.sh "$O" $tag
                   -md $MTPG -ngld 99 --spec-type draft-mtp --spec-draft-n-max 3) ;;
    ex_plain) CMD=(env PROMPT="$P" $Q/bench/baselines/exllamav3/run.sh "$O" $tag) ;;
    ex_mtp4)  CMD=(env PROMPT="$P" $Q/bench/baselines/exllamav3/run.sh "$O" $tag --mtp) ;;
    si_plain) CMD=(env PROMPT="$P" $Q/bench/baselines/vllm/run_server.sh "$O" $tag sparkinfer 18101
                   env SPARKINFER_DETERMINISTIC=1 $SI/build/server/sparkinfer_server -m $GT --tokenizer $GT/tokenizer.json
                   --host 127.0.0.1 --port 18101 --ctx 4096) ;;
    *) echo "unknown tag $tag" >&2; return 1 ;;
  esac
}

needs() {  # files a tag depends on (for DRY checks)
  case "$1" in
    lc_srv) echo "$LL/llama-server $Q40 $Q/bench/baselines/llamacpp/run_server.sh" ;;
    lc_mtp2|lc_mtp3) echo "$LL/llama-server $Q40 $MTPG $Q/bench/baselines/llamacpp/run_server.sh" ;;
    ex_plain|ex_mtp4) echo "$Q/bench/baselines/exllamav3/run.sh $HOME/Codes/baselines/exllamav3" ;;
    si_plain) echo "$SI/build/server/sparkinfer_server $GT/tokenizer.json $Q/bench/baselines/vllm/run_server.sh" ;;
  esac
}

for P in $Q/bench/prompts/suite/*.txt; do
  stem=$(basename "$P" .txt)
  if [ -n "${PROMPTS:-}" ] && ! [[ " $PROMPTS " == *" ${stem%%_*} "* ]]; then continue; fi
  O=$OUTROOT/$stem
  for tag in $TAGS; do
    cmd_for "$tag" "$O" "$P" || exit 1
    if [ "$DRY" = 1 ]; then
      echo "[dry] $stem/$tag: $LOCK $H/run_cfg.sh $O $tag ${CMD[*]}"
      for f in $(needs "$tag") "$P"; do [ -e "$f" ] && echo "      ok  $f" || echo "      MISSING $f"; done
      continue
    fi
    mkdir -p "$O"
    if [ -f "$O/$tag.h2h_meta.txt" ] && grep -q "rc=0" "$O/$tag.h2h_meta.txt"; then echo "skip $stem/$tag"; continue; fi
    echo "=== $(date -Is) $stem/$tag"
    $LOCK $H/run_cfg.sh "$O" "$tag" "${CMD[@]}"
    echo "    rc=$? $(date -Is)"
  done
done
