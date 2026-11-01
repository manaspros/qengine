#!/usr/bin/env bash
# Block until the GPU is quiet (util < 5% and < 2,200 MiB used = desktop only) for 3 consecutive
# 1 s samples, then print the reading. Gives up after 120 s with a non-zero exit (run invalid).
for i in $(seq 120); do
  r=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  u=${r%%,*}; m=${r##*, }
  if [ "$u" -lt 5 ] && [ "$m" -lt 2200 ]; then ok=$((${ok:-0}+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "idle: util ${u}%, ${m} MiB"; exit 0; }
  sleep 1
done
echo "GPU not idle: $r" >&2; exit 1
