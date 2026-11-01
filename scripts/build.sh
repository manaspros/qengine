#!/usr/bin/env bash
# Build qengine CUDA targets into qengine/build/. nvcc 13.4, sm_120 (RTX 5090).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVCC="${NVCC:-${CUDA_HOME:-/opt/cuda}/bin/nvcc}"
ARCH="${ARCH:-sm_120}"
mkdir -p "$ROOT/build"
FLAGS=(-O3 -std=c++17 -arch="$ARCH" -lineinfo)

# BF16 decode kernels extension qengine._C (csrc/{bindings.cpp,basic.cu,gdn.cu,attn.cu}).
# `scripts/build.sh c` builds only this extension.
PY="${PY:-$ROOT/../.venv/bin/python}"; [ -x "$PY" ] || PY=python3
MAX_JOBS="${MAX_JOBS:-8}" ARCH="$ARCH" CUDA_HOME="${CUDA_HOME:-/opt/cuda}" "$PY" "$ROOT/python/build_c.py" >"$ROOT/build/build_c.log" 2>&1 \
    || { tail -40 "$ROOT/build/build_c.log"; exit 1; }
echo "built: qengine._C"
[[ "${1:-}" == "c" ]] && exit 0

"$NVCC" "${FLAGS[@]}" -o "$ROOT/build/membw" "$ROOT/bench/membw/membw.cu"
echo "built: $ROOT/build/membw"

# W4A16 GEMV extension qengine._w4 (separate step, owned by the W4 work).
"$ROOT/scripts/build_w4.sh"
