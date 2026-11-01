#!/usr/bin/env bash
# Build the W4A16 GEMV torch extension qengine._w4 -> python/qengine/_w4*.so
# (csrc/gemv_w4.cu + csrc/gemv_w4_bind.cpp; nvcc 13.4, sm_120a; no ninja/setuptools needed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
PY="${PY:-$ROOT/../.venv/bin/python}"; [ -x "$PY" ] || PY=python3
OUT="$ROOT/build/w4"; mkdir -p "$OUT"
read -r TORCH PYINC EXT ABI < <("$PY" -c "
import torch, os, sysconfig
print(os.path.dirname(torch.__file__), sysconfig.get_paths()['include'],
      sysconfig.get_config_var('EXT_SUFFIX'), int(torch._C._GLIBCXX_USE_CXX11_ABI))")
INC=(-I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC" -I"$CUDA_HOME/include" -I"$ROOT/csrc")
"$NVCC" -O3 -std=c++17 -arch=sm_120a -lineinfo -Xcompiler -fPIC -Xptxas -v -I"$ROOT/csrc" \
  -c "$ROOT/csrc/gemv_w4.cu" -o "$OUT/gemv_w4.o" 2> "$OUT/ptxas.log" || { cat "$OUT/ptxas.log"; exit 1; }
g++ -O2 -std=c++20 -fPIC -D_GLIBCXX_USE_CXX11_ABI=$ABI -DTORCH_EXTENSION_NAME=_w4 \
  -DTORCH_API_INCLUDE_EXTENSION_H "${INC[@]}" -c "$ROOT/csrc/gemv_w4_bind.cpp" -o "$OUT/bind.o"
g++ -shared -o "$ROOT/python/qengine/_w4$EXT" "$OUT/gemv_w4.o" "$OUT/bind.o" \
  -L"$TORCH/lib" -L"$CUDA_HOME/lib64" -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lcudart
echo "built: $ROOT/python/qengine/_w4$EXT"
