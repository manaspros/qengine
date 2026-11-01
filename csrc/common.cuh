// Shared helpers for qengine._C kernels (BF16 decode path, sm_120).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace qe {

using bf16 = __nv_bfloat16;

// PDL: allow the next kernel in the stream (if launched with programmatic stream serialization,
// e.g. the W4 GEMV with pdl=True) to start now and prefetch its weights. Its griddepcontrol.wait
// still waits for this whole grid to finish and its writes to be visible, so this is always safe;
// it is a no-op when the next kernel is launched normally.
// g_pdl_trigger (set from Python, _C.set_pdl_trigger) switches it at run time for A/B tests.
// One copy per translation unit (no -rdc); each .cu exports a setter, see set_pdl_trigger().
static __device__ int g_pdl_trigger = 1;
__device__ __forceinline__ void pdl_trigger() {
  if (g_pdl_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// PDL, consumer side: wait until the previous kernel (prerequisite grid) has finished and its
// writes are visible. No-op when this kernel was launched normally.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Host: launch with programmatic stream serialization when pdl_launch_flag() is set, so this
// kernel can be scheduled while the previous one finishes (it waits in pdl_wait()).
inline bool& pdl_launch_flag() {
  static bool f = true;  // default on (A/B: +2.9% plain, +3.3% MTP k=3 on the 27B)
  return f;
}
template <typename... KArgs, typename... Args>
inline void qe_launch(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t c = {};
  c.gridDim = grid;
  c.blockDim = block;
  c.dynamicSmemBytes = smem;
  c.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = at;
  c.numAttrs = pdl_launch_flag() ? 1 : 0;
  cudaLaunchKernelEx(&c, kern, static_cast<KArgs>(args)...);
}

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
// Round-to-nearest-even to bf16, as torch does for every bf16 op output.
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }
// Round an fp32 value through bf16 and back: models one torch bf16 op.
__device__ __forceinline__ float rb(float x) { return b2f(f2b(x)); }

// torch silu on fp32: x / (1 + exp(-x)) with accurate expf (no fast-math in this build).
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Block-wide sum; `red` must hold >= 32 floats. All threads get the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();  // protect `red` against a previous use
  if (lane == 0) red[w] = v;
  __syncthreads();
  v = (lane < nw) ? red[lane] : 0.f;
  return warp_sum(v);
}

// Unpack 8 bf16 packed in a uint4 to floats.
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

}  // namespace qe
