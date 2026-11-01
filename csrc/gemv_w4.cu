// qengine W4A16 GEMV for decode: out[n, m] = sum_k x[n, k] * W[m, k],  n < N <= 8.
//
// ---------------------------------------------------------------------------------------------
// Packed layout (produced by python/qengine/quant.py)
// ---------------------------------------------------------------------------------------------
// W is symmetric int4, q in [-8, 7], stored as the nibble u = q + 8. One fp16 scale per
// (row, 128-wide k group): W[m, k] = (u - 8) * scale[m, k / 128].
// M is padded to Mp (multiple of 16), K to Kp (multiple of 128); padding nibbles are 8 (= 0.0).
//
// The matrix is cut into tiles of 16 rows; a tile is cut into "chunks" of 64 k. A chunk
// (16 x 64 = 1024 weights = 512 bytes) is stored as 32 lanes x 16 bytes, so one warp-wide
// 128-bit load fetches a whole chunk, fully coalesced:
//
//   qw (uint4) index = (tile * (Kp/64) + chunk) * 32 + lane
//
// Lane l = 4*g + t (g = 0..7, t = 0..3) holds rows g and g+8 of the tile and the 16 contiguous
// k values  k0 = chunk*64 + 16*t + 0..15. Its 4 words are the 4 mma k-steps s = 0..3; word s
// covers k = chunk*64 + 16*t + 4*s + {0,1,2,3} and its 8 nibbles are
//
//   nibble: 0          1            2          3            4          5            6          7
//   value : W[g][k+0]  W[g+8][k+0]  W[g][k+2]  W[g+8][k+2]  W[g][k+1]  W[g+8][k+1]  W[g][k+3]  W[g+8][k+3]
//
// so that (word >> 4*j) & 0x000F000F | 0x43004300, read as bf16x2, is exactly the j-th register
// of the A fragment of mma.m16n8k16 (row-major A): a0 = (row g, k+0..1), a1 = (row g+8, k+0..1),
// a2 = (row g, k+2..3), a3 = (row g+8, k+2..3). 0x4300 is bf16 128.0 whose ulp is 1, so OR-ing a
// nibble into the low mantissa bits gives 128 + u exactly; subtracting 136 gives q exactly.
//
// The mma's k index is permuted relative to real k (mma k 2t,2t+1 <-> real 4t+0,1 and
// mma k 2t+8,2t+9 <-> real 4t+2,3 inside a 16-k step; steps interleaved by 4). A dot product
// does not care about the order of k, as long as A and B use the same permutation, and this
// one makes the B fragment (x) of lane (g, t) for the whole chunk 16 contiguous bf16
// x[n = g][chunk*64 + 16t .. +15]: two 16-byte loads, no shuffles.
//
// scales (fp16) index = (tile * (Kp/128) + group) * 16 + row_in_tile   (32 contiguous bytes)
//
// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------
// grid = (Mp/16 tiles, ksplit). A block of WARPS warps owns one tile and a contiguous range of
// 128-k groups; the warps take groups round-robin, U groups (U KB of weights) in flight per warp.
// Per group: 2 x 16-byte weight loads, 8 x mma.m16n8k16 (bf16 in, fp32 accumulate) into a
// per-group accumulator, then acc += scale[row] * group_acc in fp32 (scale applied exactly once,
// in fp32). N <= 8 activation rows use the mma's n = 8 columns (rows n >= N are zero).
// Warps are reduced through shared memory. ksplit > 1: fp32 atomicAdd into a workspace, the last
// block of the tile (arrival counter) applies the epilogue, writes bf16 and re-zeroes the
// workspace and counter (so the kernel is CUDA-graph safe with persistent buffers).
// Epilogue SWIGLU: tile rows 0..7 are gate rows, 8..15 the matching up rows (quant.py
// interleaves them), so a lane holds gate and up of the same output in registers:
// out = silu(gate) * up, computed in fp32.
// PDL: weights do not depend on the previous kernel, so the first batch of weight loads is
// issued before griddepcontrol.wait; only x (and the output/workspace) wait for the producer.
#include "gemv_w4.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qengine_w4 {

namespace {

constexpr uint32_t kMask = 0x000F000Fu;
constexpr uint32_t kMagic = 0x43004300u;   // bf16x2 (128, 128)
constexpr uint32_t kBias = 0x43084308u;    // bf16x2 (136, 136)

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));  // (a & b) | c
  return r;
}

__device__ __forceinline__ uint32_t bf16x2_sub(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One mma k-step: dequantize one packed word into the A fragment and multiply by x pairs b0, b1.
__device__ __forceinline__ void step(float (&c)[4], uint32_t w, uint32_t b0, uint32_t b1) {
  const uint32_t a0 = bf16x2_sub(lop3_and_or(w, kMask, kMagic), kBias);
  const uint32_t a1 = bf16x2_sub(lop3_and_or(w >> 4, kMask, kMagic), kBias);
  const uint32_t a2 = bf16x2_sub(lop3_and_or(w >> 8, kMask, kMagic), kBias);
  const uint32_t a3 = bf16x2_sub(lop3_and_or(w >> 12, kMask, kMagic), kBias);
  mma_bf16(c, a0, a1, a2, a3, b0, b1);
}

template <int V>
__device__ __forceinline__ uint4 ld_weights(const uint4* p, uint64_t policy) {
  uint4 r;
  if constexpr (V == 0) {
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  } else if constexpr (V == 1) {
    asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  } else if constexpr (V == 2) {
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  } else {
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(policy));
  }
  return r;
}

__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

__device__ __forceinline__ uint16_t f2bf(float v) {
  // round-to-nearest-even float -> bf16, NaN preserved
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// x fragment for lane (g, t), chunk c: 16 bf16 x[g][c*64 + 16t ..]. Masked for k >= K, n >= N.
__device__ __forceinline__ void load_x(uint32_t (&xr)[8], const GemvParams& p, int g, int t,
                                       int chunk, bool vec_ok) {
  const int k = chunk * 64 + 16 * t;
  if (g >= p.N) {
#pragma unroll
    for (int i = 0; i < 8; ++i) xr[i] = 0u;
    return;
  }
  const uint16_t* xp = p.x + (long)g * p.ldx + k;
  if (vec_ok && k + 16 <= p.K) {
    const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(xp));
    const uint4 v1 = __ldg(reinterpret_cast<const uint4*>(xp) + 1);
    xr[0] = v0.x; xr[1] = v0.y; xr[2] = v0.z; xr[3] = v0.w;
    xr[4] = v1.x; xr[5] = v1.y; xr[6] = v1.z; xr[7] = v1.w;
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t lo = (k + 2 * i < p.K) ? __ldg(xp + 2 * i) : 0u;
      const uint32_t hi = (k + 2 * i + 1 < p.K) ? __ldg(xp + 2 * i + 1) : 0u;
      xr[i] = lo | (hi << 16);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Work split, determinism
// ---------------------------------------------------------------------------------------------
// The weight stream is a sequence of "units" (unit = one 128-k group of one 16-row tile = 1 KB
// of int4 + 32 B of scales), numbered in memory order u = tile * G + group. A block processes
// "segments" = (tile, contiguous group range); its WARPS warps take the segment's groups
// round-robin (U groups in flight per warp), so a block always streams one contiguous region.
//   mode 0: grid (tiles, ksplit); block (x, y) = one segment: tile x, groups split y of ksplit.
//   mode 2: persistent grid of B = SMs x occupancy blocks; block b owns units
//           [total*b/B, total*(b+1)/B), cut at tile boundaries into segments (block stream-K:
//           every block streams the same bytes, no wave-quantization tail).
// Bitwise determinism (no float atomics anywhere): a segment's partial is reduced inside the
// block in a fixed order (warp 0..WARPS-1). A segment covering the whole tile is stored
// directly. Otherwise the partial goes to its own workspace slot, an integer arrival counter
// (counting groups) elects whichever contributor completes the tile, and that finalizer sums
// the tile's slots in a fixed contributor order (ksplit index / block index), applies the
// epilogue and stores bf16, then re-zeroes the counter. The result depends only on the launch
// configuration (never on timing), and not on N: the mma's n columns are independent and no
// launch decision looks at N, so row r of an N-row call equals the 1-row call bit for bit.
// ---------------------------------------------------------------------------------------------

template <int EPI>
__device__ __forceinline__ void write_out(const GemvParams& p, int tile, const float (&a)[4], int g, int t) {
  const int r0 = tile * 16 + g, r1 = r0 + 8, n0 = 2 * t, n1 = 2 * t + 1;
  if constexpr (EPI == EPI_NONE) {
    if (n0 < p.N) {
      if (r0 < p.M) p.out[(long)n0 * p.ldo + r0] = f2bf(a[0]);
      if (r1 < p.M) p.out[(long)n0 * p.ldo + r1] = f2bf(a[2]);
    }
    if (n1 < p.N) {
      if (r0 < p.M) p.out[(long)n1 * p.ldo + r0] = f2bf(a[1]);
      if (r1 < p.M) p.out[(long)n1 * p.ldo + r1] = f2bf(a[3]);
    }
  } else {
    const int j = tile * 8 + g;  // gate row g, up row g+8 of this tile
    if (j < p.M) {
      if (n0 < p.N) p.out[(long)n0 * p.ldo + j] = f2bf(silu(a[0]) * a[2]);
      if (n1 < p.N) p.out[(long)n1 * p.ldo + j] = f2bf(silu(a[1]) * a[3]);
    }
  }
}

// Accumulate groups [gbeg, gend) of `tile` into acc (this warp's share: gbeg + warp + k*WARPS).
template <int WARPS, int U, int V>
__device__ __forceinline__ void accumulate(const GemvParams& p, int tile, int gbeg, int gend,
                                           int warp, int lane, float (&acc)[4], bool& waited,
                                           uint64_t policy, bool vec_ok) {
  const int g = lane >> 2, t = lane & 3;
  const int G = p.Kp >> 7;
  const uint4* wt = p.qw + (long)tile * (p.Kp >> 6) * 32 + lane;
  const __half* st = reinterpret_cast<const __half*>(p.scales) + (long)tile * G * 16;
  for (int g0 = gbeg + warp; g0 < gend; g0 += WARPS * U) {
    uint4 w[U][2];
    float s_lo[U], s_hi[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int gg = g0 + u * WARPS;
      if (gg < gend) {
        w[u][0] = ld_weights<V>(wt + (long)(2 * gg) * 32, policy);
        w[u][1] = ld_weights<V>(wt + (long)(2 * gg + 1) * 32, policy);
        s_lo[u] = __half2float(st[gg * 16 + g]);
        s_hi[u] = __half2float(st[gg * 16 + g + 8]);
      }
    }
    // PDL: the weights above do not depend on the previous kernel; x does.
    if (!waited) { griddep_wait(); waited = true; }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int gg = g0 + u * WARPS;
      if (gg < gend) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t xr[8];
          load_x(xr, p, g, t, 2 * gg + h, vec_ok);
          step(c, w[u][h].x, xr[0], xr[1]);
          step(c, w[u][h].y, xr[2], xr[3]);
          step(c, w[u][h].z, xr[4], xr[5]);
          step(c, w[u][h].w, xr[6], xr[7]);
        }
        acc[0] = fmaf(s_lo[u], c[0], acc[0]);
        acc[1] = fmaf(s_lo[u], c[1], acc[1]);
        acc[2] = fmaf(s_hi[u], c[2], acc[2]);
        acc[3] = fmaf(s_hi[u], c[3], acc[3]);
      }
    }
  }
}

// Fixed-order block reduction; result valid in warp 0. Ends with __syncthreads-safe smem reuse.
template <int WARPS>
__device__ __forceinline__ void block_reduce(float (&acc)[4], float (*red)[4][32], int warp, int lane) {
  if constexpr (WARPS > 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float s = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < WARPS; ++w2) s += red[w2][i][lane];
        acc[i] = s;
      }
    }
    __syncthreads();
  }
}

// Warp 0: publish a partial segment (cnt groups of `tile`) in workspace slot `slot`; if it
// completes the tile, sum all of the tile's slots in the fixed order given by Slots and store.
template <int EPI, typename Slots>
__device__ __forceinline__ void finish_partial(const GemvParams& p, int tile, const float (&a)[4],
                                               int cnt, int G, int slot, int lane, Slots slots) {
  const int g = lane >> 2, t = lane & 3;
  float* ws = p.ws + (long)slot * 128;
#pragma unroll
  for (int i = 0; i < 4; ++i) __stcg(ws + i * 32 + lane, a[i]);
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = (atomicAdd(p.counters + tile, cnt) + cnt == G);   // integer: exact
  last = __shfl_sync(0xffffffffu, last, 0);
  if (!last) return;
  __threadfence();
  float f[4] = {0.f, 0.f, 0.f, 0.f};
  slots([&](int s) {
    const float* w = p.ws + (long)s * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] += __ldcg(w + i * 32 + lane);
  });
  write_out<EPI>(p, tile, f, g, t);
  if (lane == 0) p.counters[tile] = 0;
}

// mode 0: block (tile, y) of (tiles, ksplit)
template <int WARPS, int U, int V, int EPI>
__global__ void __launch_bounds__(WARPS * 32)
gemv_w4_tile(const GemvParams p) {
  griddep_launch();  // dependents (PDL) may start and prefetch their own weights now
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int G = p.Kp >> 7;
  const int ks = gridDim.y, y = blockIdx.y;
  const int gb = (int)((long)G * y / ks), ge = (int)((long)G * (y + 1) / ks);
  const bool vec_ok = ((p.ldx & 7) == 0) && ((reinterpret_cast<uintptr_t>(p.x) & 15) == 0);
  uint64_t policy = 0;
  if constexpr (V == 3) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  __shared__ float red[WARPS][4][32];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  bool waited = false;
  accumulate<WARPS, U, V>(p, tile, gb, ge, warp, lane, acc, waited, policy, vec_ok);
  griddep_wait();  // output / workspace writes must follow the producer
  block_reduce<WARPS>(acc, red, warp, lane);
  if (warp != 0) return;
  if (ge - gb == G) { write_out<EPI>(p, tile, acc, lane >> 2, lane & 3); return; }
  finish_partial<EPI>(p, tile, acc, ge - gb, G, tile * ks + y, lane, [&](auto&& f) {
    for (int s = 0; s < ks; ++s) f(tile * ks + s);
  });
}

// mode 2: persistent block stream-K
template <int WARPS, int U, int V, int EPI>
__global__ void __launch_bounds__(WARPS * 32)
gemv_w4_bsk(const GemvParams p) {
  griddep_launch();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int G = p.Kp >> 7;
  // 32-bit index math: the host only selects mode 2 when total * B < 2^31.
  const int total = (p.Mp >> 4) * G;
  const int B = gridDim.x;
  const int b = blockIdx.x;
  const int b0 = total * b / B, b1 = total * (b + 1) / B;
  const bool vec_ok = ((p.ldx & 7) == 0) && ((reinterpret_cast<uintptr_t>(p.x) & 15) == 0);
  uint64_t policy = 0;
  if constexpr (V == 3) asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  __shared__ float red[WARPS][4][32];
  bool waited = false;
  const int first_tile = b0 / G;
  int tile = first_tile;
  for (int seg = b0; seg < b1; ++tile) {
    const int seg_end = min(b1, (tile + 1) * G);
    const int grp0 = seg - tile * G;
    const int n = seg_end - seg;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    accumulate<WARPS, U, V>(p, tile, grp0, grp0 + n, warp, lane, acc, waited, policy, vec_ok);
    griddep_wait();
    block_reduce<WARPS>(acc, red, warp, lane);
    if (warp == 0) {
      if (n == G) {
        write_out<EPI>(p, tile, acc, lane >> 2, lane & 3);
      } else {
        // Partial segment: slot 0 = this block's first tile, slot 1 = its last tile. The blocks
        // covering a tile are consecutive; the finalizer sums them in ascending block order.
        finish_partial<EPI>(p, tile, acc, n, G, 2 * b + (tile == first_tile ? 0 : 1), lane,
                            [&](auto&& f) {
          const int u_lo = tile * G, u_hi = u_lo + G - 1;
          int c = (int)((long)u_lo * B / total);            // first guess, then fix up
          while (c > 0 && total * c / B > u_lo) --c;
          while (total * (c + 1) / B <= u_lo) ++c;
          for (; c < B && total * c / B <= u_hi; ++c)
            f(2 * c + ((total * c / B) / G == tile ? 0 : 1));
        });
      }
    }
    seg = seg_end;
  }
}

using KernelFn = void (*)(const GemvParams);

template <int WARPS, int U, int V>
KernelFn pick_epi(int epi, int mode) {
  if (mode == 2) return epi == EPI_SWIGLU ? gemv_w4_bsk<WARPS, U, V, EPI_SWIGLU> : gemv_w4_bsk<WARPS, U, V, EPI_NONE>;
  return epi == EPI_SWIGLU ? gemv_w4_tile<WARPS, U, V, EPI_SWIGLU> : gemv_w4_tile<WARPS, U, V, EPI_NONE>;
}
template <int WARPS, int U>
KernelFn pick_v(int v, int epi, int mode) {
  switch (v) {
    case 1: return pick_epi<WARPS, U, 1>(epi, mode);
    case 2: return pick_epi<WARPS, U, 2>(epi, mode);
    case 3: return pick_epi<WARPS, U, 3>(epi, mode);
    default: return pick_epi<WARPS, U, 0>(epi, mode);
  }
}
template <int WARPS>
KernelFn pick_u(int u, int v, int epi, int mode) {
  switch (u) {
    case 1: return pick_v<WARPS, 1>(v, epi, mode);
    case 4: return pick_v<WARPS, 4>(v, epi, mode);
    default: return pick_v<WARPS, 2>(v, epi, mode);
  }
}
KernelFn pick(int warps, int u, int v, int epi, int mode) {
  switch (warps) {
    case 4: return pick_u<4>(u, v, epi, mode);
    case 16: return pick_u<16>(u, v, epi, mode);
    default: return pick_u<8>(u, v, epi, mode);
  }
}

int occupancy(KernelFn fn, int threads) {
  static std::unordered_map<const void*, int> cache;  // host-side, per kernel instantiation
  auto it = cache.find((const void*)fn);
  if (it != cache.end()) return it->second;
  int b = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, fn, threads, 0);
  b = std::max(b, 1);
  cache[(const void*)fn] = b;
  return b;
}

int sm_count() {
  static int sms = 0;
  if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  return sms;
}

}  // namespace

int auto_ksplit(int Mp, int Kp, int N, const LaunchCfg& cfg) {
  const int warps = cfg.warps > 0 ? cfg.warps : 8;
  (void)N;  // never depends on N (row r of an N-row call must equal the 1-row call bitwise)
  const int tiles = Mp / 16, G = Kp / 128;
  int ks = 1;
  if ((long)tiles * G >= 8192) return 1;  // only tiny matrices are split (see plan_gemv_w4)
  // Small M: split K across blocks until the grid has >= ~2 blocks per SM (170 SMs), while
  // keeping >= one full round of groups per warp.
  while (tiles * ks < 340 && G / (ks * 2) >= warps) ks *= 2;
  return ks;
}

// Experiment knob (shape-keyed, never N-keyed): QE_W4_CFG="TxG:mode,warps,unroll,ksplit,bps,min_units;..."
// T = Mp/16 tiles, G = Kp/128 groups. Overrides the auto policy and explicit args for that shape.
struct ShapeOverride { long T, G; int mode, warps, unroll, ksplit, bps, min_units; };
static const std::vector<ShapeOverride>& shape_overrides() {
  static std::vector<ShapeOverride> v = [] {
    std::vector<ShapeOverride> r;
    const char* e = std::getenv("QE_W4_CFG");
    if (!e) return r;
    std::string s(e);
    size_t i = 0;
    while (i < s.size()) {
      size_t j = s.find(';', i);
      if (j == std::string::npos) j = s.size();
      ShapeOverride o{};
      if (std::sscanf(s.substr(i, j - i).c_str(), "%ldx%ld:%d,%d,%d,%d,%d,%d", &o.T, &o.G, &o.mode, &o.warps,
                      &o.unroll, &o.ksplit, &o.bps, &o.min_units) == 8)
        r.push_back(o);
      i = j + 1;
    }
    return r;
  }();
  return v;
}

LaunchPlan plan_gemv_w4(const GemvParams& p, const LaunchCfg& cfg_in) {
  LaunchPlan pl;
  LaunchCfg cfg = cfg_in;
  const long tiles = p.Mp / 16, G = p.Kp / 128, units = tiles * G;
  for (const auto& o : shape_overrides())
    if (o.T == tiles && o.G == G) {
      cfg.mode = o.mode; cfg.warps = o.warps; cfg.unroll = o.unroll; cfg.ksplit = o.ksplit;
      cfg.blocks_per_sm = o.bps; cfg.min_units = o.min_units;
    }
  // Auto policy (from bench/gemv sweeps). Depends only on the
  // weight shape, never on N, so results are bitwise independent of the batch size.
  //   tiny (< 8,192 units ~ 8 MB, k/v_proj, in_proj_a/b): mode 0 + split-K, 8 warps
  //   huge (> 200k units, lm_head): mode 0, 8 warps (many waves, no tail)
  //   few long tiles (<= 512 tiles and K >= 6144, e.g. out/o_proj, down, mtp.fc): mode 0, 16 warps,
  //     unroll 1, no split (a tile never straddles blocks: no partials at all)
  //   otherwise: mode 2 (block stream-K), 8 warps, unroll 1
  // (unroll 1 everywhere: full-token sequence 8.134 vs 8.151 ms at N=1, 8.306 vs 8.374 at N=8)
  int amode, awarps, aunroll;
  if (units < 8192) { amode = 0; awarps = 8; aunroll = 1; }
  else if (units > 200000) { amode = 0; awarps = 8; aunroll = 1; }
  else if (tiles <= 512 && G >= 48) { amode = 0; awarps = 16; aunroll = 1; }
  else { amode = 2; awarps = 8; aunroll = 1; }
  pl.mode = (cfg.mode == 0 || cfg.mode == 2) ? cfg.mode : amode;
  if (cfg.warps <= 0) cfg.warps = (pl.mode == amode) ? awarps : 8;
  if (cfg.unroll <= 0) cfg.unroll = (pl.mode == amode) ? aunroll : 1;
  if (cfg.ksplit <= 0 && cfg_in.mode < 0 && amode == 0 && units >= 8192) cfg.ksplit = 1;
  if (pl.mode == 2 && units * (long)sm_count() * 16 >= (1L << 31)) pl.mode = 0;  // 32-bit math in mode 2
  const int warps = (cfg.warps == 4 || cfg.warps == 16) ? cfg.warps : 8;
  pl.fn = (const void*)pick(warps, cfg.unroll, cfg.variant, p.epilogue, pl.mode);
  pl.threads = warps * 32;
  if (pl.mode == 2) {
    const int bps = cfg.blocks_per_sm > 0 ? cfg.blocks_per_sm : occupancy((KernelFn)pl.fn, pl.threads);
    const long min_units = (long)std::max(1, cfg.min_units) * warps;  // per block
    const long blocks = std::max<long>(1, std::min<long>((long)sm_count() * bps, (units + min_units - 1) / min_units));
    pl.grid_x = (int)blocks;
    pl.grid_y = 1;
    pl.ws_floats = blocks * 2 * 128;
  } else {
    int ks = cfg.ksplit > 0 ? cfg.ksplit : auto_ksplit(p.Mp, p.Kp, p.N, cfg);
    ks = (int)std::max<long>(1, std::min<long>(ks, G));
    pl.grid_x = (int)tiles;
    pl.grid_y = ks;
    pl.ws_floats = ks > 1 ? tiles * ks * 128 : 0;
  }
  pl.counters = tiles;
  return pl;
}

cudaError_t launch_gemv_w4(GemvParams p, const LaunchCfg& cfg, cudaStream_t stream) {
  const LaunchPlan pl = plan_gemv_w4(p, cfg);
  p.ksplit = pl.grid_y;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(pl.grid_x, pl.grid_y, 1);
  lc.blockDim = dim3(pl.threads, 1, 1);
  lc.dynamicSmemBytes = 0;
  lc.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = attr;
  lc.numAttrs = cfg.pdl ? 1 : 0;
  return cudaLaunchKernelEx(&lc, (KernelFn)pl.fn, p);
}

}  // namespace qengine_w4
