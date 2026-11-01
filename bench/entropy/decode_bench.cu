// Decode-only gate microbenchmark for entropy-coded W4 codes (bench/entropy/ec.py format).
// dec_kernel: decodes every unit (16 rows x 512 k, lane streams of 256 nibbles) into the exact
// packed words of python/qengine/quant.py and hashes them per unit (hash_out[u], exact check vs torch).
// raw_kernel: same work split / hash on the raw packed qw (baseline).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define WIN 12
#define THR 28
#define GPU 4
#define NCHK (16 * GPU)

constexpr int WARPS = 8;

__device__ __forceinline__ uint32_t lanemask_lt() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

__device__ __forceinline__ uint32_t warp_add(uint32_t v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int MAXR>  // MAXR = max 16-byte rows per lane per unit (unit <= 32*MAXR*16 bytes)
__global__ void __launch_bounds__(WARPS * 32)
dec_kernel(const uint4* __restrict__ stream, const uint32_t* __restrict__ off16, int U,
           const uint16_t* __restrict__ lut_g, uint32_t* __restrict__ hash_out) {
  __shared__ uint16_t lut[1 << WIN];
  __shared__ uint4 sbuf[WARPS][32 * MAXR];
  for (int i = threadIdx.x; i < (1 << WIN); i += blockDim.x) lut[i] = lut_g[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t lt = lanemask_lt();
  const long B = gridDim.x;
  const int b0 = (int)((long)U * blockIdx.x / B), b1 = (int)((long)U * (blockIdx.x + 1) / B);
  uint4 pre[MAXR];
  int u = b0 + warp;
  auto load = [&](int uu) {
    const uint32_t o = off16[uu], n = off16[uu + 1] - o;
#pragma unroll
    for (int r = 0; r < MAXR; ++r) {
      const uint32_t i = r * 32 + lane;
      if (i < n) pre[r] = __ldg(stream + o + i);
    }
  };
  if (u < b1) load(u);
  for (; u < b1; u += WARPS) {
    const uint32_t n = off16[u + 1] - off16[u];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
      if (r * 32 + lane < (int)n) sbuf[warp][r * 32 + lane] = pre[r];
    __syncwarp();
    if (u + WARPS < b1) load(u + WARPS);  // prefetch the next unit while decoding this one
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(sbuf[warp]);
    uint64_t buf = 0;
    int bits = 0, nxt = 0;
    uint32_t h = 0;
#pragma unroll 1
    for (int grp = 0; grp < GPU; ++grp) {
      uint32_t w[8];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const bool need = bits < THR;
        const uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) {
          buf |= (uint64_t)sw[nxt + __popc(m & lt)] << bits;
          bits += 32;
        }
        nxt += __popc(m);
        uint32_t acc = 0;
        int prod = 0;
        while (prod < 4) {
          const uint32_t e = lut[(uint32_t)buf & ((1u << WIN) - 1)];
          const bool pair = (e >> 15) && prod <= 2;
          const uint32_t syms = pair ? (e & 0xFFu) : (e & 0xFu);
          const int used = pair ? (int)((e >> 11) & 0xFu) : (int)((e >> 8) & 0x7u);
          acc |= syms << (4 * prod);
          prod += pair ? 2 : 1;
          buf >>= used;
          bits -= used;
        }
        if (c & 1) w[c >> 1] |= acc << 16; else w[c >> 1] = acc;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) h += w[i] * (uint32_t)(lane * 32 + grp * 8 + i + 1);
    }
    h = warp_add(h);
    if (lane == 0) hash_out[u] = h;
    __syncwarp();
  }
}


// v2: same format; branch-free fast path (2 pair lookups per 4 symbols, generic loop only when a
// lane's pair does not fit the window), 32-bit lo/hi bit buffer with funnel shifts.
template <int MAXR>
__global__ void __launch_bounds__(WARPS * 32)
dec2_kernel(const uint4* __restrict__ stream, const uint32_t* __restrict__ off16, int U,
            const uint16_t* __restrict__ lut_g, uint32_t* __restrict__ hash_out) {
  __shared__ uint16_t lut[1 << WIN];
  __shared__ uint4 sbuf[WARPS][32 * MAXR];
  for (int i = threadIdx.x; i < (1 << WIN); i += blockDim.x) lut[i] = lut_g[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t lt = lanemask_lt();
  const long B = gridDim.x;
  const int b0 = (int)((long)U * blockIdx.x / B), b1 = (int)((long)U * (blockIdx.x + 1) / B);
  uint4 pre[MAXR];
  int u = b0 + warp;
  auto load = [&](int uu) {
    const uint32_t o = off16[uu], n = off16[uu + 1] - o;
#pragma unroll
    for (int r = 0; r < MAXR; ++r) {
      const uint32_t i = r * 32 + lane;
      if (i < n) pre[r] = __ldg(stream + o + i);
    }
  };
  if (u < b1) load(u);
  constexpr uint32_t WM = (1u << WIN) - 1;
  for (; u < b1; u += WARPS) {
    const uint32_t n = off16[u + 1] - off16[u];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
      if (r * 32 + lane < (int)n) sbuf[warp][r * 32 + lane] = pre[r];
    __syncwarp();
    if (u + WARPS < b1) load(u + WARPS);
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(sbuf[warp]);
    uint32_t lo = 0, hi = 0;
    int bits = 0, nxt = 0;
    uint32_t h = 0;
#pragma unroll 1
    for (int grp = 0; grp < GPU; ++grp) {
      uint32_t w[8];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const bool need = bits < THR;
        const uint32_t m = __ballot_sync(0xffffffffu, need);
        if (need) {
          const uint32_t x = sw[nxt + __popc(m & lt)];
          lo |= x << bits;                       // bits < 28: valid bits all in lo, hi == 0
          hi = __funnelshift_l(x, 0u, bits);     // x >> (32 - bits), 0 when bits == 0
          bits += 32;
        }
        nxt += __popc(m);
        // fast path: two pair lookups
        const uint32_t e1 = lut[lo & WM];
        const uint32_t u1 = (e1 >> 11) & 0xFu;
        const uint32_t lo1 = __funnelshift_r(lo, hi, u1), hi1 = hi >> u1;
        const uint32_t e2 = lut[lo1 & WM];
        const uint32_t u2 = (e2 >> 11) & 0xFu;
        const bool ok = (e1 & e2) >> 15;
        uint32_t acc;
        if (__all_sync(0xffffffffu, ok)) {
          acc = (e1 & 0xFFu) | ((e2 & 0xFFu) << 8);
          lo = __funnelshift_r(lo1, hi1, u2);
          hi = hi1 >> u2;
          bits -= (int)(u1 + u2);
        } else {
          if (ok) {
            acc = (e1 & 0xFFu) | ((e2 & 0xFFu) << 8);
            lo = __funnelshift_r(lo1, hi1, u2);
            hi = hi1 >> u2;
            bits -= (int)(u1 + u2);
          } else {  // generic loop (rare): identical consumption rule to ec.decode_ref
            acc = 0;
            int prod = 0;
            while (prod < 4) {
              const uint32_t e = lut[lo & WM];
              const bool pair = (e >> 15) && prod <= 2;
              const uint32_t used = pair ? ((e >> 11) & 0xFu) : ((e >> 8) & 0x7u);
              acc |= (pair ? (e & 0xFFu) : (e & 0xFu)) << (4 * prod);
              prod += pair ? 2 : 1;
              lo = __funnelshift_r(lo, hi, used);
              hi >>= used;
              bits -= (int)used;
            }
          }
        }
        if (c & 1) w[c >> 1] |= acc << 16; else w[c >> 1] = acc;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) h += w[i] * (uint32_t)(lane * 32 + grp * 8 + i + 1);
    }
    h = warp_add(h);
    if (lane == 0) hash_out[u] = h;
    __syncwarp();
  }
}

// v3: v2 + branch-free refill (unconditional in-bounds smem read, selects); fast path (2 pair lookups per 4 symbols, generic loop only when a
// lane's pair does not fit the window), 32-bit lo/hi bit buffer with funnel shifts.
template <int MAXR>
__global__ void __launch_bounds__(WARPS * 32)
dec3_kernel(const uint4* __restrict__ stream, const uint32_t* __restrict__ off16, int U,
            const uint16_t* __restrict__ lut_g, uint32_t* __restrict__ hash_out) {
  __shared__ uint16_t lut[1 << WIN];
  __shared__ uint4 sbuf[WARPS][32 * MAXR];
  for (int i = threadIdx.x; i < (1 << WIN); i += blockDim.x) lut[i] = lut_g[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t lt = lanemask_lt();
  const long B = gridDim.x;
  const int b0 = (int)((long)U * blockIdx.x / B), b1 = (int)((long)U * (blockIdx.x + 1) / B);
  uint4 pre[MAXR];
  int u = b0 + warp;
  auto load = [&](int uu) {
    const uint32_t o = off16[uu], n = off16[uu + 1] - o;
#pragma unroll
    for (int r = 0; r < MAXR; ++r) {
      const uint32_t i = r * 32 + lane;
      if (i < n) pre[r] = __ldg(stream + o + i);
    }
  };
  if (u < b1) load(u);
  constexpr uint32_t WM = (1u << WIN) - 1;
  for (; u < b1; u += WARPS) {
    const uint32_t n = off16[u + 1] - off16[u];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
      if (r * 32 + lane < (int)n) sbuf[warp][r * 32 + lane] = pre[r];
    __syncwarp();
    if (u + WARPS < b1) load(u + WARPS);
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(sbuf[warp]);
    uint32_t lo = 0, hi = 0;
    int bits = 0, nxt = 0;
    uint32_t h = 0;
#pragma unroll 1
    for (int grp = 0; grp < GPU; ++grp) {
      uint32_t w[8];
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const bool need = bits < THR;
        const uint32_t m = __ballot_sync(0xffffffffu, need);
        {
          // every lane reads (index stays inside sbuf: nxt + 31 < 32 * MAXR * 4); non-refilling
          // lanes discard the word. No divergent branch, no reconvergence bookkeeping.
          const uint32_t x = need ? sw[nxt + __popc(m & lt)] : 0u;
          lo |= x << (bits & 31);
          hi = need ? __funnelshift_l(x, 0u, bits & 31) : hi;
          bits += need ? 32 : 0;
        }
        nxt += __popc(m);
        // fast path: two pair lookups
        const uint32_t e1 = lut[lo & WM];
        const uint32_t u1 = (e1 >> 11) & 0xFu;
        const uint32_t lo1 = __funnelshift_r(lo, hi, u1), hi1 = hi >> u1;
        const uint32_t e2 = lut[lo1 & WM];
        const uint32_t u2 = (e2 >> 11) & 0xFu;
        const bool ok = (e1 & e2) >> 15;
        uint32_t acc;
        if (__all_sync(0xffffffffu, ok)) {
          acc = (e1 & 0xFFu) | ((e2 & 0xFFu) << 8);
          lo = __funnelshift_r(lo1, hi1, u2);
          hi = hi1 >> u2;
          bits -= (int)(u1 + u2);
        } else {
          if (ok) {
            acc = (e1 & 0xFFu) | ((e2 & 0xFFu) << 8);
            lo = __funnelshift_r(lo1, hi1, u2);
            hi = hi1 >> u2;
            bits -= (int)(u1 + u2);
          } else {  // generic loop (rare): identical consumption rule to ec.decode_ref
            acc = 0;
            int prod = 0;
            while (prod < 4) {
              const uint32_t e = lut[lo & WM];
              const bool pair = (e >> 15) && prod <= 2;
              const uint32_t used = pair ? ((e >> 11) & 0xFu) : ((e >> 8) & 0x7u);
              acc |= (pair ? (e & 0xFFu) : (e & 0xFu)) << (4 * prod);
              prod += pair ? 2 : 1;
              lo = __funnelshift_r(lo, hi, used);
              hi >>= used;
              bits -= (int)used;
            }
          }
        }
        if (c & 1) w[c >> 1] |= acc << 16; else w[c >> 1] = acc;
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) h += w[i] * (uint32_t)(lane * 32 + grp * 8 + i + 1);
    }
    h = warp_add(h);
    if (lane == 0) hash_out[u] = h;
    __syncwarp();
  }
}

// Raw baseline: unit u = 4 KB contiguous packed qw (8 chunks x 32 lanes x uint4).
__global__ void __launch_bounds__(WARPS * 32)
raw_kernel(const uint4* __restrict__ qw, int U, uint32_t* __restrict__ hash_out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long B = gridDim.x;
  const int b0 = (int)((long)U * blockIdx.x / B), b1 = (int)((long)U * (blockIdx.x + 1) / B);
  for (int u = b0 + warp; u < b1; u += WARPS) {
    uint4 v[8];
#pragma unroll
    for (int hh = 0; hh < 8; ++hh) {
      asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
          : "=r"(v[hh].x), "=r"(v[hh].y), "=r"(v[hh].z), "=r"(v[hh].w)
          : "l"(qw + ((long)u * 8 + hh) * 32 + lane));
    }
    uint32_t h = 0;
#pragma unroll
    for (int hh = 0; hh < 8; ++hh) {
      const uint32_t base = lane * 32 + hh * 4 + 1;
      h += v[hh].x * base + v[hh].y * (base + 1) + v[hh].z * (base + 2) + v[hh].w * (base + 3);
    }
    h = warp_add(h);
    if (lane == 0) hash_out[u] = h;
  }
}

static int sm_count() {
  int s = 0;
  cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0);
  return s;
}

void decode(torch::Tensor stream, torch::Tensor off16, torch::Tensor lut, torch::Tensor hash,
            int64_t maxr, int64_t blocks_per_sm, int64_t version) {
  const int U = (int)hash.numel();
  auto s = c10::cuda::getCurrentCUDAStream();
  const int grid = sm_count() * (int)blocks_per_sm;
  auto args = std::make_tuple((const uint4*)stream.data_ptr(), (const uint32_t*)off16.data_ptr(), U,
                              (const uint16_t*)lut.data_ptr(), (uint32_t*)hash.data_ptr());
#define L(R) if (version == 3) dec3_kernel<R><<<grid, WARPS * 32, 0, s>>>((const uint4*)stream.data_ptr(), \
      (const uint32_t*)off16.data_ptr(), U, (const uint16_t*)lut.data_ptr(), (uint32_t*)hash.data_ptr()); \
  else if (version == 2) dec2_kernel<R><<<grid, WARPS * 32, 0, s>>>((const uint4*)stream.data_ptr(), \
      (const uint32_t*)off16.data_ptr(), U, (const uint16_t*)lut.data_ptr(), (uint32_t*)hash.data_ptr()); \
  else dec_kernel<R><<<grid, WARPS * 32, 0, s>>>((const uint4*)stream.data_ptr(), \
      (const uint32_t*)off16.data_ptr(), U, (const uint16_t*)lut.data_ptr(), (uint32_t*)hash.data_ptr())
  switch (maxr) {
    case 4: L(4); break;
    case 5: L(5); break;
    case 6: L(6); break;
    case 7: L(7); break;
    case 8: L(8); break;
    default: TORCH_CHECK(false, "maxr in 4..8");
  }
#undef L
  (void)args;
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode launch failed");
}

void raw(torch::Tensor qw, torch::Tensor hash, int64_t blocks_per_sm) {
  const int U = (int)hash.numel();
  raw_kernel<<<sm_count() * (int)blocks_per_sm, WARPS * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)qw.data_ptr(), U, (uint32_t*)hash.data_ptr());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "raw launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode);
  m.def("raw", &raw);
}
