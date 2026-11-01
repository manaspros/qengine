// qengine M-1 Task 2: DRAM bandwidth microbenchmark (RTX 5090, sm_120).
//
// Measures three things per buffer size:
//   copy  : dst[i] = src[i] with 16-byte (int4) loads/stores. Bytes counted = read + write.
//   read  : read-only reduction (sum of uint32 words) -> closest to a W4 GEMV, which streams
//           weights and writes almost nothing. Bytes counted = read only.
//   d2d   : cudaMemcpyAsync device-to-device (driver copy engine / copy kernel), read + write.
//   readblk: (added 2026-11-01) read-only, GEMV-like access: the buffer is cut into contiguous
//           chunks (argv[3] KiB, default 32 = one 27B W4 tile row of K=5120 + slack); block c
//           reads chunks c, c+grid, ...; inside a chunk the block's 8 warps read 16 B per lane,
//           consecutive warps 512 B apart, 8 loads in flight per thread (same pattern as the
//           qengine._w4 tile kernel). Bytes counted = read only.
//   readseg: read-only, persistent: block b reads the contiguous range [n*b/B, n*(b+1)/B)
//           with the same intra-block pattern (like the _w4 block stream-K mode).
// Why: the W4 GEMV measured 1,742 GB/s on lm_head, above `read` (grid-stride: at any moment
// the resident warps touch addresses spread over the whole buffer -> poor DRAM page locality).
//
// Correctness first: copy result is compared word-for-word on the device, the reduction is
// compared to the closed-form sum of the fill pattern (exact, integer arithmetic).
//
// Timing: ~1 s of warm-up load (an idle 5090 needs it before clocks settle), then ITERS timed
// launches per size with cudaEvents; the per-size number is the median over those launches.
// One process = one "run"; the driver script runs the binary 5 times for n=5.
//
// Safety: every kernel is a grid-stride loop bounded by n (no out-of-range access for any n);
// each launch moves at most a few GB and takes a few ms, so the desktop never stalls long.

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  std::exit(1); } } while (0)

// Fill word i with a cheap hash of i so that a wrong index in copy/read is detected.
__host__ __device__ inline uint32_t pattern(uint64_t i) {
  return (uint32_t)(i * 2654435761ull) ^ (uint32_t)(i >> 32);
}

__constant__ int g_ld = 0;

__global__ void fill_kernel(uint32_t* p, uint64_t n_words) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n_words;
       i += (uint64_t)gridDim.x * blockDim.x)
    p[i] = pattern(i);
}

// Copy: each thread handles UNROLL int4 vectors per grid-stride step so that UNROLL independent
// 16-byte loads are in flight before the first store (memory-level parallelism).
template <int UNROLL>
__global__ void copy_kernel(const int4* __restrict__ src, int4* __restrict__ dst, uint64_t n_vec) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  // Main loop: all UNROLL indices i, i+stride, ... are < n_vec, so no per-load mask needed.
  for (; i + (UNROLL - 1) * stride < n_vec; i += UNROLL * stride) {
    int4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = __ldcs(&src[i + u * stride]);  // streaming load
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) __stcs(&dst[i + u * stride], v[u]);    // streaming store
  }
  // Tail: masked one vector at a time.
  for (; i < n_vec; i += stride) dst[i] = src[i];
}

// Read-only: sum all uint32 words (mod 2^64) into one counter. Per-thread partial sums, warp
// shuffle reduction, one atomicAdd per warp: the atomics are ~1e5 total, negligible vs GBs read.
template <int UNROLL>
__global__ void read_kernel(const int4* __restrict__ src, uint64_t n_vec,
                            unsigned long long* __restrict__ out) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  unsigned long long acc = 0;
  for (; i + (UNROLL - 1) * stride < n_vec; i += UNROLL * stride) {
    int4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = __ldcs(&src[i + u * stride]);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc += (uint64_t)(uint32_t)v[u].x + (uint32_t)v[u].y + (uint32_t)v[u].z + (uint32_t)v[u].w;
  }
  for (; i < n_vec; i += stride) {
    int4 v = src[i];
    acc += (uint64_t)(uint32_t)v.x + (uint32_t)v.y + (uint32_t)v.z + (uint32_t)v.w;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

// GEMV-like reads. chunk_vec = int4 per chunk; LOADS loads in flight per thread.
// g_ld (argv[4]): 0 = __ldcs (ld.global.cs, as `read`), 1 = ld.global.nc.L1::no_allocate (as
// the W4 GEMV's weight stream).
__device__ __forceinline__ int4 ld_na(const int4* p) {
  int4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
template <int LOADS>
__device__ __forceinline__ unsigned long long read_range(const int4* __restrict__ src, uint64_t lo,
                                                         uint64_t hi) {
  unsigned long long acc = 0;
  for (uint64_t j = lo + threadIdx.x; j < hi; j += (uint64_t)blockDim.x * LOADS) {
    int4 v[LOADS];
#pragma unroll
    for (int u = 0; u < LOADS; ++u) {
      const uint64_t k = j + (uint64_t)u * blockDim.x;
      v[u] = k < hi ? (g_ld ? ld_na(&src[k]) : __ldcs(&src[k])) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < LOADS; ++u)
      acc += (uint64_t)(uint32_t)v[u].x + (uint32_t)v[u].y + (uint32_t)v[u].z + (uint32_t)v[u].w;
  }
  return acc;
}

template <int LOADS>
__global__ void readblk_kernel(const int4* __restrict__ src, uint64_t n_vec, uint64_t chunk_vec,
                               unsigned long long* __restrict__ out) {
  unsigned long long acc = 0;
  for (uint64_t c = blockIdx.x; c * chunk_vec < n_vec; c += gridDim.x)
    acc += read_range<LOADS>(src, c * chunk_vec, min(n_vec, (c + 1) * chunk_vec));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

template <int LOADS>
__global__ void readseg_kernel(const int4* __restrict__ src, uint64_t n_vec,
                               unsigned long long* __restrict__ out) {
  const uint64_t lo = n_vec * blockIdx.x / gridDim.x, hi = n_vec * (blockIdx.x + 1) / gridDim.x;
  unsigned long long acc = read_range<LOADS>(src, lo, hi);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

__global__ void compare_kernel(const uint32_t* a, const uint32_t* b, uint64_t n,
                               unsigned long long* mism) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
       i += (uint64_t)gridDim.x * blockDim.x)
    if (a[i] != b[i]) atomicAdd(mism, 1ull);
}

static double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Sweep mode (decode-like footprint): membw sweep <total_gib> <window_mib> [iters=10]
//   Fills a total_gib buffer, then each sample reads ALL of it as consecutive window_mib
//   windows, one readblk kernel per window, back-to-back (like the ~260 GEMVs of one decode
//   token streaming 13.6 GB). GB/s = total bytes / sample time; prints median/min/max.
//   Found 2026-11-01: sustained read bandwidth depends on the footprint touched (2.6 GB of
//   rotating GEMV weights: 1,744 GB/s; 5.2 GB: 1,693 GB/s), so the decode-relevant number is
//   measured at the decode footprint.
static int sweep_main(int argc, char** argv);

// Usage: membw [blocks_per_sm=8] [max_size_mib=4096] [chunk_kib=32] [ld=0] [launches_per_sample=1]
// launches_per_sample > 1: each timed sample is that many back-to-back launches (per-launch
// ramp-up/tail amortized, like consecutive GEMVs in a CUDA graph); GB/s = bytes*L / time.
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "sweep") return sweep_main(argc, argv);
  const int ITERS = 30;
  constexpr int UNROLL = 4;
  const int THREADS = 256;
  int dev = 0, sms = 0;
  CK(cudaSetDevice(dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  int l2 = 0; CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  const int blocks_per_sm = argc > 1 ? atoi(argv[1]) : 8;
  const int blocks = sms * blocks_per_sm;

  // Sizes per buffer in MiB. 64 MiB and below can partly hit L2 (L2 printed below);
  // >= 256 MiB is DRAM-only. Copy at 4 GiB uses 8 GiB of VRAM.
  // argv[2] (optional) caps the largest size, for a small first/sanitizer run.
  const uint64_t all_sizes[] = {16, 64, 256, 512, 1024, 2048, 4096};
  const uint64_t cap_mib = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4096;
  std::vector<uint64_t> sizes_mib;
  for (uint64_t s : all_sizes) if (s <= cap_mib) sizes_mib.push_back(s);
  if (sizes_mib.empty()) { fprintf(stderr, "cap below smallest size\n"); return 1; }
  const uint64_t max_bytes = sizes_mib.back() << 20;
  const uint64_t chunk_kib = argc > 3 ? strtoull(argv[3], nullptr, 10) : 32;
  const uint64_t chunk_vec = chunk_kib * 1024 / 16;
#ifndef MEMBW_LOADS
#define MEMBW_LOADS 8
#endif
  constexpr int LOADS = MEMBW_LOADS;  // loads in flight per thread for readblk/readseg
  const int ld = argc > 4 ? atoi(argv[4]) : 0;
  const int LPS = argc > 5 ? std::max(1, atoi(argv[5])) : 1;
  CK(cudaMemcpyToSymbol(g_ld, &ld, sizeof(int)));

  uint32_t *a, *b;
  unsigned long long* d_acc;
  CK(cudaMalloc(&a, max_bytes));
  CK(cudaMalloc(&b, max_bytes));
  CK(cudaMalloc(&d_acc, sizeof(unsigned long long)));
  fill_kernel<<<blocks, THREADS>>>(a, max_bytes / 4);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  printf("# SMs=%d L2=%d bytes blocks=%d threads=%d unroll=%d iters=%d chunk_kib=%llu loads=%d ld=%s "
         "launches_per_sample=%d\n", sms, l2, blocks, THREADS, UNROLL, ITERS,
         (unsigned long long)chunk_kib, LOADS, ld ? "nc.L1::no_allocate" : "cs", LPS);

  // Warm-up: ~1 s of copy load at the largest size so clocks are at their loaded state.
  {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.0) {
      copy_kernel<UNROLL><<<blocks, THREADS>>>((const int4*)a, (int4*)b, max_bytes / 16);
      CK(cudaDeviceSynchronize());
    }
  }

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  printf("size_mib,kind,median_gbps,min_gbps,max_gbps\n");
  for (uint64_t mib : sizes_mib) {
    const uint64_t bytes = mib << 20, n_vec = bytes / 16, n_words = bytes / 4;

    // ---- correctness: copy ----
    CK(cudaMemset(b, 0, bytes));
    copy_kernel<UNROLL><<<blocks, THREADS>>>((const int4*)a, (int4*)b, n_vec);
    CK(cudaGetLastError());
    CK(cudaMemset(d_acc, 0, 8));
    compare_kernel<<<blocks, THREADS>>>(a, b, n_words, d_acc);
    unsigned long long mism = 0;
    CK(cudaMemcpy(&mism, d_acc, 8, cudaMemcpyDeviceToHost));
    // ---- correctness: read (exact integer sum vs host closed form) ----
    CK(cudaMemset(d_acc, 0, 8));
    read_kernel<UNROLL><<<blocks, THREADS>>>((const int4*)a, n_vec, d_acc);
    CK(cudaGetLastError());
    unsigned long long got = 0, want = 0;
    CK(cudaMemcpy(&got, d_acc, 8, cudaMemcpyDeviceToHost));
    for (uint64_t i = 0; i < n_words; ++i) want += pattern(i);
    unsigned long long got_blk = 0, got_seg = 0;
    CK(cudaMemset(d_acc, 0, 8));
    readblk_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a, n_vec, chunk_vec, d_acc);
    CK(cudaMemcpy(&got_blk, d_acc, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemset(d_acc, 0, 8));
    readseg_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a, n_vec, d_acc);
    CK(cudaMemcpy(&got_seg, d_acc, 8, cudaMemcpyDeviceToHost));
    if (got_blk != want || got_seg != want) {
      fprintf(stderr, "FAIL size=%llu MiB readblk got=%llu readseg got=%llu want=%llu\n",
              (unsigned long long)mib, got_blk, got_seg, want);
      return 3;
    }
    if (mism != 0 || got != want) {
      fprintf(stderr, "FAIL size=%llu MiB copy_mismatches=%llu read got=%llu want=%llu\n",
              (unsigned long long)mib, mism, got, want);
      return 2;
    }

    for (int kind = 0; kind < 5; ++kind) {
      std::vector<float> ms(ITERS);
      for (int it = 0; it < ITERS; ++it) {
        CK(cudaEventRecord(e0));
        for (int l = 0; l < LPS; ++l) {
        if (kind == 0)
          copy_kernel<UNROLL><<<blocks, THREADS>>>((const int4*)a, (int4*)b, n_vec);
        else if (kind == 1)
          read_kernel<UNROLL><<<blocks, THREADS>>>((const int4*)a, n_vec, d_acc);
        else if (kind == 2)
          CK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice));
        else if (kind == 3)
          readblk_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a, n_vec, chunk_vec, d_acc);
        else
          readseg_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a, n_vec, d_acc);
        }
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms[it], e0, e1));
      }
      CK(cudaGetLastError());
      const double moved = (kind == 0 || kind == 2 ? 2.0 : 1.0) * bytes * LPS;  // read-only kinds: reads
      const float mn = *std::min_element(ms.begin(), ms.end());
      const float mx = *std::max_element(ms.begin(), ms.end());
      const char* names[] = {"copy", "read", "d2d", "readblk", "readseg"};
      printf("%llu,%s,%.1f,%.1f,%.1f\n", (unsigned long long)mib, names[kind],
             moved / (median(ms) * 1e-3) / 1e9, moved / (mx * 1e-3) / 1e9,
             moved / (mn * 1e-3) / 1e9);
    }
  }
  printf("# correctness: copy, read, readblk, readseg PASS at all sizes\n");
  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(d_acc));
  return 0;
}

static int sweep_main(int argc, char** argv) {
  const double total_gib = argc > 2 ? atof(argv[2]) : 13.0;
  const uint64_t win_mib = argc > 3 ? strtoull(argv[3], nullptr, 10) : 64;
  const int iters = argc > 4 ? atoi(argv[4]) : 10;
  constexpr int LOADS = 2, THREADS = 256;
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const uint64_t win = win_mib << 20;
  const uint64_t nwin = (uint64_t)(total_gib * (1ull << 30)) / win;
  const uint64_t bytes = nwin * win;
  uint32_t* a;
  unsigned long long* d_acc;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&d_acc, 8));
  fill_kernel<<<sms * 8, THREADS>>>(a, bytes / 4);
  CK(cudaGetLastError());
  // correctness: whole-buffer sum through the windows equals the closed form
  CK(cudaMemset(d_acc, 0, 8));
  const uint64_t wvec = win / 16, chunk_vec = 32 * 1024 / 16;
  const int blocks = sms * 8;
  for (uint64_t w = 0; w < nwin; ++w)
    readblk_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a + w * wvec, wvec, chunk_vec, d_acc);
  unsigned long long got = 0, want = 0;
  CK(cudaMemcpy(&got, d_acc, 8, cudaMemcpyDeviceToHost));
  for (uint64_t i = 0; i < bytes / 4; ++i) want += pattern(i);
  if (got != want) { fprintf(stderr, "FAIL sweep sum got=%llu want=%llu\n", got, want); return 4; }
  // warm-up ~1 s
  auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.0) {
    for (uint64_t w = 0; w < nwin; ++w)
      readblk_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a + w * wvec, wvec, chunk_vec, d_acc);
    CK(cudaDeviceSynchronize());
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<float> ms(iters);
  for (int it = 0; it < iters; ++it) {
    CK(cudaEventRecord(e0));
    for (uint64_t w = 0; w < nwin; ++w)
      readblk_kernel<LOADS><<<blocks, THREADS>>>((const int4*)a + w * wvec, wvec, chunk_vec, d_acc);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms[it], e0, e1));
  }
  CK(cudaGetLastError());
  const float mn = *std::min_element(ms.begin(), ms.end()), mx = *std::max_element(ms.begin(), ms.end());
  printf("sweep,%.2f,%llu,%llu,%.1f,%.1f,%.1f\n", bytes / 1e9, (unsigned long long)win_mib,
         (unsigned long long)nwin, bytes / (median(ms) * 1e-3) / 1e9, bytes / (mx * 1e-3) / 1e9,
         bytes / (mn * 1e-3) / 1e9);
  CK(cudaFree(a));
  CK(cudaFree(d_acc));
  return 0;
}
