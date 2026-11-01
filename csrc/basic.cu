// Small decode kernels: embedding gather, RMSNorm1p (+ fused residual add), BF16 GEMV for 1..8
// activation rows, SwiGLU, per-row argmax, and the device-side bookkeeping of a step
// (speculative accept / commit) so that a whole decode or verify cycle is one CUDA graph.
//
// Exactness rule for multi-row (verify) passes: every kernel computes row r of an n-row pass with
// exactly the same fp32 operation order as a 1-row pass. A verify pass therefore reproduces plain
// decode bit for bit, and greedy speculative output equals greedy output.
#include "common.cuh"
#include "ops.h"

namespace qe {

// ---------------------------------------------------------------------------------------------
// Embedding: out[r, :] = table[tok[r]] for r < n.
// ---------------------------------------------------------------------------------------------
__global__ void embed_kernel(const bf16* __restrict__ table, const int* __restrict__ tok,
                             bf16* __restrict__ out, int H, int V, long ldo) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y;
  const int t = tok[r];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H) return;
  // Out-of-range token ids (corrupt input) produce zeros instead of reading out of bounds.
  out[r * ldo + i] = (t >= 0 && t < V) ? table[(size_t)t * H + i] : f2b(0.f);
}

// ---------------------------------------------------------------------------------------------
// RMSNorm1p, one block per row:
//   y = x_f32 * rsqrt(mean(x_f32^2) + eps) * (1 + w_f32), one rounding to bf16.
// With `delta` != nullptr the residual add is fused first: x = bf16(x + delta) is written back
// to `x` (the residual stream), then normalized into `out`.
// ---------------------------------------------------------------------------------------------
__global__ void rmsnorm1p_kernel(bf16* __restrict__ x, const bf16* __restrict__ delta,
                                 const bf16* __restrict__ w, bf16* __restrict__ out, int H,
                                 long ldx, long ldd, long ldo, float eps) {
  pdl_trigger();
  pdl_wait();
  __shared__ float red[32];
  bf16* xr = x + blockIdx.x * ldx;
  bf16* orow = out + blockIdx.x * ldo;
  float ss = 0.f;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    float v = b2f(xr[c]);
    if (delta) {
      v = rb(v + b2f(delta[blockIdx.x * ldd + c]));  // torch: residual + x rounds to bf16
      xr[c] = f2b(v);  // same thread re-reads this element below: no cross-thread hazard
    }
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + eps);
  for (int c = threadIdx.x; c < H; c += blockDim.x)
    orow[c] = f2b(b2f(xr[c]) * inv * (1.0f + b2f(w[c])));
}

// Vectorized single-pass variant (H % 8 == 0, H / 8 <= 1024 threads, 16-byte aligned rows): each
// thread keeps its 8 elements in registers, so global memory is read once. Same fp32 math and
// per-element rounding as rmsnorm1p_kernel; only the order of the sum of squares differs.
__global__ void rmsnorm1p_vec_kernel(bf16* __restrict__ x, const bf16* __restrict__ delta,
                                     const bf16* __restrict__ w, bf16* __restrict__ out, int H,
                                     long ldx, long ldd, long ldo, float eps) {
  pdl_trigger();
  pdl_wait();
  __shared__ float red[32];
  const int c8 = threadIdx.x;  // owns elements 8*c8 .. 8*c8+7
  const bool act = c8 * 8 < H;
  uint4* xr = reinterpret_cast<uint4*>(x + blockIdx.x * ldx) + c8;
  float v[8];
  float ss = 0.f;
  if (act) {
    unpack8(*xr, v);
    if (delta) {
      float dd[8];
      unpack8(reinterpret_cast<const uint4*>(delta + blockIdx.x * ldd)[c8], dd);
      uint4 pk;
      __nv_bfloat162* ph = reinterpret_cast<__nv_bfloat162*>(&pk);
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = rb(v[i] + dd[i]);  // residual + x rounds to bf16
#pragma unroll
      for (int i = 0; i < 4; ++i) ph[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
      *xr = pk;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(v[i], v[i], ss);
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + eps);
  if (act) {
    float wf[8];
    unpack8(reinterpret_cast<const uint4*>(w)[c8], wf);
    uint4 pk;
    __nv_bfloat162* ph = reinterpret_cast<__nv_bfloat162*>(&pk);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ph[i] = __floats2bfloat162_rn(v[2 * i] * inv * (1.0f + wf[2 * i]), v[2 * i + 1] * inv * (1.0f + wf[2 * i + 1]));
    reinterpret_cast<uint4*>(out + blockIdx.x * ldo)[c8] = pk;
  }
}

// ---------------------------------------------------------------------------------------------
// BF16 GEMV for NR = 1..8 activation rows: out[r, n] = sum_k W[n,k] x[r,k]. One warp per weight
// row; each 16-byte weight load is reused for all NR activation rows (the weights are read once,
// which is what makes a verify pass cheap). Activations are read through the read-only cache.
// Per lane the k order is the same for every NR, so row r is bit-identical to an NR = 1 call.
// Requires K % 8 == 0 and 16-byte aligned rows (checked on the host).
// ---------------------------------------------------------------------------------------------
constexpr int GEMV_WARPS = 8;

template <int NR>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
gemv_bf16_kernel(const bf16* __restrict__ W, const bf16* __restrict__ x, bf16* __restrict__ out,
                 int N, int K, long ldx, long ldo) {
  pdl_trigger();
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
  if (row >= N) return;
  const int K8 = K >> 3;
  const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * K);
  float acc[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) acc[r] = 0.f;
  int k = lane;
  // 4 independent 16-byte weight loads in flight per lane to hide DRAM latency
  for (; k + 96 < K8; k += 128) {
    const uint4 a[4] = {__ldg(w4 + k), __ldg(w4 + k + 32), __ldg(w4 + k + 64), __ldg(w4 + k + 96)};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      float wf[8];
      unpack8(a[u], wf);
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        float xf[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(x + r * ldx) + k + 32 * u), xf);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r] = fmaf(wf[j], xf[j], acc[r]);
      }
    }
  }
  for (; k < K8; k += 32) {  // tail (masked by the loop bound)
    float wf[8];
    unpack8(__ldg(w4 + k), wf);
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      float xf[8];
      unpack8(__ldg(reinterpret_cast<const uint4*>(x + r * ldx) + k), xf);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[r] = fmaf(wf[j], xf[j], acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) out[r * ldo + row] = f2b(s);
  }
}

// ---------------------------------------------------------------------------------------------
// SwiGLU, per row: gu = [gate(I) | up(I)]; out = bf16(bf16(silu(g)) * u).
// ---------------------------------------------------------------------------------------------
__global__ void swiglu_kernel(const bf16* __restrict__ gu, bf16* __restrict__ out, int I,
                              long ldg, long ldo) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= I) return;
  const bf16* g = gu + r * ldg;
  const float s = rb(silu_f(b2f(g[i])));  // torch silu on a bf16 tensor rounds its output
  out[r * ldo + i] = f2b(s * b2f(g[I + i]));
}

// ---------------------------------------------------------------------------------------------
// Argmax with torch semantics (NaN counts as the maximum, ties go to the lowest index), per row.
// Pass 1: grid (nb, rows), each block reduces a strided slice. Pass 2: one block per row.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) {
  const bool an = isnan(a), bn = isnan(b);
  if (an != bn) return an;
  if (an) return ia < ib;
  if (a != b) return a > b;
  return ia < ib;
}

__device__ __forceinline__ void block_argmax(float& v, int& i, float* sv, int* si) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o);
    int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) { v = ov; i = oi; }
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if (lane == 0) { sv[w] = v; si[w] = i; }
  __syncthreads();
  if (w == 0) {
    v = lane < nw ? sv[lane] : -INFINITY;
    i = lane < nw ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, v, o);
      int oi = __shfl_xor_sync(0xffffffffu, i, o);
      if (better(ov, oi, v, i)) { v = ov; i = oi; }
    }
  }
}

__global__ void argmax_pass1(const bf16* __restrict__ x, int n, long ldx, float* pv, int* pi) {
  pdl_trigger();
  pdl_wait();
  __shared__ float sv[32];
  __shared__ int si[32];
  const int r = blockIdx.y, nb = gridDim.x;
  const bf16* xr = x + r * ldx;
  float v = -INFINITY;
  int i = 0x7fffffff;  // sentinel index loses every tie
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += nb * blockDim.x) {
    float xv = b2f(xr[k]);
    if (better(xv, k, v, i)) { v = xv; i = k; }
  }
  block_argmax(v, i, sv, si);
  if (threadIdx.x == 0) { pv[r * nb + blockIdx.x] = v; pi[r * nb + blockIdx.x] = i; }
}

// idmap (optional): out = idmap[argmax] (pruned draft vocabulary -> vocabulary id)
__global__ void argmax_pass2(const float* pv, const int* pi, int nb, int* out, const int* idmap) {
  pdl_trigger();
  pdl_wait();
  __shared__ float sv[32];
  __shared__ int si[32];
  const int r = blockIdx.x;
  float v = -INFINITY;
  int i = 0x7fffffff;
  for (int k = threadIdx.x; k < nb; k += blockDim.x)
    if (better(pv[r * nb + k], pi[r * nb + k], v, i)) { v = pv[r * nb + k]; i = pi[r * nb + k]; }
  block_argmax(v, i, sv, si);
  if (threadIdx.x == 0) {
    const int a = (i == 0x7fffffff) ? 0 : i;  // all -inf from sentinel → index 0
    out[r] = idmap ? idmap[a] : a;
  }
}

// ---------------------------------------------------------------------------------------------
// Step bookkeeping. Device scalars (int32): ctl[0] = pos (position of in_tok[0]),
// ctl[1] = pending (GDN rows staged by the previous pass, replayed by the next), ctl[2] = mode
// (0 = prefill: in_tok are given tokens; 1 = decode/verify: in_tok = [last token, drafts...]),
// ctl[3] = m (index of the last accepted row of this pass), ctl[4] = cycles, ctl[5] = accepted
// drafts (both statistics).
//
// accept: decide how many rows of this pass are kept.
//   verify: m = longest prefix with in_tok[i+1] == am[i] (draft i+1 matches the target), so rows
//           0..m are correct and am[m] is the bonus token.
//   prefill: m = n - 1 (all rows are given tokens).
//   next_tok[r] = the true token after row r (MTP input): am[r] when verifying (equal to the
//   draft for accepted rows), else in_tok[r+1] (in_tok[n] < 0 means "unknown": use am).
//   Generated tokens are logged at their positions in `log`.
// ---------------------------------------------------------------------------------------------
__global__ void accept_kernel(const int* in_tok, const int* am, int n, int* ctl, int* next_tok,
                              int* log, int max_len) {
  pdl_trigger();
  pdl_wait();
  const int P = ctl[0], mode = ctl[2];
  int m = n - 1;
  if (mode == 1) {
    m = 0;
    while (m < n - 1 && in_tok[m + 1] == am[m]) ++m;
    for (int i = 0; i <= m; ++i)
      if (P + 1 + i < max_len) log[P + 1 + i] = am[i];
    ctl[4] += 1;
    ctl[5] += m;
    ctl[16 + m] += 1;  // histogram of accepted drafts per cycle (acceptance per depth)
  } else if (in_tok[n] < 0 && P + n < max_len) {
    log[P + n] = am[n - 1];  // last prompt chunk: its last row produces the first new token
  }
  for (int r = 0; r < n; ++r) {
    if (mode == 1) next_tok[r] = am[r];
    else if (r + 1 < n) next_tok[r] = in_tok[r + 1];
    else next_tok[r] = in_tok[n] >= 0 ? in_tok[n] : am[n - 1];
  }
  ctl[3] = m;
}

// commit (end of the graph): advance pos past the kept rows, mark them pending for the GDN
// replay, and set up the next pass input [bonus, drafts...].
__global__ void commit_kernel(int* ctl, int* in_tok, const int* am, const int* drafts, int k) {
  pdl_trigger();
  pdl_wait();
  const int m = ctl[3];
  ctl[0] += m + 1;
  ctl[1] = m + 1;
  in_tok[0] = am[m];
  for (int j = 0; j < k; ++j) in_tok[1 + j] = drafts[j];
}

// mpos = pos + m + 1 + off (MTP chain position) or pos (MTP rows pass, off < 0); copy row m of
// src [n, H] (row stride lds) into dst [H] for the chain input hidden, and the row-m draft token.
__global__ void mtp_select_kernel(const int* ctl, int* mpos, int off, const bf16* src, long lds,
                                  bf16* dst, int H, const int* mtp_am, int* draft0) {
  pdl_trigger();
  pdl_wait();
  const int m = ctl[3];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    mpos[0] = off < 0 ? ctl[0] : ctl[0] + m + 1 + off;
    if (draft0) draft0[0] = mtp_am[m];
  }
  if (src)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x)
      dst[i] = src[m * lds + i];
}

}  // namespace qe

// ------------------------------------- host launchers ----------------------------------------
using namespace qe;

void set_pdl_trigger_basic(int v) { cudaMemcpyToSymbol(g_pdl_trigger, &v, sizeof(int)); }
void set_pdl_trigger_gdn(int v);
void set_pdl_trigger_attn(int v);
void set_pdl_launch(bool on) { pdl_launch_flag() = on; }
void set_pdl_trigger(bool on) {
  const int v = on ? 1 : 0;
  set_pdl_trigger_basic(v);
  set_pdl_trigger_gdn(v);
  set_pdl_trigger_attn(v);
}

void launch_embed(const void* table, const int* tok, void* out, int n, int H, int V, long ldo,
                  cudaStream_t s) {
  qe_launch(embed_kernel, dim3(dim3((H + 255) / 256, n)), dim3(256), 0, s, (const bf16*)table, tok, (bf16*)out, H, V, ldo);
}

void launch_rmsnorm1p(void* x, const void* delta, const void* w, void* out, int rows, int H,
                      long ldx, long ldd, long ldo, float eps, cudaStream_t s) {
  const bool vec = H % 8 == 0 && H / 8 <= 1024 && ldx % 8 == 0 && ldo % 8 == 0 && (!delta || ldd % 8 == 0) &&
                   ((uintptr_t)x % 16 == 0) && ((uintptr_t)out % 16 == 0) && ((uintptr_t)w % 16 == 0) &&
                   (!delta || (uintptr_t)delta % 16 == 0);
  if (vec) {
    const int threads = ((H / 8 + 31) / 32) * 32;
    qe_launch(rmsnorm1p_vec_kernel, dim3(rows), dim3(threads), 0, s, (bf16*)x, (const bf16*)delta, (const bf16*)w, (bf16*)out, H,
                                                  ldx, ldd, ldo, eps);
    return;
  }
  int threads = H >= 1024 ? 1024 : ((H + 31) / 32) * 32;
  qe_launch(rmsnorm1p_kernel, dim3(rows), dim3(threads), 0, s, (bf16*)x, (const bf16*)delta, (const bf16*)w,
                                            (bf16*)out, H, ldx, ldd, ldo, eps);
}

void launch_gemv_bf16(const void* W, const void* x, void* out, int nr, int N, int K, long ldx,
                      long ldo, cudaStream_t s) {
  const dim3 grid((N + GEMV_WARPS - 1) / GEMV_WARPS), block(GEMV_WARPS * 32);
  const bf16 *w = (const bf16*)W, *xx = (const bf16*)x;
  bf16* o = (bf16*)out;
  switch (nr) {
#define QE_CASE(R) case R: qe_launch(gemv_bf16_kernel<R>, dim3(grid), dim3(block), 0, s, w, xx, o, N, K, ldx, ldo); break;
    QE_CASE(1) QE_CASE(2) QE_CASE(3) QE_CASE(4) QE_CASE(5) QE_CASE(6) QE_CASE(7) QE_CASE(8)
#undef QE_CASE
  }
}

void launch_swiglu(const void* gu, void* out, int n, int I, long ldg, long ldo, cudaStream_t s) {
  qe_launch(swiglu_kernel, dim3(dim3((I + 255) / 256, n)), dim3(256), 0, s, (const bf16*)gu, (bf16*)out, I, ldg, ldo);
}

void launch_argmax(const void* x, int rows, int n, long ldx, float* pv, int* pi, int nb, int* out,
                   const int* idmap, cudaStream_t s) {
  qe_launch(argmax_pass1, dim3(dim3(nb, rows)), dim3(256), 0, s, (const bf16*)x, n, ldx, pv, pi);
  qe_launch(argmax_pass2, dim3(rows), dim3(1024), 0, s, pv, pi, nb, out, idmap);
}

void launch_accept(const int* in_tok, const int* am, int n, int* ctl, int* next_tok, int* log,
                   int max_len, cudaStream_t s) {
  qe_launch(accept_kernel, dim3(1), dim3(1), 0, s, in_tok, am, n, ctl, next_tok, log, max_len);
}

void launch_commit(int* ctl, int* in_tok, const int* am, const int* drafts, int k, cudaStream_t s) {
  qe_launch(commit_kernel, dim3(1), dim3(1), 0, s, ctl, in_tok, am, drafts, k);
}

void launch_mtp_select(const int* ctl, int* mpos, int off, const void* src, long lds, void* dst,
                       int H, const int* mtp_am, int* draft0, cudaStream_t s) {
  qe_launch(mtp_select_kernel, dim3(src ? (H + 255) / 256 : 1), dim3(256), 0, s, ctl, mpos, off, (const bf16*)src, lds,
                                                              (bf16*)dst, H, mtp_am, draft0);
}

// ---- L2 persistence controls (experiment: keep MTP draft weights in L2 across draft steps) ----
void l2_persist_limit(size_t bytes) {
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
  cudaCtxResetPersistingL2Cache();  // drop lines persisted under a previous setting
}
// Access-policy window on the current stream (captured into kernel nodes during graph capture):
// accesses to [base, base + bytes) persist in the set-aside L2; bytes = 0 clears the window.
void stream_access_window(uintptr_t base, size_t bytes, float hit_ratio, cudaStream_t s) {
  cudaStreamAttrValue v = {};
  v.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(base);
  v.accessPolicyWindow.num_bytes = bytes;
  v.accessPolicyWindow.hitRatio = bytes ? hit_ratio : 0.f;
  v.accessPolicyWindow.hitProp = bytes ? cudaAccessPropertyPersisting : cudaAccessPropertyNormal;
  v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v);
}
