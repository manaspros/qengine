// Gated DeltaNet decode / verify step for n = 1..8 new tokens, one kernel per
// layer, one block per value head, with "lazy rollback":
//
//   committed state S (fp32, in global memory) is the state after position pos - pending.
//   1. replay: the `pending` rows staged by the previous pass (rows that the pass kept) are
//      applied to S from their staged (k, v, decay, beta); S is written back once.
//   2. new rows t = 0..n-1 at positions pos + t run from registers: conv1d + SiLU, l2norm,
//      gating, delta rule, gated RMSNorm output. Their (k, v, decay, beta) are staged; S is not
//      written. The next pass replays only the rows that were accepted.
// Plain decode is n = 1 with pending = 1. The per-token update is the same code in the replay
// and the new-row path, so the committed state equals plain sequential decode bit for bit.
//
// Per value head j (qk head i = j / (Hv/Hk), repeat_interleave mapping), per token:
//   conv: out_c = sum_w conv_w[c][w] * [x_{t-3}, x_{t-2}, x_{t-1}, x_t][w], bf16, then SiLU, bf16
//   q = l2norm(q)/sqrt(dk), k = l2norm(k) (fp32); beta = bf16(sigmoid(b_j));
//   decay = exp(-exp(A_log_j) * softplus(a_j + dt_bias_j))
//   S = decay S; kv = S^T k; delta = beta (v - kv); S += k delta^T; o = S^T q
//   out = bf16(bf16(w * bf16(norm(o))) * silu(z_j))
//
// Conv history is a ring of 16 slots [16][C] keyed by absolute position (slot = pos & 15). A pass
// reads history positions < pos from the ring (never written in this pass) and earlier rows of
// this pass from its own input rows, and writes the slots of positions pos..pos+n-1. With n <= 8
// the read slots (pos-3..pos-1) and written slots never alias, so blocks sharing a qk head need no
// synchronization. Rejected rows leave stale slots at positions >= the new pos; the next pass
// rewrites them before anyone reads them. Only the block with j % r == 0 writes q/k channels.
#include "common.cuh"
#include "ops.h"

namespace qe {

constexpr int DK = 128, DV = 128, RING = 16, MAXN = 8;

// The 128 state rows of column d are split over Q thread groups (Q = 2, 4, 8): thread (d, h)
// owns rows h*RP .. h*RP+RP-1 with RP = 128/Q; partial dot products are summed over the Q
// groups in a fixed order. Threads per block = 128 * Q.
template <int Q>
__device__ __forceinline__ float sum_q(float (*red)[DV], int d) {
  float s = red[0][d];
#pragma unroll
  for (int i = 1; i < Q; ++i) s += red[i][d];
  return s;
}

// One delta-rule state update for the column this thread owns.
// Identical instruction sequence for replayed and new rows.
template <int Q>
__device__ __forceinline__ void gdn_update(float (&sreg)[128 / Q], const float* ks, const float* vs,
                                           float decay, float beta, float (*red2)[DV], int d, int h) {
  constexpr int RP = 128 / Q;
  float part = 0.f;
#pragma unroll
  for (int ii = 0; ii < RP; ++ii) {
    sreg[ii] = sreg[ii] * decay;
    part = fmaf(sreg[ii], ks[h * RP + ii], part);
  }
  red2[h][d] = part;
  __syncthreads();
  const float kv = sum_q<Q>(red2, d);
  const float delta = (vs[d] - kv) * beta;
  __syncthreads();  // everyone has read red2 before it is reused
#pragma unroll
  for (int ii = 0; ii < RP; ++ii) sreg[ii] = fmaf(ks[h * RP + ii], delta, sreg[ii]);
}

template <int Q>
__global__ void __launch_bounds__(128 * Q)
gdn_step_n_kernel(const bf16* __restrict__ qkv,      // rows [n][ldp]: raw in_proj_qkv output (C)
                  const bf16* __restrict__ z,        // rows [n][ldp]: Hv*DV
                  const bf16* __restrict__ b,        // rows [n][ldp]: Hv
                  const bf16* __restrict__ a,        // rows [n][ldp]: Hv
                  long ldp,
                  bf16* __restrict__ ring,           // [16][C]
                  const bf16* __restrict__ conv_w,   // [C][4]
                  const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,  // [Hv]
                  const bf16* __restrict__ norm_w,   // [DV]
                  float* __restrict__ S,             // [Hv][DK][DV] committed state
                  float* __restrict__ st_k,          // [8][Hv][DK] staged normalized k
                  float* __restrict__ st_v,          // [8][Hv][DV] staged v
                  float* __restrict__ st_g,          // [8][Hv][2]  staged (decay, beta)
                  const int* __restrict__ ctl,       // ctl[0] = pos, ctl[1] = pending
                  bf16* __restrict__ out, long ldo,  // rows [n][ldo]: Hv*DV
                  int n, int Hk, int Hv, float eps, int early) {
  // PDL: everything up to pdl_wait() reads only data this kernel wrote in an earlier graph replay
  // (state, staged rows, conv ring) or constants, so it overlaps the producing in_proj GEMV.
  // The projection rows (qkv, z, b, a) are read only after pdl_wait().
  pdl_trigger();
  // bit 1: phase-parallel rows (below), used for n > 1 only (at n = 1 the per-row path is ~1% faster
  // end to end); both paths give bit-identical rows
  const bool phases = early & 2;
  early &= 1;
  if (!early) pdl_wait();  // A/B switch bit 0: 0 waits at entry (no overlap, no prefetch)
  __shared__ float qs[DK], ks[DK], vs[DV];
  constexpr int RP = 128 / Q, NT = 128 * Q;
  const bool PREF = NT >= 3 * DK && early;  // one conv channel per thread: prefetch weights/history
  __shared__ float red2[Q][DV];  // partial sums of the Q row groups
  __shared__ float red[32];
  __shared__ float sh_beta, sh_decay;

  const int j = blockIdx.x;  // value head
  const int r = Hv / Hk;
  const int i = j / r;       // qk head (repeat_interleave)
  const int C = 2 * Hk * DK + Hv * DV;
  const int P = ctl[0], pend = ctl[1];
  const int tid = threadIdx.x, d = tid & (DV - 1), h = tid >> 7;
  float* Sj = S + (size_t)j * DK * DV;

  float sreg[RP];  // this thread's column d, rows h*RP.. of S_j (coalesced along d)
#pragma unroll
  for (int ii = 0; ii < RP; ++ii) sreg[ii] = Sj[(h * RP + ii) * DV + d];

  // ---- 1. replay the rows kept by the previous pass ----
  for (int t = 0; t < pend; ++t) {
    if (tid < DK) ks[tid] = st_k[((size_t)t * Hv + j) * DK + tid];
    else if (tid < DK + DV) vs[tid - DK] = st_v[((size_t)t * Hv + j) * DV + tid - DK];
    __syncthreads();
    gdn_update<Q>(sreg, ks, vs, st_g[(t * Hv + j) * 2], st_g[(t * Hv + j) * 2 + 1], red2, d, h);
    __syncthreads();  // ks/vs are overwritten next iteration
  }
  if (pend > 0) {
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) Sj[(h * RP + ii) * DV + d] = sreg[ii];
  }
  // prefetch (independent of the producer): this thread's conv channel weights and the committed
  // history at positions P-3..P-1, gate parameters, norm weight
  float pcw[4] = {0.f, 0.f, 0.f, 0.f}, phist[3] = {0.f, 0.f, 0.f};
  int pch = 0;
  if (PREF) {
    if (tid < 3 * DK) {
      const int part = tid / DK, c = tid % DK;
      pch = part == 0 ? i * DK + c : part == 1 ? Hk * DK + i * DK + c : 2 * Hk * DK + j * DV + c;
#pragma unroll
      for (int w = 0; w < 4; ++w) pcw[w] = b2f(conv_w[pch * 4 + w]);
#pragma unroll
      for (int w = 0; w < 3; ++w) {
        const int q = P - 3 + w;
        phist[w] = q >= 0 ? b2f(ring[(size_t)(q & (RING - 1)) * C + pch]) : 0.f;
      }
    }
  }
  const float pAlog = b2f(A_log[j]), pdtb = b2f(dt_bias[j]);
  const float pnw = tid < DV ? b2f(norm_w[tid]) : 0.f;
  if (early) pdl_wait();  // the producer's outputs (qkv, z, b, a) are visible from here on

  // ---- 2. new rows, state kept in registers ----
  if (NT >= 3 * DK + MAXN && NT / 32 >= 2 * MAXN && phases) {
    // Phases over all rows at once; only the recurrence itself is serial per row.
    __shared__ float qa[MAXN][DK], ka[MAXN][DK], va[MAXN][DV], ga[MAXN][2];
    float zr[MAXN];
    // A. conv + SiLU for every row (thread tid < 384 owns one channel), ring writes, gates, z
    if (tid < 3 * DK) {
      const int part = tid / DK, c = tid % DK;
      const int ch = PREF ? pch : part == 0 ? i * DK + c : part == 1 ? Hk * DK + i * DK + c : 2 * Hk * DK + j * DV + c;
      float cw[4], h3[3];
#pragma unroll
      for (int w = 0; w < 4; ++w) cw[w] = PREF ? pcw[w] : b2f(conv_w[ch * 4 + w]);
#pragma unroll
      for (int w = 0; w < 3; ++w) {
        const int q = P - 3 + w;
        h3[w] = PREF ? phist[w] : q >= 0 ? b2f(ring[(size_t)(q & (RING - 1)) * C + ch]) : 0.f;
      }
      float xv[MAXN];
      bf16 xb[MAXN];
#pragma unroll
      for (int t = 0; t < MAXN; ++t)
        if (t < n) { xb[t] = qkv[t * ldp + ch]; xv[t] = b2f(xb[t]); }
#pragma unroll
      for (int t = 0; t < MAXN; ++t) {
        if (t >= n) break;
        // history x at positions P+t-3 .. P+t-1: earlier rows of this pass, else committed ring
        const float x0 = t >= 3 ? xv[t - 3] : h3[t];
        const float x1 = t >= 2 ? xv[t - 2] : h3[t + 1];
        const float x2 = t >= 1 ? xv[t - 1] : h3[2];
        float acc = cw[0] * x0;  // HF buffer order [x_{t-3}, x_{t-2}, x_{t-1}, x_t]
        acc = fmaf(cw[1], x1, acc);
        acc = fmaf(cw[2], x2, acc);
        acc = fmaf(cw[3], xv[t], acc);
        const float y = rb(silu_f(rb(acc)));  // conv output rounds to bf16, silu output rounds again
        if (part == 0) qa[t][c] = y; else if (part == 1) ka[t][c] = y; else va[t][c] = y;
        // v channels are unique to this block; q/k channels are shared by r blocks: one writes
        if (part == 2 || (j % r) == 0) ring[(size_t)((P + t) & (RING - 1)) * C + ch] = xb[t];
      }
    } else if (tid < 3 * DK + MAXN) {
      const int t = tid - 3 * DK;
      if (t < n) {
        ga[t][1] = rb(sigmoid_f(b2f(b[t * ldp + j])));  // b.sigmoid() on a bf16 tensor -> bf16
        const float x = b2f(a[t * ldp + j]) + pdtb;
        const float sp = x > 20.f ? x : log1pf(expf(x));  // F.softplus(beta=1, threshold=20)
        ga[t][0] = expf(-expf(pAlog) * sp);                // exp(g)
      }
    }
    if (tid < DV) {
#pragma unroll
      for (int t = 0; t < MAXN; ++t)
        if (t < n) zr[t] = b2f(z[t * ldp + j * DV + tid]);
    }
    __syncthreads();
    {  // B. l2norm of q_t and k_t for all rows: warp w normalizes vector w (q of row w/2 or k)
      const int lane = tid & 31, w = tid >> 5;
      if (w < 2 * n) {
        float* vec = (w & 1) ? ka[w >> 1] : qa[w >> 1];
        // same summation order as the per-row path: butterfly sum of each 32-element slice, then
        // the 4 slice sums in order (bit-identical results)
        float e[4], part[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { e[u] = vec[lane + 32 * u]; part[u] = warp_sum(e[u] * e[u]); }
        const float ss = part[0] + part[1] + part[2] + part[3];
        const float inv = rsqrtf(ss + 1e-6f);
#pragma unroll
        for (int u = 0; u < 4; ++u)  // HF: l2norm, then q / sqrt(dk)
          vec[lane + 32 * u] = (w & 1) ? e[u] * inv : (e[u] * inv) / sqrtf((float)DK);
      }
    }
    __syncthreads();
    // C. stage every row for a possible replay by the next pass (own region of this block)
    for (int t = 0; t < n; ++t) {
      if (tid < DK) st_k[((size_t)t * Hv + j) * DK + tid] = ka[t][tid];
      else if (tid < DK + DV) st_v[((size_t)t * Hv + j) * DV + tid - DK] = va[t][tid - DK];
      else if (tid == DK + DV) { st_g[(t * Hv + j) * 2] = ga[t][0]; st_g[(t * Hv + j) * 2 + 1] = ga[t][1]; }
    }
    // D. the recurrence, row by row
#pragma unroll
    for (int t = 0; t < MAXN; ++t) {
      if (t >= n) break;
      gdn_update<Q>(sreg, ka[t], va[t], ga[t][0], ga[t][1], red2, d, h);
      float part = 0.f;
#pragma unroll
      for (int ii = 0; ii < RP; ++ii) part = fmaf(sreg[ii], qa[t][h * RP + ii], part);
      red2[h][d] = part;
      __syncthreads();
      float o = 0.f;  // gated RMSNorm over the 128 outputs (threads 0..127)
      if (tid < DV) o = rb(sum_q<Q>(red2, tid));  // core_attn_out cast to bf16
      const float ss = block_sum(tid < DV ? o * o : 0.f, red);
      if (tid < DV) {
        const float nn = rb(o * rsqrtf(ss / DV + eps));  // norm(x).to(bf16)
        const float y = rb(pnw * nn);                      // bf16 weight * bf16 -> bf16
        out[t * ldo + j * DV + tid] = f2b(y * silu_f(zr[t]));
      }
    }
    return;
  }
  for (int t = 0; t < n; ++t) {
    const bf16* xrow = qkv + t * ldp;
    for (int idx = tid; idx < 3 * DK; idx += NT) {
      const int part = idx / DK, c = idx % DK;  // part 0 = q, 1 = k, 2 = v
      const int ch = PREF ? pch : part == 0 ? i * DK + c : part == 1 ? Hk * DK + i * DK + c : 2 * Hk * DK + j * DV + c;
      float hist[3];  // x at positions P+t-3, P+t-2, P+t-1
#pragma unroll
      for (int w = 0; w < 3; ++w) {
        const int back = 3 - w, tt = t - back, q = P + tt;
        hist[w] = tt >= 0 ? b2f(qkv[tt * ldp + ch])                         // earlier row of this pass
                : PREF ? phist[3 + tt]                                         // prefetched history
                : q >= 0 ? b2f(ring[(size_t)(q & (RING - 1)) * C + ch])       // committed history
                         : 0.f;                                                // before the sequence
      }
      const bf16 xt = xrow[ch];
      float cw[4];
#pragma unroll
      for (int w = 0; w < 4; ++w) cw[w] = PREF ? pcw[w] : b2f(conv_w[ch * 4 + w]);
      // HF buffer order [x_{t-3}, x_{t-2}, x_{t-1}, x_t] against weights w[0..3]
      float acc = cw[0] * hist[0];
      acc = fmaf(cw[1], hist[1], acc);
      acc = fmaf(cw[2], hist[2], acc);
      acc = fmaf(cw[3], b2f(xt), acc);
      const float y = rb(silu_f(rb(acc)));  // conv output rounds to bf16, silu output rounds again
      if (part == 0) qs[c] = y; else if (part == 1) ks[c] = y; else vs[c] = y;
      // v channels are unique to this block; q/k channels are shared by r blocks: one writes
      if (part == 2 || (j % r) == 0) ring[(size_t)((P + t) & (RING - 1)) * C + ch] = xt;
    }
    if (tid == 0) {
      sh_beta = rb(sigmoid_f(b2f(b[t * ldp + j])));  // b.sigmoid() on a bf16 tensor -> bf16
      const float x = b2f(a[t * ldp + j]) + pdtb;
      const float sp = x > 20.f ? x : log1pf(expf(x));  // F.softplus(beta=1, threshold=20)
      sh_decay = expf(-expf(pAlog) * sp);                // exp(g)
    }
    __syncthreads();
    {  // l2norm: threads 0..127 own q, 128..255 own k (4 warps each); others idle here
      const int lane = tid & 31, w = tid >> 5;
      const float v = tid < DK ? qs[tid] : tid < 2 * DK ? ks[tid - DK] : 0.f;
      const float ss = warp_sum(v * v);
      if (lane == 0 && w < 8) red[w] = ss;
      __syncthreads();
      const float sq = red[0] + red[1] + red[2] + red[3];
      const float sk = red[4] + red[5] + red[6] + red[7];
      __syncthreads();
      if (tid < DK) qs[tid] = (v * rsqrtf(sq + 1e-6f)) / sqrtf((float)DK);  // HF: l2norm, then / sqrt(dk)
      else if (tid < 2 * DK) ks[tid - DK] = v * rsqrtf(sk + 1e-6f);
    }
    __syncthreads();
    const float decay = sh_decay, beta = sh_beta;
    // stage this row for a possible replay by the next pass (own region of this block)
    if (tid < DK) st_k[((size_t)t * Hv + j) * DK + tid] = ks[tid];
    else if (tid < DK + DV) st_v[((size_t)t * Hv + j) * DV + tid - DK] = vs[tid - DK];
    if (tid == 0) { st_g[(t * Hv + j) * 2] = decay; st_g[(t * Hv + j) * 2 + 1] = beta; }
    gdn_update<Q>(sreg, ks, vs, decay, beta, red2, d, h);
    float part = 0.f;
#pragma unroll
    for (int ii = 0; ii < RP; ++ii) part = fmaf(sreg[ii], qs[h * RP + ii], part);
    red2[h][d] = part;
    __syncthreads();
    // gated RMSNorm over the 128 outputs (threads 0..127)
    float o = 0.f;
    if (tid < DV) o = rb(sum_q<Q>(red2, tid));  // core_attn_out cast to bf16
    const float ss = block_sum(tid < DV ? o * o : 0.f, red);
    if (tid < DV) {
      const float nn = rb(o * rsqrtf(ss / DV + eps));  // norm(x).to(bf16)
      const float y = rb(pnw * nn);                      // bf16 weight * bf16 -> bf16
      out[t * ldo + j * DV + tid] = f2b(y * silu_f(b2f(z[t * ldp + j * DV + tid])));
    }
    __syncthreads();  // qs/ks/vs/red2 reused by the next row
  }
}

}  // namespace qe

using namespace qe;

bool& gdn_early_flag() {
  static bool f = true;
  return f;
}
void set_gdn_early(bool on) { gdn_early_flag() = on; }
bool& gdn_phase_flag() {
  static bool f = true;
  return f;
}
void set_gdn_phases(bool on) { gdn_phase_flag() = on; }

void set_pdl_trigger_gdn(int v) { cudaMemcpyToSymbol(g_pdl_trigger, &v, sizeof(int)); }

void launch_gdn_step_n(const void* qkv, const void* z, const void* b, const void* a, long ldp,
                       void* ring, const void* conv_w, const void* A_log, const void* dt_bias,
                       const void* norm_w, float* S, float* st_k, float* st_v, float* st_g,
                       const int* ctl, void* out, long ldo, int n, int Hk, int Hv, float eps,
                       int q, cudaStream_t s) {
#define QE_GDN(Q) qe_launch(gdn_step_n_kernel<Q>, dim3(Hv), dim3(128 * Q), 0, s,                                      \
      (const bf16*)qkv, (const bf16*)z, (const bf16*)b, (const bf16*)a, ldp, (bf16*)ring,         \
      (const bf16*)conv_w, (const bf16*)A_log, (const bf16*)dt_bias, (const bf16*)norm_w, S, st_k, \
      st_v, st_g, ctl, (bf16*)out, ldo, n, Hk, Hv, eps, (gdn_early_flag() ? 1 : 0) | (gdn_phase_flag() && n > 1 ? 2 : 0))
  if (q == 8) QE_GDN(8); else if (q == 4) QE_GDN(4); else QE_GDN(2);
#undef QE_GDN
}
