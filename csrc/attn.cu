// Gated full attention, decode / verify for n = 1..8 rows. Row t is the token at
// position pos + t; it attends causally to cache entries [0, pos + t]. Three kernels per layer:
//   attn_prep    : q/k RMSNorm1p per head + partial RoPE (first rotary_dim dims, rotate-half),
//                  q -> q_out, k/v -> KV cache at position pos. Grid = Hq + Hkv blocks.
//   attn_split   : flash-decoding. Grid (Hq, NS); block s handles a contiguous slice of the
//                  positions [0, pos] and writes an unnormalized partial (o, max, sum).
//   attn_combine : merges the NS partials, rounds to bf16, multiplies by bf16(sigmoid(gate)).
// head_dim is fixed at 256 (both 0.8B and 27B): one thread per dim in prep, 8 dims per lane in
// the split kernel (a 256-dim bf16 row is exactly 32 lanes x 16 bytes).
#include "common.cuh"
#include "ops.h"

namespace qe {

constexpr int HD = 256;

__global__ void __launch_bounds__(HD)
attn_prep_kernel(const bf16* __restrict__ qg,      // rows [n][ldp]: [Hq][2*HD], per head [q | gate]
                 const bf16* __restrict__ k,       // rows [n][ldp]: [Hkv][HD]
                 const bf16* __restrict__ v,       // rows [n][ldp]: [Hkv][HD]
                 long ldp,
                 const bf16* __restrict__ qnw, const bf16* __restrict__ knw,  // [HD]
                 const float* __restrict__ inv_freq,  // [rd/2], built on the CPU like HF
                 const int* __restrict__ pos_ptr,
                 bf16* __restrict__ q_out,          // rows [n][Hq*HD]
                 bf16* __restrict__ kc, bf16* __restrict__ vc,  // [Hkv][max_len][HD]
                 int Hq, int max_len, int rd, float eps) {
  pdl_trigger();
  pdl_wait();
  __shared__ float xn[HD];
  __shared__ float red[32];
  const int hb = blockIdx.x, c = threadIdx.x;
  const bool is_q = hb < Hq;
  const int h = is_q ? hb : hb - Hq;
  const int t = blockIdx.y, pos = pos_ptr[0] + t;  // row t -> absolute position
  qg += t * ldp; k += t * ldp; v += t * ldp; q_out += (size_t)t * Hq * HD;
  const float x = b2f(is_q ? qg[(size_t)h * 2 * HD + c] : k[(size_t)h * HD + c]);
  const float inv = rsqrtf(block_sum(x * x, red) / HD + eps);
  // RMSNorm1p output is a bf16 tensor in HF
  xn[c] = rb(x * inv * (1.0f + b2f(is_q ? qnw[c] : knw[c])));
  __syncthreads();
  float y = xn[c];
  if (c < rd) {
    const int half = rd >> 1;
    const float f = (float)pos * inv_freq[c % half];  // fp32, then cos/sin cast to bf16
    const float cs = rb(cosf(f)), sn = rb(sinf(f));
    const float rot = c < half ? -xn[c + half] : xn[c - half];  // rotate_half
    y = rb(rb(xn[c] * cs) + rb(rot * sn));  // bf16 elementwise: every op rounds
  }
  if (is_q) {
    q_out[(size_t)h * HD + c] = f2b(y);
  } else if (pos < max_len) {  // masked: never write past the cache
    kc[((size_t)h * max_len + pos) * HD + c] = f2b(y);
    vc[((size_t)h * max_len + pos) * HD + c] = v[(size_t)h * HD + c];
  }
}

constexpr int SPLIT_WARPS = 8;

__global__ void __launch_bounds__(SPLIT_WARPS * 32)
attn_split_kernel(const bf16* __restrict__ q,                       // rows [n][Hq*HD]
                  const bf16* __restrict__ kc, const bf16* __restrict__ vc,  // [Hkv][max_len][HD]
                  const int* __restrict__ pos_ptr,
                  float* __restrict__ part_o,   // [n][Hq][NS][HD]
                  float* __restrict__ part_ml,  // [n][Hq][NS][2] (max, sum)
                  int Hq, int Hkv, int max_len, float scale) {
  pdl_trigger();
  pdl_wait();
  __shared__ float sm[SPLIT_WARPS], sl[SPLIT_WARPS];
  __shared__ float so[SPLIT_WARPS][HD];
  const int h = blockIdx.x, s = blockIdx.y, NS = gridDim.y;
  const int g = h / (Hq / Hkv);  // GQA: q head h reads kv head h / group
  const int t = blockIdx.z;
  q += (size_t)t * Hq * HD;
  part_o += (size_t)t * Hq * NS * HD;
  part_ml += (size_t)t * Hq * NS * 2;
  // valid cache entries for row t; same split boundaries as a 1-row pass at that position
  const int n = min(pos_ptr[0] + t + 1, max_len);
  const int chunk = (n + NS - 1) / NS;
  const int start = s * chunk, end = min(n, start + chunk);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;

  float qf[8];
  unpack8(reinterpret_cast<const uint4*>(q + (size_t)h * HD)[lane], qf);
  float m = -INFINITY, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const bf16* kb = kc + (size_t)g * max_len * HD;
  const bf16* vb = vc + (size_t)g * max_len * HD;
  for (int p = start + w; p < end; p += SPLIT_WARPS) {  // one warp per position
    float kf[8], vf[8];
    unpack8(reinterpret_cast<const uint4*>(kb + (size_t)p * HD)[lane], kf);
    float d = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) d = fmaf(qf[e], kf[e], d);
    const float sc = warp_sum(d) * scale;
    const float mn = fmaxf(m, sc);
    const float corr = expf(m - mn);  // m = -inf on the first position -> 0
    const float pe = expf(sc - mn);
    unpack8(reinterpret_cast<const uint4*>(vb + (size_t)p * HD)[lane], vf);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = fmaf(pe, vf[e], acc[e] * corr);
    l = l * corr + pe;
    m = mn;
  }
  // merge the 8 warps of this block
  if (lane == 0) { sm[w] = m; sl[w] = l; }
#pragma unroll
  for (int e = 0; e < 8; ++e) so[w][lane * 8 + e] = acc[e];
  __syncthreads();
  const int c = threadIdx.x;  // 256 threads = 256 dims
  float M = -INFINITY;
  for (int i = 0; i < SPLIT_WARPS; ++i) M = fmaxf(M, sm[i]);
  float L = 0.f, O = 0.f;
  if (M != -INFINITY) {  // empty slice -> (o = 0, m = -inf, l = 0)
    for (int i = 0; i < SPLIT_WARPS; ++i) {
      const float f = expf(sm[i] - M);  // warps with m = -inf contribute 0
      L += sl[i] * f;
      O += so[i][c] * f;
    }
  }
  const size_t base = (size_t)h * NS + s;
  part_o[base * HD + c] = O;
  if (c == 0) { part_ml[base * 2] = M; part_ml[base * 2 + 1] = L; }
}

__global__ void __launch_bounds__(HD)
attn_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                    const bf16* __restrict__ qg, long ldp,  // gate lives in qg[h][HD..2*HD)
                    bf16* __restrict__ out, long ldo, int Hq, int NS) {
  pdl_trigger();
  pdl_wait();
  const int h = blockIdx.x, c = threadIdx.x, t = blockIdx.y;
  part_o += (size_t)t * Hq * NS * HD;
  part_ml += (size_t)t * Hq * NS * 2;
  qg += t * ldp;
  out += t * ldo;
  float M = -INFINITY;
  for (int s = 0; s < NS; ++s) M = fmaxf(M, part_ml[((size_t)h * NS + s) * 2]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < NS; ++s) {
    const float ms = part_ml[((size_t)h * NS + s) * 2];
    if (ms == -INFINITY) continue;
    const float f = expf(ms - M);
    L += part_ml[((size_t)h * NS + s) * 2 + 1] * f;
    O += part_o[((size_t)h * NS + s) * HD + c] * f;
  }
  const float o = rb(O / L);  // attention output is a bf16 tensor in HF
  const float sg = rb(sigmoid_f(b2f(qg[(size_t)h * 2 * HD + HD + c])));  // sigmoid(gate) bf16
  out[(size_t)h * HD + c] = f2b(o * sg);
}

}  // namespace qe

using namespace qe;

void set_pdl_trigger_attn(int v) { cudaMemcpyToSymbol(g_pdl_trigger, &v, sizeof(int)); }

void launch_attn_prep(const void* qg, const void* k, const void* v, long ldp, const void* qnw,
                      const void* knw, const float* inv_freq, const int* pos, void* q_out, void* kc,
                      void* vc, int n, int Hq, int Hkv, int max_len, int rd, float eps, cudaStream_t s) {
  qe_launch(attn_prep_kernel, dim3(dim3(Hq + Hkv, n)), dim3(HD), 0, s, (const bf16*)qg, (const bf16*)k, (const bf16*)v, ldp,
                                                    (const bf16*)qnw, (const bf16*)knw, inv_freq, pos,
                                                    (bf16*)q_out, (bf16*)kc, (bf16*)vc, Hq, max_len, rd, eps);
}

void launch_attn_decode(const void* q, const void* kc, const void* vc, const int* pos,
                        const void* qg, long ldp, float* part_o, float* part_ml, void* out, long ldo,
                        int n, int Hq, int Hkv, int max_len, int NS, float scale, cudaStream_t s) {
  qe_launch(attn_split_kernel, dim3(dim3(Hq, NS, n)), dim3(SPLIT_WARPS * 32), 0, s, 
      (const bf16*)q, (const bf16*)kc, (const bf16*)vc, pos, part_o, part_ml, Hq, Hkv, max_len, scale);
  qe_launch(attn_combine_kernel, dim3(dim3(Hq, n)), dim3(HD), 0, s, part_o, part_ml, (const bf16*)qg, ldp, (bf16*)out, ldo,
                                                 Hq, NS);
}
