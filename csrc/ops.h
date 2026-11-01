// Host launchers for qengine._C (BF16 decode / verify kernels, n = 1..8 rows). Pointers are raw
// device pointers; `ld*` are row strides in elements. Shapes are validated in bindings.cpp.
#pragma once
#include <cuda_runtime.h>

void launch_embed(const void* table, const int* tok, void* out, int n, int H, int V, long ldo,
                  cudaStream_t s);
void launch_rmsnorm1p(void* x, const void* delta, const void* w, void* out, int rows, int H,
                      long ldx, long ldd, long ldo, float eps, cudaStream_t s);
void launch_gemv_bf16(const void* W, const void* x, void* out, int nr, int N, int K, long ldx,
                      long ldo, cudaStream_t s);
void launch_swiglu(const void* gu, void* out, int n, int I, long ldg, long ldo, cudaStream_t s);
void launch_argmax(const void* x, int rows, int n, long ldx, float* pv, int* pi, int nb, int* out,
                   const int* idmap, cudaStream_t s);
void launch_accept(const int* in_tok, const int* am, int n, int* ctl, int* next_tok, int* log,
                   int max_len, cudaStream_t s);
void launch_commit(int* ctl, int* in_tok, const int* am, const int* drafts, int k, cudaStream_t s);
void launch_mtp_select(const int* ctl, int* mpos, int off, const void* src, long lds, void* dst,
                       int H, const int* mtp_am, int* draft0, cudaStream_t s);
void launch_gdn_step_n(const void* qkv, const void* z, const void* b, const void* a, long ldp,
                       void* ring, const void* conv_w, const void* A_log, const void* dt_bias,
                       const void* norm_w, float* S, float* st_k, float* st_v, float* st_g,
                       const int* ctl, void* out, long ldo, int n, int Hk, int Hv, float eps,
                       int q, cudaStream_t s);
void launch_attn_prep(const void* qg, const void* k, const void* v, long ldp, const void* qnw,
                      const void* knw, const float* inv_freq, const int* pos, void* q_out, void* kc,
                      void* vc, int n, int Hq, int Hkv, int max_len, int rd, float eps, cudaStream_t s);
void launch_attn_decode(const void* q, const void* kc, const void* vc, const int* pos,
                        const void* qg, long ldp, float* part_o, float* part_ml, void* out, long ldo,
                        int n, int Hq, int Hkv, int max_len, int NS, float scale, cudaStream_t s);
void set_pdl_trigger(bool on);
void set_pdl_launch(bool on);
void set_gdn_early(bool on);
void set_gdn_phases(bool on);
void l2_persist_limit(size_t bytes);
void stream_access_window(uintptr_t base, size_t bytes, float hit_ratio, cudaStream_t s);
