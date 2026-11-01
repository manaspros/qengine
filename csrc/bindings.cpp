// torch bindings for qengine._C. Every op writes into caller-provided tensors (no allocation),
// launches on the current CUDA stream, and is therefore safe to capture in a CUDA graph.
// Row-batched ops take 2-D views [n, width] with stride(1) == 1 and any row stride (so slices of
// a fused projection buffer can be passed directly). A 1-D tensor counts as one row.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "ops.h"

namespace {
cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream().stream(); }
const auto BF = torch::kBFloat16;
const auto I32 = torch::kInt32;
const auto F32 = torch::kFloat32;

void chk(const torch::Tensor& t, torch::ScalarType dt, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dt, name, " has wrong dtype ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
// Row view: returns (rows, width, row stride).
struct Rows { int n; long w; long ld; };
Rows rows(const torch::Tensor& t, torch::ScalarType dt, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt, name, ": wrong device/dtype");
  if (t.dim() == 1) { TORCH_CHECK(t.stride(0) == 1, name, " not contiguous"); return {1, t.size(0), t.size(0)}; }
  TORCH_CHECK(t.dim() == 2 && t.stride(1) == 1, name, " must be [n, w] with unit inner stride");
  return {(int)t.size(0), t.size(1), t.stride(0)};
}
void chk16(const void* p, long ld, const char* name) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(p) % 16 == 0 && ld % 8 == 0, name, " must be 16-byte aligned rows");
}
}  // namespace

// out[r] = table[tok[r]], r < out rows; tok int32 with >= rows entries.
void embed(torch::Tensor table, torch::Tensor tok, torch::Tensor out) {
  chk(table, BF, "table"); TORCH_CHECK(tok.is_cuda() && tok.scalar_type() == I32);
  auto o = rows(out, BF, "out");
  TORCH_CHECK(o.w == table.size(1) && tok.numel() >= o.n);
  launch_embed(table.data_ptr(), tok.data_ptr<int>(), out.data_ptr(), o.n, o.w, table.size(0), o.ld, cur());
}

// x: [rows, H]; delta optional (fused residual add, x updated in place).
void rmsnorm1p(torch::Tensor x, c10::optional<torch::Tensor> delta, torch::Tensor w,
               torch::Tensor out, double eps) {
  chk(w, BF, "w");
  auto X = rows(x, BF, "x"), O = rows(out, BF, "out");
  TORCH_CHECK(X.w == w.numel() && O.w == X.w && O.n == X.n);
  const void* d = nullptr;
  long ldd = 0;
  if (delta.has_value()) {
    auto D = rows(*delta, BF, "delta");
    TORCH_CHECK(D.n == X.n && D.w == X.w);
    d = delta->data_ptr(); ldd = D.ld;
  }
  launch_rmsnorm1p(x.data_ptr(), d, w.data_ptr(), out.data_ptr(), X.n, X.w, X.ld, ldd, O.ld, (float)eps, cur());
}

// out[n, N] = x[n, K] @ W[N, K]^T, n = 1..8.
void gemv_bf16(torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  chk(W, BF, "W");
  auto X = rows(x, BF, "x"), O = rows(out, BF, "out");
  TORCH_CHECK(W.dim() == 2 && X.w == W.size(1) && O.w == W.size(0) && X.n == O.n);
  TORCH_CHECK(X.n >= 1 && X.n <= 8, "1..8 rows");
  TORCH_CHECK(W.size(1) % 8 == 0, "K must be a multiple of 8");
  chk16(W.data_ptr(), W.size(1), "W"); chk16(x.data_ptr(), X.ld, "x");
  launch_gemv_bf16(W.data_ptr(), x.data_ptr(), out.data_ptr(), X.n, W.size(0), W.size(1), X.ld, O.ld, cur());
}

void swiglu(torch::Tensor gu, torch::Tensor out) {
  auto G = rows(gu, BF, "gu"), O = rows(out, BF, "out");
  TORCH_CHECK(G.w == 2 * O.w && G.n == O.n);
  launch_swiglu(gu.data_ptr(), out.data_ptr(), O.n, O.w, G.ld, O.ld, cur());
}

// out[r] = argmax(x[r]); ws_v/ws_i: workspaces with rows*nb entries, nb = ws numel / rows.
void argmax(torch::Tensor x, torch::Tensor ws_v, torch::Tensor ws_i, torch::Tensor out,
            c10::optional<torch::Tensor> idmap) {
  auto X = rows(x, BF, "x");
  chk(ws_v, F32, "ws_v"); chk(ws_i, I32, "ws_i"); chk(out, I32, "out");
  const int nb = ws_v.numel() / X.n;
  TORCH_CHECK(ws_v.numel() == ws_i.numel() && nb >= 1 && out.numel() >= X.n);
  const int* mp = nullptr;
  if (idmap.has_value()) { chk(*idmap, I32, "idmap"); TORCH_CHECK(idmap->numel() == X.w); mp = idmap->data_ptr<int>(); }
  launch_argmax(x.data_ptr(), X.n, X.w, X.ld, ws_v.data_ptr<float>(), ws_i.data_ptr<int>(), nb,
                out.data_ptr<int>(), mp, cur());
}

void accept(torch::Tensor in_tok, torch::Tensor am, int64_t n, torch::Tensor ctl, torch::Tensor next_tok,
            torch::Tensor log) {
  for (auto* t : {&in_tok, &am, &ctl, &next_tok, &log}) chk(*t, I32, "int buffer");
  TORCH_CHECK(in_tok.numel() > n && am.numel() >= n && next_tok.numel() >= n && ctl.numel() >= 24);
  launch_accept(in_tok.data_ptr<int>(), am.data_ptr<int>(), n, ctl.data_ptr<int>(),
                next_tok.data_ptr<int>(), log.data_ptr<int>(), log.numel(), cur());
}

void commit(torch::Tensor ctl, torch::Tensor in_tok, torch::Tensor am, torch::Tensor drafts, int64_t k) {
  for (auto* t : {&ctl, &in_tok, &am, &drafts}) chk(*t, I32, "int buffer");
  TORCH_CHECK(in_tok.numel() > k && drafts.numel() >= k);
  launch_commit(ctl.data_ptr<int>(), in_tok.data_ptr<int>(), am.data_ptr<int>(), drafts.data_ptr<int>(), k, cur());
}

// mpos = off < 0 ? pos : pos + m + 1 + off; optional: dst = src[m], draft0 = mtp_am[m].
void mtp_select(torch::Tensor ctl, torch::Tensor mpos, int64_t off, c10::optional<torch::Tensor> src,
                c10::optional<torch::Tensor> dst, c10::optional<torch::Tensor> mtp_am,
                c10::optional<torch::Tensor> draft0) {
  chk(ctl, I32, "ctl"); chk(mpos, I32, "mpos");
  const void* sp = nullptr; void* dp = nullptr; long lds = 0; int H = 0;
  if (src.has_value()) {
    auto S = rows(*src, BF, "src");
    chk(*dst, BF, "dst");
    sp = src->data_ptr(); dp = dst->data_ptr(); lds = S.ld; H = S.w;
    TORCH_CHECK(dst->numel() == H);
  }
  launch_mtp_select(ctl.data_ptr<int>(), mpos.data_ptr<int>(), off, sp, lds, dp, H,
                    mtp_am.has_value() ? mtp_am->data_ptr<int>() : nullptr,
                    draft0.has_value() ? draft0->data_ptr<int>() : nullptr, cur());
}

// GDN n-row step with lazy rollback; qkv/z/b/a are row views of one projection buffer.
void gdn_step_n(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
                torch::Tensor ring, torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
                torch::Tensor norm_w, torch::Tensor S, torch::Tensor st_k, torch::Tensor st_v,
                torch::Tensor st_g, torch::Tensor ctl, torch::Tensor out, int64_t Hk, double eps,
                int64_t q) {
  auto Q = rows(qkv, BF, "qkv"), Z = rows(z, BF, "z"), B = rows(b, BF, "b"), A = rows(a, BF, "a");
  auto O = rows(out, BF, "out");
  TORCH_CHECK(Q.ld == Z.ld && Q.ld == B.ld && Q.ld == A.ld, "qkv/z/b/a must share a row stride");
  for (auto& [t, nm] : {std::pair{&ring, "ring"}, {&conv_w, "conv_w"}, {&A_log, "A_log"},
                        {&dt_bias, "dt_bias"}, {&norm_w, "norm_w"}})
    chk(*t, BF, nm);
  chk(S, F32, "S"); chk(st_k, F32, "st_k"); chk(st_v, F32, "st_v"); chk(st_g, F32, "st_g"); chk(ctl, I32, "ctl");
  const int n = Q.n, Hv = S.size(0);
  TORCH_CHECK(n >= 1 && n <= 8 && Z.n == n && B.n == n && A.n == n && O.n == n);
  TORCH_CHECK(S.dim() == 3 && S.size(1) == 128 && S.size(2) == 128, "dk = dv = 128 only");
  TORCH_CHECK(Hv % Hk == 0);
  const long C = 2 * Hk * 128 + Hv * 128;
  TORCH_CHECK(Q.w == C && ring.numel() == 16 * C && conv_w.numel() == 4 * C);
  TORCH_CHECK(Z.w == Hv * 128 && O.w == Hv * 128 && B.w == Hv && A.w == Hv);
  TORCH_CHECK(A_log.numel() == Hv && dt_bias.numel() == Hv && norm_w.numel() == 128);
  TORCH_CHECK(st_k.numel() == 8L * Hv * 128 && st_v.numel() == 8L * Hv * 128 && st_g.numel() == 8L * Hv * 2);
  launch_gdn_step_n(qkv.data_ptr(), z.data_ptr(), b.data_ptr(), a.data_ptr(), Q.ld, ring.data_ptr(),
                    conv_w.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(), norm_w.data_ptr(),
                    S.data_ptr<float>(), st_k.data_ptr<float>(), st_v.data_ptr<float>(),
                    st_g.data_ptr<float>(), ctl.data_ptr<int>(), out.data_ptr(), O.ld, n, Hk, Hv,
                    (float)eps, (int)q, cur());
}

// pos: int32 tensor whose element 0 is the position of row 0.
void attn_prep(torch::Tensor qg, torch::Tensor k, torch::Tensor v, torch::Tensor qnw,
               torch::Tensor knw, torch::Tensor inv_freq, torch::Tensor pos, torch::Tensor q_out,
               torch::Tensor kc, torch::Tensor vc, double eps) {
  auto QG = rows(qg, BF, "qg"), K = rows(k, BF, "k"), V = rows(v, BF, "v"), QO = rows(q_out, BF, "q_out");
  TORCH_CHECK(QG.ld == K.ld && QG.ld == V.ld, "qg/k/v must share a row stride");
  for (auto& [t, nm] : {std::pair{&qnw, "qnw"}, {&knw, "knw"}, {&kc, "kc"}, {&vc, "vc"}}) chk(*t, BF, nm);
  chk(inv_freq, F32, "inv_freq"); TORCH_CHECK(pos.is_cuda() && pos.scalar_type() == I32);
  TORCH_CHECK(kc.dim() == 3 && kc.size(2) == 256, "head_dim 256 only");
  const int n = QG.n, Hkv = kc.size(0), max_len = kc.size(1), Hq = QO.w / 256;
  TORCH_CHECK(n >= 1 && n <= 8 && K.n == n && V.n == n && QO.n == n && QO.ld == QO.w);
  TORCH_CHECK(QG.w == Hq * 512 && K.w == Hkv * 256 && V.w == Hkv * 256);
  TORCH_CHECK(vc.sizes() == kc.sizes() && qnw.numel() == 256 && knw.numel() == 256);
  const int rd = inv_freq.numel() * 2;
  TORCH_CHECK(rd <= 256);
  launch_attn_prep(qg.data_ptr(), k.data_ptr(), v.data_ptr(), QG.ld, qnw.data_ptr(), knw.data_ptr(),
                   inv_freq.data_ptr<float>(), pos.data_ptr<int>(), q_out.data_ptr(), kc.data_ptr(),
                   vc.data_ptr(), n, Hq, Hkv, max_len, rd, (float)eps, cur());
}

// q: [n, Hq*256] contiguous; part_o: fp32 [>=n, Hq, NS, 256]; part_ml [>=n, Hq, NS, 2];
// out: [n, Hq*256] (gated by sigmoid of the gate half of qg rows).
void attn_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor pos,
                 torch::Tensor qg, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out) {
  auto Qr = rows(q, BF, "q"), QG = rows(qg, BF, "qg"), O = rows(out, BF, "out");
  chk(kc, BF, "kc"); chk(vc, BF, "vc"); chk(part_o, F32, "part_o"); chk(part_ml, F32, "part_ml");
  TORCH_CHECK(pos.is_cuda() && pos.scalar_type() == I32);
  chk16(q.data_ptr(), Qr.ld, "q"); chk16(kc.data_ptr(), 256, "kc"); chk16(vc.data_ptr(), 256, "vc");
  const int n = Qr.n, Hq = Qr.w / 256, Hkv = kc.size(0), max_len = kc.size(1);
  TORCH_CHECK(Qr.ld == Qr.w && QG.n == n && O.n == n && Hq % Hkv == 0);
  TORCH_CHECK(part_o.dim() == 4 && part_o.size(0) >= n && part_o.size(1) == Hq && part_o.size(3) == 256);
  const int NS = part_o.size(2);
  TORCH_CHECK(part_ml.numel() == part_o.size(0) * Hq * NS * 2 && O.w == Hq * 256 && QG.w == Hq * 512);
  launch_attn_decode(q.data_ptr(), kc.data_ptr(), vc.data_ptr(), pos.data_ptr<int>(), qg.data_ptr(), QG.ld,
                     part_o.data_ptr<float>(), part_ml.data_ptr<float>(), out.data_ptr(), O.ld, n, Hq,
                     Hkv, max_len, NS, 1.0f / 16.0f /* 256^-0.5 */, cur());
}

void stream_window(int64_t base, int64_t bytes, double hit) {
  stream_access_window((uintptr_t)base, (size_t)bytes, (float)hit, cur());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("l2_persist_limit", [](int64_t b) { l2_persist_limit((size_t)b); });
  m.def("stream_window", &stream_window, "access-policy window on the current stream (bytes=0 clears)");
  m.def("embed", &embed);
  m.def("set_gdn_phases", &set_gdn_phases, "GDN kernel: phase-parallel rows (A/B switch)");
  m.def("set_gdn_early", &set_gdn_early, "GDN kernel: overlap state/replay/prefetch with the producer (A/B switch)");
  m.def("set_pdl_launch", &set_pdl_launch, "launch own kernels with programmatic stream serialization (A/B)");
  m.def("set_pdl_trigger", &set_pdl_trigger, "own kernels call griddepcontrol.launch_dependents (A/B switch)");
  m.def("rmsnorm1p", &rmsnorm1p, py::arg("x"), py::arg("delta"), py::arg("w"), py::arg("out"), py::arg("eps"));
  m.def("gemv_bf16", &gemv_bf16);
  m.def("swiglu", &swiglu);
  m.def("argmax", &argmax, py::arg("x"), py::arg("ws_v"), py::arg("ws_i"), py::arg("out"),
        py::arg("idmap") = py::none());
  m.def("accept", &accept);
  m.def("commit", &commit);
  m.def("mtp_select", &mtp_select, py::arg("ctl"), py::arg("mpos"), py::arg("off"), py::arg("src") = py::none(),
        py::arg("dst") = py::none(), py::arg("mtp_am") = py::none(), py::arg("draft0") = py::none());
  m.def("gdn_step_n", &gdn_step_n, py::arg("qkv"), py::arg("z"), py::arg("b"), py::arg("a"), py::arg("ring"),
        py::arg("conv_w"), py::arg("A_log"), py::arg("dt_bias"), py::arg("norm_w"), py::arg("S"), py::arg("st_k"),
        py::arg("st_v"), py::arg("st_g"), py::arg("ctl"), py::arg("out"), py::arg("Hk"), py::arg("eps"),
        py::arg("q") = 2);
  m.def("attn_prep", &attn_prep);
  m.def("attn_decode", &attn_decode);
}
