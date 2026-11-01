// torch binding for qengine._w4 (W4A16 GEMV). Kernel and layout: gemv_w4.cu.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <vector>

#include "gemv_w4.h"

namespace {

// Partial-sum workspace (fp32 slots, contents irrelevant) and arrival counters ([Mp/16], left
// zeroed by the kernel) per device, reusable by any later call and by captured CUDA graphs. Buffers are
// only ever grown; old ones are kept alive because a captured graph may still point at them.
struct Workspace {
  std::vector<at::Tensor> keep;
  at::Tensor ws, counters;
};
Workspace& workspace(int device) {
  static std::vector<Workspace> per_dev(64);
  return per_dev.at(device);
}

void run(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& scales, at::Tensor& out,
         int64_t M, int epilogue, int64_t warps, int64_t unroll, int64_t ksplit, bool pdl,
         int64_t variant, int64_t mode, int64_t bps, int64_t min_units) {
  TORCH_CHECK(x.is_cuda() && qw.is_cuda() && scales.is_cuda() && out.is_cuda(), "CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16 (W4A16 kernel)");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bf16");
  TORCH_CHECK(qw.scalar_type() == at::kInt && qw.dim() == 4 && qw.size(2) == 32 && qw.size(3) == 4,
              "qw must be int32 [Mp/16, Kp/64, 32, 4] from quant.quantize_w4");
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 3 && scales.size(2) == 16,
              "scales must be fp16 [Mp/16, Kp/128, 16]");
  TORCH_CHECK(qw.is_contiguous() && scales.is_contiguous(), "packed tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "x [N,K], out [N,M]");
  TORCH_CHECK(x.stride(1) == 1 && out.stride(1) == 1, "x and out need unit stride along K / M");
  const int64_t N = x.size(0), K = x.size(1);
  TORCH_CHECK(N >= 1 && N <= 8, "N must be 1..8");
  TORCH_CHECK(out.size(0) == N && out.size(1) == M, "out shape must be [N, M]");
  const int64_t Mp = qw.size(0) * 16, Kp = qw.size(1) * 64;
  TORCH_CHECK(Kp % 128 == 0 && scales.size(0) == qw.size(0) && scales.size(1) == Kp / 128,
              "qw / scales shapes disagree");
  TORCH_CHECK(K <= Kp && K > Kp - 128, "x has K=", K, " but the packed weight has Kp=", Kp);
  if (epilogue == qengine_w4::EPI_SWIGLU) {
    TORCH_CHECK(M <= Mp / 2 && M > Mp / 2 - 8, "SwiGLU: out width I must match packed rows/2");
  } else {
    TORCH_CHECK(M <= Mp && M > Mp - 16, "out width M must match packed rows");
  }
  TORCH_CHECK(x.data_ptr() != out.data_ptr(), "in-place not supported");

  qengine_w4::GemvParams p{};
  p.qw = reinterpret_cast<const uint4*>(qw.data_ptr());
  p.scales = reinterpret_cast<const uint16_t*>(scales.data_ptr());
  p.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  p.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  p.M = (int)M; p.K = (int)K; p.Mp = (int)Mp; p.Kp = (int)Kp; p.N = (int)N;
  p.ldx = x.stride(0); p.ldo = out.stride(0);
  p.ksplit = 0;
  p.epilogue = epilogue;

  qengine_w4::LaunchCfg cfg;
  cfg.warps = (int)warps; cfg.unroll = (int)unroll; cfg.ksplit = (int)ksplit;
  cfg.pdl = pdl; cfg.variant = (int)variant;
  cfg.mode = (int)mode; cfg.blocks_per_sm = (int)bps; cfg.min_units = (int)min_units;
  TORCH_CHECK(cfg.warps == 0 || cfg.warps == 4 || cfg.warps == 8 || cfg.warps == 16, "warps in {0 (auto),4,8,16}");
  TORCH_CHECK(cfg.unroll == 0 || cfg.unroll == 1 || cfg.unroll == 2 || cfg.unroll == 4, "unroll in {0 (auto),1,2,4}");
  TORCH_CHECK(cfg.mode == -1 || cfg.mode == 0 || cfg.mode == 2, "mode in {-1 (auto), 0, 2}");

  const c10::cuda::CUDAGuard guard(x.device());
  const qengine_w4::LaunchPlan pl = qengine_w4::plan_gemv_w4(p, cfg);
  auto& W = workspace(x.get_device());
  if (!W.ws.defined() || W.ws.numel() < pl.ws_floats) {
    if (W.ws.defined()) W.keep.push_back(W.ws);
    W.ws = at::empty({std::max<long>(pl.ws_floats, 1 << 18)}, x.options().dtype(at::kFloat));
  }
  if (!W.counters.defined() || W.counters.numel() < pl.counters) {
    if (W.counters.defined()) W.keep.push_back(W.counters);
    W.counters = at::zeros({std::max<long>(pl.counters, 1 << 14)}, x.options().dtype(at::kInt));
  }
  p.ws = W.ws.data_ptr<float>();
  p.counters = W.counters.data_ptr<int>();
  const cudaError_t e = qengine_w4::launch_gemv_w4(p, cfg, c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == cudaSuccess, "gemv_w4 launch failed: ", cudaGetErrorString(e));
}

void gemv(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& scales, at::Tensor out,
          int64_t M, int64_t warps, int64_t unroll, int64_t ksplit, bool pdl, int64_t variant, int64_t mode, int64_t bps, int64_t min_units) {
  run(x, qw, scales, out, M, qengine_w4::EPI_NONE, warps, unroll, ksplit, pdl, variant, mode, bps, min_units);
}

void gemv_swiglu(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& scales,
                 at::Tensor out, int64_t I, int64_t warps, int64_t unroll, int64_t ksplit,
                 bool pdl, int64_t variant, int64_t mode, int64_t bps, int64_t min_units) {
  run(x, qw, scales, out, I, qengine_w4::EPI_SWIGLU, warps, unroll, ksplit, pdl, variant, mode, bps, min_units);
}

int64_t auto_ksplit(int64_t Mp, int64_t Kp, int64_t N, int64_t warps) {
  qengine_w4::LaunchCfg cfg;
  cfg.warps = (int)warps;
  return qengine_w4::auto_ksplit((int)Mp, (int)Kp, (int)N, cfg);
}

}  // namespace

PYBIND11_MODULE(_w4, m) {
  m.doc() = "qengine W4A16 GEMV (int4 g128 symmetric, fp16 scales, bf16 activations, N=1..8)";
  m.def("gemv", &gemv, "out[N,M] = x[N,K] @ W^T (W packed by quant.quantize_w4)",
        py::arg("x"), py::arg("qw"), py::arg("scales"), py::arg("out"), py::arg("M"),
        py::arg("warps") = 0, py::arg("unroll") = 0, py::arg("ksplit") = 0,
        py::arg("pdl") = true, py::arg("variant") = 0, py::arg("mode") = -1,
        py::arg("blocks_per_sm") = 0, py::arg("min_units") = 4);
  m.def("gemv_swiglu", &gemv_swiglu,
        "out[N,I] = silu(x @ Wg^T) * (x @ Wu^T) (packed by quant.quantize_w4_gate_up)",
        py::arg("x"), py::arg("qw"), py::arg("scales"), py::arg("out"), py::arg("I"),
        py::arg("warps") = 0, py::arg("unroll") = 0, py::arg("ksplit") = 0,
        py::arg("pdl") = true, py::arg("variant") = 0, py::arg("mode") = -1,
        py::arg("blocks_per_sm") = 0, py::arg("min_units") = 4);
  m.def("auto_ksplit", &auto_ksplit, py::arg("Mp"), py::arg("Kp"), py::arg("N"),
        py::arg("warps") = 8);
}
