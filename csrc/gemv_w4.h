// qengine W4A16 GEMV / skinny GEMM for decode (N = 1..8 activation rows).
// Layout and kernel design: see the header comment in gemv_w4.cu.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace qengine_w4 {

enum Epilogue : int { EPI_NONE = 0, EPI_SWIGLU = 1 };

struct GemvParams {
  const uint4* qw;          // packed int4 weights, [Mp/16][Kp/64][32 lanes][4 words]
  const uint16_t* scales;   // fp16 scales, [Mp/16][Kp/128][16]
  const uint16_t* x;        // bf16 activations, N rows, row stride ldx (elements)
  uint16_t* out;            // bf16 output, N rows, row stride ldo (elements)
  float* ws;                // fp32 partial-sum slots (LaunchPlan::ws_floats), any contents
  int* counters;            // arrival counters [Mp/16], all zero between calls (kernel re-zeroes)
  int M;                    // logical output rows (EPI_SWIGLU: intermediate size I)
  int K;                    // logical reduction length
  int Mp, Kp;               // padded (multiples of 16 / 128)
  int N;                    // activation rows, 1..8
  long ldx, ldo;
  int ksplit;               // blocks along K (>= 1)
  int epilogue;             // Epilogue
};

struct LaunchCfg {
  int warps = 0;    // warps per block (4, 8, 16; 0 = auto by shape)
  int unroll = 0;   // groups (1 KB each) in flight per warp (1, 2, 4; 0 = auto by shape)
  int ksplit = 0;   // 0 = auto
  bool pdl = true;  // programmatic dependent launch (prefetch weights before griddepcontrol.wait)
  int variant = 0;  // load flavour for experiments: 0 = ld.global.nc.L1::no_allocate,
                    // 1 = plain ld.global.nc, 2 = nc.L1::no_allocate.L2::256B,
                    // 3 = nc.L1::no_allocate + L2 evict_first policy
  int mode = -1;         // -1 = auto, 0 = one 16-row tile per block (+ksplit),
                         // 2 = block-level stream-K (persistent). (modes 1 and 3 were slower and removed)
  int blocks_per_sm = 0; // mode 2: 0 = max occupancy
  int min_units = 4;     // mode 2: minimum 1-KB units per warp (caps the grid for small shapes)
};

// Resolved launch (deterministic function of shape + cfg; never of N).
struct LaunchPlan {
  int mode = 0;
  int grid_x = 1, grid_y = 1, threads = 256;
  long ws_floats = 0;   // required size of GemvParams::ws
  long counters = 0;    // required size of GemvParams::counters
  const void* fn = nullptr;
};
LaunchPlan plan_gemv_w4(const GemvParams& p, const LaunchCfg& cfg);

// Choose ksplit for a shape when cfg.ksplit == 0.
int auto_ksplit(int Mp, int Kp, int N, const LaunchCfg& cfg);

// Launch. Returns cudaSuccess or the launch error.
cudaError_t launch_gemv_w4(GemvParams p, const LaunchCfg& cfg, cudaStream_t stream);

}  // namespace qengine_w4
