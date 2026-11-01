"""qengine._w4 W4A16 GEMV vs torch: out = x @ dequantize_w4(W).T computed in fp32.

Tolerance (per element), with S = |x| @ |W~|^T (fp32):
    |out - ref| <= 2^-8 * |ref| + K * 2^-24 * S
- 2^-8 * |ref|: the kernel's only rounding beyond fp32 is the final fp32 -> bf16 store; bf16 has
  an 8-bit significand (7 stored + implicit), so round-to-nearest is <= 2^-8 relative.
- K * 2^-24 * S: worst-case bound for summing K fp32 products in a different order (Higham,
  gamma_K with unit roundoff 2^-24), which covers the tensor-core accumulation order, the
  per-group scale multiply and the split-K atomics. Products themselves are exact (int4 * bf16
  fits in fp32).
On top of the bound, >= 90% of the outputs must be bit-identical to bf16(fp32 reference): the
accumulation-order difference (~1e-6 relative) only flips the bf16 rounding near ties or under
heavy cancellation, so a kernel that is merely "within a loose bound" fails. (A first version
required median relative error <= 2^-9; that was a wrong derivation, since bf16 rounding alone
reaches 2^-8, and the kernel was bit-exact on the failing case.)
bf16 is the only activation dtype (W4A16 design, the model runs bf16): fp16/fp32 x are rejected.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("qengine._w4")
from qengine import _w4  # noqa: E402
from qengine.quant import dequantize_w4, gemv, quantize_w4, quantize_w4_gate_up  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"

H, I, V = 5120, 17408, 248320
SHAPES_27B = {  # (M, K) of every linear layer of Qwen3.8-27B (+ the fused concatenations)
    "gdn.in_proj_qkv": (10240, H),
    "gdn.in_proj_z": (6144, H),
    "gdn.in_proj_a/b": (48, H),
    "gdn.qkvzab_fused": (16480, H),
    "gdn.out_proj": (H, 6144),
    "attn.q_proj": (12288, H),
    "attn.k/v_proj": (1024, H),
    "attn.qkv_fused": (14336, H),
    "attn.o_proj": (H, 6144),
    "mlp.gate/up_proj": (I, H),
    "mlp.down_proj": (H, I),
    "lm_head": (V, H),
}
ODD_SHAPES = [(1, 1), (1, 128), (7, 127), (16, 129), (127, 1000), (1000, 4097), (33, 32768),
              (4097, 7), (17, 255), (100, 136)]


def _bound(x: torch.Tensor, Wd: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    S = x.float().abs() @ Wd.abs().T
    return 2.0**-8 * ref.abs() + x.shape[1] * 2.0**-24 * S


def _check(out: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor, what: str):
    o = out.float()
    nan_ref = torch.isnan(ref)
    assert torch.equal(torch.isnan(o), nan_ref), f"{what}: NaN pattern differs"
    inf_ref = torch.isinf(ref)
    assert torch.equal(torch.isinf(o), inf_ref), f"{what}: inf pattern differs"
    assert torch.equal(o[inf_ref], ref[inf_ref]), f"{what}: inf sign differs"
    fin = ~(nan_ref | inf_ref)
    err = (o - ref).abs()[fin]
    bad = err > bound[fin]
    assert not bad.any(), (f"{what}: {int(bad.sum())} elements over bound, worst err "
                           f"{err.max().item():.3e}")
    if fin.sum() >= 8:
        exact = (o[fin] == ref[fin].to(torch.bfloat16).float()).float().mean().item()
        assert exact >= 0.9, f"{what}: only {exact:.3f} of outputs equal bf16(reference)"


def _rand_w(M, K, scale=0.02, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, K, generator=g, device=DEV) * scale).to(torch.bfloat16)


def _rand_x(N, K, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(N, K, generator=g, device=DEV).to(torch.bfloat16)


# ------------------------------------------------------------------ packing / layout


def _naive_pack(W: torch.Tensor):
    """Slow reference packer written directly from the layout described in python/qengine/quant.py."""
    M, K = W.shape
    Mp, Kp = (M + 15) // 16 * 16, (K + 127) // 128 * 128
    Wp = torch.zeros(Mp, Kp, dtype=torch.float64)
    Wp[:M, :K] = W.double().cpu()
    G = Kp // 128
    s16 = torch.empty(Mp, G, dtype=torch.float16)
    u = torch.empty(Mp, Kp, dtype=torch.int64)
    for m in range(Mp):
        for j in range(G):
            blk = Wp[m, 128 * j:128 * j + 128]
            s = torch.tensor(float(blk.abs().max()) * 2 / 15).to(torch.float16)
            s16[m, j] = s
            sf = float(s)
            for kk in range(128):
                u[m, 128 * j + kk] = 8 if sf == 0 else int(min(15, max(0, round(float(blk[kk]) / sf) + 8)))
    qw = torch.zeros(Mp // 16, Kp // 64, 32, 4, dtype=torch.int64)
    for T in range(Mp // 16):
        for c in range(Kp // 64):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for s in range(4):
                    k0 = 64 * c + 16 * t + 4 * s
                    order = [(g, k0), (g + 8, k0), (g, k0 + 2), (g + 8, k0 + 2),
                             (g, k0 + 1), (g + 8, k0 + 1), (g, k0 + 3), (g + 8, k0 + 3)]
                    word = 0
                    for j, (r, k) in enumerate(order):
                        word |= int(u[16 * T + r, k]) << (4 * j)
                    qw[T, c, lane, s] = word
    sc = s16.view(Mp // 16, 16, G).permute(0, 2, 1).contiguous()
    return qw, sc


def test_pack_matches_naive_spec():
    # Python's round() is half-to-even like torch.round, so codes must match exactly.
    W = _rand_w(19, 200, seed=3)
    p = quantize_w4(W)
    qw, sc = _naive_pack(W)
    assert torch.equal(p.qw.cpu().to(torch.int64) & 0xFFFFFFFF, qw)
    assert torch.equal(p.scales.cpu(), sc)


@pytest.mark.parametrize("M,K", [(16, 128), (37, 1000), (1024, 5120)])
def test_roundtrip_error(M, K):
    W = _rand_w(M, K, seed=4)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    assert Wd.shape == (M, K)
    s = p.scales.float().permute(0, 2, 1).reshape(-1, p.scales.shape[1])[:M]
    s = s.repeat_interleave(128, dim=1)[:, :K]
    err = (Wd - W.float()).abs()
    # RTN error <= s/2; the fp16 rounding of s can push |w|/s up to 7.5*(1 + 2^-11): + |w| 2^-10
    assert (err <= 0.5 * s + W.float().abs() * 2.0**-10 + 1e-12).all()


# ------------------------------------------------------------------ GEMV correctness


@pytest.mark.parametrize("name", list(SHAPES_27B))
def test_27b_shapes_all_N(name):
    M, K = SHAPES_27B[name]
    W = _rand_w(M, K, seed=5)
    p = quantize_w4(W)
    del W
    Wd = dequantize_w4(p)
    x = _rand_x(8, K, seed=6)
    ref_all = x.float() @ Wd.T
    bound_all = _bound(x, Wd, ref_all)
    for N in range(1, 9):
        out = gemv(x[:N].contiguous(), p)
        _check(out, ref_all[:N], bound_all[:N], f"{name} N={N}")


@pytest.mark.parametrize("M,K", ODD_SHAPES)
def test_odd_shapes(M, K):
    W = _rand_w(M, K, seed=7)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(8, K, seed=8)
    for N in range(1, 9):
        ref = x[:N].float() @ Wd.T
        out = gemv(x[:N].contiguous(), p)
        _check(out, ref, _bound(x[:N], Wd, ref), f"{M}x{K} N={N}")


@pytest.mark.parametrize("mode,ksplit,min_units", [(0, 1, 4), (0, 2, 4), (0, 3, 4), (0, 7, 4),
                                                  (0, 40, 4), (2, 0, 1), (2, 0, 4), (-1, 0, 4)])
@pytest.mark.parametrize("warps,unroll", [(4, 1), (8, 2), (8, 4), (16, 1)])
def test_launch_configs(mode, ksplit, min_units, warps, unroll):
    M, K = 1000, 5120
    W = _rand_w(M, K, seed=9)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(5, K, seed=10)
    ref = x.float() @ Wd.T
    for variant in range(4):
        for pdl in (False, True):
            out = gemv(x, p, ksplit=ksplit, warps=warps, unroll=unroll, variant=variant, pdl=pdl,
                       mode=mode, min_units=min_units)
            _check(out, ref, _bound(x, Wd, ref),
                   f"mode={mode} ks={ksplit} w={warps} u={unroll} v={variant}")


@pytest.mark.parametrize("kw", [dict(mode=0, ksplit=16), dict(mode=2, min_units=1)])
def test_splitk_workspace_reset_repeated(kw):
    # the split-K workspace/counters must be left zeroed: repeated calls give the same answer
    W = _rand_w(64, 8192, seed=11)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(3, 8192, seed=12)
    ref = x.float() @ Wd.T
    for _ in range(20):
        out = gemv(x, p, **kw)
        _check(out, ref, _bound(x, Wd, ref), f"repeat {kw}")


def test_cuda_graph_replay():
    W = _rand_w(1024, 5120, seed=13)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(2, 5120, seed=14)
    out = torch.empty(2, 1024, dtype=torch.bfloat16, device=DEV)
    gemv(x, p, out, mode=0, ksplit=8)  # allocates workspace outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemv(x, p, out, mode=0, ksplit=8)
        gemv(x, p, out, mode=2, min_units=1)
        gemv(x, p, out)
    for seed in range(3):
        x.copy_(_rand_x(2, 5120, seed=100 + seed))
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        ref = x.float() @ Wd.T
        _check(out, ref, _bound(x, Wd, ref), f"graph replay {seed}")


# ------------------------------------------------------------------ fused gate/up + SwiGLU


@pytest.mark.parametrize("Iw,K", [(I, H), (37, 200), (8, 128), (1000, 4097)])
def test_swiglu(Iw, K):
    Wg, Wu = _rand_w(Iw, K, seed=15), _rand_w(Iw, K, seed=16)
    p = quantize_w4_gate_up(Wg, Wu)
    del Wg, Wu
    gd, ud = dequantize_w4(p)
    x = _rand_x(8, K, seed=17)
    for N in range(1, 9):
        xs = x[:N].contiguous()
        g = xs.float() @ gd.T
        u = xs.float() @ ud.T
        ref = torch.nn.functional.silu(g) * u
        # error of silu(g)*u from errors eg, eu in g, u (bounded as above, without the output
        # rounding term): |d(silu(g)u)| <= |silu'(g)| |u| eg + |silu(g)| eu, silu' <= 1.1;
        # plus the bf16 output rounding 2^-8 |ref|.
        Kb = K * 2.0**-24
        eg = Kb * (xs.float().abs() @ gd.abs().T)
        eu = Kb * (xs.float().abs() @ ud.abs().T)
        bound = 2.0**-8 * ref.abs() + 1.1 * u.abs() * eg + torch.nn.functional.silu(g).abs() * eu + 1e-30
        out = gemv(xs, p)
        assert out.shape == (N, Iw)
        _check(out, ref, bound, f"swiglu {Iw}x{K} N={N}")


# ------------------------------------------------------------------ special values / strides


def test_nan_inf_rows():
    M, K = 300, 1000
    W = _rand_w(M, K, seed=18)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(4, K, seed=19)
    x[1, 17] = float("nan")
    x[2, 900] = float("inf")
    x[3, 5] = float("-inf")
    x[3, 6] = float("inf")
    ref = x.float() @ Wd.T
    out = gemv(x, p)
    _check(out, ref, _bound(x, Wd, ref).nan_to_num(posinf=0.0), "nan/inf")
    assert torch.isnan(out[1]).all()


@pytest.mark.parametrize("mag", [1e-30, 1e-4, 1e4, 1e30])
def test_extreme_magnitudes(mag):
    M, K = 256, 2048
    W = _rand_w(M, K, seed=20)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = (_rand_x(3, K, seed=21).float() * mag).to(torch.bfloat16)
    ref = x.float() @ Wd.T
    out = gemv(x, p)
    _check(out, ref, _bound(x, Wd, ref), f"mag {mag}")


def test_weight_extremes():
    # large and tiny weight groups, all-zero groups, single outliers per group
    M, K = 64, 1024
    W = _rand_w(M, K, seed=22).float()
    W[0] *= 1e3
    W[1] *= 1e-3
    W[2, :128] = 0
    W[3, 5] = 50.0
    W = W.to(torch.bfloat16)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    x = _rand_x(2, K, seed=23)
    ref = x.float() @ Wd.T
    _check(gemv(x, p), ref, _bound(x, Wd, ref), "weight extremes")


@pytest.mark.parametrize("pad,offset", [(64, 0), (3, 0), (0, 1), (8, 3)])
def test_noncontiguous_x_and_out(pad, offset):
    M, K = 500, 1024
    W = _rand_w(M, K, seed=24)
    p = quantize_w4(W)
    Wd = dequantize_w4(p)
    big = _rand_x(4, K + pad + offset, seed=25)
    x = big[:, offset:offset + K]           # row stride K+pad+offset; misaligned when offset odd
    assert not x.is_contiguous()
    obig = torch.full((4, M + 40), 7.0, dtype=torch.bfloat16, device=DEV)
    out = obig[:, 8:8 + M]
    gemv(x, p, out)
    ref = x.float() @ Wd.T
    _check(out, ref, _bound(x, Wd, ref), f"strided pad={pad} off={offset}")
    assert (obig[:, :8] == 7).all() and (obig[:, 8 + M:] == 7).all(), "wrote outside out"


def test_rejects_bad_inputs():
    W = _rand_w(32, 256)
    p = quantize_w4(W)
    x = _rand_x(1, 256)
    with pytest.raises(RuntimeError):
        _w4.gemv(x.half(), p.qw, p.scales, torch.empty(1, 32, dtype=torch.bfloat16, device=DEV), 32)
    with pytest.raises(RuntimeError):
        _w4.gemv(_rand_x(9, 256), p.qw, p.scales, torch.empty(9, 32, dtype=torch.bfloat16, device=DEV), 32)
    with pytest.raises(RuntimeError):
        _w4.gemv(_rand_x(1, 384), p.qw, p.scales, torch.empty(1, 32, dtype=torch.bfloat16, device=DEV), 32)


# ------------------------------------------------------------------ bitwise determinism


DET_CFGS = [dict(), dict(warps=4), dict(warps=16), dict(unroll=1), dict(unroll=4),
            dict(warps=16, unroll=1), dict(mode=0), dict(mode=0, ksplit=2), dict(mode=0, ksplit=7),
            dict(mode=2), dict(mode=2, min_units=1), dict(mode=2, warps=4, unroll=1),
            dict(variant=1), dict(pdl=False)]


def _bits(t):
    return t.view(torch.int16).clone()


@pytest.mark.parametrize("M,K", [(5120, 17408), (1024, 5120), (1000, 4097), (16480, 5120)])
def test_bitwise_repeatable(M, K):
    """Identical inputs -> identical bits, every config (no float atomics, fixed reduction order)."""
    W = _rand_w(M, K, seed=30)
    p = quantize_w4(W)
    x = _rand_x(8, K, seed=31)
    out = torch.empty(8, M, dtype=torch.bfloat16, device=DEV)
    for kw in DET_CFGS:
        gemv(x, p, out, **kw)
        ref = _bits(out)
        for _ in range(30):
            out.fill_(0)
            gemv(x, p, out, **kw)
            assert torch.equal(_bits(out), ref), f"{M}x{K} {kw}: run differs bitwise"


@pytest.mark.parametrize("M,K", [(5120, 17408), (1024, 5120), (127, 1000), (16480, 5120)])
def test_row_independent_of_N(M, K):
    """Row r of an N-row call equals the 1-row call on x[r], bit for bit, for every N."""
    W = _rand_w(M, K, seed=32)
    p = quantize_w4(W)
    x = _rand_x(8, K, seed=33)
    for kw in DET_CFGS:
        single = [_bits(gemv(x[r:r + 1].contiguous(), p, **kw)) for r in range(8)]
        for N in range(2, 9):
            out = gemv(x[:N].contiguous(), p, **kw)
            for r in range(N):
                assert torch.equal(_bits(out[r:r + 1]), single[r]), f"{M}x{K} {kw} N={N} row {r}"


def test_swiglu_bitwise_and_row_independent():
    Wg, Wu = _rand_w(I, H, seed=34), _rand_w(I, H, seed=35)
    p = quantize_w4_gate_up(Wg, Wu)
    del Wg, Wu
    x = _rand_x(8, H, seed=36)
    for kw in DET_CFGS:
        full = _bits(gemv(x, p, **kw))
        for _ in range(10):
            assert torch.equal(_bits(gemv(x, p, **kw)), full), f"swiglu {kw} repeat"
        for r in range(8):
            assert torch.equal(_bits(gemv(x[r:r + 1].contiguous(), p, **kw)), full[r:r + 1]), \
                f"swiglu {kw} row {r}"
