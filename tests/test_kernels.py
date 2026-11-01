"""Unit tests: every qengine._C kernel vs python/qengine/reference.py (itself validated vs HF).

Tolerances (BF16 ulp = 2^-8 relative, i.e. 2^-7 between neighbours near 1):
- embedding, swiglu, argmax, fused residual add: exact (same elementwise ops, same rounding).
- rmsnorm1p: <= 1 bf16 ulp. Only the fp32 sum-of-squares order differs, which can move a value
  across a rounding boundary.
- gemv: |y - y_exact| <= 2^-8 |y_exact| + K * 2^-23 * sum|w x|  (one bf16 rounding of the output
  plus fp32 accumulation error), with y_exact in fp64.
- gdn_step / attention: chains of bf16 roundings (conv -> silu, o -> norm -> weight -> gate) can
  each flip by one ulp when the fp32 sums are reordered, so outputs are checked to 2^-6 relative
  (2 ulp) + small atol, and the fraction of elements off by more than 1 ulp is bounded.
"""
import math

import pytest
import torch

from qengine import reference as R

C = pytest.importorskip("qengine._C")
dev = "cuda"
BF = torch.bfloat16


def ulp_close(a, b, ulps=1):
    """|a-b| <= ulps * bf16 spacing at max(|a|,|b|) (NaN must match NaN)."""
    a, b = a.float(), b.float()
    nan_ok = torch.isnan(a) == torch.isnan(b)
    m = torch.maximum(a.abs(), b.abs())
    spacing = torch.where(m > 0, 2.0 ** (torch.floor(torch.log2(m.clamp_min(1e-38))) - 7), 1e-38)
    ok = ((a - b).abs() <= ulps * spacing) | (a == b) | (torch.isnan(a) & torch.isnan(b))
    return bool((ok & nan_ok).all()), (a - b).abs().nan_to_num().max().item()


# ---------------------------------------------------------------------------------- embedding
@pytest.mark.parametrize("V,H", [(7, 8), (1000, 1024), (248320, 1024), (33, 5120)])
def test_embed(V, H):
    tab = torch.randn(V, H, device=dev).to(BF)
    tokens = torch.tensor([3 % V, V - 1, 0, V + 5, -1], dtype=torch.int32, device=dev)
    out = torch.empty(5, H, dtype=BF, device=dev)
    C.embed(tab, tokens, out)
    for i in range(3):
        assert torch.equal(out[i], tab[tokens[i]])
    assert torch.equal(out[3:], torch.zeros_like(out[3:]))  # out of range -> zeros, no fault


# ---------------------------------------------------------------------------------- rmsnorm1p
@pytest.mark.parametrize("rows,H", [(1, 1024), (1, 5120), (1, 1000), (8, 256), (3, 7), (1, 127), (4, 4097)])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 300.0])
def test_rmsnorm1p(rows, H, scale):
    x = (torch.randn(rows, H, device=dev) * scale).to(BF)
    w = (torch.randn(H, device=dev) * 0.3).to(BF)
    out = torch.empty_like(x)
    C.rmsnorm1p(x, None, w, out, 1e-6)
    ok, err = ulp_close(out, R.rmsnorm1p(x, w, 1e-6))
    assert ok, err


def test_rmsnorm1p_fused_residual_and_nan():
    H = 1024
    h = torch.randn(4, H, device=dev).to(BF)
    d = torch.randn(4, H, device=dev).to(BF)
    d[2, 5] = float("nan")  # NaN stays in its row
    d[3, 0] = float("inf")
    w = torch.randn(H, device=dev).to(BF)
    h_ref = h + d
    out = torch.empty_like(h)
    C.rmsnorm1p(h, d, w, out, 1e-6)
    assert torch.equal(h.nan_to_num(9.0), h_ref.nan_to_num(9.0))  # residual add exact
    ok, err = ulp_close(out, R.rmsnorm1p(h_ref, w, 1e-6))
    assert ok, err
    assert torch.isnan(out[2]).all() and not torch.isnan(out[:2]).any()


# ---------------------------------------------------------------------------------- gemv
@pytest.mark.parametrize("N,K", [(1, 8), (7, 64), (127, 1000), (1000, 1024), (4097, 1024), (6144, 1024),
                                 (1024, 3584), (5120, 17408), (248320, 1024), (33, 24)])
def test_gemv_bf16(N, K):
    W = torch.randn(N, K, device=dev).to(BF)
    x = torch.randn(K, device=dev).to(BF)
    out = torch.empty(N, dtype=BF, device=dev)
    C.gemv_bf16(W, x, out)
    exact = (W.double() @ x.double())
    bound = 2.0 ** -8 * exact.abs() + K * 2.0 ** -23 * (W.double().abs() @ x.double().abs()) + 1e-30
    err = (out.double() - exact).abs()
    assert (err <= bound).all(), (err / bound).max().item()
    # and vs torch (cuBLAS): within 1 ulp almost everywhere
    ok_frac = ((out.float() - R.linear(x, W).float()).abs() <= 2.0 ** -7 * out.float().abs() + 1e-6).float().mean()
    assert ok_frac > 0.99


@pytest.mark.parametrize("nr", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("N,K", [(7, 64), (1000, 1024), (6144, 1024), (1024, 3584), (33, 24)])
def test_gemv_bf16_rows_bit_exact(nr, N, K):
    """Row r of an nr-row call == a 1-row call on that row, bit for bit (verify exactness),
    also with a row stride larger than K (a view into a wider buffer)."""
    W = torch.randn(N, K, device=dev).to(BF)
    xb = torch.randn(nr, K + 16, device=dev).to(BF)
    x = xb[:, :K]
    ob = torch.empty(nr, N + 8, dtype=BF, device=dev)
    out = ob[:, :N]
    C.gemv_bf16(W, x, out)
    one = torch.empty(N, dtype=BF, device=dev)
    for r in range(nr):
        C.gemv_bf16(W, x[r].contiguous(), one)
        assert torch.equal(out[r], one), r


def test_gemv_nan_propagates():
    W = torch.randn(64, 128, device=dev).to(BF)
    x = torch.randn(128, device=dev).to(BF)
    x[3] = float("nan")
    out = torch.empty(64, dtype=BF, device=dev)
    C.gemv_bf16(W, x, out)
    assert torch.isnan(out).all()


# ---------------------------------------------------------------------------------- swiglu
@pytest.mark.parametrize("I", [1, 7, 127, 3584, 17408, 4097])
def test_swiglu(I):
    gu = (torch.randn(2 * I, device=dev) * 4).to(BF)
    gu[0] = float("-inf") if I > 1 else gu[0]
    out = torch.empty(I, dtype=BF, device=dev)
    C.swiglu(gu, out)
    ref = torch.nn.functional.silu(gu[:I]) * gu[I:]
    ok, err = ulp_close(out, ref)
    assert ok, err
    gu2 = gu.repeat(3, 1)
    out2 = torch.empty(3, I, dtype=BF, device=dev)
    C.swiglu(gu2, out2)
    assert torch.equal(out2.view(torch.int16), out.expand(3, I).view(torch.int16))  # bitwise (NaN-safe)


# ---------------------------------------------------------------------------------- argmax
def _argmax(x, nb=256):
    out = torch.empty(1, dtype=torch.int32, device=dev)
    C.argmax(x, torch.empty(nb, device=dev), torch.empty(nb, dtype=torch.int32, device=dev), out)
    return int(out)


def test_argmax_rows():
    x = torch.randn(5, 248320, device=dev).to(BF)
    out = torch.empty(5, dtype=torch.int32, device=dev)
    C.argmax(x, torch.empty(5 * 256, device=dev), torch.empty(5 * 256, dtype=torch.int32, device=dev), out)
    assert out.tolist() == x.argmax(-1).tolist()


@pytest.mark.parametrize("n", [1, 7, 127, 1000, 4097, 32768, 248320])
def test_argmax(n):
    x = torch.randn(n, device=dev).to(BF)
    assert _argmax(x) == int(x.argmax())
    x[:] = 1.0  # all ties -> lowest index
    assert _argmax(x) == 0
    if n > 3:
        x = torch.randn(n, device=dev).to(BF)
        x[n // 2] = 50.0
        x[n - 1] = 50.0  # tie between two maxima -> lower index
        assert _argmax(x) == n // 2 == int(x.argmax())
        x[n - 2] = float("nan")  # torch: NaN is the max
        assert _argmax(x) == n - 2 == int(x.argmax())
        x = torch.full((n,), float("-inf"), device=dev).to(BF)
        assert _argmax(x) == 0 == int(x.argmax())


# ---------------------------------------------------------------------------------- gdn_step
def _gdn_cfg(hk, hv):
    return R.Config(hidden=0, layers=1, inter=0, vocab=0, eps=1e-6, layer_types=["linear_attention"],
                    hq=1, hkv=1, head_dim=256, rope_theta=1e7, rotary_dim=64, hk=hk, hv=hv, dk=128,
                    dv=128, conv_k=4, tied=True)


class GDN:
    """One GDN layer driven through C.gdn_step_n with its own ctl (pos, pending)."""

    def __init__(self, cfg, W, Cd, q=2):
        self.cfg, self.W, self.q = cfg, W, q
        hv = cfg.hv
        self.ring = torch.zeros(16, Cd, dtype=BF, device=dev)
        self.S = torch.zeros(hv, 128, 128, device=dev)
        self.st = [torch.zeros(8, hv, 128, device=dev), torch.zeros(8, hv, 128, device=dev),
                   torch.zeros(8, hv, 2, device=dev)]
        self.ctl = torch.zeros(8, dtype=torch.int32, device=dev)

    def run(self, rows, keep):
        """rows: [n, C+Vd+2hv] projection rows. Returns outputs [n, hv*128]; keeps `keep` rows."""
        cfg = self.cfg
        n = rows.shape[0]
        c0, c1 = cfg.conv_dim, cfg.conv_dim + cfg.value_dim
        out = torch.empty(n, cfg.hv * 128, dtype=BF, device=dev)
        C.gdn_step_n(rows[:, :c0], rows[:, c0:c1], rows[:, c1:c1 + cfg.hv], rows[:, c1 + cfg.hv:], self.ring,
                     self.W["conv1d.weight"], self.W["A_log"], self.W["dt_bias"], self.W["norm.weight"], self.S,
                     *self.st, self.ctl, out, cfg.hk, 1e-6, q=self.q)
        self.ctl[0] += keep
        self.ctl[1] = keep
        return out


def _gdn_setup(hk, hv, seed=0):
    torch.manual_seed(seed)
    cfg = _gdn_cfg(hk, hv)
    Cd = cfg.conv_dim
    W = {"conv1d.weight": (torch.randn(Cd, 1, 4, device=dev) * 0.5).to(BF),
         "A_log": torch.empty(hv, device=dev).uniform_(-3, 2.7).to(BF),
         "dt_bias": torch.randn(hv, device=dev).to(BF),
         "norm.weight": (1 + 0.2 * torch.randn(128, device=dev)).to(BF)}
    return cfg, W, Cd


def _gdn_rows(cfg, n):
    Cd, hv = cfg.conv_dim, cfg.hv
    w = Cd + cfg.value_dim + 2 * hv
    buf = torch.empty(n, w + 24, dtype=BF, device=dev)  # wider buffer: rows are strided views
    buf[:, :Cd] = (torch.randn(n, Cd, device=dev) * 2).to(BF)
    buf[:, Cd:Cd + cfg.value_dim] = torch.randn(n, cfg.value_dim, device=dev).to(BF)
    buf[:, Cd + cfg.value_dim:Cd + cfg.value_dim + hv] = (torch.randn(n, hv, device=dev) * 2).to(BF)
    buf[:, Cd + cfg.value_dim + hv:w] = (torch.randn(n, hv, device=dev) * 3).to(BF)
    return buf[:, :w]


def _gdn_ref_rows(cfg, W, rows, conv, S):
    """Reference over rows from the given state (modified in place). Returns outputs [n, hv*128]."""
    Cd, hv = cfg.conv_dim, cfg.hv
    c1 = Cd + cfg.value_dim
    return torch.stack([R.deltanet_core(r[:Cd], r[Cd:c1], r[c1:c1 + hv], r[c1 + hv:], W, "", conv, S, cfg)
                        for r in rows])


@pytest.mark.parametrize("q", [2, 4, 8])
@pytest.mark.parametrize("hk,hv", [(16, 16), (16, 48), (2, 2), (1, 3)])
def test_gdn_step_n_vs_reference(hk, hv, q):
    """Speculative pattern: passes of n rows, keep m+1; every row (kept or not) vs the reference
    run from the committed state; committed state vs reference."""
    cfg, W, Cd = _gdn_setup(hk, hv)
    g = GDN(cfg, W, Cd, q)
    conv_ref = torch.zeros(Cd, 3, dtype=BF, device=dev)
    S_ref = torch.zeros(hv, 128, 128, device=dev)
    rng = torch.Generator().manual_seed(1)
    worst = 0.0
    for it, n in enumerate([1, 4, 8, 3, 1, 1, 5, 8, 2, 4]):
        rows = _gdn_rows(cfg, n)
        if it == 4:
            rows[0, Cd + cfg.value_dim + hv] = 30.0  # softplus threshold branch (a of head 0)
        keep = int(torch.randint(1, n + 1, (1,), generator=rng))
        S_before = S_ref.clone()
        out = g.run(rows, keep)
        # kernel S is now the committed state before this pass (pending rows replayed)
        torch.testing.assert_close(g.S, S_before, rtol=1e-3, atol=1e-4)
        c2, S2 = conv_ref.clone(), S_ref.clone()
        ref = _gdn_ref_rows(cfg, W, rows, c2, S2)
        diff = (out.float() - ref.float()).abs()
        assert (diff <= 2.0 ** -6 * ref.float().abs() + 1e-3).all(), (it, diff.max().item())
        assert (diff > 2.0 ** -7 * ref.float().abs() + 1e-6).float().mean().item() < 0.01
        worst = max(worst, diff.max().item())
        _gdn_ref_rows(cfg, W, rows[:keep], conv_ref, S_ref)  # advance committed reference state
    print(f"gdn hk={hk} hv={hv} max|dout| {worst:.2e}")


@pytest.mark.parametrize("q", [2, 4, 8])
@pytest.mark.parametrize("hk,hv", [(16, 48), (2, 2)])
def test_gdn_lazy_rollback_bit_exact(hk, hv, q):
    """Kept rows of n-row passes (with rejected rows thrown away) == plain 1-row decode of the
    same tokens, bit for bit, outputs and committed state."""
    cfg, W, Cd = _gdn_setup(hk, hv, seed=3)
    spec, plain = GDN(cfg, W, Cd, q), GDN(cfg, W, Cd, q)
    rng = torch.Generator().manual_seed(7)
    for it, n in enumerate([8, 1, 3, 8, 8, 2, 5, 1, 7, 4, 8]):
        rows = _gdn_rows(cfg, n)
        keep = int(torch.randint(1, n + 1, (1,), generator=rng))
        out = spec.run(rows, keep)
        for t in range(keep):
            o1 = plain.run(rows[t:t + 1], 1)
            assert torch.equal(out[t], o1[0]), (it, t)
    # one more 1-row pass each replays the last kept rows: committed states must be identical
    last = _gdn_rows(cfg, 1)
    a, b = spec.run(last, 1), plain.run(last, 1)
    assert torch.equal(a, b)
    assert torch.equal(spec.S, plain.S)


def test_gdn_step_nan_isolated_to_head():
    cfg, W, Cd = _gdn_setup(2, 2)
    g = GDN(cfg, W, Cd)
    rows = _gdn_rows(cfg, 1)
    rows[0, 2 * 256 + 128 + 5] = float("nan")  # a v channel of head 1
    out = g.run(rows, 1)
    assert torch.isnan(out[0, 128:]).all()
    assert not torch.isnan(out[0, :128]).any()


# ---------------------------------------------------------------------------------- attention
def _attn_cfg(hq, hkv):
    return R.Config(hidden=0, layers=1, inter=0, vocab=0, eps=1e-6, layer_types=["full_attention"],
                    hq=hq, hkv=hkv, head_dim=256, rope_theta=1e7, rotary_dim=64, hk=1, hv=1, dk=128,
                    dv=128, conv_k=4, tied=True)


def _inv_freq(cfg):
    rd = cfg.rotary_dim
    return (1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))).to(dev)


@pytest.mark.parametrize("hq,hkv,steps,start,n", [(8, 2, 20, 0, 1), (24, 4, 6, 0, 1), (8, 2, 3, 4094, 1),
                                                    (1, 1, 5, 125, 1), (8, 2, 24, 0, 8), (24, 4, 12, 100, 3)])
def test_attention(hq, hkv, steps, start, n):
    """Passes of n rows at positions start, start+n, ...: q/k vs reference (<= 1 ulp), output of
    every row vs reference attention over the causal prefix, and each row bit-identical to a
    1-row pass at the same position."""
    torch.manual_seed(1)
    cfg = _attn_cfg(hq, hkv)
    max_len = start + steps
    W = {"q_norm.weight": (0.3 * torch.randn(256, device=dev)).to(BF),
         "k_norm.weight": (0.3 * torch.randn(256, device=dev)).to(BF)}
    kc = torch.zeros(hkv, max_len, 256, dtype=BF, device=dev)
    vc = torch.zeros_like(kc)
    kc[:, :start] = torch.randn(hkv, start, 256, device=dev).to(BF)
    vc[:, :start] = torch.randn(hkv, start, 256, device=dev).to(BF)
    kc_ref, vc_ref = kc.clone(), vc.clone()
    inv = _inv_freq(cfg)
    NS = 16
    pos_t = torch.zeros(1, dtype=torch.int32, device=dev)
    po = torch.empty(8, hq, NS, 256, device=dev)
    pml = torch.empty(8, hq, NS, 2, device=dev)
    for p0 in range(start, start + steps, n):
        nn = min(n, start + steps - p0)
        w = hq * 512 + 2 * hkv * 256
        proj = torch.empty(nn, w + 8, dtype=BF, device=dev)
        proj[:, :hq * 512] = (torch.randn(nn, hq * 512, device=dev) * 3).to(BF)
        proj[:, hq * 512:hq * 512 + hkv * 256] = (torch.randn(nn, hkv * 256, device=dev) * 3).to(BF)
        proj[:, hq * 512 + hkv * 256:w] = torch.randn(nn, hkv * 256, device=dev).to(BF)
        qg, k, v = proj[:, :hq * 512], proj[:, hq * 512:hq * 512 + hkv * 256], proj[:, hq * 512 + hkv * 256:w]
        refs, qrs = [], []
        for t in range(nn):
            q_r, gate, k_r, v_r = R.attention_qkv_from_proj(qg[t], k[t], v[t], W, "", p0 + t, cfg)
            kc_ref[:, p0 + t], vc_ref[:, p0 + t] = k_r, v_r
            qrs.append(q_r.reshape(-1))
        for t in range(nn):
            gate = qg[t].view(hq, 512)[:, 256:].reshape(-1)
            refs.append(R.attention_core(qrs[t].view(hq, 256), gate, kc_ref, vc_ref, p0 + t + 1, cfg))
        pos_t.fill_(p0)
        q_out = torch.empty(nn, hq * 256, dtype=BF, device=dev)
        C.attn_prep(qg, k, v, W["q_norm.weight"], W["k_norm.weight"], inv, pos_t, q_out, kc, vc, 1e-6)
        ok, err = ulp_close(q_out, torch.stack(qrs), ulps=1)
        assert ok, ("q", p0, err)
        ok, err = ulp_close(kc[:, p0:p0 + nn], kc_ref[:, p0:p0 + nn], ulps=1)
        assert ok, ("k", p0, err)
        assert torch.equal(vc[:, p0:p0 + nn], vc_ref[:, p0:p0 + nn])
        # attention on the reference cache so the check is independent of q/k flips
        qr = torch.stack(qrs).contiguous()
        out = torch.empty(nn, hq * 256, dtype=BF, device=dev)
        C.attn_decode(qr, kc_ref, vc_ref, pos_t, qg, po, pml, out)
        for t in range(nn):
            diff = (out[t].float() - refs[t].float()).abs()
            assert (diff <= 2.0 ** -6 * refs[t].float().abs() + 2e-3).all(), (p0 + t, diff.max().item())
            # bit-exact vs a 1-row pass at that position
            o1 = torch.empty(1, hq * 256, dtype=BF, device=dev)
            C.attn_decode(qr[t:t + 1], kc_ref, vc_ref, pos_t + t, qg[t:t + 1], po, pml, o1)
            assert torch.equal(o1[0], out[t])


def test_rope_matches_hf_mrope():
    """for text positions the interleaved mRoPE of HF equals 1-D RoPE."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from transformers import AutoConfig
    from conftest import M08
    tc = AutoConfig.from_pretrained(M08).text_config
    rot = Qwen3_5TextRotaryEmbedding(tc).to(dev)
    cfg = R.Config.from_dir(M08)
    pos = torch.tensor([0, 1, 5, 127, 2047, 100000], device=dev)
    x = torch.zeros(1, len(pos), 8, dtype=BF, device=dev)
    cos, sin = rot(x, pos.view(1, 1, -1).expand(3, 1, -1))
    for i, p in enumerate(pos.tolist()):
        c, s = R.rope_cos_sin(p, cfg, dev)
        assert torch.equal(cos[0, i], c) and torch.equal(sin[0, i], s)
