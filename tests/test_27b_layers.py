"""per-layer checks on Qwen3.8-27B, each layer loaded alone from safetensors.

The 0.8B has Hv = Hk = 16, so it never exercises the DeltaNet repeat_interleave head mapping
(27B: Hv = 48, Hk = 16) nor the 27B attention shapes (Hq 24 / Hkv 4). Here HF modules
(transformers 5.20.1) are built for single layers, fed T tokens of realistic hidden states
(prefill path, no cache), and compared with reference.py (recurrent step loop) and with the CUDA
kernels (gemv + gdn_step / attn kernels, step by step).

Tolerance: HF's prefill uses the chunked delta rule / SDPA, so it differs from the recurrent
step path by reassociation, not by math: relative L2 error < 2e-2 and per-token cosine > 0.999.
A head-mapping bug (tile instead of repeat_interleave, wrong q/gate split, wrong GQA group) gives
relative errors of order 1. Reference vs CUDA (same algorithm) must agree to < 1e-2 rel L2.
"""
import pytest
import torch
from conftest import M27

from qengine import reference as R

pytestmark = pytest.mark.skipif(not (M27 / "model-00018-of-00018.safetensors").exists(),
                                reason="27B weights missing")
dev = "cuda"
BF = torch.bfloat16
T = 32


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def min_cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min().item()


@pytest.fixture(scope="module")
def cfgs():
    from transformers import AutoConfig
    tc = AutoConfig.from_pretrained(M27).text_config
    tc._attn_implementation = "sdpa"
    return tc, R.Config.from_dir(M27)


def _inputs(W, p, cfg):
    torch.manual_seed(0)
    h = torch.randn(T, cfg.hidden, device=dev).to(BF) * 0.5
    return R.rmsnorm1p(h, W[p + "input_layernorm.weight"], cfg.eps)  # realistic mixer input


def _load_sub(W, prefix):
    return {k[len(prefix):]: v for k, v in W.items() if k.startswith(prefix)}


def test_27b_config_head_layout(cfgs):
    _, cfg = cfgs
    assert (cfg.hk, cfg.hv, cfg.dk, cfg.dv, cfg.hq, cfg.hkv, cfg.head_dim) == (16, 48, 128, 128, 24, 4, 256)
    assert not cfg.tied and cfg.rotary_dim == 64 and cfg.layer_types[3] == "full_attention"


@torch.no_grad()
def test_27b_deltanet_layer0(cfgs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from qengine import _C
    tc, cfg = cfgs
    W = R.load_weights(M27, names=lambda n: n.startswith("layers.0."))
    p = "layers.0.linear_attn."
    x = _inputs(W, "layers.0.", cfg)
    hf = Qwen3_5GatedDeltaNet(tc, 0).to(dev, BF).eval()
    hf.load_state_dict(_load_sub(W, p), strict=True)
    y_hf = hf(x[None])[0]
    # reference: recurrent step loop
    conv = torch.zeros(cfg.conv_dim, 3, dtype=BF, device=dev)
    S = torch.zeros(cfg.hv, cfg.dk, cfg.dv, device=dev)
    y_ref = torch.stack([R.deltanet_step(x[t], W, p, conv, S, cfg) for t in range(T)])
    # CUDA: gemv projections + gdn_step_n, run as a speculative pattern (passes of 1..5 rows,
    # all kept) so the 27B head mapping is checked through the multi-row path too
    ring = torch.zeros(16, cfg.conv_dim, dtype=BF, device=dev)
    Sc = torch.zeros_like(S)
    st = [torch.zeros(8, cfg.hv, 128, device=dev), torch.zeros(8, cfg.hv, 128, device=dev),
          torch.zeros(8, cfg.hv, 2, device=dev)]
    ctl = torch.zeros(8, dtype=torch.int32, device=dev)
    Win = torch.cat([W[p + "in_proj_qkv.weight"], W[p + "in_proj_z.weight"], W[p + "in_proj_b.weight"],
                     W[p + "in_proj_a.weight"]])
    c0, c1 = cfg.conv_dim, cfg.conv_dim + cfg.value_dim
    ys, t = [], 0
    for n in [1, 3, 5, 2, 8, 4, 1, 8]:
        n = min(n, T - t)
        if n == 0:
            break
        pr = torch.empty(n, Win.shape[0], dtype=BF, device=dev)
        _C.gemv_bf16(Win, x[t:t + n], pr)
        o = torch.empty(n, cfg.value_dim, dtype=BF, device=dev)
        _C.gdn_step_n(pr[:, :c0], pr[:, c0:c1], pr[:, c1:c1 + cfg.hv], pr[:, c1 + cfg.hv:], ring,
                      W[p + "conv1d.weight"].contiguous(), W[p + "A_log"], W[p + "dt_bias"], W[p + "norm.weight"],
                      Sc, *st, ctl, o, cfg.hk, cfg.eps)
        ctl[0] += n
        ctl[1] = n
        y = torch.empty(n, cfg.hidden, dtype=BF, device=dev)
        _C.gemv_bf16(W[p + "out_proj.weight"], o, y)
        ys.extend(y)
        t += n
    y_cu = torch.stack(ys)
    print(f"\n27B DeltaNet L0: ref vs HF rel {rel(y_ref, y_hf):.2e} cos {min_cos(y_ref, y_hf):.5f}; "
          f"CUDA vs ref rel {rel(y_cu, y_ref):.2e}; CUDA vs HF rel {rel(y_cu, y_hf):.2e}")
    assert rel(y_ref, y_hf) < 2e-2 and min_cos(y_ref, y_hf) > 0.999
    assert rel(y_cu, y_ref) < 1e-2 and rel(y_cu, y_hf) < 2e-2
    # negative control: the wrong (tiled) qk->v head mapping must be caught by this comparison
    conv.zero_(); S.zero_()
    y_bad = torch.stack([R.deltanet_step(x[t], W, p, conv, S, cfg, _tile=True) for t in range(T)])
    print(f"negative control (tiled v heads) rel vs HF {rel(y_bad, y_hf):.2e}")
    assert rel(y_bad, y_hf) > 0.1


@torch.no_grad()
def test_27b_attention_layer3(cfgs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding
    from qengine import _C
    tc, cfg = cfgs
    W = R.load_weights(M27, names=lambda n: n.startswith("layers.3."))
    p = "layers.3.self_attn."
    x = _inputs(W, "layers.3.", cfg)
    hf = Qwen3_5Attention(tc, 3).to(dev, BF).eval()
    hf.load_state_dict(_load_sub(W, p), strict=True)
    rot = Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos_ids = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
    y_hf = hf(x[None], position_embeddings=rot(x[None], pos_ids), attention_mask=None)[0][0]
    kc = torch.zeros(cfg.hkv, T, cfg.head_dim, dtype=BF, device=dev)
    vc = torch.zeros_like(kc)
    y_ref = torch.stack([R.attention_step(x[t], W, p, kc, vc, t, cfg) for t in range(T)])
    # CUDA
    kc2, vc2 = torch.zeros_like(kc), torch.zeros_like(vc)
    mk = lambda n: torch.empty(n, dtype=BF, device=dev)
    qg, k, v, q, o, y = (mk(cfg.hq * 512), mk(cfg.hkv * 256), mk(cfg.hkv * 256), mk(cfg.hq * 256),
                         mk(cfg.hq * 256), mk(cfg.hidden))
    rd = cfg.rotary_dim
    inv = (1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))).to(dev)
    po = torch.empty(8, cfg.hq, 16, 256, device=dev)
    pml = torch.empty(8, cfg.hq, 16, 2, device=dev)
    pos = torch.zeros(1, dtype=torch.int32, device=dev)
    Wqkv = torch.cat([W[p + "q_proj.weight"], W[p + "k_proj.weight"], W[p + "v_proj.weight"]])
    q0, k1 = cfg.hq * 512, cfg.hq * 512 + cfg.hkv * 256
    ys, t = [], 0
    for n in [1, 4, 8, 2, 8, 1, 8]:  # multi-row passes (verify path)
        n = min(n, T - t)
        if n == 0:
            break
        pos.fill_(t)
        pr = torch.empty(n, Wqkv.shape[0], dtype=BF, device=dev)
        _C.gemv_bf16(Wqkv, x[t:t + n], pr)
        q = torch.empty(n, cfg.hq * 256, dtype=BF, device=dev)
        _C.attn_prep(pr[:, :q0], pr[:, q0:k1], pr[:, k1:], W[p + "q_norm.weight"], W[p + "k_norm.weight"], inv,
                     pos, q, kc2, vc2, cfg.eps)
        o = torch.empty(n, cfg.hq * 256, dtype=BF, device=dev)
        _C.attn_decode(q, kc2, vc2, pos, pr[:, :q0], po, pml, o)
        y = torch.empty(n, cfg.hidden, dtype=BF, device=dev)
        _C.gemv_bf16(W[p + "o_proj.weight"], o, y)
        ys.extend(y)
        t += n
    y_cu = torch.stack(ys)
    print(f"\n27B attention L3: ref vs HF rel {rel(y_ref, y_hf):.2e} cos {min_cos(y_ref, y_hf):.5f}; "
          f"CUDA vs ref rel {rel(y_cu, y_ref):.2e}; CUDA vs HF rel {rel(y_cu, y_hf):.2e}")
    assert rel(y_ref, y_hf) < 2e-2 and min_cos(y_ref, y_hf) > 0.999
    assert rel(y_cu, y_ref) < 1e-2 and rel(y_cu, y_hf) < 2e-2


@torch.no_grad()
def test_27b_mtp_layer(cfgs):
    """MTP: weights load with the expected shapes; its decoder layer (a gated-attention layer,
    VMTP 118-124) matches an HF Qwen3_5DecoderLayer of type full_attention; fc input order
    [embedding, hidden] is checked against the shape only (HF has no MTP)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
    tc, cfg = cfgs
    W = R.load_weights(M27, names=lambda n: n.startswith("mtp."))
    assert W["mtp.fc.weight"].shape == (cfg.hidden, 2 * cfg.hidden)
    for n in ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
        assert W[n].shape == (cfg.hidden,)
    hf = Qwen3_5DecoderLayer(tc, 3).to(dev, BF).eval()  # layer 3 is full_attention
    hf.load_state_dict(_load_sub(W, "mtp.layers.0."), strict=True)
    rot = Qwen3_5TextRotaryEmbedding(tc).to(dev)
    torch.manual_seed(0)
    h = torch.randn(T, cfg.hidden, device=dev).to(BF)
    pos_ids = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
    y_hf = hf(h[None], position_embeddings=rot(h[None], pos_ids), attention_mask=None)[0]
    mst = R.MTPState(cfg, T)
    ys = []
    for t in range(T):
        mst.pos = t
        ys.append(R.decoder_layer(h[t], W, 0, mst, cfg, prefix="mtp.layers."))
    y_ref = torch.stack(ys)
    print(f"\n27B MTP layer: ref vs HF decoder layer rel {rel(y_ref, y_hf):.2e}")
    assert rel(y_ref, y_hf) < 2e-2
    # full MTP head math runs (lm_head is the main model's, loaded alone: 2.5 GB bf16)
    W.update(R.load_weights(M27, names=lambda n: n in ("embed_tokens.weight", "lm_head.weight")))
    logits, _ = R.mtp_step(W, R.MTPState(cfg, 4), h[0], 1000, 0, cfg)
    assert logits.shape == (cfg.vocab,) and torch.isfinite(logits.float()).all()
