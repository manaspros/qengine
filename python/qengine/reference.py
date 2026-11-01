"""Pure-torch reference of (qwen3_5 text path), one decode step at a time.

Mirrors HF transformers 5.20.1 `modeling_qwen3_5.py` on its *decode* path (T = 1, cache present),
with the model loaded in BF16 (HF casts every parameter, including the 0.8B's F32 `A_log` and
`linear_attn.norm.weight`, to BF16). Every place HF rounds to BF16 is reproduced here, so the
reference and HF differ only by summation order inside matmuls / reductions.

Prefill is a loop of decode steps (fast prefill is a later milestone).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
F32 = torch.float32


# ----------------------------------------------------------------------------------------------
# Config and weights
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    hidden: int
    layers: int
    inter: int
    vocab: int
    eps: float
    layer_types: list[str]
    # gated attention
    hq: int
    hkv: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    # gated deltanet
    hk: int  # linear_num_key_heads
    hv: int  # linear_num_value_heads
    dk: int
    dv: int
    conv_k: int
    tied: bool
    mtp_layers: int = 0
    eos: list[int] = field(default_factory=list)

    @property
    def key_dim(self) -> int:
        return self.hk * self.dk

    @property
    def value_dim(self) -> int:
        return self.hv * self.dv

    @property
    def conv_dim(self) -> int:  # channels of the depthwise conv = q + k + v
        return 2 * self.key_dim + self.value_dim

    @staticmethod
    def from_dir(path: str | Path) -> "Config":
        raw = json.loads((Path(path) / "config.json").read_text())
        t = raw.get("text_config", raw)
        rp = t["rope_parameters"]
        eos = t.get("eos_token_id", [])
        gen = Path(path) / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id", eos)
        return Config(
            hidden=t["hidden_size"], layers=t["num_hidden_layers"], inter=t["intermediate_size"],
            vocab=t["vocab_size"], eps=t["rms_norm_eps"], layer_types=list(t["layer_types"]),
            hq=t["num_attention_heads"], hkv=t["num_key_value_heads"], head_dim=t["head_dim"],
            rope_theta=float(rp["rope_theta"]),
            rotary_dim=int(t["head_dim"] * rp.get("partial_rotary_factor", 1.0)),
            hk=t["linear_num_key_heads"], hv=t["linear_num_value_heads"],
            dk=t["linear_key_head_dim"], dv=t["linear_value_head_dim"],
            conv_k=t["linear_conv_kernel_dim"],
            tied=bool(raw.get("tie_word_embeddings", t.get("tie_word_embeddings", False))),
            mtp_layers=t.get("mtp_num_hidden_layers", 0),
            eos=eos if isinstance(eos, list) else [eos],
        )


PREFIX = "model.language_model."


def load_weights(path: str | Path, device="cuda", names=None, dtype=BF16) -> dict[str, torch.Tensor]:
    """Load text (and mtp) tensors from safetensors. Keys lose the `model.language_model.` prefix.

    `names`: optional predicate on the short name, to load a subset (e.g. one 27B layer).
    Floating tensors are cast to `dtype` (BF16, as HF does with dtype=bfloat16).
    """
    from safetensors import safe_open

    path = Path(path)
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        files = sorted(path.glob("*.safetensors"))
        wmap = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    wmap[k] = f.name
    by_file: dict[str, list[str]] = {}
    for k, f in wmap.items():
        if k.startswith("model.visual") or k.startswith("model.vision"):
            continue
        short = k[len(PREFIX):] if k.startswith(PREFIX) else k
        if names is not None and not names(short):
            continue
        by_file.setdefault(f, []).append(k)
    out = {}
    for f, keys in by_file.items():
        with safe_open(path / f, "pt", device="cpu") as h:
            for k in keys:
                t = h.get_tensor(k)
                if t.is_floating_point():
                    t = t.to(dtype)
                short = k[len(PREFIX):] if k.startswith(PREFIX) else k
                out[short] = t.to(device)
    return out


# ----------------------------------------------------------------------------------------------
# Elementary ops (each mirrors an HF function)
# ----------------------------------------------------------------------------------------------
def rmsnorm1p(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """zero-centred RMSNorm, y = norm(x_f32) * (1 + w_f32), cast to x.dtype."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(x.dtype)


def rmsnorm_gated(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """y = w * norm(x).to(bf16); y = y * silu(z_f32); cast bf16."""
    dt = x.dtype
    xf = x.float()
    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dt)
    y = w * n  # both bf16 -> bf16 rounding, as HF (weight was cast to bf16 at load)
    y = y * F.silu(z.float())  # promotes to fp32
    return y.to(dt)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 x @ W^T (fp32 accumulate in cuBLAS, one rounding to bf16)."""
    return F.linear(x, w)


def rope_cos_sin(pos: int, cfg: Config, device, dtype=BF16):
    """1-D RoPE over rotary_dim dims (text positions make mRoPE 1-D). cos/sin computed in
    fp32 then cast to the activation dtype (HF 181-186)."""
    rd = cfg.rotary_dim
    # HF builds inv_freq on the CPU at init, then moves it; do the same so the bits match
    inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=F32) / rd))).to(device)
    freqs = torch.tensor(float(pos), device=device) * inv_freq  # fp32, same op order as HF
    emb = torch.cat([freqs, freqs])
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., head_dim] bf16; rotate-half on the first rotary_dim dims, pass the rest (HF 652-676).
    The arithmetic is bf16 elementwise (each op rounds), exactly as HF."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return torch.cat([xr * cos + rot * sin, xp], dim=-1)


# ----------------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------------
class State:
    """Per-sequence decode state. conv: [conv_dim, conv_k-1] bf16 (oldest first, HF layout);
    S: [hv, dk, dv] fp32; K/V cache: [hkv, max_len, head_dim] bf16."""

    def __init__(self, cfg: Config, max_len: int, device="cuda", n_attn_extra: int = 0):
        self.pos = 0
        self.conv, self.S, self.k, self.v = {}, {}, {}, {}
        for l, t in enumerate(cfg.layer_types):
            if t == "linear_attention":
                self.conv[l] = torch.zeros(cfg.conv_dim, cfg.conv_k - 1, dtype=BF16, device=device)
                self.S[l] = torch.zeros(cfg.hv, cfg.dk, cfg.dv, dtype=F32, device=device)
            else:
                self.k[l] = torch.zeros(cfg.hkv, max_len, cfg.head_dim, dtype=BF16, device=device)
                self.v[l] = torch.zeros(cfg.hkv, max_len, cfg.head_dim, dtype=BF16, device=device)


# ----------------------------------------------------------------------------------------------
# Layers (one token)
# ----------------------------------------------------------------------------------------------
def deltanet_step(x, W: dict, p: str, conv_state, S, cfg: Config, lin=linear, _tile=False):
    """for one token. x [H] bf16. Updates conv_state and S in place. Returns [H] bf16."""
    qkv = lin(x, W[p + "in_proj_qkv.weight"])  # [conv_dim]
    z = lin(x, W[p + "in_proj_z.weight"])
    b = lin(x, W[p + "in_proj_b.weight"])
    a = lin(x, W[p + "in_proj_a.weight"])
    o = deltanet_core(qkv, z, b, a, W, p, conv_state, S, cfg, _tile)
    return lin(o, W[p + "out_proj.weight"])


def deltanet_core(qkv, z, b, a, W, p, conv_state, S, cfg: Config, _tile=False):
    """Everything between the input projections and out_proj (the CUDA gdn_step kernel's job).
    Returns the gated-normed output [hv*dv] bf16."""
    z = z.view(cfg.hv, cfg.dv)
    # causal_conv1d_update (HF 236-250): conv over [state, x] in bf16, then silu (bf16 round each)
    wc = W[p + "conv1d.weight"].view(cfg.conv_dim, cfg.conv_k)
    buf = torch.cat([conv_state, qkv[:, None]], dim=-1)  # [C, 4] bf16
    conv_state.copy_(buf[:, 1:])
    conv = (buf.float() * wc.float()).sum(-1).to(BF16)  # F.conv1d bf16: fp32 accumulate, 1 rounding
    qkv = F.silu(conv)  # bf16 in -> bf16 out
    q, k, v = torch.split(qkv, [cfg.key_dim, cfg.key_dim, cfg.value_dim])
    q = q.view(cfg.hk, cfg.dk)
    k = k.view(cfg.hk, cfg.dk)
    v = v.view(cfg.hv, cfg.dv)
    beta = b.sigmoid()  # bf16
    g = -W[p + "A_log"].float().exp() * F.softplus(a.float() + W[p + "dt_bias"])  # fp32 [hv]
    r = cfg.hv // cfg.hk
    if r > 1 and _tile:  # WRONG mapping (j % hk), only for the negative-control test
        q, k = q.repeat(r, 1), k.repeat(r, 1)
    elif r > 1:  # repeat_interleave: v head j uses qk head j // r
        q = q.repeat_interleave(r, dim=0)
        k = k.repeat_interleave(r, dim=0)
    qf = l2norm(q.float()) / math.sqrt(cfg.dk)
    kf = l2norm(k.float())
    vf, bf = v.float(), beta.float()
    # torch_recurrent_gated_delta_rule (HF 466-477), per v head, fp32 state
    S.mul_(g.exp()[:, None, None])
    kv = (S * kf[:, :, None]).sum(dim=1)  # [hv, dv]
    delta = (vf - kv) * bf[:, None]
    S.add_(kf[:, :, None] * delta[:, None, :])
    o = (S * qf[:, :, None]).sum(dim=1).to(BF16)  # [hv, dv], cast to input dtype
    return rmsnorm_gated(o, z, W[p + "norm.weight"], cfg.eps).reshape(-1)


def attention_qkv(x, W, p, pos, cfg: Config, lin=linear):
    """q (normed, roped) [hq, hd], gate [hq*hd], k (normed, roped) [hkv, hd], v [hkv, hd]."""
    qg = lin(x, W[p + "q_proj.weight"])
    return attention_qkv_from_proj(qg, lin(x, W[p + "k_proj.weight"]), lin(x, W[p + "v_proj.weight"]),
                                   W, p, pos, cfg)


def attention_qkv_from_proj(qg, k, v, W, p, pos, cfg: Config):
    hd = cfg.head_dim
    qg = qg.view(cfg.hq, 2 * hd)  # per-head interleaved [q | gate]
    q, gate = qg[:, :hd], qg[:, hd:].reshape(-1)
    q = rmsnorm1p(q, W[p + "q_norm.weight"], cfg.eps)
    k = rmsnorm1p(k.view(cfg.hkv, hd), W[p + "k_norm.weight"], cfg.eps)
    v = v.view(cfg.hkv, hd)
    cos, sin = rope_cos_sin(pos, cfg, qg.device)
    return apply_rope(q, cos, sin), gate, apply_rope(k, cos, sin), v


def attention_core(q, gate, kc, vc, n, cfg: Config):
    """softmax(q K^T / sqrt(hd)) V over cache positions [0, n), GQA group = hq/hkv, fp32 softmax;
    output * sigmoid(gate) in bf16 (HF 804). Returns [hq*hd] bf16."""
    grp = cfg.hq // cfg.hkv
    K = kc[:, :n].float().repeat_interleave(grp, 0)  # [hq, n, hd]
    V = vc[:, :n].float().repeat_interleave(grp, 0)
    s = torch.einsum("hd,hnd->hn", q.float(), K) * cfg.head_dim ** -0.5
    pr = torch.softmax(s, dim=-1)
    o = torch.einsum("hn,hnd->hd", pr, V).to(BF16).reshape(-1)
    return o * torch.sigmoid(gate)


def attention_step(x, W, p, kc, vc, pos, cfg: Config, lin=linear):
    q, gate, k, v = attention_qkv(x, W, p, pos, cfg, lin)
    kc[:, pos] = k
    vc[:, pos] = v
    o = attention_core(q, gate, kc, vc, pos + 1, cfg)
    return lin(o, W[p + "o_proj.weight"])


def mlp(x, W, p, lin=linear):
    """SwiGLU; silu(g) and the product each round to bf16 (HF 821-824)."""
    g = lin(x, W[p + "gate_proj.weight"])
    u = lin(x, W[p + "up_proj.weight"])
    return lin(F.silu(g) * u, W[p + "down_proj.weight"])


def decoder_layer(h, W, l, st: State, cfg: Config, prefix="layers.", lin=linear):
    p = f"{prefix}{l}."
    x = rmsnorm1p(h, W[p + "input_layernorm.weight"], cfg.eps)
    if cfg.layer_types[l] == "linear_attention" and prefix == "layers.":
        x = deltanet_step(x, W, p + "linear_attn.", st.conv[l], st.S[l], cfg, lin)
    else:
        x = attention_step(x, W, p + "self_attn.", st.k[l], st.v[l], st.pos, cfg, lin)
    h = h + x
    x = rmsnorm1p(h, W[p + "post_attention_layernorm.weight"], cfg.eps)
    return h + mlp(x, W, p + "mlp.", lin)


def lm_head(W, cfg):
    return W["embed_tokens.weight"] if cfg.tied else W["lm_head.weight"]


@torch.no_grad()
def decode_step(W, st: State, tok: int, cfg: Config, lin=linear, return_hidden=False):
    """One token through the model; returns bf16 logits [vocab] (HF returns bf16 logits)."""
    h = W["embed_tokens.weight"][tok]
    for l in range(cfg.layers):
        h = decoder_layer(h, W, l, st, cfg, lin=lin)
    st.pos += 1
    hf = rmsnorm1p(h, W["norm.weight"], cfg.eps)
    logits = lin(hf, lm_head(W, cfg))
    return (logits, hf) if return_hidden else logits  # hidden after the final norm (MTP input)


@torch.no_grad()
def run_sequence(W, cfg: Config, ids: list[int], max_len=None, keep_logits=True):
    """Prefill-by-steps: feed every token of `ids`, return logits [len(ids), vocab] (bf16, CPU
    argmax-friendly) or None."""
    st = State(cfg, max_len or len(ids))
    out = []
    for t in ids:
        lg = decode_step(W, st, t, cfg)
        if keep_logits:
            out.append(lg)
    return torch.stack(out) if keep_logits else None


@torch.no_grad()
def greedy(W, cfg: Config, prompt: list[int], n_new: int):
    st = State(cfg, len(prompt) + n_new)
    for t in prompt[:-1]:
        decode_step(W, st, t, cfg)
    tok, gen = prompt[-1], []
    for _ in range(n_new):
        tok = int(decode_step(W, st, tok, cfg).argmax())
        gen.append(tok)
    return gen


# ----------------------------------------------------------------------------------------------
# MTP head, math from vLLM qwen3_5_mtp.py; not used by HF
# ----------------------------------------------------------------------------------------------
class MTPState:
    def __init__(self, cfg: Config, max_len: int, device="cuda"):
        self.pos = 0
        self.k = {0: torch.zeros(cfg.hkv, max_len, cfg.head_dim, dtype=BF16, device=device)}
        self.v = {0: torch.zeros(cfg.hkv, max_len, cfg.head_dim, dtype=BF16, device=device)}
        self.conv, self.S = {}, {}


@torch.no_grad()
def mtp_step(W, mst: MTPState, h_prev: torch.Tensor, next_tok: int, pos: int, cfg: Config, lin=linear):
    """One MTP draft row (vLLM qwen3_5_mtp.py 160-193 convention):
    h_prev = main-model hidden state AFTER the final norm at position `pos` (the vector the
    lm_head consumed), next_tok = the token at pos + 1; the MTP attention layer runs at `pos`
    (vLLM keeps the target positions). For chained drafts pass the returned hidden (after
    mtp.norm) as h_prev. Returns (logits [vocab] bf16, hidden after mtp.norm [H])."""
    e = rmsnorm1p(W["embed_tokens.weight"][next_tok], W["mtp.pre_fc_norm_embedding.weight"], cfg.eps)
    hh = rmsnorm1p(h_prev, W["mtp.pre_fc_norm_hidden.weight"], cfg.eps)
    u = lin(torch.cat([e, hh]), W["mtp.fc.weight"])  # order [embedding, hidden]
    mst.pos = pos
    u = decoder_layer(u, W, 0, mst, cfg, prefix="mtp.layers.", lin=lin)
    hn = rmsnorm1p(u, W["mtp.norm.weight"], cfg.eps)
    return lin(hn, lm_head(W, cfg)), hn


@torch.no_grad()
def run_sequence_layerwise(layer_W, cfg: Config, ids: list[int], embed: torch.Tensor, lin=linear):
    """Same math as run_sequence, but layer-major: all tokens go through layer 0, then layer 1,
    ... (each layer's recurrent / KV state only depends on that layer's inputs), so only one
    layer's weights are needed at a time. layer_W(l) returns that layer's weights with the usual
    `layers.{l}.` names. Returns the hidden states [T, H] BEFORE the final norm."""
    T = len(ids)
    h = embed[torch.tensor(ids, device=embed.device)].clone()
    st = State(cfg, T, device=embed.device)
    for l in range(cfg.layers):
        W = layer_W(l)
        for t in range(T):
            st.pos = t
            h[t] = decoder_layer(h[t], W, l, st, cfg, lin=lin)
        del W
    return h
