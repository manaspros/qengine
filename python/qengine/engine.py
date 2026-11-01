"""qengine decode engine (BF16 or pluggable linear backend, batch 1, qwen3_5 text path).

Passes. A pass runs n = 1..8 rows (tokens at positions pos .. pos+n-1) through the model:
  * plain decode   : n = 1, the row is the last generated token.
  * verify (spec)  : n = k+1, rows = [last token, draft_1 .. draft_k]; row r's argmax is the
                     target's token for position pos+r+1; the longest prefix of matching drafts is
                     kept plus the target's own next token (the "bonus").
  * prefill        : n <= 8 prompt tokens per pass (`prefill`), all rows kept.
Every pass is one CUDA graph (one graph per (n, mtp, k) variant, captured on first use) and all
bookkeeping happens on the device (`_C.accept` / `_C.commit`): one replay = one decode step or
one full speculative cycle (verify + MTP rows + k-1 chained drafts), no host sync inside.

Rollback. Attention K/V rows of rejected tokens are simply overwritten later (pos only advances
over kept rows). Gated DeltaNet uses lazy rollback: a pass stages its
rows' (k, v, decay, beta) and writes no state; the next pass replays the kept rows into the
committed state first. Conv history is a 16-slot ring keyed by position (no rollback needed).
Every kernel computes row r of an n-row pass with the same fp32 op order as a 1-row pass, so
speculative greedy output equals plain greedy output token for token.

MTP (vLLM convention): row r of the MTP pass takes the main model's hidden state
after the final norm at position pos+r and the true next token (am[r]); its attention layer runs
at position pos+r. Chained drafts feed back the hidden state after mtp.norm.

Linear layers go through a pluggable backend (see `BF16Linear`):

    handle = backend.prepare(W)      # W: [N, K] bf16 CUDA tensor, called once at load
    backend(x, handle, out)          # x: [n, K] bf16, out: [n, N] bf16, n = 1..8; both 2-D with
                                     # stride(1) == 1 and any row stride (views into a buffer);
                                     # must not allocate or sync (captured in CUDA graphs);
                                     # row r of out must not depend on n (bit-exact verify).

Projections that read the same input are fused at load (row concatenation, numerically identical
for a row-wise GEMV): DeltaNet [qkv | z | b | a], attention [q+gate | k | v], MLP [gate | up].
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import reference as R
from . import _C

BF16 = torch.bfloat16
I32 = torch.int32
NMAX = 8  # max rows per pass (verify depth k <= 7)


class BF16Linear:
    """Default backend: BF16 GEMV for 1..8 rows (weights read once per pass, fp32 accumulate)."""

    name = "bf16"

    def prepare(self, w: torch.Tensor):
        return w.contiguous()

    def __call__(self, x: torch.Tensor, w, out: torch.Tensor) -> None:
        _C.gemv_bf16(w, x, out)


class TorchLinear:
    """cuBLAS backend for debugging (graph-capturable; rows may differ from n=1 by summation
    order, so it does not guarantee spec == plain bit-exactness)."""

    name = "torch"

    def prepare(self, w):
        return w.contiguous()

    def __call__(self, x, w, out):
        torch.matmul(x, w.t(), out=out) if out.is_contiguous() else out.copy_(x @ w.t())


class W4Linear:
    """Adapter over the W4A16 GEMV (python/qengine/quant.py + qengine._w4).
    This class is the only place that touches that API, so kernel-side changes stay local.

    - prepare(W) -> W4Packed (RTN int4, g128, fp16 scales), quantized on the GPU.
    - prepare_gate_up(Wg, Wu) / gate_up(x, h, out): fused SwiGLU epilogue, out = silu(g) * u.
    - The kernel is bitwise deterministic in every config (no float atomics; partials summed in
      a fixed order) and row r of an n-row call equals the 1-row call bitwise, so
      verify == plain decode (tests/test_gemv_w4.py::test_bitwise_repeatable / _row_independent).
    """

    name = "w4"
    cache_tag = "w4-rtn-g128-v1"  # bump when the packed layout or the quantizer changes

    # Launch options: empty = the kernel's shape-based auto policy (mode/warps/unroll/ksplit,
    # never dependent on n) with PDL on.
    DEFAULT_KW = {}

    def __init__(self, ksplit_max=64, **kw):
        self.ksplit_max = ksplit_max
        self.kw = {**self.DEFAULT_KW, **kw}  # passed through to _w4 (warps, unroll, pdl, ...)

    def prepare(self, w):
        from .quant import quantize_w4
        return quantize_w4(w)

    def prepare_gate_up(self, wg, wu):
        from .quant import quantize_w4_gate_up
        return quantize_w4_gate_up(wg, wu)

    def _ks(self, p, n):
        from . import _w4
        ks = _w4.auto_ksplit(p.qw.shape[0] * 16, p.qw.shape[1] * 64, n, self.kw.get("warps", 8))
        return min(ks, self.ksplit_max)

    def __call__(self, x, p, out):
        from . import _w4
        _w4.gemv(x, p.qw, p.scales, out, p.M, ksplit=self._ks(p, x.shape[0]), **self.kw)

    def gate_up(self, x, p, out):
        from . import _w4
        _w4.gemv_swiglu(x, p.qw, p.scales, out, p.M, ksplit=self._ks(p, x.shape[0]), **self.kw)

    @torch.no_grad()
    def prepare_from_grid(self, Wq: torch.Tensor, rows_per_step: int = 2048):
        """Pack a matrix whose values already lie on quant.py's grid (q * s, q in [-8, 7], fp16
        s per 128-group, stored rounded to bf16, e.g. the GPTQ lm_head from bench/quality/ppl.py)
        WITHOUT re-quantizing. Per group: recover s and q (fp16 scale candidates near
        absmax/7 and absmax/8, keep the one that reproduces the stored values best), then build
        an fp32 matrix that quantize_w4 maps to exactly (s, q): entries q * s, except q = -8
        entries become -7.5 s (rounds half-even to -8) or, if the group has no -8, one q = 7
        entry becomes 7.5 s (rounds to 8, clamps to 7), so absmax = 7.5 s pins the scale.
        Returns (packed, stats) with the fraction of weights reproduced exactly."""
        from .quant import quantize_w4
        M, K = Wq.shape
        assert K % 128 == 0
        Wp = torch.empty(M, K, dtype=torch.float32, device=Wq.device)
        exact, total, fallback = 0, 0, 0
        for r0 in range(0, M, rows_per_step):
            v = Wq[r0:r0 + rows_per_step].float().view(-1, K // 128, 128)  # [r, G, 128]
            A = v.abs().amax(-1, keepdim=True)
            cands = []
            for qm in (7.0, 8.0):
                s0 = (A / qm).to(torch.float16)
                for d in (-2, -1, 0, 1, 2):  # neighbouring fp16 values of the estimate
                    bits = s0.view(torch.int16) + d
                    cands.append(bits.view(torch.float16).float())
            best_s, best_err = None, None
            for s in cands:
                s_safe = torch.where(s > 0, s, torch.ones_like(s))
                q = torch.clamp(torch.round(v / s_safe), -8, 7)
                err = ((q * s_safe).to(torch.bfloat16).float() - v).abs().sum(-1, keepdim=True)
                if best_s is None:
                    best_s, best_err = s_safe, err
                else:
                    better = err < best_err
                    best_s = torch.where(better, s_safe, best_s)
                    best_err = torch.min(err, best_err)
            s = best_s
            q = torch.clamp(torch.round(v / s), -8, 7)
            exact += ((q * s).to(torch.bfloat16).float() == v).sum().item()
            total += v.numel()
            w = q * s
            has_m8 = (q == -8).any(-1, keepdim=True)
            w = torch.where(q == -8, -7.5 * s, w)
            # groups without a -8: put 7.5 s on one entry whose code becomes 7 (or -7.5 s -> -8).
            # If the group has a 7 that entry is exact. Otherwise (the GPTQ scale came from
            # weights that were updated later) move the single entry closest to a pin value:
            # the largest q to 7 or the smallest q to -8, whichever moves it less (one weight
            # per such group changes by that many steps; the rest of the group stays exact).
            qmax = q.amax(-1, keepdim=True)
            qmin = q.amin(-1, keepdim=True)
            up = (7 - qmax) <= (qmin + 8)  # pin with +7.5 s on the max entry, else -7.5 s on the min
            idx_max = (q == qmax).int().cumsum(-1) == 1
            idx_min = (q == qmin).int().cumsum(-1) == 1
            pin_p = idx_max & (q == qmax) & up & ~has_m8
            pin_n = idx_min & (q == qmin) & ~up & ~has_m8
            w = torch.where(pin_p, 7.5 * s, w)
            w = torch.where(pin_n, -7.5 * s, w)
            fallback += (~has_m8 & (qmax < 7) & (A > 0)).sum().item()
            qeff = torch.where(pin_p, torch.full_like(q, 7.0), torch.where(pin_n, torch.full_like(q, -8.0), q))
            exact -= ((q * s).to(torch.bfloat16).float() == v).sum().item()
            exact += ((qeff * s).to(torch.bfloat16).float() == v).sum().item()
            Wp[r0:r0 + rows_per_step] = w.view(-1, K)
        packed = quantize_w4(Wp)
        del Wp
        return packed, {"exact_frac": exact / total, "groups_with_moved_pin": fallback}

    @staticmethod
    def dequantize(p):
        """fp32 weights the kernel effectively multiplies with ((gate, up) for gate_up packs)."""
        from .quant import dequantize_w4
        return dequantize_w4(p)


class Engine:
    ATTN_SPLITS = 16  # flash-decoding splits per head (fixed so graphs are static)
    GDN_Q = 4  # GDN kernel: each state column split over Q thread groups (bench/engine/micro_kernels.py)
    # ctl indices (device int32 scalars, see csrc/basic.cu accept_kernel)
    POS, PENDING, MODE, M, CYCLES, ACCEPTED = range(6)

    def __init__(self, path: str | Path, max_len: int = 4096, linear=None, device="cuda", mtp=True,
                 lm_head: str = "same", cache_dir: str | Path | None = None,
                 gptq_lm_head: str | Path = Path.home() / "models/ppl-cache/lm_head_w4gptq.pt",
                 draft_vocab: int | None = None, mtp_weights: str | Path | None = None):
        """lm_head: "same" (use `linear`), "bf16" (BF16 GEMV even when `linear` is W4),
        "w4gptq" (W4 backend; the GPTQ lm_head from bench/quality/ppl.py, `gptq_lm_head` file, packed
        exactly on its grid by W4Linear.prepare_from_grid).
        mtp_weights: file with alternate mtp.* weights (BF16 dict, e.g. bench/distill/train_mtp.py
        output), quantized like the stock ones; not cached.
        draft_vocab: MTP drafts score only this many most frequent tokens (draft_vocab.py,
        FR-Spec); verify keeps the full lm_head, so output is unchanged.
        cache_dir: for backends with a `cache_tag` (W4), prepared weights are stored there per
        layer on first load and read back on later loads. Weights are streamed from safetensors
        one layer at a time (at most ~1 layer of BF16 on the GPU)."""
        self.cfg = cfg = R.Config.from_dir(path)
        self.max_len = max_len
        self.linear = linear or BF16Linear()
        self.lm_lin = BF16Linear() if lm_head == "bf16" else self.linear
        self.dev = dev = torch.device(device)
        self.path = Path(path)
        tag = getattr(self.linear, "cache_tag", None)
        cache = Path(cache_dir) / f"{self.path.name}-{tag}" if (cache_dir and tag) else None
        self.cache = cache
        load = lambda f: torch.load(f, map_location=dev, weights_only=False)
        save = (lambda obj, name: torch.save(obj, cache / name)) if cache else (lambda obj, name: None)
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        use_cache = bool(cache) and (cache / "COMPLETE").exists()  # all-or-nothing

        def cached(name, build):
            if use_cache and (cache / name).exists():
                return load(cache / name)
            obj = build()
            save(obj, name)
            return obj

        def top():
            W = R.load_weights(path, device=dev, names=lambda n: n in (
                "embed_tokens.weight", "lm_head.weight", "norm.weight"))
            return {"embed": W["embed_tokens.weight"], "norm": W["norm.weight"],
                    "lm_head_bf16": R.lm_head(W, cfg)}
        T = cached("top.pt", top)
        self.embed, self.final_norm = T["embed"], T["norm"]
        self.draft_head = None
        if draft_vocab and mtp and cfg.mtp_layers > 0:
            from .draft_vocab import draft_vocab_ids
            ids = draft_vocab_ids(path, draft_vocab, cfg.vocab).to(dev)

            def build_draft():
                if lm_head == "w4gptq":
                    G = torch.load(gptq_lm_head, map_location=dev, weights_only=False)
                    rows = G[ids.long()].contiguous()
                    del G
                    return {"ids": ids, "head": self.linear.prepare_from_grid(rows)[0]}
                rows = T["lm_head_bf16"][ids.long()].contiguous()
                lin = self.lm_lin
                return {"ids": ids, "head": rows if isinstance(lin, BF16Linear) else lin.prepare(rows)}
            dd = build_draft()  # a few seconds; not cached (the weight cache is shared read-only)
            self.draft_ids, self.draft_head = dd["ids"], dd["head"]
            torch.cuda.empty_cache()
        self.lm_head_stats = None
        if lm_head == "w4gptq":
            assert hasattr(self.linear, "prepare_from_grid"), "w4gptq needs the W4 backend"
            del T["lm_head_bf16"]
            torch.cuda.empty_cache()

            def build_gptq():
                G = torch.load(gptq_lm_head, map_location=dev, weights_only=False)
                packed, st = self.linear.prepare_from_grid(G)
                return {"packed": packed, "stats": st, "src": str(gptq_lm_head)}
            d = cached("lm_head_w4gptq.pt", build_gptq)
            self.lm_head, self.lm_head_stats = d["packed"], d["stats"]
        elif self.lm_lin is self.linear and getattr(self.linear, "name", "") != "bf16":
            self.lm_head = cached("lm_head.pt", lambda: self.linear.prepare(T["lm_head_bf16"]))
            del T["lm_head_bf16"]
        else:
            self.lm_head = T["lm_head_bf16"].contiguous()
        del T
        self.layers = []
        for l, t in enumerate(cfg.layer_types):
            pre = f"layers.{l}."
            self.layers.append(cached(f"layer{l:02d}.pt", lambda: self._layer_weights(
                R.load_weights(path, device=dev, names=lambda n: n.startswith(pre)), pre, t)))
            torch.cuda.empty_cache()
        self.has_mtp = mtp and cfg.mtp_layers > 0
        if self.has_mtp:
            def build_mtp():
                if mtp_weights is not None:  # alternate MTP weights (e.g. self-distilled), same names
                    W = {k: v.to(dev) for k, v in torch.load(mtp_weights, map_location="cpu").items()}
                else:
                    W = R.load_weights(path, device=dev, names=lambda n: n.startswith("mtp."))
                M = self._layer_weights(W, "mtp.layers.0.", "full_attention")
                M.update(fc=self.linear.prepare(W["mtp.fc.weight"]), norm=W["mtp.norm.weight"],
                         pre_e=W["mtp.pre_fc_norm_embedding.weight"], pre_h=W["mtp.pre_fc_norm_hidden.weight"])
                return M
            self.mtp = build_mtp() if mtp_weights is not None else cached("mtp.pt", build_mtp)
        if cache and not use_cache:
            (cache / "COMPLETE").write_text(f"{tag} lm_head={lm_head} mtp={self.has_mtp}\n")
        torch.cuda.empty_cache()
        self.l2_win = None              # MTP weight window persisted in L2 during drafting (pin_mtp_l2)
        self.mtp_lin = self.linear      # MTP layer + fc backend (BF16 variant: set_mtp_variant)
        self.draft_lin = self.lm_lin    # draft head backend
        self.mtp_prenorm = False        # MTP hidden input: False = post final norm (vLLM)
        self._alloc()
        self.graphs = {}
        if self.has_mtp and isinstance(self.linear, W4Linear):
            # MTP attention weights (54 MB W4) in one buffer with a persisting L2 window on the MTP
            # part of every graph; the 60 MB set-aside is enabled by generate_spec and released by
            # generate (A/B: MTP k4 +0.4%, draft phase -7%; plain unaffected)
            self.pin_mtp_l2(True)
            _C.l2_persist_limit(0)

    def _layer_weights(self, W, p, t, lin=None):
        lin = lin or self.linear
        L = {"type": t, "in_norm": W[p + "input_layernorm.weight"],
             "post_norm": W[p + "post_attention_layernorm.weight"],
             "down": lin.prepare(W[p + "mlp.down_proj.weight"])}
        g, u = W.pop(p + "mlp.gate_proj.weight"), W.pop(p + "mlp.up_proj.weight")
        if hasattr(lin, "gate_up"):  # backend fuses silu(g) * u into the GEMV epilogue
            L["gate_up"], L["fused_swiglu"] = lin.prepare_gate_up(g, u), True
        else:
            L["gate_up"], L["fused_swiglu"] = lin.prepare(torch.cat([g, u])), False
        del g, u
        if t == "linear_attention":
            q = p + "linear_attn."
            L["in_proj"] = lin.prepare(torch.cat([W[q + "in_proj_qkv.weight"], W[q + "in_proj_z.weight"],
                                                  W[q + "in_proj_b.weight"], W[q + "in_proj_a.weight"]]))
            L["out"] = lin.prepare(W[q + "out_proj.weight"])
            L["conv_w"] = W[q + "conv1d.weight"].contiguous()
            L["A_log"], L["dt_bias"], L["norm_w"] = W[q + "A_log"], W[q + "dt_bias"], W[q + "norm.weight"]
        else:
            q = p + "self_attn."
            L["in_proj"] = lin.prepare(torch.cat([W[q + "q_proj.weight"], W[q + "k_proj.weight"],
                                                  W[q + "v_proj.weight"]]))
            L["out"] = lin.prepare(W[q + "o_proj.weight"])
            L["q_norm"], L["k_norm"] = W[q + "q_norm.weight"], W[q + "k_norm.weight"]
        return L

    # ------------------------------------------------------------------------------------------
    def set_mtp_variant(self, mtp_bf16: bool, draft_bf16: bool):
        """A/B helper: MTP layer + fc in BF16 (original weights, BF16 GEMV) instead of the main
        backend, and/or the pruned draft head in BF16 (rows of the original lm_head). Graphs must
        be re-captured (self.graphs = {})."""
        if not hasattr(self, "_mtp_w4"):
            self._mtp_w4, self._draft_w4 = self.mtp, self.draft_head
        if mtp_bf16 and not hasattr(self, "_mtp_bf16"):
            W = R.load_weights(self.path, device=self.dev, names=lambda n: n.startswith("mtp."))
            bl = BF16Linear()
            M = self._layer_weights(W, "mtp.layers.0.", "full_attention", lin=bl)
            M.update(fc=bl.prepare(W["mtp.fc.weight"]), norm=W["mtp.norm.weight"],
                     pre_e=W["mtp.pre_fc_norm_embedding.weight"], pre_h=W["mtp.pre_fc_norm_hidden.weight"])
            M["kc"], M["vc"] = self._mtp_w4["kc"], self._mtp_w4["vc"]
            self._mtp_bf16 = M
        if draft_bf16 and not hasattr(self, "_draft_bf16"):
            W = R.load_weights(self.path, device=self.dev, names=lambda n: n == ("embed_tokens.weight" if self.cfg.tied
                                                                                  else "lm_head.weight"))
            self._draft_bf16 = next(iter(W.values()))[self.draft_ids.long()].contiguous()
            del W
            torch.cuda.empty_cache()
        self.mtp = self._mtp_bf16 if mtp_bf16 else self._mtp_w4
        self.mtp_lin = BF16Linear() if mtp_bf16 else self.linear
        self.draft_head = self._draft_bf16 if draft_bf16 else self._draft_w4
        self.draft_lin = BF16Linear() if draft_bf16 else self.lm_lin

    # ------------------------------------------------------------------------------------------
    def pin_mtp_l2(self, on: bool, names=("in_proj", "out"), limit=60 << 20):
        """Experiment: keep MTP weights resident in L2 across the k draft steps of a cycle. The
        chosen W4 matrices are re-homed into one contiguous buffer (one access-policy window); the
        MTP part of each graph runs with a persisting window, cleared before commit. on=False
        removes the window and the L2 set-aside. Graphs must be re-captured (self.graphs = {})."""
        if on and getattr(self, "_l2_buf", None) is None:
            mats = [self.mtp[n] for n in names]
            parts = [t for m in mats for t in (m.qw, m.scales)]
            sizes = [(t.numel() * t.element_size() + 255) // 256 * 256 for t in parts]
            buf = torch.empty(sum(sizes), dtype=torch.uint8, device=self.dev)
            off, views = 0, []
            for t, sz in zip(parts, sizes):
                v = buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
                v.copy_(t)
                views.append(v)
                off += sz
            for i, n in enumerate(names):
                old = self.mtp[n]
                self.mtp[n] = type(old)(views[2 * i], views[2 * i + 1], old.M, old.K, old.gate_up)
            self._l2_buf = buf
        self.l2_win = (self._l2_buf.data_ptr(), self._l2_buf.numel()) if on else None
        _C.l2_persist_limit(limit if on else 0)

    # ------------------------------------------------------------------------------------------
    def _alloc(self):
        cfg, dev, ml, N = self.cfg, self.dev, self.max_len, NMAX
        H, I, V = cfg.hidden, cfg.inter, cfg.vocab
        z = lambda *s, dt=BF16: torch.zeros(*s, dtype=dt, device=dev)
        self.ctl = z(32, dt=I32)  # [16 + m]: histogram of accepted drafts per cycle
        self.in_tok = z(N + 1, dt=I32)   # pass input tokens (+ in_tok[n]: next prompt token or -1)
        self.am = z(N, dt=I32)           # per-row argmax of the main model
        self.next_tok = z(N, dt=I32)     # true next token per row (MTP input)
        self.drafts = z(N, dt=I32)
        self.mtp_am = z(N, dt=I32)
        self.mpos = z(1, dt=I32)
        self.log = z(ml, dt=I32)         # generated tokens at their positions
        self.h, self.xn, self.mo = z(N, H), z(N, H), z(N, H)
        self.hid = z(N, H)               # main hidden after the final norm (lm_head + MTP input)
        self.gdn_proj_n = cfg.conv_dim + cfg.value_dim + 2 * cfg.hv
        self.attn_proj_n = cfg.hq * 2 * cfg.head_dim + 2 * cfg.hkv * cfg.head_dim
        pw = max(self.gdn_proj_n, self.attn_proj_n)
        self.proj = z(N, (pw + 7) // 8 * 8)  # row stride multiple of 8 (16-byte aligned rows)
        self.mix = z(N, max(cfg.value_dim, cfg.hq * cfg.head_dim))
        self.gu, self.act = z(N, 2 * I), z(N, I)
        self.logits = z(N, V)
        self.am_v, self.am_i = z(N * 256, dt=torch.float32), z(N * 256, dt=I32)
        self.q = z(N, cfg.hq * cfg.head_dim)
        self.part_o = z(N, cfg.hq, self.ATTN_SPLITS, cfg.head_dim, dt=torch.float32)
        self.part_ml = z(N, cfg.hq, self.ATTN_SPLITS, 2, dt=torch.float32)
        rd = cfg.rotary_dim  # inv_freq built on the CPU exactly as HF does, then moved
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))).to(dev)
        for L in self.layers:
            if L["type"] == "linear_attention":
                L["ring"] = z(16, cfg.conv_dim)
                L["S"] = z(cfg.hv, cfg.dk, cfg.dv, dt=torch.float32)
                L["st_k"] = z(N, cfg.hv, cfg.dk, dt=torch.float32)
                L["st_v"] = z(N, cfg.hv, cfg.dv, dt=torch.float32)
                L["st_g"] = z(N, cfg.hv, 2, dt=torch.float32)
            else:
                L["kc"], L["vc"] = z(cfg.hkv, ml, cfg.head_dim), z(cfg.hkv, ml, cfg.head_dim)
        if self.has_mtp:
            self.mtp["kc"], self.mtp["vc"] = z(cfg.hkv, ml, cfg.head_dim), z(cfg.hkv, ml, cfg.head_dim)
            self.cat = z(N, 2 * H)
            self.mh, self.mout, self.chain_h = z(N, H), z(N, H), z(1, H)

    def reset(self):
        """Empty sequence. KV caches need no clearing (only [0, pos] is ever read)."""
        for L in self.layers:
            if "ring" in L:
                L["ring"].zero_()
                L["S"].zero_()
        self.ctl.zero_()

    # ------------------------------------------------------------------------------------------
    def _layer(self, L, n, h, eps, first, pos, lin=None):
        """One decoder layer on rows h[:n] (residual stream). The previous sublayer's output in
        self.mo is added by this layer's first norm (fused residual add) unless `first`."""
        cfg, lin = self.cfg, lin or self.linear
        xn, mo = self.xn[:n], self.mo[:n]
        _C.rmsnorm1p(h, None if first else mo, L["in_norm"], xn, eps)
        if L["type"] == "linear_attention":
            pr = self.proj[:n, :self.gdn_proj_n]
            lin(xn, L["in_proj"], pr)
            c0, c1 = cfg.conv_dim, cfg.conv_dim + cfg.value_dim
            out = self.mix[:n, :cfg.value_dim]
            _C.gdn_step_n(pr[:, :c0], pr[:, c0:c1], pr[:, c1:c1 + cfg.hv], pr[:, c1 + cfg.hv:], L["ring"],
                          L["conv_w"], L["A_log"], L["dt_bias"], L["norm_w"], L["S"], L["st_k"], L["st_v"],
                          L["st_g"], self.ctl, out, cfg.hk, eps, q=self.GDN_Q)
        else:
            pr = self.proj[:n, :self.attn_proj_n]
            lin(xn, L["in_proj"], pr)
            q0 = cfg.hq * 2 * cfg.head_dim
            k1 = q0 + cfg.hkv * cfg.head_dim
            qg = pr[:, :q0]
            q = self.q[:n]
            _C.attn_prep(qg, pr[:, q0:k1], pr[:, k1:], L["q_norm"], L["k_norm"], self.inv_freq, pos, q,
                         L["kc"], L["vc"], eps)
            out = self.mix[:n, :cfg.hq * cfg.head_dim]
            _C.attn_decode(q, L["kc"], L["vc"], pos, qg, self.part_o, self.part_ml, out)
        lin(out, L["out"], mo)
        _C.rmsnorm1p(h, mo, L["post_norm"], xn, eps)
        if L["fused_swiglu"]:
            lin.gate_up(xn, L["gate_up"], self.act[:n])
        else:
            lin(xn, L["gate_up"], self.gu[:n])
            _C.swiglu(self.gu[:n], self.act[:n])
        lin(self.act[:n], L["down"], mo)

    def _main(self, n):
        eps = self.cfg.eps
        h = self.h[:n]
        _C.embed(self.embed, self.in_tok, h)
        for i, L in enumerate(self.layers):
            self._layer(L, n, h, eps, i == 0, self.ctl)  # ctl[0] = pos
        _C.rmsnorm1p(h, self.mo[:n], self.final_norm, self.hid[:n], eps)  # post-norm hidden
        self.lm_lin(self.hid[:n], self.lm_head, self.logits[:n])
        _C.argmax(self.logits[:n], self.am_v[:n * 256], self.am_i[:n * 256], self.am)
        _C.accept(self.in_tok, self.am, n, self.ctl, self.next_tok, self.log)

    def _mtp(self, n, hid, tok):
        """MTP rows: hid [n, H] hidden (post-norm), tok [>= n] int32 next tokens, at mpos + r.
        Output: self.mout[:n] (after mtp.norm), self.mtp_am[:n] (draft tokens)."""
        M, eps, H = self.mtp, self.cfg.eps, self.cfg.hidden
        mh, cat = self.mh[:n], self.cat[:n]
        _C.embed(self.embed, tok, mh)  # mh used as scratch for the embedding rows
        _C.rmsnorm1p(mh, None, M["pre_e"], cat[:, :H], eps)
        _C.rmsnorm1p(hid, None, M["pre_h"], cat[:, H:], eps)
        self.mtp_lin(cat, M["fc"], mh)
        self._layer(M, n, mh, eps, True, self.mpos, self.mtp_lin)
        _C.rmsnorm1p(mh, self.mo[:n], M["norm"], self.mout[:n], eps)  # mh = pre-norm residual
        if self.draft_head is not None:  # pruned draft vocabulary (FR-Spec)
            lg = self.logits[:n, :self.draft_ids.numel()]
            self.draft_lin(self.mout[:n], self.draft_head, lg)
            _C.argmax(lg, self.am_v[:n * 256], self.am_i[:n * 256], self.mtp_am, self.draft_ids)
        else:
            self.lm_lin(self.mout[:n], self.lm_head, self.logits[:n])
            _C.argmax(self.logits[:n], self.am_v[:n * 256], self.am_i[:n * 256], self.mtp_am)

    def _pass(self, n, mtp_rows, k):
        """Graph body: main pass on n rows; optionally MTP over the n rows (fills the MTP cache,
        gives draft 1 from the last kept row) and k-1 chained drafts; then commit."""
        self._main(n)
        win = getattr(self, "l2_win", None) if mtp_rows else None
        if win:
            _C.stream_window(win[0], win[1], 1.0)
        if mtp_rows:
            _C.mtp_select(self.ctl, self.mpos, -1)  # mpos = pos
            # MTP input hidden: main model after the final norm (vLLM convention) or before it
            self._mtp(n, (self.h if self.mtp_prenorm else self.hid)[:n], self.next_tok)
            chain_src = self.mh if self.mtp_prenorm else self.mout  # before / after mtp.norm
            if k > 0:  # draft 1 = MTP argmax of row m; chain input = its hidden
                _C.mtp_select(self.ctl, self.mpos, 0, chain_src[:n], self.chain_h, self.mtp_am, self.drafts[:1])
            for j in range(1, k):
                if j > 1:
                    _C.mtp_select(self.ctl, self.mpos, j - 1)  # mpos = pos + m + j
                self._mtp(1, self.chain_h, self.drafts[j - 1:j])
                self.drafts[j:j + 1].copy_(self.mtp_am[:1])
                self.chain_h.copy_(chain_src[:1])
        if win:
            _C.stream_window(0, 0, 0.0)
        _C.commit(self.ctl, self.in_tok, self.am, self.drafts, k)

    def graph(self, n, mtp_rows=False, k=0):
        """CUDA graph for one pass variant; captured on first use. Capturing runs the body once
        eagerly (warm-up), which changes the sequence state: call reset() afterwards."""
        key = (n, mtp_rows, k)
        assert 1 <= n <= NMAX and (not mtp_rows or self.has_mtp)
        if key not in self.graphs:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._pass(n, mtp_rows, k)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._pass(n, mtp_rows, k)
            torch.cuda.synchronize()
            self.graphs[key] = g
        return self.graphs[key]

    # ------------------------------------------------------------------------------------------
    def _chunks(self, seq, size):
        return [seq[i:i + size] for i in range(0, len(seq), size)]

    def _prefill_keys(self, P, mtp, k, chunk=NMAX):
        sizes = [len(c) for c in self._chunks(list(range(P)), chunk)]
        keys = {(s, mtp, 0) for s in sizes[:-1]}
        keys.add((sizes[-1], mtp, k))
        return keys

    def _ensure(self, keys):
        new = [k for k in keys if k not in self.graphs]
        for key in new:
            self.graph(*key)
        if new:
            self.reset()

    def prefill(self, prompt: list[int], mtp=False, k=0, chunk=NMAX):
        """Feed the prompt in passes of `chunk` rows (prefill mode). The last row's argmax is the
        first generated token (log[len(prompt)]); with mtp/k the drafts for the first verify are
        ready. Leaves the engine in decode/verify mode. Assumes reset() and graphs exist."""
        chunks = self._chunks(prompt, chunk)
        self.ctl[self.MODE].fill_(0)
        buf = torch.full((NMAX + 1,), -1, dtype=I32)
        for ci, c in enumerate(chunks):
            last = ci == len(chunks) - 1
            buf.fill_(-1)
            buf[:len(c)] = torch.tensor(c, dtype=I32)
            if not last:
                buf[len(c)] = chunks[ci + 1][0]
            self.in_tok.copy_(buf)
            self.graph(len(c), mtp, k if last else 0).replay()
        self.ctl[self.MODE].fill_(1)

    @torch.no_grad()
    def generate(self, prompt: list[int], n_new: int, timing=False, chunk=NMAX):
        """Plain greedy: prefill in passes of `chunk`, then n_new - 1 one-row decode passes.
        Returns n_new token ids (EOS not checked). timing=True adds (prefill_s, decode_s)."""
        P = len(prompt)
        assert P + n_new + 1 <= self.max_len
        if self.l2_win:
            _C.l2_persist_limit(0)  # no L2 set-aside for plain decode
        self._ensure(self._prefill_keys(P, False, 0, chunk) | {(1, False, 0)})
        self.reset()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        self.prefill(prompt, chunk=chunk)
        e1.record()
        g = self.graphs[(1, False, 0)]
        for _ in range(n_new - 1):
            g.replay()
        e2.record()
        torch.cuda.synchronize()
        gen = self.log[P:P + n_new].tolist()
        if timing:
            return gen, e0.elapsed_time(e1) / 1e3, e1.elapsed_time(e2) / 1e3
        return gen

    @torch.no_grad()
    def generate_spec(self, prompt: list[int], n_new: int, k: int, timing=False, chunk=NMAX):
        """Greedy speculative decoding with k MTP drafts per cycle (verify n = k+1 rows).
        Output equals `generate` token for token. Returns tokens (and with timing:
        prefill_s, decode_s, stats dict with cycles / accepted drafts)."""
        assert self.has_mtp and 1 <= k <= NMAX - 1
        P = len(prompt)
        assert P + n_new + 2 * NMAX + 2 <= self.max_len
        if self.l2_win:
            _C.l2_persist_limit(60 << 20)  # set-aside for the persisting MTP weight window
        self._ensure(self._prefill_keys(P, True, k, chunk) | {(k + 1, True, k)})
        self.reset()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        self.prefill(prompt, mtp=True, k=k, chunk=chunk)
        self.ctl[self.CYCLES:self.ACCEPTED + 1].zero_()
        self.ctl[16:].zero_()
        e1.record()
        g = self.graphs[(k + 1, True, k)]
        target = P + n_new - 1  # log[P + n_new - 1] must be written: need pos >= target
        pos = P
        while pos < target:
            # each cycle advances pos by 1..k+1; replay as many as cannot overshoot by > 1 cycle
            for _ in range(max(1, (target - pos) // (k + 1))):
                g.replay()
            pos = int(self.ctl[self.POS].item())  # one host sync per batch of cycles
        e2.record()
        torch.cuda.synchronize()
        gen = self.log[P:P + n_new].tolist()
        if not timing:
            return gen
        c = self.ctl.tolist()
        stats = {"cycles": c[self.CYCLES], "accepted": c[self.ACCEPTED],
                 "accept_rate": c[self.ACCEPTED] / max(1, c[self.CYCLES] * k),
                 "tokens_per_cycle": 1 + c[self.ACCEPTED] / max(1, c[self.CYCLES])}
        hist = c[16:16 + k + 1]  # cycles that accepted exactly m drafts
        reach = [sum(hist[d:]) for d in range(k + 1)]  # cycles that accepted >= d drafts
        stats["accept_by_depth"] = [reach[d] / max(1, reach[d - 1]) for d in range(1, k + 1)]  # P(d | d-1)
        return gen, e0.elapsed_time(e1) / 1e3, e1.elapsed_time(e2) / 1e3, stats

    @torch.no_grad()
    def sequence_logits(self, seq: list[int], keep="full", chunk=NMAX):
        """Teacher forcing: feed `seq` in prefill passes and return per-position logits [T, V]
        bf16 (keep='full') or argmax ids [T] (keep='argmax')."""
        T = len(seq)
        chunks = self._chunks(seq, chunk)
        self._ensure({(len(c), False, 0) for c in chunks})
        self.reset()
        out = torch.empty((T, self.cfg.vocab) if keep == "full" else (T,), dtype=BF16 if keep == "full" else I32,
                          device=self.dev)
        buf = torch.full((NMAX + 1,), -1, dtype=I32)
        for ci, c in enumerate(chunks):
            buf.fill_(-1)
            buf[:len(c)] = torch.tensor(c, dtype=I32)
            if ci + 1 < len(chunks):
                buf[len(c)] = chunks[ci + 1][0]
            self.in_tok.copy_(buf)
            self.graph(len(c), False, 0).replay()
            o = ci * chunk
            if keep == "full":
                out[o:o + len(c)].copy_(self.logits[:len(c)])
            else:
                out[o:o + len(c)].copy_(self.am[:len(c)])
        torch.cuda.synchronize()
        return out

    @torch.no_grad()
    def mtp_rows_logits(self, seq: list[int]):
        """Validation helper: teacher-forced pass over seq (one row at a time) with MTP rows on;
        returns (main post-norm hidden [T, H], MTP logits [T, V]) where MTP row t uses the main
        hidden at t and next token seq[t+1] (am at the last position)."""
        self._ensure({(1, True, 0)})
        self.reset()
        T, H = len(seq), self.cfg.hidden
        hid = torch.empty(T, H, dtype=BF16, device=self.dev)
        lg = torch.empty(T, self.cfg.vocab, dtype=BF16, device=self.dev)
        self.ctl[self.MODE].fill_(0)
        buf = torch.full((NMAX + 1,), -1, dtype=I32)
        for t in range(T):
            buf.fill_(-1)
            buf[0] = seq[t]
            if t + 1 < T:
                buf[1] = seq[t + 1]
            self.in_tok.copy_(buf)
            self.graph(1, True, 0).replay()
            hid[t].copy_(self.hid[0])
            lg[t].copy_(self.logits[0])  # the MTP pass reuses the logits buffer last
        torch.cuda.synchronize()
        return hid, lg
