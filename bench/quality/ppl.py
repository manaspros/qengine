"""wikitext-2 perplexity of Qwen3.8-27B, BF16 vs W4 (g128 RTN fake-quant).

Protocol (fixed; every variant sees the same token windows)
-----------------------------------------------------------
- Data: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test` (parquet from the HF hub,
  revision recorded in `data/meta.json`). Text = "\\n\\n".join(rows) (HF perplexity guide).
- Tokenizer: the model's own (`~/models/Qwen3.8-27B`), no BOS / special tokens added.
- Windows: context L = 2048, non-overlapping, all complete windows (the tail < L is dropped).
  Each window starts from an empty state; the loss covers positions 1..L-1 of every window
  (L-1 predicted tokens per window). ppl = exp(mean NLL over all predicted tokens).
- Forward: HF transformers 5.20.1 `Qwen3_5DecoderLayer` modules (prefill path: chunked gated
  delta rule in fp32, SDPA causal attention), BF16 activations, one decoder layer resident on the
  GPU at a time (the BF16 27B does not fit in 32 GB, spec R4). Final RMSNorm + lm_head in BF16,
  cross entropy on logits upcast to fp32 (as HF's loss).
- Variants (decoder weights): `bf16` = checkpoint; `w4rtn` = every nn.Linear in every decoder
  layer replaced by `dequantize_w4(quantize_w4(W))` (qengine.quant: int4 symmetric, group 128 along
  K, fp16 scale; the engine quantizes exactly these: GDN in_proj_qkv/z/b/a + out_proj, attention
  q/k/v/o_proj, MLP gate/up/down). The dequantized weight is rounded to BF16 for F.linear (rel.
  error 2^-9, negligible next to 4-bit error). Embedding, norms, conv1d, A_log, dt_bias stay BF16.
  lm_head is evaluated both ways (`bf16`, `w4`) on top of each decoder variant (Q6).

Running (resumable; every GPU job holds scripts/gpu_lock.sh and lasts <= --budget seconds):
    python bench/quality/ppl.py fetch            # data -> ~/models/ppl-cache/data (needs pyarrow*)
    python bench/quality/ppl.py check            # HF-layer prefill vs reference.py (27B, 64 tok)
    python bench/quality/ppl.py run --weights bf16,w4rtn
* `fetch` runs under `uv run --no-project --with pyarrow --with huggingface_hub` so the project
  venv is not modified; it writes plain text that the rest of the script reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # qengine/
CACHE = Path.home() / "models" / "ppl-cache"
DATA = CACHE / "data"
MODEL = Path.home() / "models" / "Qwen3.8-27B"
LOCK = ROOT / "scripts" / "gpu_lock.sh"
WIKI_REPO = "Salesforce/wikitext"
WIKI_FILE = "wikitext-2-raw-v1/{split}-00000-of-00001.parquet"


def sfx(split: str) -> str:
    """File-name suffix: the test split keeps the original (suffix-less) cache names."""
    return "" if split == "test" else f"_{split}"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def cmd_fetch(a):
    """Download the official parquet and write the joined raw text (no torch needed)."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    DATA.mkdir(parents=True, exist_ok=True)
    rev = HfApi().dataset_info(WIKI_REPO).sha
    for split in ("test", "train"):     # train: calibration data for GPTQ only, never scored
        fn = WIKI_FILE.format(split=split)
        f = hf_hub_download(WIKI_REPO, fn, repo_type="dataset", revision=rev, cache_dir=DATA / "hub")
        rows = pq.read_table(f).column("text").to_pylist()
        text = "\n\n".join(rows)
        (DATA / f"wikitext2_{split}.txt").write_text(text)
        meta = {"repo": WIKI_REPO, "file": fn, "revision": rev, "rows": len(rows),
                "chars": len(text), "sha256_text": hashlib.sha256(text.encode()).hexdigest()}
        (DATA / f"meta{sfx(split)}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta, indent=1))


def windows(L: int, max_windows: int | None, split: str = "test"):
    """[N, L] int64 token windows (cached)."""
    import torch
    f = DATA / f"ids_L{L}{sfx(split)}.pt"
    if not f.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        ids = tok((DATA / f"wikitext2_{split}.txt").read_text(), add_special_tokens=False).input_ids
        ids = torch.tensor(ids, dtype=torch.int64)
        n = ids.numel() // L
        torch.save({"total_tokens": ids.numel(), "win": ids[: n * L].view(n, L).clone()}, f)
    d = torch.load(f)
    w = d["win"]
    if max_windows:
        w = w[:max_windows]
    return w, d["total_tokens"]


# ----------------------------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------------------------
def _setup():
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    from transformers import AutoConfig
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 delta-rule matmuls stay true fp32
    torch.backends.cudnn.allow_tf32 = False
    tc = AutoConfig.from_pretrained(MODEL).text_config
    tc._attn_implementation = "sdpa"
    return tc


def layer_weights(l: int):
    from qengine import reference as R
    p = f"layers.{l}."
    W = R.load_weights(MODEL, device="cuda", names=lambda n: n.startswith(p))
    return {k[len(p):]: v for k, v in W.items()}


def fake_w4(W, rows: int = 16384):
    """dequantize_w4(quantize_w4(W)) row-slice by row-slice (rows are independent; bounds memory)."""
    import torch
    from qengine.quant import dequantize_w4, quantize_w4
    out = torch.empty_like(W)
    for r in range(0, W.shape[0], rows):
        out[r:r + rows] = dequantize_w4(quantize_w4(W[r:r + rows])).to(W.dtype)
    return out


def fake_w8(W, rows: int = 16384):
    """int8 symmetric, group 128 along K, fp16 scale = absmax/127 (comparison point for Q6)."""
    import torch
    out = torch.empty_like(W)
    M, K = W.shape
    Kp = (K + 127) // 128 * 128
    for r in range(0, M, rows):
        Wp = torch.nn.functional.pad(W[r:r + rows].float(), (0, Kp - K)).view(-1, Kp // 128, 128)
        sc = (Wp.abs().amax(-1, keepdim=True) / 127).half().float()
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))
        q = torch.clamp(torch.round(Wp / sc), -127, 127) * sc
        out[r:r + rows] = q.view(-1, Kp)[:, :K].to(W.dtype)
    return out


GPTQ_HEAD = CACHE / "lm_head_w4gptq.pt"


def gptq_w4(W, Hinv, group: int = 128):
    """GPTQ (Frantar et al. 2022, reference algorithm, no act-order) onto quant.py's exact grid:
    per (row, 128-group) scale = fp16(2*absmax/15) of the *updated* weights at the group start,
    u = clamp(round(w/s) + 8, 0, 15), w~ = (u - 8) * s. W: fp32 [M, K] (K % 128 == 0);
    Hinv: upper Cholesky factor of the damped inverse Hessian. Returns w~ fp32 [M, K]."""
    import torch
    W = W.clone()
    M, K = W.shape
    Q = torch.empty_like(W)
    for i1 in range(0, K, group):                      # block = group: scale uses updated W
        i2 = i1 + group
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err = torch.empty_like(W1)
        s = (W1.abs().amax(-1) * (2.0 / 15.0)).half().float()
        zero = s == 0
        s = torch.where(zero, torch.ones_like(s), s)
        for i in range(group):
            w = W1[:, i]
            q = (torch.clamp(torch.round(w / s) + 8, 0, 15) - 8) * s
            q = torch.where(zero, torch.zeros_like(q), q)
            Q[:, i1 + i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q


def cmd_gptq_head(a):
    """GPTQ the lm_head with H = sum x x^T over final-normed hidden states of wikitext-2 *train*
    windows (from a `run --split train --weights w4rtn --max-windows N` pass). Saves bf16 w~."""
    import torch
    _setup()
    from qengine import reference as R
    a.split = "train"
    d = run_dir(a)
    v = a.weights.split(",")[0]
    assert get_state(d, v)["next_layer"] == 64, f"calibration run {d} / {v} not finished"
    W = R.load_weights(MODEL, device="cuda", names=lambda n: n in ("norm.weight", "lm_head.weight"))
    h = torch.load(d / f"h_{v}.pt", mmap=True)
    K = h.shape[-1]
    H = torch.zeros(K, K, device="cuda", dtype=torch.float64)
    n = 0
    for i in range(h.shape[0]):
        x = R.rmsnorm1p(h[i].cuda(), W["norm.weight"], 1e-6).double()
        H += x.T @ x
        n += x.shape[0]
    H = (H / n).float()
    Wl = W["lm_head.weight"]
    H[torch.arange(K), torch.arange(K)] += a.damp * H.diagonal().mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    out = torch.empty_like(Wl)
    proxy = {"rtn": 0.0, "gptq": 0.0}
    t0 = time.time()
    for r in range(0, Wl.shape[0], 16384):
        Wr = Wl[r:r + 16384].float()
        Q = gptq_w4(Wr, Hinv)
        Qr = fake_w4(Wl[r:r + 16384]).float()
        for k, q in (("gptq", Q.to(torch.bfloat16).float()), ("rtn", Qr)):
            D = Wr - q
            proxy[k] += ((D @ H) * D).sum().item()        # output MSE proxy tr(D H D^T)
        out[r:r + 16384] = Q.to(Wl.dtype)
    # on-grid check: every group is an integer multiple of its fp16 scale in [-8, 7]
    g = out[:4096].float().view(4096, -1, 128)
    s = (g.abs().amax(-1, keepdim=True) / 8).clamp_min(1e-30)
    print(f"[gptq] {Wl.shape[0]} rows, {n} calibration tokens, {time.time() - t0:.1f}s; "
          f"proxy tr(dW H dW^T): RTN {proxy['rtn']:.4e}, GPTQ {proxy['gptq']:.4e} "
          f"({proxy['gptq'] / proxy['rtn']:.3f}x)", flush=True)
    torch.save(out.cpu(), GPTQ_HEAD)
    json.dump({"calib_dir": str(d), "variant": v, "tokens": n, "damp": a.damp, **proxy},
              open(CACHE / "lm_head_w4gptq.json", "w"), indent=1)


def is_quantized(name: str, t) -> bool:
    """Every nn.Linear weight of a decoder layer (2-D, ends in .weight; conv1d is 3-D, norms 1-D)."""
    return t.dim() == 2 and name.endswith(".weight")


def quantize_layer(sd: dict, variant: str) -> dict:
    if variant == "bf16":
        return sd
    if variant == "w4rtn":
        return {k: (fake_w4(v) if is_quantized(k, v) else v) for k, v in sd.items()}
    if variant == "w4gptq":
        raise SystemExit("w4gptq: weights come from gptq.py (not run)")
    raise SystemExit(f"unknown variant {variant}")


def build_layer(tc, l: int, sd: dict):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    with torch.device("meta"):
        m = Qwen3_5DecoderLayer(tc, l)
    m.load_state_dict(sd, strict=True, assign=True)
    return m.eval()


def rope(tc, B: int, L: int):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    re = Qwen3_5TextRotaryEmbedding(tc).cuda()
    pos = torch.arange(L, device="cuda").view(1, 1, L).expand(3, B, L)
    return re(torch.empty(1, device="cuda", dtype=torch.bfloat16), pos)


def run_layer(m, tc, h, mb: int):
    """h: [N, L, H] bf16 on GPU, updated in place, mb windows per forward."""
    import torch
    N, L, _ = h.shape
    pe = rope(tc, mb, L)
    for i in range(0, N, mb):
        x = h[i:i + mb]
        b = x.shape[0]
        cs = pe if b == mb else (pe[0][:b], pe[1][:b])   # cos/sin: [B, L, rotary_dim]
        h[i:i + mb] = m(x, position_embeddings=cs, attention_mask=None)


# ----------------------------------------------------------------------------------------------
# Resumable streaming
# ----------------------------------------------------------------------------------------------
def run_dir(a) -> Path:
    d = CACHE / f"run_L{a.ctx}_N{a.max_windows or 'all'}{sfx(a.split)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(d: Path, v: str) -> Path:
    return d / f"state_{v}.json"


def get_state(d: Path, v: str) -> dict:
    p = state_path(d, v)
    return json.loads(p.read_text()) if p.exists() else {"next_layer": 0, "seconds": 0.0}


def cmd_step(a):
    """One GPU-locked chunk: advance every variant by as many layers as fit in --budget s."""
    import torch
    t0 = time.time()
    tc = _setup()
    d = run_dir(a)
    win, _ = windows(a.ctx, a.max_windows, a.split)
    variants = a.weights.split(",")
    st = {v: get_state(d, v) for v in variants}
    nl = {s["next_layer"] for s in st.values()}
    assert len(nl) == 1, f"variants out of step: {st}"
    l0 = nl.pop()
    nlayers = tc.num_hidden_layers
    if l0 >= nlayers:
        return
    hs = {}
    for v in variants:
        f = d / f"h_{v}.pt"
        if l0 == 0:
            from qengine import reference as R
            E = R.load_weights(MODEL, device="cuda", names=lambda n: n == "embed_tokens.weight")
            hs[v] = E["embed_tokens.weight"][win.cuda()]
            del E
        else:
            hs[v] = torch.load(f, mmap=True).cuda()
    t_load = time.time() - t0
    l, per_layer = l0, []
    while l < nlayers:
        tl = time.time()
        sd = layer_weights(l)
        for v in variants:
            m = build_layer(tc, l, quantize_layer(sd, v))
            with torch.no_grad():
                run_layer(m, tc, hs[v], a.mb)
            del m
        del sd
        torch.cuda.synchronize()
        per_layer.append(time.time() - tl)
        l += 1
        for v in variants:
            assert torch.isfinite(hs[v]).all(), f"non-finite hidden state, {v} layer {l - 1}"
        if time.time() - t0 + max(per_layer) + 5.0 * len(variants) > a.budget:   # 5 s/variant to save h
            break
    for v in variants:
        tmp = d / f"h_{v}.pt.tmp"
        torch.save(hs[v].cpu(), tmp)
        os.replace(tmp, d / f"h_{v}.pt")
        s = st[v]
        s["next_layer"] = l
        s["seconds"] = s["seconds"] + (time.time() - t0)
        state_path(d, v).write_text(json.dumps(s))
    print(f"[step] layers {l0}..{l - 1} of {nlayers} ({','.join(variants)}): load {t_load:.1f}s, "
          f"{sum(per_layer) / len(per_layer):.1f}s/layer, chunk {time.time() - t0:.1f}s", flush=True)


def cmd_final(a):
    """Final norm + lm_head (bf16 and w4) + NLL for each finished decoder variant."""
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    _setup()
    from qengine import reference as R
    d = run_dir(a)
    win, total = windows(a.ctx, a.max_windows, a.split)
    W = R.load_weights(MODEL, device="cuda", names=lambda n: n in ("norm.weight", "lm_head.weight"))
    heads = {"bf16": W["lm_head.weight"]}
    hl = a.heads.split(",")
    if "w4" in hl:
        heads["w4"] = fake_w4(W["lm_head.weight"])
    if "w8" in hl:
        heads["w8"] = fake_w8(W["lm_head.weight"])
    if "w4gptq" in hl:
        heads["w4gptq"] = torch.load(GPTQ_HEAD, mmap=True).cuda()
    heads = {k: v for k, v in heads.items() if k in hl}
    res_f = d / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for v in a.weights.split(","):
        assert get_state(d, v)["next_layer"] == 64, f"{v} not finished"
        h = torch.load(d / f"h_{v}.pt", mmap=True)
        N, L, _ = h.shape
        for hn, Wh in heads.items():
            nll_w = torch.zeros(N, dtype=torch.float64)
            for i in range(N):
                x = R.rmsnorm1p(h[i].cuda(), W["norm.weight"], 1e-6)
                logits = F.linear(x[:-1], Wh).float()           # [L-1, V], bf16 -> fp32 (HF loss)
                tgt = win[i, 1:].cuda()
                nll_w[i] = F.cross_entropy(logits, tgt, reduction="sum").double().cpu()
            ntok = N * (L - 1)
            nll = nll_w.sum().item() / ntok
            res[f"{v}+head_{hn}"] = {"ppl": math.exp(nll), "nll": nll, "tokens": ntok, "windows": N,
                                     "nll_per_window": (nll_w / (L - 1)).tolist()}
            print(f"[final] decoder {v:6s} lm_head {hn:4s}: ppl {math.exp(nll):.4f} "
                  f"(nll {nll:.5f}, {ntok} tokens)", flush=True)
    res["_meta"] = meta(a, total)
    res_f.write_text(json.dumps(res, indent=1))
    print(f"[final] {time.time() - t0:.1f}s -> {res_f}")


def meta(a, total):
    import torch
    import transformers
    q = ROOT / "python" / "qengine" / "quant.py"
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return {"ctx": a.ctx, "max_windows": a.max_windows, "dataset_tokens_total": total,
            "data": json.loads((DATA / "meta.json").read_text()),
            "quant_py_sha256": hashlib.sha256(q.read_bytes()).hexdigest(),
            "git_head": git, "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "date": time.strftime("%Y-%m-%d %H:%M")}


def report(a):
    d = run_dir(a)
    res = json.loads((d / "results.json").read_text())
    import statistics
    base = res.get("bf16+head_bf16")
    print(f"\nwikitext-2 test, L={a.ctx}, {base['windows']} windows, {base['tokens']} predicted tokens")
    print(f"{'variant':24s} {'ppl':>9s} {'delta':>8s} {'paired SE':>9s}")
    for k, r in res.items():
        if k.startswith("_"):
            continue
        dp = 100 * (r["ppl"] / base["ppl"] - 1)
        diffs = [x - y for x, y in zip(r["nll_per_window"], base["nll_per_window"])]
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 and any(diffs) else 0.0
        print(f"{k:24s} {r['ppl']:9.4f} {dp:+7.2f}% {100 * se:8.3f}%")


def cmd_run(a):
    """Driver: loop locked chunks until all variants are done, then the final pass."""
    py = sys.executable
    me = str(Path(__file__).resolve())
    common = ["--ctx", str(a.ctx), "--split", a.split, "--weights", a.weights, "--mb", str(a.mb), "--budget", str(a.budget)]
    if a.max_windows:
        common += ["--max-windows", str(a.max_windows)]
    windows(a.ctx, a.max_windows, a.split)       # tokenize on CPU outside the lock
    d = run_dir(a)
    while min(get_state(d, v)["next_layer"] for v in a.weights.split(",")) < 64:
        subprocess.run([str(LOCK), py, me, "_step", *common], check=True)
        time.sleep(a.gap)               # let other GPU jobs take the lock
    if a.split != "test":               # calibration hidden states only; never scored
        return
    subprocess.run([str(LOCK), py, me, "_final", *common, "--heads", a.heads], check=True)
    report(a)


# ----------------------------------------------------------------------------------------------
# Check: HF prefill layers vs reference.py (recurrent decode path), 27B, streamed
# ----------------------------------------------------------------------------------------------
def cmd_check(a):
    """Feed T tokens through all 64 layers two ways, layer-streamed: (1) the HF prefill modules
    used by this script, (2) reference.py token by token. Per layer, both see the SAME input (the
    HF-path hidden state) so errors do not compound; report rel L2 / min cosine per layer and
    top-1 agreement of final logits (each path fed its own chain too). Also checks fake_w4 against
    the closed-form RTN formula of quant.py's docstring."""
    import torch
    tc = _setup()
    from qengine import reference as R
    cfg = R.Config.from_dir(MODEL)
    win, _ = windows(a.ctx, None)
    ids = win[0, : a.tokens].cuda()
    T = ids.numel()

    # quantizer self-check: closed form vs quant.py
    torch.manual_seed(0)
    Wt = (torch.randn(300, 1000, device="cuda") * 0.02).to(torch.bfloat16)
    Wg = torch.nn.functional.pad(Wt.float(), (0, 24)).view(300, 8, 128)
    s = (Wg.abs().amax(-1) * (2 / 15)).half().float()
    q = (torch.clamp(torch.round(Wg / s[..., None]) + 8, 0, 15) - 8) * s[..., None]
    ref_q = q.view(300, 1024)[:, :1000]
    got = fake_w4(Wt).float()
    err_q = (got - ref_q.to(torch.bfloat16).float()).abs().max().item()
    print(f"[check] fake_w4 vs closed-form RTN: max |diff| {err_q:.3e}")
    assert err_q == 0.0

    E = R.load_weights(MODEL, device="cuda", names=lambda n: n == "embed_tokens.weight")
    h_hf = E["embed_tokens.weight"][ids][None]          # [1, T, H]
    h_ref = h_hf[0].clone()                              # independent chain
    del E
    st = R.State(cfg, T)
    worst = {"linear_attention": 0.0, "full_attention": 0.0}
    pe = rope(tc, 1, T)
    for l in range(cfg.layers):
        sd = layer_weights(l)
        m = build_layer(tc, l, sd)
        Wl = {f"layers.{l}.{k}": v for k, v in sd.items()}
        with torch.no_grad():
            out_hf = m(h_hf, position_embeddings=pe, attention_mask=None)
            # reference on the same input (fresh state for this layer)
            st1 = R.State(cfg, T)
            o1 = []
            for t in range(T):
                st1.pos = t
                o1.append(R.decoder_layer(h_hf[0, t], Wl, l, st1, cfg))
            o1 = torch.stack(o1)
            # reference on its own chain
            o2 = []
            for t in range(T):
                st.pos = t
                o2.append(R.decoder_layer(h_ref[t], Wl, l, st, cfg))
            h_ref = torch.stack(o2)
        d_ = (out_hf[0] - h_hf[0]).float()
        e = (o1 - h_hf[0]).float()
        rel = ((d_ - e).norm() / e.norm()).item()        # error on the layer's update (not residual)
        cos = torch.nn.functional.cosine_similarity(d_, e, dim=-1).min().item()
        t = tc.layer_types[l]
        worst[t] = max(worst[t], rel)
        if l < 4 or l % 16 == 15:
            print(f"[check] layer {l:2d} {t:16s} update rel L2 {rel:.2e}  min cos {cos:.6f}", flush=True)
        h_hf = out_hf
        del m, sd, Wl
    W = R.load_weights(MODEL, device="cuda", names=lambda n: n in ("norm.weight", "lm_head.weight"))
    lg_hf = torch.nn.functional.linear(R.rmsnorm1p(h_hf[0], W["norm.weight"], cfg.eps), W["lm_head.weight"])
    lg_ref = torch.nn.functional.linear(R.rmsnorm1p(h_ref, W["norm.weight"], cfg.eps), W["lm_head.weight"])
    agree = (lg_hf.argmax(-1) == lg_ref.argmax(-1)).float().mean().item()
    tgt = ids[1:]
    nll = lambda lg: torch.nn.functional.cross_entropy(lg[:-1].float(), tgt).item()
    print(f"[check] worst layer-update rel L2: GDN {worst['linear_attention']:.2e}, "
          f"attn {worst['full_attention']:.2e}")
    print(f"[check] end-to-end {T} tokens: top-1 agreement {agree:.4f}, "
          f"NLL hf-prefill {nll(lg_hf):.4f} vs reference {nll(lg_ref):.4f}")
    ok = worst["linear_attention"] < 2e-2 and worst["full_attention"] < 2e-2 and agree >= 0.95
    print("[check] PASS" if ok else "[check] FAIL")
    if not ok:
        raise SystemExit(1)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "check", "run", "report", "gptq_head", "_step", "_final"])
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None, help="default: all complete windows")
    ap.add_argument("--weights", default="bf16,w4rtn", help="decoder variants: bf16, w4rtn")
    ap.add_argument("--heads", default="bf16,w4", help="lm_head variants evaluated in the final pass")
    ap.add_argument("--mb", type=int, default=4, help="windows per layer forward")
    ap.add_argument("--budget", type=float, default=480, help="max seconds per locked GPU chunk")
    ap.add_argument("--gap", type=float, default=5, help="seconds between chunks (lock hand-off)")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="train = GPTQ calibration hidden states only (run with --max-windows)")
    ap.add_argument("--damp", type=float, default=0.01, help="GPTQ: damping, fraction of mean diag(H)")
    ap.add_argument("--tokens", type=int, default=64, help="check: tokens")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "check": cmd_check, "run": cmd_run, "report": report,
     "gptq_head": cmd_gptq_head, "_step": cmd_step, "_final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
