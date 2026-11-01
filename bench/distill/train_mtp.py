"""Self-distill the Qwen3.8-27B MTP head on the W4 engine's own greedy outputs (ideas-round2 #2).

Data: bench/distill/gen_data.py (tokens + post-final-norm hidden states per position).
Trained: the MTP decoder layer, fc, pre_fc norms and mtp.norm (fp32 master weights, BF16 compute).
Frozen: token embedding and the draft head (the 48K-row GPTQ W4 lm_head used at inference,
dequantized), so training sees exactly the head that drafts.

Forward over a whole sequence, all bases t in parallel, emulating inference exactly:
  depth 1 (MTP rows pass): position t, input (h_t, x_{t+1}), attends to depth-1 keys at <= t
  depth d >= 2 (chain):    position t+d-1, input (own post-mtp.norm output of depth d-1 at base t,
                           x_{t+d}); attends to depth-1 keys at <= t plus the chain keys of the
                           same base at positions t+1 .. t+d-1 (what the KV cache holds at that
                           point of a cycle).
Loss: CE over the draft vocabulary for the target x_{t+d+1}, only where the target lies in the
generated part (the model's own greedy choices) and is in the draft vocabulary.

Time-boxed + resumable (checkpoint in ~/models/mtp-distill/ckpt.pt): run repeatedly under
scripts/gpu_lock.sh. `--check` compares the forward with reference.mtp_step and exits.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qengine import reference as R  # noqa: E402
from qengine.draft_vocab import draft_vocab_ids  # noqa: E402

DATA = Path.home() / "models" / "mtp-distill"
M27 = Path.home() / "models" / "Qwen3.8-27B"
BF = torch.bfloat16
dev = "cuda"
NAMES = ["mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"] + [
    f"mtp.layers.0.{n}.weight" for n in
    ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
     "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]


def rms1p(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(BF)


class MTPTrain:
    def __init__(self, cfg, W):
        self.cfg = cfg
        self.p = {n: torch.nn.Parameter(W[n].float().clone()) for n in NAMES}
        rd = cfg.rotary_dim
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))).to(dev)

    def w(self, n):
        return self.p[n if n in self.p else n + ".weight"].to(BF)

    def rope(self, x, pos):  # x [T, heads, 256] bf16, pos [T]
        ang = pos.float()[:, None] * self.inv_freq
        emb = torch.cat([ang, ang], -1)
        cos, sin = emb.cos().to(BF)[:, None, :], emb.sin().to(BF)[:, None, :]
        rd = cos.shape[-1]
        xr, xp = x[..., :rd], x[..., rd:]
        rot = torch.cat([-xr[..., rd // 2:], xr[..., :rd // 2]], -1)
        return torch.cat([xr * cos + rot * sin, xp], -1)

    def qkv(self, x, pos):
        c, L = self.cfg, "mtp.layers.0.self_attn."
        T = x.shape[0]
        qg = F.linear(x, self.w(L + "q_proj")).view(T, c.hq, 512)
        q, gate = qg[..., :256], qg[..., 256:].reshape(T, -1)
        q = self.rope(rms1p(q, self.w(L + "q_norm"), c.eps), pos)
        k = self.rope(rms1p(F.linear(x, self.w(L + "k_proj")).view(T, c.hkv, 256), self.w(L + "k_norm"), c.eps), pos)
        v = F.linear(x, self.w(L + "v_proj")).view(T, c.hkv, 256)
        return q, gate, k, v

    def layer(self, u, pos, ctx):
        """u [T, H] layer input for depth d at positions pos; ctx = list of (k, v) of depths
        1..d-1 for the same bases (k [T, Hkv, 256]); returns (layer out pre-norm, (k, v))."""
        c, L = self.cfg, "mtp.layers.0."
        x = rms1p(u, self.w(L + "input_layernorm"), c.eps)
        q, gate, k, v = self.qkv(x, pos)
        grp = c.hq // c.hkv
        T = u.shape[0]
        K1, V1 = (ctx[0] if ctx else (k, v))
        qh = q.float().transpose(0, 1)                                           # [Hq, T, 256]
        k1 = K1.float().repeat_interleave(grp, 1).transpose(0, 1)                # [Hq, T, 256]
        v1 = V1.float().repeat_interleave(grp, 1).transpose(0, 1)
        s_main = torch.einsum("htd,hjd->htj", qh, k1) / 16.0
        mask = torch.ones(T, T, dtype=torch.bool, device=u.device).tril()
        s_main = s_main.masked_fill(~mask, float("-inf"))
        extras = [ctx_kv for ctx_kv in ctx[1:]] + ([(k, v)] if ctx else [])  # chain keys of the same base
        if extras:
            ke = [kk.float().repeat_interleave(grp, 1).transpose(0, 1) for kk, _ in extras]
            ve = [vv.float().repeat_interleave(grp, 1).transpose(0, 1) for _, vv in extras]
            s_ex = torch.stack([(qh * kk).sum(-1) / 16.0 for kk in ke], -1)      # [Hq, T, E]
            s = torch.cat([s_main, s_ex], -1)
            pr = torch.softmax(s, -1)
            o = torch.einsum("htj,hjd->htd", pr[..., :T], v1)
            for i, vv in enumerate(ve):
                o = o + pr[..., T + i:T + i + 1] * vv
        else:
            o = torch.einsum("htj,hjd->htd", torch.softmax(s_main, -1), v1)
        o = o.transpose(0, 1).reshape(T, -1).to(BF) * torch.sigmoid(gate)
        h = u + F.linear(o, self.w(L + "self_attn.o_proj"))
        x2 = rms1p(h, self.w(L + "post_attention_layernorm"), c.eps)
        mlp = F.linear(F.silu(F.linear(x2, self.w(L + "mlp.gate_proj"))) * F.linear(x2, self.w(L + "mlp.up_proj")),
                       self.w(L + "mlp.down_proj"))
        return h + mlp, (k, v)

    def forward(self, tokens, hidden, embed, depths):
        """Returns list of post-mtp.norm outputs per depth, each [T, H] indexed by base t
        (rows past the valid range are garbage and masked by the loss)."""
        c, T = self.cfg, tokens.shape[0]
        H = c.hidden
        base = torch.arange(T, device=dev)
        outs, ctx, hin = [], [], hidden
        for d in range(1, depths + 1):
            tok_in = tokens[(base + d).clamp(max=T - 1)]
            e = rms1p(embed[tok_in], self.w("mtp.pre_fc_norm_embedding"), c.eps)
            hh = rms1p(hin, self.w("mtp.pre_fc_norm_hidden"), c.eps)
            u = F.linear(torch.cat([e, hh], -1), self.w("mtp.fc"))
            y, kv = self.layer(u, base + d - 1, ctx)
            ctx = ctx + [kv]
            out = rms1p(y, self.w("mtp.norm"), c.eps)
            outs.append(out)
            hin = out
        return outs


def load_seq(f):
    d = torch.load(f)
    return d["tokens"].long().to(dev), d["prompt_len"], d["hidden"].to(dev)


def losses(model, seq, embed, head, vmap, depths):
    tokens, plen, hidden = seq
    T = tokens.shape[0]
    outs = model.forward(tokens, hidden, embed, depths)
    res = []
    base = torch.arange(T, device=dev)
    for d, out in enumerate(outs, start=1):
        tgt_pos = base + d + 1
        valid = (tgt_pos < T) & (tgt_pos >= plen)
        tgt = vmap[tokens[tgt_pos.clamp(max=T - 1)]]
        valid = valid & (tgt >= 0)
        logits = F.linear(out[valid], head).float()
        res.append((F.cross_entropy(logits, tgt[valid], reduction="sum"), (logits.argmax(-1) == tgt[valid]).sum(),
                    valid.sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--depths", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cfg = R.Config.from_dir(M27)
    W = R.load_weights(M27, names=lambda n: n.startswith("mtp.") or n == "embed_tokens.weight")
    embed = W.pop("embed_tokens.weight")
    ids = draft_vocab_ids(M27, 49152, cfg.vocab).to(dev).long()
    G = torch.load(Path.home() / "models/ppl-cache/lm_head_w4gptq.pt", map_location=dev, weights_only=False)
    head = G[ids].contiguous()  # GPTQ grid values (what the W4 draft head computes), bf16
    del G
    torch.cuda.empty_cache()
    vmap = torch.full((cfg.vocab,), -1, dtype=torch.long, device=dev)
    vmap[ids] = torch.arange(ids.numel(), device=dev)
    model = MTPTrain(cfg, W)
    files = sorted(DATA.glob("seq_*.pt"))
    rnd = random.Random(0)
    order = files[:]
    rnd.shuffle(order)
    n_val = max(16, len(order) // 10)
    val, train = order[:n_val], order[n_val:]

    if a.check:
        return check(model, cfg, W, embed, head, ids, val[0])

    @torch.no_grad()
    def evaluate():
        tot = [[0.0, 0, 0] for _ in range(a.depths)]
        for f in val:
            for d, (l, c, n) in enumerate(losses(model, load_seq(f), embed, head, vmap, a.depths)):
                tot[d][0] += l.item(); tot[d][1] += c.item(); tot[d][2] += n.item()
        return [t[0] / max(1, t[2]) for t in tot], [t[1] / max(1, t[2]) for t in tot]

    params = list(model.p.values())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = int(a.epochs * len(train) / a.accum)
    ck = DATA / "ckpt.pt"
    state = {"step": 0, "micro": 0, "best": None, "history": []}
    if ck.exists() and not a.fresh:
        s = torch.load(ck, map_location=dev, weights_only=False)
        for n in NAMES:
            model.p[n].data.copy_(s["params"][n])
        opt.load_state_dict(s["opt"])
        state = s["state"]
        print(f"resumed at step {state['step']}", flush=True)
    else:
        vl, va = evaluate()
        state["history"].append((0, vl, va))
        state["best"] = (sum(va), 0)
        torch.save({n: model.p[n].detach().to(BF).cpu() for n in NAMES}, DATA / "mtp_best.pt")
        print(f"step 0 (stock): val CE {[round(x, 3) for x in vl]} top1 {[round(x, 4) for x in va]}", flush=True)

    def lr_at(step):
        warm = 10
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm)))))

    weights = [1.0, 0.8, 0.6, 0.5][:a.depths]
    while state["step"] < total and time.time() - t0 < a.budget:
        f = train[(state["micro"]) % len(train)]
        res = losses(model, load_seq(f), embed, head, vmap, a.depths)
        loss = sum(w * l / max(1, n.item()) for w, (l, _, n) in zip(weights, res)) / a.accum
        loss.backward()
        state["micro"] += 1
        if state["micro"] % a.accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % a.eval_every == 0 or state["step"] == total:
                vl, va = evaluate()
                state["history"].append((state["step"], vl, va))
                tag = ""
                if sum(va) > state["best"][0]:
                    state["best"] = (sum(va), state["step"])
                    torch.save({n: model.p[n].detach().to(BF).cpu() for n in NAMES}, DATA / "mtp_best.pt")
                    tag = " (best)"
                print(f"step {state['step']}/{total}: val CE {[round(x, 3) for x in vl]} top1 "
                      f"{[round(x, 4) for x in va]}{tag}", flush=True)
    torch.save({"params": {n: model.p[n].detach() for n in NAMES}, "opt": opt.state_dict(), "state": state}, ck)
    print(f"saved checkpoint at step {state['step']}/{total}; best {state['best']}; {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def check(model, cfg, W, embed, head, ids, f):
    """Forward vs reference.mtp_step: depth 1 for bases 0..39, depth 2 for bases 20..29."""
    tokens, plen, hidden = load_seq(f)
    tokens, hidden = tokens[:48], hidden[:48]
    outs = model.forward(tokens, hidden, embed, 2)
    Wr = dict(W)
    Wr["embed_tokens.weight"] = embed
    Wr["lm_head.weight"] = head  # mtp_step also returns logits; the 48K draft rows suffice
    lin = lambda x, w: F.linear(x, w)
    agree1 = agree2 = 0
    mst = R.MTPState(cfg, 64)
    ref1 = []
    for t in range(40):
        hn = R.mtp_step(Wr, mst, hidden[t], int(tokens[t + 1]), t, cfg, lin=lin)[1]
        ref1.append(hn)
        agree1 += int((F.linear(hn, head).argmax() == F.linear(outs[0][t], head).argmax()).item())
    rel1 = ((torch.stack(ref1).float() - outs[0][:40].float()).norm() / torch.stack(ref1).float().norm()).item()
    rels = []
    for t in range(20, 30):
        mst = R.MTPState(cfg, 64)
        for j in range(t + 1):
            hn = R.mtp_step(Wr, mst, hidden[j], int(tokens[j + 1]), j, cfg, lin=lin)[1]
        h2 = R.mtp_step(Wr, mst, hn, int(tokens[t + 2]), t + 1, cfg, lin=lin)[1]
        rels.append(((h2.float() - outs[1][t].float()).norm() / h2.float().norm()).item())
        agree2 += int((F.linear(h2, head).argmax() == F.linear(outs[1][t], head).argmax()).item())
    print(f"check: depth1 rel {rel1:.2e}, top1 agree {agree1}/40; depth2 rel max {max(rels):.2e}, top1 agree {agree2}/10")


if __name__ == "__main__":
    main()
