"""qengine MTP evaluation suite: every prompt in bench/prompts/suite/ (chat template, greedy,
1,024 new tokens), n runs each of plain decode and MTP k in --ks, interleaved per round.
Optional --mtp-weights adds a second MTP head ("distilled") measured interleaved with the stock
head on the same loaded engine. Every speculative output is checked == plain greedy.

One JSON per prompt in --outdir (resumable: existing files are skipped), so the suite can run as
one GPU-locked job per prompt:
  for i in 1 2 3 4 5 6 7 8; do scripts/gpu_lock.sh python bench/engine/bench_suite.py --only $i ...; done
  python bench/engine/bench_suite.py --aggregate --outdir ...   # per-prompt table + geomeans
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def distinct_ngrams(ids, n=8):
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / max(1, len(g))


def smi():
    q = "clocks.sm,clocks.mem,power.draw,memory.used,driver_version"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip()


def med(v):
    return {"median": statistics.median(v), "min": min(v), "max": max(v)}


def run_prompt(eng, tok, f, a, heads):
    import torch
    text = f.read_text()
    prompt = tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True,
                                     tokenize=True)
    if hasattr(prompt, "input_ids"):
        prompt = prompt["input_ids"]
    prompt = list(prompt)
    plain = eng.generate(prompt, a.new)
    t0 = time.time()
    while time.time() - t0 < 1.5:  # clocks up
        eng.generate(prompt, 128)
    ks = [int(k) for k in a.ks.split(",")]
    for name, mw in heads.items():  # capture graphs
        eng.mtp = mw
        eng.graphs = {k: g for k, g in eng.graphs.items() if not k[1]}
        for k in ks:
            eng.generate_spec(prompt, 32, k)
    res = {"plain": []}
    for _ in range(a.n):
        _, _, d = eng.generate(prompt, a.new, timing=True)
        res["plain"].append(((a.new - 1) / d, {}))
        for k in ks:
            for name, mw in heads.items():
                if eng.mtp is not mw:
                    eng.mtp = mw
                    eng.graphs = {kk: g for kk, g in eng.graphs.items() if not kk[1]}
                g, _, d, st = eng.generate_spec(prompt, a.new, k, timing=True)
                assert g == plain, f"{f.name}: {name} k={k} differs from plain greedy"
                res.setdefault(f"{name}_k{k}", []).append(((a.new - 1) / d, st))
    out = {"prompt": f.name, "prompt_tokens": len(prompt), "new_tokens": a.new, "n": a.n,
           "distinct_8grams": distinct_ngrams(plain), "output_head": tok.decode(plain[:80]),
           "spec_equals_plain": True, "smi_end": smi(), "configs": {}}
    for key, v in res.items():
        tps = [x[0] for x in v]
        c = {"tok_s": med(tps)}
        if v[0][1]:
            st = v[0][1]  # acceptance is identical across runs (deterministic)
            c.update(tokens_per_cycle=st["tokens_per_cycle"], accept_rate=st["accept_rate"],
                     accept_by_depth=st["accept_by_depth"])
        out["configs"][key] = c
    return out


def aggregate(outdir):
    files = sorted(p for p in Path(outdir).glob("*.json") if p.name != "summary.json")
    rows = [json.loads(p.read_text()) for p in files]
    keys = sorted({k for r in rows for k in r["configs"]}, key=lambda s: (s != "plain", s))
    print("| prompt | tokens | distinct 8-grams | " + " | ".join(keys) + " |")
    print("|---|---:|---:|" + "---:|" * len(keys))
    for r in rows:
        cells = []
        for k in keys:
            c = r["configs"].get(k)
            cells.append("-" if c is None else f"{c['tok_s']['median']:.1f}" +
                         (f" ({c['tokens_per_cycle']:.2f})" if "tokens_per_cycle" in c else ""))
        print(f"| {r['prompt']} | {r['prompt_tokens']} | {r['distinct_8grams']:.1%} | " + " | ".join(cells) + " |")
    geo = {}
    for k in keys:
        v = [r["configs"][k]["tok_s"]["median"] for r in rows if k in r["configs"]]
        if len(v) == len(rows):
            geo[k] = math.prod(v) ** (1 / len(v))
    print("| **geomean** | | | " + " | ".join(f"**{geo[k]:.1f}**" if k in geo else "-" for k in keys) + " |")
    heads = sorted({k.rsplit("_k", 1)[0] for k in keys if k != "plain"})
    best = {}
    for h in heads:  # best single k per head over the suite, and per-prompt best k (oracle)
        hk = {k: geo[k] for k in geo if k.startswith(h + "_k")}
        if hk:
            kb = max(hk, key=hk.get)
            oracle = math.prod(max(r["configs"][k]["tok_s"]["median"] for k in hk) for r in rows) ** (1 / len(rows))
            best[h] = {"best_k": kb, "geomean": hk[kb], "per_prompt_best_k_geomean": oracle}
            print(f"{h}: best single k {kb} geomean {hk[kb]:.2f}; per-prompt best k geomean {oracle:.2f}")
    depth = {}
    for k in keys:
        if k == "plain":
            continue
        accs = [r["configs"][k]["accept_by_depth"] for r in rows if k in r["configs"]]
        depth[k] = [statistics.mean(a[d] for a in accs) for d in range(len(accs[0]))]
        print(f"{k}: mean acceptance by depth {[round(x, 3) for x in depth[k]]}")
    summary = {"prompts": [r["prompt"] for r in rows], "geomean": geo, "best": best, "mean_accept_by_depth": depth,
               "per_prompt": rows}
    (Path(outdir) / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-27B"))
    ap.add_argument("--suite", default=str(ROOT / "bench/prompts/suite"))
    ap.add_argument("--outdir", default=str(ROOT / "build/suite"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--new", type=int, default=1024)
    ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--draft-vocab", type=int, default=49152)
    ap.add_argument("--mtp-weights", default=None, help="extra MTP head measured as 'distilled'")
    ap.add_argument("--only", default=None, help="comma list of prompt-file prefixes (e.g. 1,2) to run")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate(a.outdir)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.suite).glob("*.txt"))
    if a.only:
        sel = a.only.split(",")
        files = [f for f in files if f.name.split("_")[0] in sel]
    files = [f for f in files if not (out / (f.stem + ".json")).exists()]
    if not files:
        print("nothing to do")
        return
    import torch
    from transformers import AutoTokenizer
    from qengine import reference as R
    from qengine.engine import Engine, W4Linear
    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(a.model, max_len=2048, linear=W4Linear(), lm_head="w4gptq", draft_vocab=a.draft_vocab or None,
                 cache_dir=str(Path.home() / "models/qengine-cache"), mtp_weights=a.mtp_weights)
    heads = {}
    if a.mtp_weights:  # engine holds the extra head; rebuild the stock one next to it (shared MTP KV cache)
        extra = eng.mtp
        W = R.load_weights(a.model, device="cuda", names=lambda n: n.startswith("mtp."))
        stock = eng._layer_weights(W, "mtp.layers.0.", "full_attention")
        stock.update(fc=eng.linear.prepare(W["mtp.fc.weight"]), norm=W["mtp.norm.weight"],
                     pre_e=W["mtp.pre_fc_norm_embedding.weight"], pre_h=W["mtp.pre_fc_norm_hidden.weight"],
                     kc=extra["kc"], vc=extra["vc"])
        del W
        torch.cuda.empty_cache()
        heads = {"stock": stock, "distilled": extra}
    else:
        heads = {"stock": eng.mtp}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    for f in files:
        r = run_prompt(eng, tok, f, a, heads)
        r.update(commit=commit, mtp_weights=a.mtp_weights, draft_vocab=a.draft_vocab, cmd=" ".join(sys.argv))
        (out / (f.stem + ".json")).write_text(json.dumps(r, indent=1))
        line = ", ".join(f"{k} {c['tok_s']['median']:.1f}" for k, c in r["configs"].items())
        print(f"{f.name}: distinct {r['distinct_8grams']:.1%} | {line}", flush=True)


if __name__ == "__main__":
    main()
