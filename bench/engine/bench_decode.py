"""qengine decode benchmark: fixed prompt, greedy, 1,024 generated tokens, n runs,
median + min/max, ~1 s GPU warm-up, clocks/driver/VRAM/commit recorded.

Usage: scripts/gpu_lock.sh .venv/bin/python bench/engine/bench_decode.py --model ~/models/Qwen3.5-0.8B
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from qengine.engine import Engine, W4Linear  # noqa: E402


def smi():
    q = "clocks.sm,clocks.mem,power.draw,memory.used,driver_version,utilization.gpu"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def distinct_ngrams(ids, n=8):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.5-0.8B"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--new", type=int, default=1024)
    ap.add_argument("--out", default=None)
    ap.add_argument("--w4", action="store_true", help="W4A16 backend (qengine._w4) with the packed cache")
    ap.add_argument("--lm-head", default=None, choices=["same", "bf16", "w4gptq"],
                    help="default: w4gptq with --w4 (criterion-2 configuration), else same")
    ap.add_argument("--cache", default=str(Path.home() / "models/qengine-cache"))
    ap.add_argument("--prompt", default=str(ROOT / "bench/prompts/prose.txt"))
    ap.add_argument("--raw", action="store_true", help="feed the prompt text without the chat template")
    ap.add_argument("--draft-vocab", type=int, default=49152,
                    help="pruned MTP draft vocabulary size (draft_vocab.py; 0 = full lm_head)")
    ap.add_argument("--spec", default="", help="comma list of MTP draft depths k to also run, e.g. 1,2,3,4")
    a = ap.parse_args()
    if a.lm_head is None:
        a.lm_head = "w4gptq" if a.w4 else "same"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True,
                                text=True, cwd=ROOT).stdout.strip())
    tok = AutoTokenizer.from_pretrained(a.model)
    text = Path(a.prompt).read_text()
    if a.raw:
        prompt = tok(text).input_ids
    else:  # chat template, exactly as bench/h2h/h2h_qengine.py (and every baseline) does
        prompt = tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True,
                                         tokenize=True)
        if hasattr(prompt, "input_ids"):
            prompt = prompt["input_ids"]
        prompt = list(prompt)
    eng = Engine(a.model, max_len=len(prompt) + a.new + 32, linear=W4Linear() if a.w4 else None,
                 lm_head=a.lm_head, cache_dir=a.cache if a.w4 else None, draft_vocab=a.draft_vocab or None)
    print("idle:", smi())
    t0 = time.time()  # ~1 s warm-up so clocks settle (also captures the graphs)
    while time.time() - t0 < 1.5:
        eng.generate(prompt, 128)
    configs = [0] + [int(k) for k in a.spec.split(",") if k]
    results, ref_gen = {}, None
    for k in configs:
        if k:
            eng.generate_spec(prompt, 128, k)  # capture this k's graphs before timing
        runs = []
        for i in range(a.n):
            if k == 0:
                gen, pre_s, dec_s = eng.generate(prompt, a.new, timing=True)
                st = {}
            else:
                gen, pre_s, dec_s, st = eng.generate_spec(prompt, a.new, k, timing=True)
            ref_gen = ref_gen or gen
            assert gen == ref_gen, f"k={k}: output differs from plain greedy"
            # the first new token comes from the prefill; decode time covers the other new-1
            runs.append({"decode_tok_s": (a.new - 1) / dec_s, "prefill_tok_s": len(prompt) / pre_s,
                         **st, "smi": smi()})
            print(f"k={k} run {i}: decode {runs[-1]['decode_tok_s']:.2f} tok/s, prefill "
                  f"{runs[-1]['prefill_tok_s']:.0f} tok/s {st} | {runs[-1]['smi']}")
        tps = [r["decode_tok_s"] for r in runs]
        results[k] = {"median": statistics.median(tps), "min": min(tps), "max": max(tps), "runs": runs}
        print(f"k={k}: decode tok/s median {results[k]['median']:.2f} (min {min(tps):.2f}, max {max(tps):.2f})")
    rep = distinct_ngrams(ref_gen)
    print(f"output check: {len(ref_gen)} tokens, distinct 8-grams {rep:.1%}" +
          ("  WARNING: repetitive output, tok/s not representative" if rep < 0.5 else ""))
    print("output head:", repr(tok.decode(ref_gen[:60])))
    res = {"model": a.model, "prompt": a.prompt, "chat_template": not a.raw, "distinct_8grams": rep, "commit": commit, "dirty_tree": dirty, "prompt_tokens": len(prompt),
           "new_tokens": a.new, "n": a.n, "backend": eng.linear.name, "lm_head": a.lm_head, "draft_vocab": a.draft_vocab,
           "w4_launch": getattr(eng.linear, "kw", None), "lm_head_stats": eng.lm_head_stats, "results": results,
           "max_mem_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20, "cmd": " ".join(sys.argv)}
    print(f"commit {commit}{' +dirty' if dirty else ''}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
