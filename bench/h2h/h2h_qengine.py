"""Head-to-head variant of bench/engine/bench_decode.py (same timing, warm-up and engine calls), with
(a) a selectable prompt file, (b) optional chat template (same 156-token prompt the other engines
get), (c) non-asserting checks: outputs identical across runs per k, and spec output == plain.
Usage: h2h_qengine.py --prompt FILE [--chat] --spec 2,3,4 --out JSON
"""
import argparse, json, statistics, subprocess, sys, time
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


ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-27B"))
ap.add_argument("--prompt", required=True)
ap.add_argument("--chat", action="store_true")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--new", type=int, default=1024)
ap.add_argument("--spec", default="")
ap.add_argument("--plain", type=int, default=1, help="also run k=0 (plain)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
tok = AutoTokenizer.from_pretrained(a.model)
text = Path(a.prompt).read_text()
if a.chat:
    prompt = tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True,
                                     tokenize=True)
    if hasattr(prompt, "input_ids"):
        prompt = prompt["input_ids"]
    prompt = list(prompt)
else:
    prompt = tok(text).input_ids  # identical to bench_decode.py
eng = Engine(a.model, max_len=len(prompt) + a.new + 32, linear=W4Linear(), lm_head="w4gptq",
             cache_dir=str(Path.home() / "models/qengine-cache"))
print("prompt tokens", len(prompt), "| idle:", smi(), flush=True)
t0 = time.time()
while time.time() - t0 < 1.5:
    eng.generate(prompt, 128)
configs = ([0] if a.plain else []) + [int(k) for k in a.spec.split(",") if k]
results, plain_gen = {}, None
for k in configs:
    if k:
        eng.generate_spec(prompt, 128, k)
    runs, gens = [], []
    for i in range(a.n):
        if k == 0:
            gen, pre_s, dec_s = eng.generate(prompt, a.new, timing=True); st = {}
        else:
            gen, pre_s, dec_s, st = eng.generate_spec(prompt, a.new, k, timing=True)
        gens.append(gen)
        runs.append({"decode_tok_s": (a.new - 1) / dec_s, "prefill_tok_s": len(prompt) / pre_s, **st, "smi": smi()})
        print(f"k={k} run {i}: {runs[-1]['decode_tok_s']:.2f} tok/s {st} | {runs[-1]['smi']}", flush=True)
    if k == 0:
        plain_gen = gens[0]
    tps = [r["decode_tok_s"] for r in runs]
    ident = all(g == gens[0] for g in gens)
    eq_plain = None if plain_gen is None else all(g == plain_gen for g in gens)
    first_diff = None
    if plain_gen is not None and not eq_plain:
        first_diff = next(j for j, (x, y) in enumerate(zip(gens[0], plain_gen)) if x != y)
    results[k] = {"median": statistics.median(tps), "min": min(tps), "max": max(tps), "runs": runs,
                  "outputs_identical": ident, "equals_plain": eq_plain, "first_diff_vs_plain": first_diff}
    if k:
        results[k]["accept_rate_median"] = statistics.median(r["accept_rate"] for r in runs)
        results[k]["tokens_per_cycle_median"] = statistics.median(r["tokens_per_cycle"] for r in runs)
    print(f"k={k}: median {results[k]['median']:.2f} (min {min(tps):.2f}, max {max(tps):.2f}); identical {ident}; "
          f"== plain {eq_plain}", flush=True)
out = {"commit": commit, "prompt": a.prompt, "chat": a.chat, "prompt_tokens": len(prompt), "new": a.new,
       "results": results, "max_mem_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20,
       "output_head": tok.decode(plain_gen[:120]) if plain_gen else tok.decode(gens[0][:120]),
       "cmd": " ".join(sys.argv)}
Path(a.out).write_text(json.dumps(out, indent=1))
