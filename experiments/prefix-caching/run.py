"""Lesson 26: prefix caching on vs off, Qwen2.5-1.5B bf16, vLLM on one RTX 5090.
Run in an env with vLLM 0.32.0:  python run.py [e1|e2]   (VLLM=/path/to/vllm to pick the binary)

For each config we start a fresh `vllm serve` (so the cache starts empty), warm it up with a
different random seed (so warm-up prompts do NOT pre-fill the cache with the measured prefixes),
run `vllm bench serve`, and read the hit counters from /metrics before and after.

E1 latency   : 1 user at a time, one shared 1,000-token prefix + 24 unique tokens, outputs 16 / 512.
E2 throughput: everyone at once (rate inf, 256 in flight), 1,024 input tokens with share s drawn
               from 8 shared prefixes (the article's setup), 256 output; plus a 16-output control.
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
VLLM = os.environ.get("VLLM", "vllm")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PORT = 8026
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def metrics():
    """Sum of vLLM's prefix-cache counters: tokens queried and tokens found in the cache."""
    txt = urllib.request.urlopen(f"http://localhost:{PORT}/metrics").read().decode()
    get = lambda name: sum(float(v) for v in re.findall(rf"^{name}{{[^}}]*}} ([0-9.e+]+)", txt, re.M))
    return get("vllm:prefix_cache_queries_total"), get("vllm:prefix_cache_hits_total")


def serve(cache: bool):
    args = [VLLM, "serve", MODEL, "--port", str(PORT), "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.6", "--max-model-len", "4096"]
    if not cache:
        args.append("--no-enable-prefix-caching")  # V1 turns prefix caching on by default
    p = subprocess.Popen(args, stdout=open(os.path.join(OUT, "server.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(300):  # wait until /health answers (model load + torch.compile + graph capture)
        try:
            if p.poll() is not None: sys.exit("server exited, see results/server.log")
            urllib.request.urlopen(f"http://localhost:{PORT}/health"); return p
        except Exception:
            time.sleep(1)
    p.kill(); sys.exit("server did not start")


def bench(tag, prefix, suffix, out, n, conc, nprefix, seed=0, save=True):
    args = [VLLM, "bench", "serve", "--backend", "vllm", "--model", MODEL, "--port", str(PORT),
            "--dataset-name", "prefix_repetition",
            "--prefix-repetition-prefix-len", str(prefix), "--prefix-repetition-suffix-len", str(suffix),
            "--prefix-repetition-num-prefixes", str(nprefix), "--prefix-repetition-output-len", str(out),
            "--num-prompts", str(n), "--max-concurrency", str(conc), "--request-rate", "inf",
            "--ignore-eos",  # exactly `out` tokens per request, so runs are comparable
            "--seed", str(seed), "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,99"]
    if save:
        args += ["--save-result", "--result-dir", OUT, "--result-filename", f"{tag}.json"]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL if not save else open(os.path.join(OUT, f"{tag}.log"), "w"))


def run(tag, cache, **kw):
    p = serve(cache)
    try:
        # warm-up: >=1 s of real load (CLAUDE.md), different seed -> different random prefixes
        bench("warm", kw["prefix"], kw["suffix"], kw["out"], max(64, kw["conc"]), kw["conc"], kw["nprefix"], seed=999, save=False)
        q0, h0 = metrics()
        bench(tag, **kw)
        q1, h1 = metrics()
        r = json.load(open(os.path.join(OUT, f"{tag}.json")))
        r["hit_rate"] = (h1 - h0) / (q1 - q0) if q1 > q0 else 0.0
        r["cache"] = cache
        json.dump(r, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
        print(f"{tag:24s} hit {r['hit_rate']:6.1%}  req/s {r['request_throughput']:7.2f}  "
              f"TTFT p50 {r['median_ttft_ms']:8.1f} p99 {r['p99_ttft_ms']:8.1f} ms  "
              f"E2E p50 {r['median_e2el_ms']:8.1f} ms  out tok/s {r['output_throughput']:8.0f}", flush=True)
    finally:
        p.terminate(); p.wait()


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "e1"):
        for out in (16, 512):
            for cache in (False, True):
                run(f"e1_out{out}_{'on' if cache else 'off'}", cache,
                    prefix=1000, suffix=24, out=out, n=40, conc=1, nprefix=1)
    if only in ("all", "e2"):
        for share in (0, 50, 75, 90):
            pre = round(1024 * share / 100)
            run(f"e2_share{share}_on", True, prefix=pre, suffix=1024 - pre, out=256, n=2048, conc=256, nprefix=8)
        run("e2_share90_off", False, prefix=922, suffix=102, out=256, n=2048, conc=256, nprefix=8)
        for cache in (False, True):  # control: short outputs -> prefill is most of the work
            run(f"e2_share90_out16_{'on' if cache else 'off'}", cache,
                prefix=922, suffix=102, out=16, n=4096, conc=256, nprefix=8)
