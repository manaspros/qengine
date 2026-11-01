"""Batch-1 decode benchmark against an OpenAI-compatible server (vLLM, SparkInfer).

Same protocol as the llama.cpp / exllamav3 baselines: fixed prompt (bench/prompts/prose.txt) sent
as one user message through /v1/chat/completions (the server applies the model's chat template,
thinking left at the template default), greedy (temperature 0), exactly 1,024 new tokens
(ignore_eos), 1 warm-up request discarded + 5 timed, sequential (batch 1).

Streaming. Decode tok/s (client) = (completion_tokens - 1) / (t_last_chunk - t_first_chunk): the
first token is the end of prefill, so this is the post-first-token rate, the same definition as
llama-server's predicted_per_second and exllamav3's time_generate. With speculation a chunk can
carry several tokens; completion_tokens comes from the final usage chunk.

Speculation counters are read from the server's Prometheus /metrics before and after each request
(only counters that exist are diffed), so acceptance is per request.

Usage: bench_openai.py --port P --prompt FILE --out JSON [--engine vllm|sparkinfer] [--runs 6]
"""
import argparse, json, re, statistics, time, urllib.request

ROOFLINE = 124.5  # tok/s, W4 g128 at 1,692 GB/s measured read bandwidth

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--engine", default="vllm")
ap.add_argument("--model", default=None, help="model id; default: first id from /v1/models")
ap.add_argument("--runs", type=int, default=6, help="run 0 is warm-up")
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()
base = f"http://127.0.0.1:{args.port}"


def get(path):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return r.read().decode()


def metrics():
    """Prometheus counters whose names mention speculation / drafts, summed over label sets."""
    out = {}
    try:
        text = get("/metrics")
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name = m.group(1)
        if re.search(r"spec|draft|accept", name) and not name.endswith("_created"):
            out[name] = out.get(name, 0.0) + float(m.group(3))
    return out


model = args.model or json.loads(get("/v1/models"))["data"][0]["id"]
user = open(args.prompt).read().strip()
body = {"model": model, "messages": [{"role": "user", "content": user}],
        "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0, "seed": 0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
if args.engine == "vllm":
    body["top_k"] = -1  # greedy is temperature 0; no truncation
else:
    body["top_k"] = 1


def one():
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    t_end = time.perf_counter()
    n = usage["completion_tokens"]
    return {"completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "ttft_s": stamps[0] - t0, "decode_s": stamps[-1] - stamps[0], "chunks": len(stamps),
            "decode_tps": (n - 1) / (stamps[-1] - stamps[0]), "wall_s": t_end - t0,
            "e2e_tps": n / (t_end - t0), "usage": usage, "content": "".join(text)}


runs = []
for i in range(args.runs):
    m0 = metrics(); r = one(); m1 = metrics()
    r["spec_counters"] = {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] - m0.get(k, 0.0) != 0}
    u = r["usage"]
    if "generation_ms" in u and "ttft_ms" in u:  # SparkInfer engine-side timing
        r["engine_decode_tps"] = (r["completion_tokens"] - 1) / ((u["generation_ms"] - u["ttft_ms"]) / 1000)
    runs.append(r)
    print(f"run {i}{' (warm-up)' if i == 0 else ''}: {r['completion_tokens']} tok in {r['chunks']} chunks, "
          f"decode {r['decode_tps']:.2f} tok/s, ttft {1000 * r['ttft_s']:.0f} ms"
          + (f", engine {r['engine_decode_tps']:.2f}" if "engine_decode_tps" in r else "")
          + (f", spec {r['spec_counters']}" if r["spec_counters"] else ""), flush=True)

timed = runs[1:]
dec = [r["decode_tps"] for r in timed]
e2e = [r["e2e_tps"] for r in timed]
same = all(r["content"] == timed[0]["content"] for r in timed)
summary = {"model": model, "prompt_tokens": timed[0]["prompt_tokens"],
           "generated": [r["completion_tokens"] for r in timed],
           "decode_tps": dec, "decode_median": statistics.median(dec), "decode_min": min(dec),
           "decode_max": max(dec), "e2e_tps": e2e, "e2e_median": statistics.median(e2e),
           "ttft_ms": [1000 * r["ttft_s"] for r in timed], "chunks": [r["chunks"] for r in timed],
           "outputs_identical": same, "warmup_decode_tps": runs[0]["decode_tps"],
           "output_head": timed[0]["content"][:400], "request_body": {k: v for k, v in body.items() if k != "messages"}}
if "engine_decode_tps" in timed[0]:
    eng = [r["engine_decode_tps"] for r in timed]
    summary.update(engine_decode_tps=eng, engine_decode_median=statistics.median(eng))
# tokens per chunk ~ tokens per decode step when speculating (each step streams what it accepted)
summary["tokens_per_chunk"] = [r["completion_tokens"] / r["chunks"] for r in timed]
keys = sorted({k for r in timed for k in r["spec_counters"]})
if keys:
    summary["spec_counters"] = {k: [r["spec_counters"].get(k, 0) for r in timed] for k in keys}
json.dump({"summary": summary, "runs": runs}, open(args.out, "w"), indent=1)
open(args.out.replace(".json", ".txt"), "w").write(timed[0]["content"])  # for greedy-output diffs
print(f"prompt {summary['prompt_tokens']} tok, generated {summary['generated']}")
print(f"decode median {summary['decode_median']:.2f} tok/s (min {min(dec):.2f}, max {max(dec):.2f}) "
      f"= {100 * summary['decode_median'] / ROOFLINE:.1f}% of roofline")
if "engine_decode_median" in summary:
    print(f"engine-side decode median {summary['engine_decode_median']:.2f} tok/s")
print(f"end-to-end median {summary['e2e_median']:.2f}; tokens/chunk "
      f"{statistics.median(summary['tokens_per_chunk']):.2f}; outputs identical: {same}")
if keys:
    for k in keys:
        print(f"  {k}: {summary['spec_counters'][k]}")
