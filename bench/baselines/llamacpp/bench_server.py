"""End-to-end llama-server baseline: fixed prompt (chat template applied by the server), greedy,
1,024 generated tokens (ignore_eos), 1 warm-up + 5 timed requests, prompt cache off.
Reports the server's decode tok/s (timings.predicted_per_second), prefill time, and client wall time.
Usage: python3 bench_server.py <port> <prompt file> <out json>
The server must already be running (see run_server.sh)."""
import json, statistics, sys, time, urllib.request

port, prompt_file, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
ROOFLINE = 124.5

def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

user = open(prompt_file).read().strip()
prompt = post("/apply-template", {"messages": [{"role": "user", "content": user}]})["prompt"]
body = {"prompt": prompt, "n_predict": 1024, "temperature": 0.0, "top_k": 1, "seed": 0,
        "ignore_eos": True, "cache_prompt": False, "stream": False}
runs = []
for i in range(6):  # run 0 = warm-up, discarded
    t0 = time.perf_counter(); r = post("/completion", body); wall = time.perf_counter() - t0
    t = r["timings"]
    runs.append({"predicted_n": t["predicted_n"], "decode_tps": t["predicted_per_second"],
                 "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "wall_s": wall,
                 "e2e_tps": t["predicted_n"] / wall, "content": r["content"],
                 "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")})
timed = runs[1:]
dec = [r["decode_tps"] for r in timed]; e2e = [r["e2e_tps"] for r in timed]
same = all(r["content"] == timed[0]["content"] for r in timed)
summary = {"prompt_tokens": timed[0]["prompt_n"], "generated": [r["predicted_n"] for r in timed],
           "decode_tps": dec, "decode_median": statistics.median(dec),
           "e2e_tps": e2e, "e2e_median": statistics.median(e2e),
           "prompt_ms": [r["prompt_ms"] for r in timed], "outputs_identical": same,
           "warmup_decode_tps": runs[0]["decode_tps"], "output_head": timed[0]["content"][:400]}
if timed[0]["draft_n"]:  # speculative decoding: acceptance = accepted draft tokens / drafted tokens
    acc = [r["draft_n_accepted"] / r["draft_n"] for r in timed]
    summary["acceptance"] = acc; summary["acceptance_median"] = statistics.median(acc)
    summary["draft_n"] = [r["draft_n"] for r in timed]
    summary["draft_n_accepted"] = [r["draft_n_accepted"] for r in timed]
json.dump({"summary": summary, "runs": runs}, open(out, "w"), indent=1)
print(f"prompt {summary['prompt_tokens']} tok, generated {summary['generated']}")
print(f"decode median {statistics.median(dec):.2f} tok/s (min {min(dec):.2f}, max {max(dec):.2f}) "
      f"= {100*statistics.median(dec)/ROOFLINE:.1f}% of roofline")
print(f"end-to-end (incl. prefill + HTTP) median {statistics.median(e2e):.2f} (min {min(e2e):.2f}, max {max(e2e):.2f})")
if "acceptance" in summary:
    print(f"acceptance median {summary['acceptance_median']:.3f} (min {min(acc):.3f}, max {max(acc):.3f}); "
          f"drafted {summary['draft_n']}, accepted {summary['draft_n_accepted']}")
open(out.replace(".json", ".txt"), "w").write(timed[0]["content"])  # for greedy-output diffs
print(f"prefill ms {[round(x,1) for x in summary['prompt_ms']]}; outputs identical across runs: {same}")
