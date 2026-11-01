"""Replica of the sudoingX/qwen38-mtp probe.py workload (repo commit 1e514a8), measured with the
server's own timings instead of client-side streaming deltas:
3 prompts (2 code, 1 prose), chat completions, thinking off, max_tokens 400, greedy.
1 warm-up request, then RUNS timed requests per prompt. Reports decode tok/s
(timings.predicted_per_second) per prompt (median/min/max) and over all requests, plus draft
acceptance (accepted / drafted) per prompt and aggregate.
Usage: python3 bench_probe.py <port> <out json> [runs=5]"""
import json, statistics as st, sys, urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
PROMPTS = [  # verbatim from sudoingX/qwen38-mtp probe.py
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
]

def run(prompt, max_tokens=400):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0, "top_k": 1, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    t = d["timings"]
    return {"decode_tps": t["predicted_per_second"], "predicted_n": t["predicted_n"],
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "content": d["choices"][0]["message"].get("content", "")}

run("warmup", 40)
res, allt, dn, da = {}, [], 0, 0
for p in PROMPTS:
    rs = [run(p) for _ in range(RUNS)]
    tps = [r["decode_tps"] for r in rs]; allt += tps
    pdn = sum(r["draft_n"] or 0 for r in rs); pda = sum(r["draft_n_accepted"] or 0 for r in rs)
    dn += pdn; da += pda
    same = all(r["content"] == rs[0]["content"] for r in rs)
    res[p] = {"runs": rs, "median": st.median(tps), "min": min(tps), "max": max(tps),
              "acceptance": (pda / pdn) if pdn else None, "outputs_identical": same}
    print(f"{st.median(tps):7.2f} tok/s median (min {min(tps):.2f}, max {max(tps):.2f}) | "
          f"n_gen {[r['predicted_n'] for r in rs]} | acc {res[p]['acceptance']} | identical {same} | {p[:45]}")
summary = {"overall_median": st.median(allt), "overall_mean": st.mean(allt), "min": min(allt),
           "max": max(allt), "acceptance_aggregate": (da / dn) if dn else None, "runs_per_prompt": RUNS}
print(f"OVERALL: median {summary['overall_median']:.2f} mean {summary['overall_mean']:.2f} "
      f"(min {summary['min']:.2f}, max {summary['max']:.2f}); acceptance aggregate {summary['acceptance_aggregate']}")
json.dump({"summary": summary, "prompts": res}, open(out, "w"), indent=1)
