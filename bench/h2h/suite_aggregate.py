"""Aggregate the 8-prompt suite: baselines (build/h2h_suite/<prompt>/<tag>.json, from
bench/h2h/suite_baselines.sh) and qengine (build/suite/summary.json, from bench_suite.py
--aggregate). Prints per-prompt decode tok/s medians and the geomean per configuration."""
import json
import math
import statistics as st
from pathlib import Path

Q = Path(__file__).resolve().parents[2]
B, QS = Q / "build/h2h_suite", Q / "build/suite/summary.json"
prompts = sorted(p.stem for p in (Q / "bench/prompts/suite").glob("*.txt"))
table = {}  # config -> prompt -> median tok/s
for p in prompts:
    for j in sorted((B / p).glob("*.json")) if (B / p).exists() else []:
        r = json.loads(j.read_text())
        if isinstance(r, dict) and "summary" in r:
            table.setdefault(j.stem, {})[p] = st.median(r["summary"]["decode_tps"])
if QS.exists():
    s = json.loads(QS.read_text())
    for r in s["per_prompt"]:
        for k, c in r["configs"].items():
            table.setdefault(f"qengine_{k}", {})[Path(r["prompt"]).stem] = c["tok_s"]["median"]
print("| config | " + " | ".join(prompts) + " | geomean |")
print("|---|" + "---:|" * (len(prompts) + 1))
for cfg, v in sorted(table.items()):
    cells = [f"{v[p]:.1f}" if p in v else "-" for p in prompts]
    geo = math.prod(v.values()) ** (1 / len(v)) if len(v) == len(prompts) else None
    print(f"| {cfg} | " + " | ".join(cells) + f" | {geo:.1f} |" if geo else f"| {cfg} | " + " | ".join(cells) + " | (incomplete) |")
