"""Summarize bench/gemv/run.sh output: median over the 5 per-process medians, with min-max."""
import collections
import glob
import json
import statistics
import sys

out = sys.argv[1]
PEAK, NCU_PEAK = 1692.0, 1765.0
rows = collections.defaultdict(list)
seq = collections.defaultdict(list)
for f in sorted(glob.glob(f"{out}/run*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("mode") == "sequence":
            seq[(r["N"], r["fused"])].append(r)
        else:
            rows[(r["N"], r["shape"])].append(r)

print(f"# {out}  (n = runs per row; GB/s = median of per-run medians [min-max])")
print("| N | shape | n | us | GB/s | % of 1,692 | % of ncu peak 1,765 |")
print("|---|---|---|---:|---:|---:|---:|")
for (N, shape), rs in sorted(rows.items()):
    g = [r["GBps"] for r in rs]
    us = statistics.median(r["us_median"] for r in rs)
    m = statistics.median(g)
    print(f"| {N} | {shape} | {len(rs)} | {us:.2f} | {m:.0f} [{min(g):.0f}-{max(g):.0f}] | "
          f"{100*m/PEAK:.1f} | {100*m/NCU_PEAK:.1f} |")
print()
print("| N | sequence | kernels | n | ms/token | GB/s | % of 1,692 | tok/s ceiling (linear only) |")
print("|---|---|---|---|---:|---:|---:|---:|")
for (N, fused), rs in sorted(seq.items()):
    ms = [r["ms_median"] for r in rs]
    g = [r["GBps"] for r in rs]
    m = statistics.median(ms)
    print(f"| {N} | {'fused' if fused else 'unfused'} | {rs[0]['kernels']} | {len(rs)} | {m:.3f} "
          f"[{min(ms):.3f}-{max(ms):.3f}] | {statistics.median(g):.0f} | "
          f"{100*statistics.median(g)/PEAK:.1f} | {1e3/m:.1f} [{1e3/max(ms):.1f}-{1e3/min(ms):.1f}] |")
