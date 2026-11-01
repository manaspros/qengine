"""Aggregate n runs of membw: per (size, kind) median of per-run medians, min and max across runs."""
import csv, glob, statistics, sys
out = sys.argv[1]
runs = sorted(glob.glob(f"{out}/run*.csv"))
data = {}
for f in runs:
    rows = [l for l in open(f) if l[0].isdigit()]
    for size, kind, med, _, _ in csv.reader(rows):
        data.setdefault((int(size), kind), []).append(float(med))
print(f"n={len(runs)} runs; per-run value = median of 30 timed launches")
print("size_MiB kind  median  min  max (GB/s)")
for (size, kind), v in sorted(data.items()):
    print(f"{size:>8} {kind:<5} {statistics.median(v):7.1f} {min(v):7.1f} {max(v):7.1f}")
import os
if os.path.exists(f"{out}/sweep.csv"):
    sw = {}
    for l in open(f"{out}/sweep.csv"):
        if l.startswith("sweep,"):
            _, gb, win, nwin, med, mn, mx = l.strip().split(",")
            sw.setdefault((float(gb), int(win)), []).append(float(med))
    for (gb, win), v in sorted(sw.items()):
        print(f"sweep {gb:.2f} GB in {win} MiB windows: median {statistics.median(v):.1f} "
              f"(min {min(v):.1f}, max {max(v):.1f}) GB/s, n={len(v)}")
clk = [l.split(", ") for l in open(f"{out}/clocks.csv")]
busy = [c for c in clk if int(c[5].split()[0]) > 50]
if busy:
    sm = [int(c[1].split()[0]) for c in busy]; mem = [int(c[2].split()[0]) for c in busy]
    pw = [float(c[3].split()[0]) for c in busy]; t = [int(c[4]) for c in busy]
    print(f"clocks while busy (util>50%, {len(busy)} samples): SM {min(sm)}-{max(sm)} MHz "
          f"(median {statistics.median(sm)}), mem {min(mem)}-{max(mem)} MHz, "
          f"power {min(pw):.0f}-{max(pw):.0f} W, temp {min(t)}-{max(t)} C")
print(open(f"{out}/before.csv").read().strip())
