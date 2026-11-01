"""Summarize llama-bench JSON (median/min/max over repetitions) and the clock log of the run."""
import csv, json, statistics, sys

ROOFLINE = 124.5  # tok/s, W4 g128 at 1,692 GB/s measured read bandwidth

def clocks(path):
    sm, mem, pw, used = [], [], [], []
    for row in csv.reader(open(path)):
        if len(row) < 7:
            continue
        util = float(row[5].split()[0])
        if util < 50:  # only samples where the GPU was busy
            continue
        sm.append(float(row[1].split()[0])); mem.append(float(row[2].split()[0]))
        pw.append(float(row[3].split()[0])); used.append(float(row[6].split()[0]))
    if not sm:
        return "no busy samples"
    return (f"SM {min(sm):.0f}-{max(sm):.0f} (median {statistics.median(sm):.0f}) MHz, "
            f"mem {statistics.median(mem):.0f} MHz, {min(pw):.0f}-{max(pw):.0f} W, "
            f"VRAM used max {max(used):.0f} MiB, {len(sm)} busy samples")

def main():
  for r in json.load(open(sys.argv[1])):
      s = r["samples_ts"]
      test = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}" + (f"@d{r['n_depth']}" if r.get("n_depth") else "")
      med = statistics.median(s)
      print(f"{test:>12} fa={r.get('flash_attn')} ub={r.get('n_ubatch')} t={r.get('n_threads')} "
            f"n={len(s)} median {med:.2f} tok/s (min {min(s):.2f}, max {max(s):.2f})"
            + (f" = {100*med/ROOFLINE:.1f}% of roofline" if r["n_gen"] else ""))
  if len(sys.argv) > 2:
      print("clocks:", clocks(sys.argv[2]))


if __name__ == "__main__":
    main()
