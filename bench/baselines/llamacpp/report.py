"""Markdown table of every llama.cpp run in one or more output dirs (llama-bench tg runs and
llama-server runs, prose or probe workload): median (min-max), acceptance, mean accepted length,
SM clock median, VRAM max, outputs identical, commit.
Usage: python3 report.py <outdir> [<outdir> ...]"""
import glob, json, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize

ROOF = 124.5
print("| dir | tag | kind | tok/s median (min–max) | % of 124.5 | acceptance | mean len | SM MHz (median) | VRAM max MiB | identical | llama.cpp |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
for d in sys.argv[1:]:
    for meta in sorted(glob.glob(f"{d}/*.meta.txt")):
        tag = os.path.basename(meta)[:-9]
        if tag.endswith(".sx"):  # run_sxprobe.sh cross-checks, reported separately
            continue
        m = open(meta).read()
        commit = (re.search(r"llama.cpp: (\w{7})", m) or [None, "?"])[1]
        clk = summarize.clocks(f"{d}/{tag}.clocks.csv") if os.path.exists(f"{d}/{tag}.clocks.csv") else ""
        smm = (re.search(r"median (\d+)\) MHz", clk) or [None, "?"])[1]
        vram = (re.search(r"VRAM used max (\d+)", clk) or [None, "?"])[1]
        js = f"{d}/{tag}.json"
        if not os.path.exists(js) or os.path.getsize(js) == 0:
            print(f"| {d} | {tag} | ? | INCOMPLETE | | | | | | | {commit} |"); continue
        j = json.load(open(js))
        acc, ml, same = "", "", ""
        if isinstance(j, list):  # llama-bench
            r = [x for x in j if x["n_gen"]][0]; s = r["samples_ts"]; kind = "llama-bench tg"
        elif "prompts" in j:  # probe workload
            s = [r["decode_tps"] for p in j["prompts"].values() for r in p["runs"]]; kind = "server probe (3×5)"
            a = j["summary"]["acceptance_aggregate"]; acc = f"{a:.3f}" if a else ""
            same = str(all(p["outputs_identical"] for p in j["prompts"].values()))
        else:
            s = j["summary"]["decode_tps"]; kind = "server prose"
            a = j["summary"].get("acceptance_median"); acc = f"{a:.3f}" if a else ""
            same = str(j["summary"]["outputs_identical"])
        log = f"{d}/{tag}.server.log"
        if os.path.exists(log):
            lens = re.findall(r"mean len =\s+([\d.]+)", open(log).read())
            ml = lens[-1] if lens else ""
        med = statistics.median(s)
        print(f"| {os.path.basename(d)} | {tag} | {kind} | {med:.2f} ({min(s):.2f}–{max(s):.2f}) | "
              f"{100*med/ROOF:.1f}% | {acc} | {ml} | {smm} | {vram} | {same} | {commit} |")
