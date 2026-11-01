"""Collect head-to-head results from build/h2h/{r1,r2,code}/ into one table (stdout, markdown)."""
import csv, json, statistics as st, sys
from pathlib import Path

H = Path(__file__).resolve().parents[2] / "build" / "h2h"


def smi(path):
    sm, pw, mem, cap, n = [], [], [], 0, 0
    if not path.exists():
        return {}
    for row in csv.reader(open(path)):
        if len(row) < 9:
            continue
        try:
            mem.append(float(row[6].split()[0]))
            util = float(row[5].split()[0])
        except ValueError:
            continue
        if util < 50:
            continue
        n += 1
        sm.append(float(row[1].split()[0])); pw.append(float(row[3].split()[0]))
        cap += "Active" in row[7] and "Not" not in row[7]
    if not sm:
        return {"vram": max(mem) if mem else None}
    return {"sm_med": st.median(sm), "sm_min": min(sm), "pw_max": max(pw), "pw_med": st.median(pw),
            "cap_frac": cap / n, "n_busy": n, "vram": max(mem), "ge570": sum(p >= 570 for p in pw) / n}


def fmt(xs):
    return f"{st.median(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})"


def rows(rd):
    d = H / rd
    out = []
    for f in sorted(d.glob("*.h2h_meta.txt")):
        tag = f.name.split(".")[0]
        meta = f.read_text()
        s = smi(d / f"{tag}.h2h_smi.csv")
        ok = "rc=0" in meta
        if tag.startswith("qe_"):
            j = d / f"{tag}.json"
            if not j.exists():
                out.append((tag, "-", "FAILED", "", "", s)); continue
            r = json.load(open(j))
            for k, v in r["results"].items():
                tps = [x["decode_tok_s"] for x in v["runs"]]
                ident = v.get("outputs_identical", True)  # bench_decode.py asserts identity + spec==plain
                eq = v.get("equals_plain", True if k != "0" else None)
                acc = ""
                if k != "0":
                    acc = f"{st.median(x['accept_rate'] for x in v['runs']):.3f} / {st.median(x['tokens_per_cycle'] for x in v['runs']):.2f}"
                note = f"pt {r.get('prompt_tokens')}; ident {ident}; ==plain {eq}" + (
                    f" (first diff tok {v.get('first_diff_vs_plain')})" if eq is False else "")
                out.append((f"{tag} k={k}", fmt(tps), acc, note, "", s))
            continue
        j = d / f"{tag}.json"
        if not j.exists():
            out.append((tag, "-", "FAILED rc!=0" if not ok else "no json", "", "", s)); continue
        r = json.load(open(j))
        if isinstance(r, list):  # llama-bench
            for t in r:
                out.append((tag, fmt(t["samples_ts"]), "", f"tg{t['n_gen']}", "", s))
            continue
        sm = r["summary"]
        tps = sm["decode_tps"]
        acc = ""
        if "acceptance" in sm:
            dn, da = sm["draft_n"], sm["draft_n_accepted"]
            acc = f"{st.median(sm['acceptance']):.3f}"
        elif "acceptance_rate" in sm:
            acc = f"{st.median(sm['acceptance_rate']):.3f}"
        elif "spec_counters" in sm:
            c = sm["spec_counters"]
            a = c["vllm:spec_decode_num_accepted_tokens_total"]; dt = c["vllm:spec_decode_num_draft_tokens_total"]
            dr = c["vllm:spec_decode_num_drafts_total"]
            acc = f"{st.median(x / y for x, y in zip(a, dt)):.3f} / {st.median(1 + x / y for x, y in zip(a, dr)):.2f}"
        note = f"pt {sm.get('prompt_tokens')}; ident {sm.get('outputs_identical')}"
        out.append((tag, fmt(tps), acc, note, "", s))
    return out


for rd in sys.argv[1:] or ["r1", "r2", "code"]:
    print(f"\n## {rd}\n| tag | tok/s median (min–max) | accept / tok-per-step | notes | | SM med MHz | P max W | ≥570 W frac | pwr-cap frac | VRAM peak MiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, t, a, n, _, s in rows(rd):
        print(f"| {tag} | {t} | {a} | {n} | | {s.get('sm_med', 0):.0f} | {s.get('pw_max', 0):.0f} | "
              f"{s.get('ge570', 0):.2f} | {s.get('cap_frac', 0):.2f} | {s.get('vram') or 0:.0f} |")
