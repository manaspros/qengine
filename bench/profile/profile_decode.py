"""Profile qengine decode steps with Nsight Systems and summarize time per kernel class.

Run (inside the GPU lock):
  nsys profile --cuda-graph-trace=node -c cudaProfilerApi -o build/profile/<name> --force-overwrite true \
      .venv/bin/python bench/profile/profile_decode.py --w4 --steps 64
  .venv/bin/python bench/profile/profile_decode.py --summarize build/profile/<name>.nsys-rep --steps 64

The profiled region is `steps` plain decode graph replays (n = 1) after a 105-token prompt.
"""
import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CLASSES = [  # (class, substrings of the kernel name), first match wins
    ("GEMV (W4)", ["gemv_w4", "qengine_w4"]),
    ("GEMV (BF16)", ["gemv_bf16"]),
    ("GDN step", ["gdn_step"]),
    ("attention", ["attn_"]),
    ("RMSNorm", ["rmsnorm"]),
    ("argmax", ["argmax"]),
    ("swiglu", ["swiglu"]),
    ("embed", ["embed_kernel"]),
    ("bookkeeping", ["accept_kernel", "commit_kernel", "mtp_select"]),
]


def classify(name):
    for c, subs in CLASSES:
        if any(s in name for s in subs):
            return c
    return "other"


def summarize(rep, steps, weight_bytes=None):
    out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export",
                          "true", rep], capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index("Time (%)"):])))
    tot = {}
    per_kernel = []
    for r in rows:
        t_ns, n = float(r["Total Time (ns)"]), int(r["Instances"])
        c = classify(r["Name"])
        tot[c] = tot.get(c, [0.0, 0])
        tot[c][0] += t_ns
        tot[c][1] += n
        per_kernel.append((t_ns, n, r["Name"][:90]))
    all_ns = sum(v[0] for v in tot.values())
    print(f"| kernel class | us / step | % of GPU time | launches / step |")
    print(f"|---|---:|---:|---:|")
    for c, (t, n) in sorted(tot.items(), key=lambda kv: -kv[1][0]):
        print(f"| {c} | {t / steps / 1e3:.1f} | {100 * t / all_ns:.1f}% | {n / steps:.0f} |")
    print(f"| **total GPU busy** | **{all_ns / steps / 1e3:.1f}** | 100% | {sum(v[1] for v in tot.values()) / steps:.0f} |")
    if weight_bytes:
        g = tot.get("GEMV (W4)", [0, 0])[0] / steps
        print(f"\nW4 GEMV: {weight_bytes / 1e9:.3f} GB/step in {g / 1e3:.1f} us -> {weight_bytes / g:.0f} GB/s")
    print("\ntop kernels:")
    for t, n, name in sorted(per_kernel, reverse=True)[:12]:
        print(f"  {t / steps / 1e3:8.1f} us/step  {n / steps:5.1f}x  {name}")


def phases(rep):
    """Per cycle (commit_kernel to commit_kernel): verify = kernels up to accept_kernel, MTP =
    after it; busy time per phase and idle gaps between consecutive kernels."""
    out = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv", "--force-export", "true",
                          rep], capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out[out.index("Start (ns)"):])))
    k = [(float(r["Start (ns)"]), float(r["Duration (ns)"]), r["Name"]) for r in rows if r.get("Duration (ns)")]
    k.sort()
    cycles, cur = [], []
    for e in k:
        cur.append(e)
        if "commit_kernel" in e[2]:
            cycles.append(cur)
            cur = []
    cycles = cycles[1:]  # first may be partial
    import statistics as st
    agg = {"cycle_us": [], "verify_busy": [], "verify_gap": [], "mtp_busy": [], "mtp_gap": [], "between": [],
           "verify_gemv": [], "mtp_gemv": [], "launches_verify": [], "launches_mtp": []}
    prev_end = None
    for c in cycles:
        ia = next(i for i, e in enumerate(c) if "accept_kernel" in e[2])
        def busy_gap(seg):
            b = sum(d for _, d, _ in seg)
            g = sum(max(0.0, seg[i + 1][0] - (seg[i][0] + seg[i][1])) for i in range(len(seg) - 1))
            return b, g
        vb, vg = busy_gap(c[:ia + 1])
        mb, mg = busy_gap(c[ia:])
        agg["verify_busy"].append(vb); agg["verify_gap"].append(vg)
        agg["mtp_busy"].append(mb - c[ia][1]); agg["mtp_gap"].append(mg)
        agg.setdefault("verify_wall", []).append(c[ia][0] + c[ia][1] - c[0][0])
        agg.setdefault("mtp_wall", []).append(c[-1][0] + c[-1][1] - (c[ia][0] + c[ia][1]))
        agg["verify_gemv"].append(sum(d for _, d, n in c[:ia] if "gemv" in n))
        agg["mtp_gemv"].append(sum(d for _, d, n in c[ia:] if "gemv" in n))
        agg["launches_verify"].append(ia + 1); agg["launches_mtp"].append(len(c) - ia - 1)
        if prev_end is not None:
            agg["between"].append(c[0][0] - prev_end)
        agg["cycle_us"].append((c[-1][0] + c[-1][1]) - c[0][0] + (agg["between"][-1] if agg["between"] else 0))
        prev_end = c[-1][0] + c[-1][1]
    for key, v in agg.items():
        if v:
            print(f"{key:16s} median {st.median(v) / (1e3 if 'launches' not in key else 1):9.1f}{' us' if 'launches' not in key else ''}")
    print(f"cycles analysed: {len(cycles)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-27B"))
    ap.add_argument("--w4", action="store_true")
    ap.add_argument("--lm-head", default=None, help="default w4gptq with --w4")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--weight-bytes", type=float, default=None)
    ap.add_argument("--spec", type=int, default=0, help="profile speculative cycles with k drafts (0 = plain steps)")
    ap.add_argument("--draft-vocab", type=int, default=49152)
    ap.add_argument("--l2pin", type=int, default=0, help="pin MTP weights in L2 (Engine.pin_mtp_l2)")
    ap.add_argument("--phases", default=None, help="summarize: per-cycle verify / MTP / gap breakdown of a rep")
    a = ap.parse_args()
    if a.lm_head is None:
        a.lm_head = "w4gptq" if a.w4 else "same"
    if a.phases:
        return phases(a.phases)
    if a.summarize:
        return summarize(a.summarize, a.steps, a.weight_bytes)
    import torch
    from transformers import AutoTokenizer
    from qengine.engine import Engine, W4Linear
    tok = AutoTokenizer.from_pretrained(a.model)
    prompt = tok((ROOT / "bench/prompts/prose.txt").read_text()).input_ids
    eng = Engine(a.model, max_len=len(prompt) + a.steps + 300, linear=W4Linear() if a.w4 else None,
                 lm_head=a.lm_head, cache_dir=str(Path.home() / "models/qengine-cache") if a.w4 else None,
                 draft_vocab=(a.draft_vocab or None) if a.spec else None)
    # bytes of all weights a decode step reads (packed W4 + bf16 lm_head if used)
    nb = 0
    for L in eng.layers:
        for k in ("in_proj", "out", "gate_up", "down"):
            w = L[k]
            nb += w.nbytes if hasattr(w, "nbytes") and not torch.is_tensor(w) else w.numel() * 2
    lm = eng.lm_head
    lm_b = lm.nbytes if not torch.is_tensor(lm) else lm.numel() * 2
    print(f"decoder weight bytes/step {nb / 1e9:.3f} GB, lm_head {lm_b / 1e9:.3f} GB")
    if a.l2pin:
        eng.pin_mtp_l2(True)
    if a.spec:
        eng.generate_spec(prompt, 200, a.spec)  # warm-up, graphs captured
        eng.generate_spec(prompt, 8, a.spec)    # leaves a live sequence to keep decoding
        g = eng.graphs[(a.spec + 1, True, a.spec)]
    else:
        eng.generate(prompt, 200)  # warm-up, graphs captured
        eng.generate(prompt, 2)
        g = eng.graphs[(1, False, 0)]
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(a.steps):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
