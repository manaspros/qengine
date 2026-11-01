"""W4A16 GEMV benchmark (qengine._w4) on real Qwen3.8-27B shapes.

Timing method
- Each measurement replays a CUDA graph of L back-to-back launches (the engine runs decode as a
  CUDA graph, so this is the realistic launch path), timed with cudaEvents around each replay;
  time/launch = elapsed / L. Median over many replays; >= 1 s warm-up of the same graph first.
- L2 (96 MiB) cannot hold the weights between launches: consecutive launches use different
  weight copies, copies = max(2, ceil(384 MiB / size)), so a copy's reuse distance is >= 288 MiB.
- Bytes counted = packed weights + fp16 scales + x (N*K*2) + out (N*M*2).
- "sequence" mode: all 64 layers' distinct linear weights at real shapes (13 GB, nothing reused)
  + lm_head, in token order, one graph = one token. Reports the linear-only time per token and the
  implied tok/s ceiling.

Usage: python bench_gemv.py shapes [--n 1,2,4,8] [--cfg warps=8,unroll=2,...] [--json out]
       python bench_gemv.py sequence [--n 1] [--cfg ...]
Run through scripts/gpu_lock.sh after wait_idle.sh (bench/gemv/run.sh does this, n=5).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from qengine import _w4  # noqa: E402

PEAK = 1692.0  # GB/s, measured DRAM read bandwidth
H, I, V = 5120, 17408, 248320

# name: (M, K, kind)  kind: "plain" or "swiglu" (M = I, packed rows = 2*I)
SHAPES = {
    "gdn.in_proj_qkv": (10240, H, "plain"),
    "gdn.in_proj_z": (6144, H, "plain"),
    "gdn.in_proj_a": (48, H, "plain"),
    "gdn.qkvzab(fused)": (10240 + 6144 + 96, H, "plain"),
    "gdn.out_proj": (H, 6144, "plain"),
    "attn.q_proj": (12288, H, "plain"),
    "attn.k_proj": (1024, H, "plain"),
    "attn.qkv(fused)": (12288 + 2048, H, "plain"),
    "attn.o_proj": (H, 6144, "plain"),
    "mlp.gate_proj": (I, H, "plain"),
    "mlp.gate_up(swiglu)": (I, H, "swiglu"),
    "mlp.down_proj": (H, I, "plain"),
    "lm_head": (V, H, "plain"),
}


def ceil(a, b):
    return (a + b - 1) // b * b


def make_packed(rows: int, K: int, dev="cuda"):
    Mp, Kp = ceil(rows, 16), ceil(K, 128)
    qw = torch.randint(-2**31, 2**31 - 1, (Mp // 16, Kp // 64, 32, 4), dtype=torch.int32, device=dev)
    sc = (torch.rand(Mp // 16, Kp // 128, 16, device=dev) * 1e-2 + 1e-3).to(torch.float16)
    return qw, sc


def packed_bytes(qw, sc):
    return qw.numel() * 4 + sc.numel() * 2


def call(kind, x, qw, sc, out, M, cfg):
    if kind == "swiglu":
        _w4.gemv_swiglu(x, qw, sc, out, M, **cfg)
    else:
        _w4.gemv(x, qw, sc, out, M, **cfg)


def time_graph(fn_capture, launches: int, warm_s=1.0, reps=200):
    """Capture fn_capture() into a graph; warm up >= warm_s; return list of ms per launch."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn_capture()  # eager warm-up (allocates the split-K workspace before capture)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn_capture()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_s:
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) / launches for a, b in ev]


def bench_shape(name, N, cfg, budget=384 << 20):
    M, K, kind = SHAPES[name]
    rows = 2 * ceil(M, 8) if kind == "swiglu" else M
    qw0, sc0 = make_packed(rows, K)
    wbytes = packed_bytes(qw0, sc0)
    copies = max(2, math.ceil(budget / wbytes))
    ws = [(qw0, sc0)] + [make_packed(rows, K) for _ in range(copies - 1)]
    x = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(N, M, device="cuda", dtype=torch.bfloat16)
    # enough launches per graph for >= ~2 ms per replay (event resolution, launch noise)
    est_us = max(wbytes / (PEAK * 1e3), 2.0)
    launches = max(copies, ceil(int(2000 / est_us) + 1, copies))

    def body():
        for i in range(launches):
            qw, sc = ws[i % copies]
            call(kind, x, qw, sc, out, M, cfg)

    ms = time_graph(body, launches)
    nbytes = wbytes + N * K * 2 + N * M * 2
    med = statistics.median(ms)
    del ws
    torch.cuda.empty_cache()
    return {
        "shape": name, "M": M, "K": K, "kind": kind, "N": N, "cfg": cfg, "copies": copies,
        "launches_per_graph": launches, "bytes": nbytes, "us_median": med * 1e3,
        "us_min": min(ms) * 1e3, "us_max": max(ms) * 1e3,
        "GBps": nbytes / (med * 1e-3) / 1e9, "pct_peak": 100 * nbytes / (med * 1e-3) / 1e9 / PEAK,
    }


def bench_sequence(N, cfg, fused=True, warm_s=1.0, reps=30):
    """All 64 layers' linear layers + lm_head, real shapes, distinct weights, token order."""
    layers = []
    total = 0
    for l in range(64):
        if l % 4 == 3:
            mix = [("attn.qkv(fused)",)] if fused else [("attn.q_proj",), ("attn.k_proj",), ("attn.k_proj",)]
            mix += [("attn.o_proj",)]
        else:
            mix = [("gdn.qkvzab(fused)",)] if fused else [("gdn.in_proj_qkv",), ("gdn.in_proj_z",),
                                                          ("gdn.in_proj_a",), ("gdn.in_proj_a",)]
            mix += [("gdn.out_proj",)]
        mlp = [("mlp.gate_up(swiglu)",)] if fused else [("mlp.gate_proj",), ("mlp.gate_proj",)]
        mlp += [("mlp.down_proj",)]
        for (name,) in mix + mlp:
            M, K, kind = SHAPES[name]
            rows = 2 * ceil(M, 8) if kind == "swiglu" else M
            qw, sc = make_packed(rows, K)
            total += packed_bytes(qw, sc) + N * K * 2 + N * M * 2
            layers.append((name, kind, M, K, qw, sc))
    M, K, kind = SHAPES["lm_head"]
    qw, sc = make_packed(M, K)
    total += packed_bytes(qw, sc) + N * K * 2 + N * M * 2
    layers.append(("lm_head", kind, M, K, qw, sc))
    xs = {k: torch.randn(N, k, device="cuda", dtype=torch.bfloat16) for k in (H, 6144, I)}
    outs = {}
    for name, kind, M, K, _, _ in layers:
        outs.setdefault(M, torch.empty(N, M, device="cuda", dtype=torch.bfloat16))

    def body():
        for name, kind, M, K, qw, sc in layers:
            call(kind, xs[K], qw, sc, outs[M], M, cfg)

    ms = time_graph(body, 1, warm_s=warm_s, reps=reps)
    med = statistics.median(ms)
    res = {
        "mode": "sequence", "fused": fused, "N": N, "cfg": cfg, "kernels": len(layers),
        "bytes": total, "ms_median": med, "ms_min": min(ms), "ms_max": max(ms),
        "GBps": total / (med * 1e-3) / 1e9, "pct_peak": 100 * total / (med * 1e-3) / 1e9 / PEAK,
        "tok_s_linear_only": 1e3 / med,
    }
    del layers
    torch.cuda.empty_cache()
    return res


def parse_cfg(s: str) -> dict:
    cfg = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=")
        cfg[k] = (v.lower() in ("1", "true")) if k == "pdl" else int(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["shapes", "sequence"])
    ap.add_argument("--n", default="1")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--cfg", default="", help="e.g. warps=8,unroll=2,ksplit=0,pdl=1,variant=0")
    ap.add_argument("--unfused", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = []
    for cfg_s in a.cfg.split(";"):
      cfg = parse_cfg(cfg_s)
      if ";" in a.cfg:
        print(f"== {cfg_s}", flush=True)
      for N in [int(v) for v in a.n.split(",")]:
        if a.mode == "shapes":
            for name in a.shapes.split(","):
                r = bench_shape(name, N, cfg)
                print(f"N={N} {name:22s} {r['us_median']:9.2f} us  {r['GBps']:7.1f} GB/s  "
                      f"{r['pct_peak']:5.1f}%  (min {r['us_min']:.2f} max {r['us_max']:.2f})", flush=True)
                res.append(r)
        else:
            r = bench_sequence(N, cfg, fused=not a.unfused)
            print(f"N={N} sequence fused={not a.unfused} {r['kernels']} kernels {r['ms_median']:.3f} ms "
                  f"{r['GBps']:.1f} GB/s {r['pct_peak']:.1f}% -> {r['tok_s_linear_only']:.1f} tok/s "
                  f"(min {r['ms_min']:.3f} max {r['ms_max']:.3f})", flush=True)
            res.append(r)
    if a.json:
        with open(a.json, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
