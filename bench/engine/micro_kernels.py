"""Microbenchmarks of qengine._C decode kernels at 27B shapes, each captured in a CUDA graph of
48 (GDN) / 129 (RMSNorm) calls on separate buffers (state 151 MB > 96 MB L2, like a real step).
Reports us per call, median of 20 graph replays. Usage: gpu_lock.sh python bench/engine/micro_kernels.py"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import torch  # noqa: E402

from qengine import _C  # noqa: E402

dev, BF = "cuda", torch.bfloat16


def time_graph(fn, calls, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(30):  # warm-up (~clocks)
        g.replay()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / calls)
    return statistics.median(ts)


def gdn(q, n=1, L=48, hk=16, hv=48):
    C = 2 * hk * 128 + hv * 128
    w = C + hv * 128 + 2 * hv
    pr = (torch.randn(n, w, device=dev) * 0.5).to(BF)
    layers = []
    for _ in range(L):
        layers.append(dict(ring=torch.zeros(16, C, dtype=BF, device=dev), S=torch.randn(hv, 128, 128, device=dev) * .01,
                           st=[torch.zeros(8, hv, 128, device=dev), torch.zeros(8, hv, 128, device=dev),
                               torch.rand(8, hv, 2, device=dev)]))
    cw = (torch.randn(C, 1, 4, device=dev) * .3).to(BF)
    A, dt, nw = torch.zeros(hv, dtype=BF, device=dev), torch.zeros(hv, dtype=BF, device=dev), torch.ones(128, dtype=BF, device=dev)
    ctl = torch.tensor([100, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32, device=dev)
    out = torch.empty(n, hv * 128, dtype=BF, device=dev)
    c1 = C + hv * 128

    def f():
        for Ld in layers:
            _C.gdn_step_n(pr[:, :C], pr[:, C:c1], pr[:, c1:c1 + hv], pr[:, c1 + hv:], Ld["ring"], cw, A, dt, nw, Ld["S"],
                          *Ld["st"], ctl, out, hk, 1e-6, q=q)
    return time_graph(f, L)


def rms(H=5120, L=129, fused=True):
    xs = [torch.randn(1, H, device=dev).to(BF) for _ in range(L)]
    d = torch.randn(1, H, device=dev).to(BF)
    w = torch.randn(H, device=dev).to(BF)
    out = torch.empty(1, H, dtype=BF, device=dev)

    def f():
        for x in xs:
            _C.rmsnorm1p(x, d if fused else None, w, out, 1e-6)
    return time_graph(f, L)


if __name__ == "__main__":
    which = sys.argv[1:] or ["gdn", "rms"]
    if "gdn" in which:
        gdn(2)  # clock warm-up
        for q in (2, 4, 8, 2, 4, 8):
            print(f"gdn_step_n Q={q}: n=1 {gdn(q):.2f} us/layer, n=4 {gdn(q, n=4):.2f} us/layer")
    if "rms" in which:
        print(f"rmsnorm1p H=5120 fused add: {rms():.2f} us/call; plain {rms(fused=False):.2f} us/call")
