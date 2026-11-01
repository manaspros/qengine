"""Repro (not a pytest file): qengine._w4 GEMV gives different bits for identical inputs.

Layer-7 `down` weight of the cached 27B W4 model, input = the layer's own SwiGLU output; run
the same GEMV R times and count runs whose output differs bitwise from the first. Also runs the
FMA-based BF16 GEMV (qengine._C) on the same data with the dequantized weights as a control.
Usage: scripts/gpu_lock.sh .venv/bin/python tests/repro_w4_nondeterminism.py [R]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch  # noqa: E402

from qengine import _C, _w4  # noqa: E402
from qengine.quant import dequantize_w4  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
torch.manual_seed(0)
L = torch.load(Path.home() / "models/qengine-cache/Qwen3.8-27B-w4-rtn-g128-v1/layer07.pt", map_location="cuda",
               weights_only=False)
gu, dn = L["gate_up"], L["down"]
x = torch.randn(8, 5120, device="cuda").to(torch.bfloat16)
act = torch.empty(8, gu.M, dtype=torch.bfloat16, device="cuda")
_w4.gemv_swiglu(x, gu.qw, gu.scales, act, gu.M, ksplit=1)
torch.cuda.synchronize()
out = torch.empty(8, dn.M, dtype=torch.bfloat16, device="cuda")


def count(f):
    outs = [f().clone() for _ in range(R)]
    return sum(not torch.equal(outs[0].view(torch.int16), o.view(torch.int16)) for o in outs[1:])


for kw in [dict(), dict(variant=1), dict(warps=4), dict(warps=16), dict(unroll=1), dict(unroll=4)]:
    n = count(lambda: (_w4.gemv(act, dn.qw, dn.scales, out, dn.M, ksplit=1, **kw), out)[1])
    print(f"_w4.gemv {kw or 'defaults'} ksplit=1: {n}/{R - 1} runs differ")
Wd = dequantize_w4(dn).to(torch.bfloat16).contiguous()  # bf16-rounded weights: a control, not equal math
n = count(lambda: (_C.gemv_bf16(Wd, act, out), out)[1])
print(f"_C.gemv_bf16 (FMA) same input: {n}/{R - 1} runs differ")
