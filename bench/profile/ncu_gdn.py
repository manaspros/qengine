"""Driver for ncu on the GDN decode kernel at 27B shapes (48 layers, separate states so the state
streams from DRAM like a real step). Usage:
  ncu --kernel-name regex:gdn --launch-skip 96 --launch-count 16 --metrics ... python bench/profile/ncu_gdn.py N Q
(two warm-up sweeps of 48 launches are skipped)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
import torch  # noqa: E402

import micro_kernels as mk  # noqa: E402
from qengine import _C  # noqa: E402

n, q = int(sys.argv[1]), int(sys.argv[2])
if len(sys.argv) > 3:
    _C.set_gdn_early(bool(int(sys.argv[3])))
if len(sys.argv) > 4:
    _C.set_gdn_phases(bool(int(sys.argv[4])))
hk, hv, BF, dev = 16, 48, torch.bfloat16, "cuda"
C = 2 * hk * 128 + hv * 128
w = C + hv * 128 + 2 * hv
pr = (torch.randn(n, w, device=dev) * 0.5).to(BF)
L = [dict(ring=torch.zeros(16, C, dtype=BF, device=dev), S=torch.randn(hv, 128, 128, device=dev) * .01,
          st=[torch.zeros(8, hv, 128, device=dev), torch.zeros(8, hv, 128, device=dev), torch.rand(8, hv, 2, device=dev)])
     for _ in range(48)]
cw = (torch.randn(C, 1, 4, device=dev) * .3).to(BF)
A, dt, nw = torch.zeros(hv, dtype=BF, device=dev), torch.zeros(hv, dtype=BF, device=dev), torch.ones(128, dtype=BF, device=dev)
ctl = torch.tensor([100, 1, 1, 0, 0, 0, 0, 0] + [0] * 24, dtype=torch.int32, device=dev)
out = torch.empty(n, hv * 128, dtype=BF, device=dev)
c1 = C + hv * 128
for _ in range(4):
    for Ld in L:
        mk_args = (pr[:, :C], pr[:, C:c1], pr[:, c1:c1 + hv], pr[:, c1 + hv:], Ld["ring"], cw, A, dt, nw, Ld["S"], *Ld["st"], ctl, out, hk, 1e-6)
        _C.gdn_step_n(*mk_args, q=q)
torch.cuda.synchronize()
