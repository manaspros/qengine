"""W4 (int4, symmetric, group 128 along K, fp16 scale) packing for qengine._w4.

Layout (kernel side: csrc/gemv_w4.cu)
------------------------------------------------------------------------
W[M, K] is padded to Mp = ceil16(M), Kp = ceil128(K) with zeros. Per (row m, group j = k // 128):

    scale[m, j] = fp16(2 * max|W[m, 128j:128j+128]| / 15)         (GPTQ "sym" convention)
    u[m, k]     = clamp(round(W[m, k] / scale) + 8, 0, 15)          (stored nibble)
    W~[m, k]    = (u - 8) * scale                                   (what the kernel multiplies)

Packed tensors:
    qw     int32 [Mp/16, Kp/64, 32, 4]   tile (16 rows) x chunk (64 k) x lane x mma k-step
    scales fp16  [Mp/16, Kp/128, 16]     tile x group x row-in-tile

Lane l = 4g + t holds rows g and g+8 of its tile and k = 64*chunk + 16t + 0..15; word s holds
k0 = 64*chunk + 16t + 4s + {0..3} with nibble j (bits 4j..4j+3) =
    row g + 8*(j & 1),   k0 + 2*((j >> 1) & 1) + (j >> 2)
i.e. nibbles (0,4) = row g k0+{0,1}; (1,5) = row g+8 k0+{0,1}; (2,6) = row g k0+{2,3};
(3,7) = row g+8 k0+{2,3}: each pair (j, j+4) is one bf16x2 register of the mma.m16n8k16 A
fragment after `(word >> 4j) & 0x000F000F | 0x43004300` (= 128 + u) and `- 136`.

GPTQ can fill the same layout: it only has to produce u (and scale) per row/group; packing is
independent of how u was chosen. RTN is implemented here.

Gate/up fusion (SwiGLU epilogue): quantize_w4_gate_up interleaves gate and up in blocks of 8
rows (tile T rows 0..7 = gate[8T..8T+7], rows 8..15 = up[8T..8T+7]) and packs that matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP = 128


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b * b


@dataclass
class W4Packed:
    qw: torch.Tensor      # int32 [Mp/16, Kp/64, 32, 4]
    scales: torch.Tensor  # fp16  [Mp/16, Kp/128, 16]
    M: int                # logical rows (for gate_up: intermediate size I)
    K: int
    gate_up: bool = False

    @property
    def nbytes(self) -> int:
        return self.qw.numel() * 4 + self.scales.numel() * 2


def _index_map(device) -> tuple[torch.Tensor, torch.Tensor]:
    """row_in_tile[32,4,8] and k_in_chunk[32,4,8] for (lane, word, nibble)."""
    lane = torch.arange(32, device=device).view(32, 1, 1)
    s = torch.arange(4, device=device).view(1, 4, 1)
    j = torch.arange(8, device=device).view(1, 1, 8)
    g, t = lane // 4, lane % 4
    row = g + 8 * (j & 1)
    k = 16 * t + 4 * s + 2 * ((j >> 1) & 1) + (j >> 2)
    return row.expand(32, 4, 8), k.expand(32, 4, 8)


def _scales_and_codes(Wp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Wp fp32 [rows, Kp] -> (scale fp16 [rows, Kp/128], u uint8 [rows, Kp])."""
    rows, Kp = Wp.shape
    Wg = Wp.view(rows, Kp // GROUP, GROUP)
    absmax = Wg.abs().amax(dim=-1)
    s16 = (absmax * (2.0 / 15.0)).to(torch.float16)
    s = s16.float()
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    u = torch.clamp(torch.round(Wg / s_safe.unsqueeze(-1)) + 8, 0, 15).to(torch.uint8)
    u = torch.where((s > 0).unsqueeze(-1), u, torch.full_like(u, 8))
    return s16, u.view(rows, Kp)


@torch.no_grad()
def _pack(W: torch.Tensor, M: int, K: int, gate_up: bool, rows_per_step: int = 4096) -> W4Packed:
    Mp = _ceil(W.shape[0], 16)
    Kp = _ceil(K, GROUP)
    dev = W.device
    tiles, chunks, groups = Mp // 16, Kp // 64, Kp // GROUP
    qw = torch.empty(tiles, chunks, 32, 4, dtype=torch.int32, device=dev)
    scales = torch.empty(tiles, groups, 16, dtype=torch.float16, device=dev)
    row_map, k_map = _index_map(dev)
    shifts = (4 * torch.arange(8, device=dev, dtype=torch.int64)).view(1, 1, 1, 1, 8)
    for r0 in range(0, Mp, rows_per_step):          # bounded memory (lm_head is 248320 rows)
        r1 = min(r0 + rows_per_step, Mp)
        Wp = torch.zeros(r1 - r0, Kp, dtype=torch.float32, device=dev)
        if r0 < W.shape[0]:
            src = W[r0:min(r1, W.shape[0]), :K].float()
            Wp[: src.shape[0], :K] = src
        s16, u = _scales_and_codes(Wp)
        nt = (r1 - r0) // 16
        u5 = u.view(nt, 16, chunks, 64)
        # gather -> [nt, chunks, 32, 4, 8] nibbles
        nib = u5[:, row_map, :, k_map]                 # [32,4,8, nt, chunks] (advanced idx first)
        nib = nib.permute(3, 4, 0, 1, 2).to(torch.int64)
        word = (nib << shifts).sum(dim=-1)            # [nt, chunks, 32, 4] in [0, 2^32)
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        qw[r0 // 16 : r1 // 16] = word
        scales[r0 // 16 : r1 // 16] = s16.view(nt, 16, groups).permute(0, 2, 1)
    return W4Packed(qw.contiguous(), scales.contiguous(), M, K, gate_up)


def quantize_w4(W: torch.Tensor) -> W4Packed:
    """RTN symmetric int4, group 128 along K, fp16 scales. W: [M, K] (any float dtype)."""
    assert W.dim() == 2
    M, K = W.shape
    return _pack(W, M, K, gate_up=False)


def quantize_w4_gate_up(Wg: torch.Tensor, Wu: torch.Tensor) -> W4Packed:
    """Pack gate [I, K] and up [I, K] interleaved for gemv_swiglu (out = silu(x Wg^T) * x Wu^T)."""
    assert Wg.shape == Wu.shape and Wg.dim() == 2
    I, K = Wg.shape
    Ip = _ceil(I, 8)
    Wi = torch.zeros(Ip // 8, 2, 8, K, dtype=Wg.dtype, device=Wg.device)
    gpad = torch.zeros(Ip, K, dtype=Wg.dtype, device=Wg.device)
    upad = torch.zeros(Ip, K, dtype=Wg.dtype, device=Wg.device)
    gpad[:I], upad[:I] = Wg, Wu
    Wi[:, 0] = gpad.view(Ip // 8, 8, K)
    Wi[:, 1] = upad.view(Ip // 8, 8, K)
    return _pack(Wi.view(2 * Ip, K), I, K, gate_up=True)


@torch.no_grad()
def dequantize_w4(p: W4Packed) -> torch.Tensor:
    """Inverse of the packing: fp32 [M, K] (gate_up: returns (gate, up) each [I, K])."""
    tiles, chunks = p.qw.shape[0], p.qw.shape[1]
    Mp, Kp = tiles * 16, chunks * 64
    dev = p.qw.device
    row_map, k_map = _index_map(dev)
    w = p.qw.to(torch.int64) & 0xFFFFFFFF                               # [T, C, 32, 4]
    shifts = (4 * torch.arange(8, device=dev, dtype=torch.int64))
    nib = ((w.unsqueeze(-1) >> shifts) & 0xF).float() - 8.0             # [T, C, 32, 4, 8]
    out = torch.empty(tiles, 16, chunks, 64, dtype=torch.float32, device=dev)
    out[:, row_map, :, k_map] = nib.permute(2, 3, 4, 0, 1)
    out = out.view(Mp, Kp)
    s = p.scales.float().permute(0, 2, 1).reshape(Mp, Kp // GROUP)      # [Mp, G]
    out = (out.view(Mp, Kp // GROUP, GROUP) * s.unsqueeze(-1)).view(Mp, Kp)
    if p.gate_up:
        o = out.view(Mp // 16, 2, 8, Kp)
        g = o[:, 0].reshape(Mp // 2, Kp)[: p.M, : p.K]
        u = o[:, 1].reshape(Mp // 2, Kp)[: p.M, : p.K]
        return g, u
    return out[: p.M, : p.K]


def gemv(x: torch.Tensor, p: W4Packed, out: torch.Tensor | None = None, **kw) -> torch.Tensor:
    """Convenience wrapper: out[N, M] = x[N, K] @ W~^T (or silu(gate)*up for gate_up packs)."""
    from qengine import _w4

    if out is None:
        out = torch.empty(x.shape[0], p.M, dtype=torch.bfloat16, device=x.device)
    if p.gate_up:
        _w4.gemv_swiglu(x, p.qw, p.scales, out, p.M, **kw)
    else:
        _w4.gemv(x, p.qw, p.scales, out, p.M, **kw)
    return out
