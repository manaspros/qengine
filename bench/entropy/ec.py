"""Lossless entropy coding of qengine W4 nibble codes (research round 2, idea #5).

Format (one independent "unit" per (16-row tile, GPU = 4 consecutive 128-k groups) = 4 KB of
today's packed qw; v1 used 1-group units and wasted ~0.5 bit/weight at stream ends):
- Lane l of a warp owns, per unit, the 8*GPU packed words it holds in the W4 layout (python/qengine/quant.py) (chunk h
  x mma step s=0..3), i.e. 64*GPU nibbles; symbol order: word i = 4h+s, nibble j = 0..7.
  Decoding reproduces those words exactly, so a fused GEMV keeps today's dequant/mma and output.
- Static Huffman code (max length 7) over u = 0..15, read LSB-first.
- Lockstep refill schedule (data independent): before symbols 4c..4c+3 every lane with
  < THR = 28 buffered bits (28 = 4 x max length) takes ONE 32-bit word; the words are stored interleaved in the order
  (c, lane), so lane l's word is at  base + popc(ballot(need) & lanemask_lt)  (Giesen 2014,
  "interleaved entropy coders"). A lane's k-th word is bits [32k, 32k+32) of its own bitstream
  (zero-padded). Each unit's word stream is padded to a multiple of 4 words (16 B).
- offsets[u] (uint32, in words) locate unit streams; offsets[U] = total.
"""
from __future__ import annotations

import heapq

import torch

MAXLEN = 7
WIN = 12   # decoder LUT window (bits)
GPU = 4    # 128-k groups per unit
THR = 28   # refill threshold (bits), = 4 symbols x MAXLEN
NCHK = 16 * GPU  # refill check points per unit (one per 4 symbols)


def unit_words(qw: torch.Tensor) -> torch.Tensor:
    """qw int32 [T, C, 32, 4] -> [U, 32, 8*GPU] (unit = (tile, group quad), lane, word i)."""
    T, C = qw.shape[0], qw.shape[1]
    h = 2 * GPU
    assert C % h == 0, "Kp must be a multiple of 128*GPU"
    return qw.view(T, C // h, h, 32, 4).permute(0, 1, 3, 2, 4).reshape(T * (C // h), 32, 4 * h)


def nibbles(words: torch.Tensor) -> torch.Tensor:
    """[..., n] int32 words -> [..., 8n] uint8 nibbles in symbol order (word-major, nibble-minor)."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    sh = 4 * torch.arange(8, device=w.device)
    return ((w.unsqueeze(-1) >> sh) & 0xF).to(torch.uint8).flatten(-2)


def histogram(qw: torch.Tensor) -> torch.Tensor:
    return torch.bincount(nibbles(unit_words(qw)).flatten().to(torch.int64), minlength=16).cpu()


def huffman_lengths(counts, maxlen=MAXLEN):
    c = [max(int(x), 1) for x in counts]
    heap = [(c[i], i, [i]) for i in range(16)]
    heapq.heapify(heap)
    L = [0] * 16
    uid = 16
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for s in sa + sb:
            L[s] += 1
        heapq.heappush(heap, (a + b, uid, sa + sb))
        uid += 1
    assert max(L) <= maxlen, f"Huffman length {max(L)} > {maxlen}; would need length limiting"
    assert abs(sum(2.0 ** -l for l in L) - 1) < 1e-12
    return L


def canonical_lsb(L):
    """Canonical Huffman codes, bit-reversed for LSB-first reading. Returns codes[16]."""
    order = sorted(range(16), key=lambda s: (L[s], s))
    code, prev, codes = 0, L[order[0]], [0] * 16
    for i, s in enumerate(order):
        if i:
            code = (code + 1) << (L[s] - prev)
        prev = L[s]
        codes[s] = int(format(code, f"0{L[s]}b")[::-1], 2)
    return codes


def build_lut(L, codes, win=WIN):
    """uint16 [2^win]: s1 | s2<<4 | len1<<8 | lentot<<11 | pair<<15 (pair = 2nd symbol fits)."""
    def first(w, nb):
        for s in range(16):
            if L[s] <= nb and (w & ((1 << L[s]) - 1)) == codes[s]:
                return s
        return None
    lut = []
    for w in range(1 << win):
        s1 = first(w, win)
        assert s1 is not None
        s2 = first(w >> L[s1], win - L[s1])
        e = s1 | (L[s1] << 8)
        if s2 is not None:
            e |= (s2 << 4) | ((L[s1] + L[s2]) << 11) | (1 << 15)
        lut.append(e)
    return torch.tensor(lut, dtype=torch.int32).to(torch.int16)


@torch.no_grad()
def encode(qw: torch.Tensor, L, codes, batch=8192):
    """qw [T, C, 32, 4] int32 (CUDA) -> (stream int32 [W], offsets int64 [U+1] in words)."""
    dev = qw.device
    Lt = torch.tensor(L, dtype=torch.int64, device=dev)
    Ct = torch.tensor(codes, dtype=torch.int64, device=dev)
    uw = unit_words(qw)
    U = uw.shape[0]
    streams, sizes = [], []
    for b0 in range(0, U, batch):
        sym = nibbles(uw[b0:b0 + batch]).to(torch.int64)          # [B, 32, 64*GPU]
        B = sym.shape[0]
        ln, cd = Lt[sym], Ct[sym]
        pos = torch.cumsum(ln, -1) - ln                            # bit position in lane stream
        W = torch.zeros(B, 32, NCHK + 2, dtype=torch.int64, device=dev)
        val = cd << (pos & 31)
        wi = pos >> 5
        W.scatter_add_(2, wi, val & 0xFFFFFFFF)
        W.scatter_add_(2, wi + 1, val >> 32)                       # disjoint bits: add == or
        hb = ln.view(B, 32, NCHK, 4).sum(-1)                       # bits per 4 symbols
        bits = torch.zeros(B, 32, dtype=torch.int64, device=dev)
        need = torch.empty(B, NCHK, 32, dtype=torch.bool, device=dev)
        for c in range(NCHK):
            nd = bits < THR
            need[:, c] = nd
            bits = bits + 32 * nd - hb[:, :, c]
        k = torch.cumsum(need.to(torch.int64), 1) - need.to(torch.int64)   # lane's word index
        vals = torch.gather(W, 2, k.permute(0, 2, 1)).permute(0, 2, 1)     # [B, NCHK, 32]
        nf = need.reshape(B, NCHK * 32)
        n = nf.sum(1)
        npad = (n + 3) // 4 * 4
        off = torch.cumsum(npad, 0) - npad
        rank = torch.cumsum(nf.to(torch.int64), 1) - nf.to(torch.int64)
        dest = (off.unsqueeze(1) + rank)[nf]
        out = torch.zeros(int(npad.sum()), dtype=torch.int64, device=dev)
        out[dest] = vals.reshape(B, NCHK * 32)[nf]
        streams.append(torch.where(out >= 2**31, out - 2**32, out).to(torch.int32))
        sizes.append(npad)
    sizes = torch.cat(sizes)
    offsets = torch.zeros(U + 1, dtype=torch.int64, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0)
    return torch.cat(streams), offsets


def decode_ref(stream, offsets, u, L, codes, lut):
    """Pure-Python decoder of one unit (validates the format). Returns [32][8*GPU] words."""
    st = [int(x) & 0xFFFFFFFF for x in stream[int(offsets[u]):int(offsets[u + 1])].tolist()]
    lutl = [int(x) & 0xFFFF for x in lut.tolist()]
    buf, bits, words = [0] * 32, [0] * 32, [[0] * (8 * GPU) for _ in range(32)]
    nxt = 0
    for c in range(NCHK):
        need = [bits[l] < THR for l in range(32)]
        r = 0
        for l in range(32):
            if need[l]:
                buf[l] |= st[nxt + r] << bits[l]
                bits[l] += 32
                r += 1
        nxt += r
        for l in range(32):
            prod = 0
            while prod < 4:
                e = lutl[buf[l] & ((1 << WIN) - 1)]
                if (e >> 15) and prod <= 2:
                    syms, used, cnt = e & 0xFF, (e >> 11) & 0xF, 2
                else:
                    syms, used, cnt = e & 0xF, (e >> 8) & 0x7, 1
                words[l][c // 2] |= syms << (4 * (4 * (c % 2) + prod))
                buf[l] >>= used
                bits[l] -= used
                prod += cnt
    return words
