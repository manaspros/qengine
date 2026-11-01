"""Gate for entropy-coded W4 (idea #5): decode-only throughput on the full 27B linear stream.

Encodes every linear of the cached 27B W4 model (64 layers x in_proj/out/gate_up/down + GPTQ
lm_head, 13.21 GB packed) with one static Huffman table, checks the decoder bit-exactly (per-unit
hash vs torch on the raw codes), and times one persistent decode kernel over the whole stream
(11.5+ GB > 96 MiB L2, no reuse) against the same kernel structure on the raw packed codes.
Usage: scripts/gpu_lock.sh .venv/bin/python bench/entropy/run_gate.py [--reps 20] [--bps 2]
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "bench/entropy"))
import ec  # noqa: E402

CACHE = Path.home() / "models/qengine-cache/Qwen3.8-27B-w4-rtn-g128-v1"


def tensors():
    for l in range(64):
        L = torch.load(CACHE / f"layer{l:02d}.pt", map_location="cpu", weights_only=False)
        for k in ("in_proj", "out", "gate_up", "down"):
            yield f"{L['type']}.{k}", L[k]
    yield "lm_head", torch.load(CACHE / "lm_head_w4gptq.pt", map_location="cpu", weights_only=False)["packed"]


def ref_hash(qw):
    uw = ec.unit_words(qw).to(torch.int64) & 0xFFFFFFFF                    # [U, 32, 32]
    wt = (torch.arange(32, device=qw.device).view(32, 1) * 32 + torch.arange(32, device=qw.device) + 1)
    out = []
    for i in range(0, uw.shape[0], 16384):
        out.append(((uw[i:i + 16384] * wt).sum((1, 2)) & 0xFFFFFFFF))
    return torch.cat(out)


def timed(fn, reps, warm_s=1.0):
    t0 = time.time()
    while time.time() - t0 < warm_s:
        fn(); torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--bps", type=int, default=2)
    ap.add_argument("--mode", choices=["dec", "raw"], default="dec")
    ap.add_argument("--json")
    ap.add_argument("--version", type=int, default=2)
    a = ap.parse_args()
    ext = load("qe_ec_bench", [str(ROOT / "bench/entropy/decode_bench.cu")],
               extra_cuda_cflags=["-O3", "-arch=sm_120a", "-lineinfo"],
               build_directory=str((ROOT / "build/entropy_ext").mkdir(parents=True, exist_ok=True) or ROOT / "build/entropy_ext"))
    # one global Huffman table from a sample of layers (all tensors share it)
    h = torch.zeros(16, dtype=torch.int64)
    for l in (0, 21, 42, 63):
        L = torch.load(CACHE / f"layer{l:02d}.pt", map_location="cuda", weights_only=False)
        for k in ("in_proj", "out", "gate_up", "down"):
            h += ec.histogram(L[k].qw)
    Ls = ec.huffman_lengths(h); codes = ec.canonical_lsb(Ls)
    lut = ec.build_lut(Ls, codes).cuda()
    print("Huffman lengths", Ls)
    res = {"lengths": Ls, "per_type": {}}
    if a.mode == "raw":
        parts, hashes = [], []
        for name, p in tensors():
            qw = p.qw.cuda(); parts.append(qw.view(-1)); hashes.append(ref_hash(qw))
        qw_all = torch.cat(parts); del parts
        ref = torch.cat(hashes); U = ref.numel()
        out = torch.zeros(U, dtype=torch.int32, device="cuda")
        ext.raw(qw_all, out, a.bps)
        ok = torch.equal(out.to(torch.int64) & 0xFFFFFFFF, ref)
        nbytes = qw_all.numel() * 4
        ms = timed(lambda: ext.raw(qw_all, out, a.bps), a.reps)
    else:
        streams, offs, hashes, base = [], [], [], 0
        stats = {}
        for name, p in tensors():
            qw = p.qw.cuda()
            st, off = ec.encode(qw, Ls, codes)
            hashes.append(ref_hash(qw))
            assert int(off[-1]) % 4 == 0
            offs.append((off[:-1] // 4 + base).to(torch.int64))
            base += int(off[-1]) // 4
            streams.append(st)
            typ = name.split(".")[-1] if name != "lm_head" else "lm_head"
            s = stats.setdefault(typ, [0, 0])
            s[0] += st.numel() * 4; s[1] += qw.numel() * 4
            del qw
        offs.append(torch.tensor([base], dtype=torch.int64, device="cuda"))
        off16 = torch.cat(offs); assert int(off16[-1]) < 2**32
        off16 = off16.to(torch.int64)
        sizes = (off16[1:] - off16[:-1])
        maxr = int((sizes.max() + 31) // 32)
        off16 = torch.where(off16 >= 2**31, off16 - 2**32, off16).to(torch.int32)
        stream = torch.cat(streams); del streams
        ref = torch.cat(hashes); U = ref.numel()
        print(f"units {U}, max unit {int(sizes.max())*16} B (maxr {maxr}), mean {float(sizes.float().mean())*16:.0f} B")
        for typ, (c, r) in stats.items():
            res["per_type"][typ] = {"compressed_B": c, "raw_B": r, "ratio": c / r, "bits_per_weight": 8 * c / (2 * r)}
            print(f"  {typ:8s} {c/1e9:7.3f} GB / {r/1e9:7.3f} GB raw codes = {c/r:.4f} ({8*c/(2*r):.3f} bit/w)")
        out = torch.zeros(U, dtype=torch.int32, device="cuda")
        ext.decode(stream, off16, lut, out, maxr, a.bps, a.version)
        ok = torch.equal(out.to(torch.int64) & 0xFFFFFFFF, ref)
        nbytes = stream.numel() * 4 + off16.numel() * 4
        ms = timed(lambda: ext.decode(stream, off16, lut, out, maxr, a.bps, a.version), a.reps)
    weights = U * 32 * 256
    med = statistics.median(ms)
    print(f"mode {a.mode}: exact {ok}; {nbytes/1e9:.3f} GB read, {weights/1e9:.2f} G weights; "
          f"{med:.3f} ms (min {min(ms):.3f} max {max(ms):.3f}) -> {nbytes/(med*1e-3)/1e9:.1f} GB/s read, "
          f"{weights/(med*1e-3)/1e12:.3f} T weights/s")
    res.update({"version": a.version, "mode": a.mode, "exact": ok, "bytes": nbytes, "weights": weights, "ms_median": med,
                "ms_min": min(ms), "ms_max": max(ms), "GBps": nbytes / (med * 1e-3) / 1e9,
                "Tweights_s": weights / (med * 1e-3) / 1e12, "bps": a.bps})
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
