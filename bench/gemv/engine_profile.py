"""Per-shape W4 GEMV efficiency inside the real 27B engine (plain decode N=1, MTP k=4 verify N=5).

run:     nsys profile --cuda-graph-trace node --trace cuda,nvtx -o OUT python engine_profile.py run
analyze: python engine_profile.py analyze OUT.sqlite
Each GEMV is charged its exposed time = end(GEMV) - end(previous kernel) (what the engine pays,
PDL overlap included). GEMVs are identified by position after each embed launch: a main pass
has 257 (64 layers x in_proj/out/gate_up/down, then lm_head), an MTP draft pass 6
(fc, in_proj, out, gate_up, down, draft head). Bytes = packed int4 + fp16 scales (ncu: DRAM bytes
equal packed bytes at the decode footprint).
"""
import collections, sqlite3, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PEAK = 1692.0
H, I = 5120, 17408


def packed_bytes(rows, K):
    Mp, Kp = (rows + 15) // 16 * 16, (K + 127) // 128 * 128
    return Mp * Kp // 2 + Mp * (Kp // 128) * 2


MAIN = {"gdn.in_proj": packed_bytes(16480, H), "attn.in_proj": packed_bytes(14336, H),
        "out": packed_bytes(H, 6144), "gate_up": packed_bytes(2 * I, H), "down": packed_bytes(H, I),
        "lm_head": packed_bytes(248320, H)}
MTP = {"mtp.fc": packed_bytes(H, 2 * H), "mtp.in_proj": packed_bytes(14336, H), "mtp.out": packed_bytes(H, 6144),
       "mtp.gate_up": packed_bytes(2 * I, H), "mtp.down": packed_bytes(H, I), "draft_head": packed_bytes(49152, H)}


def run():
    sys.path.insert(0, str(ROOT / "python"))
    import time
    import torch
    import torch.cuda.nvtx as nvtx
    from transformers import AutoTokenizer
    from qengine.engine import Engine, W4Linear
    M = Path.home() / "models/Qwen3.8-27B"
    tok = AutoTokenizer.from_pretrained(M)
    text = (ROOT / "bench/prompts/suite/1_prose.txt").read_text()
    p = tok.apply_chat_template([{"role": "user", "content": text.strip()}], add_generation_prompt=True, tokenize=True)
    p = list(p["input_ids"] if hasattr(p, "input_ids") else p)
    eng = Engine(M, max_len=2048, linear=W4Linear(), lm_head="w4gptq", draft_vocab=49152,
                 cache_dir=str(Path.home() / "models/qengine-cache"))
    eng.generate(p, 64); eng.generate_spec(p, 64, 4)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        eng.generate(p, 64)
    torch.cuda.synchronize()
    nvtx.range_push("plain"); eng.generate(p, 96); torch.cuda.synchronize(); nvtx.range_pop()
    nvtx.range_push("spec4"); eng.generate_spec(p, 96, 4); torch.cuda.synchronize(); nvtx.range_pop()


def analyze(db):
    con = sqlite3.connect(db)
    strings = dict(con.execute("select id, value from StringIds"))
    ranges = {strings.get(t, t) if isinstance(t, int) else t: (s, e) for t, s, e in
              con.execute("select coalesce(textId, text), start, end from NVTX_EVENTS where end is not null")}
    ks = con.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    ks = [(s, e, strings[n]) for s, e, n in ks]
    out = {}
    for rname in ("plain", "spec4"):
        r0, r1 = ranges[rname]
        seq = [k for k in ks if r0 <= k[0] and k[1] <= r1]
        stats = collections.defaultdict(list)
        cur, pos, prev_end = None, 0, None
        passes, pass_t = [], []
        for s, e, n in seq:
            if "embed" in n:
                if passes:
                    pass_t[-1][1] = prev_end
                passes.append([]); pos = 0
                pass_t.append([prev_end, None])
            if "gemv_w4" in n and passes:
                exposed = e - max(prev_end or s, s if prev_end is None else prev_end)
                passes[-1].append((exposed, e - s))
            prev_end = e
        if pass_t:
            pass_t[-1][1] = prev_end
        # wall time of each main pass whose GEMVs look like decode/verify (exclude prefill chunks:
        # those have far longer GEMVs); attributed from the previous pass's end
        main_t = [(t1 - t0) / 1e3 for ps, (t0, t1) in zip(passes, pass_t) if len(ps) == 257 and t0 is not None]
        if main_t:
            main_t.sort()
            print(f"{rname}: main pass wall time median {statistics.median(main_t[:max(1, len(main_t) * 3 // 4)]):.1f} us "
                  f"(fastest 75% of {len(main_t)} passes; min {main_t[0]:.1f})")
        for ps in passes:
            if len(ps) == 257:
                for i, (ex, du) in enumerate(ps):
                    if i == 256:
                        key = "lm_head"
                    else:
                        l, role = divmod(i, 4)
                        key = ["gdn.in_proj" if l % 4 != 3 else "attn.in_proj", "out", "gate_up", "down"][role]
                    stats[key].append(ex)
            elif len(ps) == 6:
                for key, (ex, du) in zip(MTP, ps):
                    stats[key].append(ex)
        tot_b = tot_t = 0
        print(f"== {rname}: {sum(1 for p in passes if len(p) == 257)} main passes, "
              f"{sum(1 for p in passes if len(p) == 6)} MTP passes, other pass lengths "
              f"{sorted(set(len(p) for p in passes) - {257, 6})}")
        print("| shape | n | exposed us (median) | GB/s | % of 1,692 |")
        for key in list(MAIN) + list(MTP):
            if key not in stats:
                continue
            v = stats[key]; b = (MAIN | MTP)[key]
            m = statistics.median(v) / 1e3
            print(f"| {key} | {len(v)} | {m:.2f} | {b / m / 1e3:.0f} | {100 * b / m / 1e3 / PEAK:.1f} |")
            if key in MAIN:
                cnt = {"gdn.in_proj": 48, "attn.in_proj": 16, "out": 64, "gate_up": 64, "down": 64, "lm_head": 1}[key]
                tot_b += b * cnt; tot_t += m * cnt
        print(f"main-pass GEMVs: {tot_t / 1e3:.3f} ms exposed, {tot_b / tot_t / 1e3:.0f} GB/s "
              f"({100 * tot_b / tot_t / 1e3 / PEAK:.1f}%)")
        out[rname] = stats
    return out


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else analyze(sys.argv[2])
