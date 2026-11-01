"""Generate MTP self-distillation data with the qengine 27B W4 engine (resumable, time-boxed).

Per prompt i (bench/distill/prompts.py): chat template, greedy generation of NEW tokens (via
MTP spec k=4, which equals plain greedy), then a teacher-forced pass in 8-row chunks over the full
sequence to record the main model's post-final-norm hidden state at every position.
Saves ~/models/mtp-distill/seq_{i:04d}.pt = {tokens int32 [T], prompt_len, hidden bf16 [T, H]}.
Usage: gpu_lock.sh python bench/distill/gen_data.py --budget 480
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from prompts import prompts  # noqa: E402
from qengine.engine import NMAX, Engine, W4Linear  # noqa: E402

OUT = Path.home() / "models" / "mtp-distill"


@torch.no_grad()
def hidden_states(eng, seq):
    T = len(seq)
    out = torch.empty(T, eng.cfg.hidden, dtype=torch.bfloat16, device="cuda")
    chunks = [seq[i:i + NMAX] for i in range(0, T, NMAX)]
    eng._ensure({(len(c), False, 0) for c in chunks})
    eng.reset()
    buf = torch.full((NMAX + 1,), -1, dtype=torch.int32)
    for ci, c in enumerate(chunks):
        buf.fill_(-1)
        buf[:len(c)] = torch.tensor(c, dtype=torch.int32)
        if ci + 1 < len(chunks):
            buf[len(c)] = chunks[ci + 1][0]
        eng.in_tok.copy_(buf)
        eng.graph(len(c), False, 0).replay()
        out[ci * NMAX:ci * NMAX + len(c)].copy_(eng.hid[:len(c)])
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/Qwen3.8-27B"))
    ap.add_argument("--new", type=int, default=768)
    ap.add_argument("--budget", type=float, default=480, help="seconds; stop starting new prompts after this")
    a = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P = prompts()
    todo = [i for i in range(len(P)) if not (OUT / f"seq_{i:04d}.pt").exists()]
    print(f"{len(todo)} of {len(P)} prompts left", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    eng = Engine(a.model, max_len=2048, linear=W4Linear(), cache_dir=str(Path.home() / "models/qengine-cache"),
                 lm_head="w4gptq", draft_vocab=49152)
    for i in todo:
        if time.time() - t_start > a.budget:
            break
        p = tok.apply_chat_template([{"role": "user", "content": P[i]}], add_generation_prompt=True, tokenize=True)
        if hasattr(p, "input_ids"):
            p = p["input_ids"]
        p = list(p)
        gen = eng.generate_spec(p, a.new, 4)
        seq = p + gen
        h = hidden_states(eng, seq)
        torch.save({"tokens": torch.tensor(seq, dtype=torch.int32), "prompt_len": len(p), "hidden": h},
                   OUT / f"seq_{i:04d}.tmp")
        (OUT / f"seq_{i:04d}.tmp").rename(OUT / f"seq_{i:04d}.pt")
        print(f"{i}: {len(p)} + {len(gen)} tokens, {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
