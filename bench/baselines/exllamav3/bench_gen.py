"""exllamav3 end-to-end baseline through its Generator (the path TabbyAPI and examples/chat.py use).

Fixed prompt (bench/prompts/prose.txt, HF chat template via the model's tokenizer, thinking left at
the template default, same as llama-server), greedy (ArgmaxSampler), exactly 1,024 new tokens
(no stop conditions = ignore EOS), 1 warm-up job discarded + 5 timed jobs. The prompt is identical
every run, so the generator may reuse cached prefix pages; that only shortens prefill, which is
excluded from the decode number. Decode tok/s = new_tokens / time_generate, exactly as examples/chat.py reports it
(time_generate runs from the first sampled token to the last, so it includes the Python
generator loop and sampling per token, like llama-server's predicted_per_second).

Run from the exllamav3 checkout with its own venv:
  cd ~/Codes/baselines/exllamav3 && .venv/bin/python <qengine>/bench/baselines/exllamav3/bench_gen.py \
      -m <model_dir> -out <json> [--mtp] [-ndt N] [-cs 8192]
"""
import argparse, json, statistics, sys, time
import torch
from exllamav3 import Generator, Job, model_init, GreedySampler

ROOFLINE = 124.5
N_NEW = 1024
RUNS = 6  # run 0 = warm-up, discarded

parser = argparse.ArgumentParser()
model_init.add_args(parser, default_cache_size=8192, add_draft_model_args=True)
parser.add_argument("-prompt", "--prompt_file", required=True)
parser.add_argument("-out", "--out", required=True)
args = parser.parse_args()

with torch.inference_mode():
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    gen = Generator(
        model=model, cache=cache, tokenizer=tokenizer,
        draft_model=draft_model, draft_cache=draft_cache,
        num_draft_tokens=args.num_draft_tokens,
        dynamic_draft_tokens=args.dynamic_draft,
        sparse_draft_head=args.sparse_draft_head,
        draft_confidence=args.draft_confidence,
    )
    user = open(args.prompt_file).read().strip()
    ids = tokenizer.hf_chat_template([{"role": "user", "content": user}], add_generation_prompt=True)
    print(f"prompt tokens: {ids.shape[-1]}")

    runs = []
    for i in range(RUNS):
        job = Job(input_ids=ids, max_new_tokens=N_NEW, stop_conditions=[], sampler=GreedySampler())
        gen.enqueue(job)
        text, last = [], None
        t0 = time.perf_counter()
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r["stage"] == "error":
                    raise r["error"]
                text.append(r.get("text", ""))
                if r.get("eos"):
                    last = r
        wall = time.perf_counter() - t0
        n = last["new_tokens"]
        run = {"new_tokens": n, "decode_tps": n / last["time_generate"], "time_generate": last["time_generate"],
               "time_prefill": last["time_prefill"], "wall_s": wall, "e2e_tps": n / wall,
               "eos_reason": last.get("eos_reason"), "content": "".join(text)}
        if "accepted_draft_tokens" in last:
            a, rj = last["accepted_draft_tokens"], last["rejected_draft_tokens"]
            run.update(accepted_draft_tokens=a, rejected_draft_tokens=rj,
                       acceptance_rate=a / max(a + rj, 1))
        runs.append(run)
        print(f"run {i}{' (warm-up)' if i == 0 else ''}: {n} tok, decode {run['decode_tps']:.2f} tok/s, "
              f"prefill {1000 * run['time_prefill']:.1f} ms"
              + (f", accept {run['acceptance_rate']:.3f}" if 'acceptance_rate' in run else ""), flush=True)

    timed = runs[1:]
    dec = [r["decode_tps"] for r in timed]
    e2e = [r["e2e_tps"] for r in timed]
    same = all(r["content"] == timed[0]["content"] for r in timed)
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()
    summary = {"prompt_tokens": int(ids.shape[-1]), "generated": [r["new_tokens"] for r in timed],
               "decode_tps": dec, "decode_median": statistics.median(dec),
               "e2e_tps": e2e, "e2e_median": statistics.median(e2e), "outputs_identical": same,
               "bpw_layer": bpw_layer, "bpw_head": bpw_head, "model_vram_bytes": vram_bits / 8,
               "torch_max_allocated": torch.cuda.max_memory_allocated(),
               "warmup_decode_tps": runs[0]["decode_tps"], "output_head": timed[0]["content"][:400],
               "args": vars(args)}
    if "acceptance_rate" in timed[0]:
        summary["acceptance_rate"] = [r["acceptance_rate"] for r in timed]
    json.dump({"summary": summary, "runs": runs}, open(args.out, "w"), indent=1)
    print(f"bitrate {bpw_layer:.3f} bpw layers / {bpw_head:.2f} bpw head, model {vram_bits / 8e9:.2f} GB")
    print(f"decode median {statistics.median(dec):.2f} tok/s (min {min(dec):.2f}, max {max(dec):.2f}) "
          f"= {100 * statistics.median(dec) / ROOFLINE:.1f}% of roofline")
    print(f"end-to-end median {statistics.median(e2e):.2f} (min {min(e2e):.2f}, max {max(e2e):.2f}); "
          f"outputs identical: {same}")
