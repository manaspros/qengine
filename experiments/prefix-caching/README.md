# Prefix caching on one RTX 5090: how much does a cache hit buy?

A side experiment, separate from the qengine engine: vLLM 0.32.0 serving Qwen2.5-1.5B-Instruct (bf16)
on an RTX 5090 (driver 615.78.08), prefix caching on vs off. The workload has the same shape as
[tokenwatt's H100 measurement](https://www.tokenwatt.io/blog/prefix-cache-revenue-per-gpu-hour.html)
(1,024 input / 256 output tokens, 8 shared prefixes), so the ratios can be compared.

## Throughput: 2,048 requests, 256 in flight

| Shared prefix | Cache hit rate | Requests/s | vs 0% | Mean time per output token | p99 time to first token |
|---|---:|---:|---:|---:|---:|
| 0% | 0% | 24.9 | 1.00× | 37.8 ms | 5,632 ms |
| 50% | 49.8% | 44.6 | 1.79× | 21.1 ms | 2,731 ms |
| 75% | 74.7% | 62.3 | 2.51× | 15.1 ms | 1,437 ms |
| 90% | 88.7% | 79.2 | **3.19×** | 11.9 ms | 770 ms |
| 90%, cache off | 0% | 25.5 | — | 36.8 ms | 5,510 ms |
| 90%, 16-token outputs | off → on | 59.4 → 319.8 | **5.4×** | — | 3,833 → 693 ms |

## Latency: one request at a time, 1,000-token shared prefix

| Output | First token, off → on | Whole request, off → on |
|---|---|---|
| 16 tokens | 21.1 → 8.2 ms (2.6×) | 67.4 → 54.5 ms (1.24×) |
| 512 tokens | 21.7 → 8.8 ms (2.5×) | 1,619 → 1,609 ms (1.006×) |

## Why the gain varies

A cache hit skips prefill and nothing else: every output token still needs its own decode step.
So the speedup follows Amdahl's law, with *f* = the share of GPU time spent on prefill:

    speedup = 1 / (1 − f × hit rate)

- With 3.19× at an 88.7% hit rate, *f* ≈ 0.77. At 256 requests in flight this small dense model is
  compute-bound in both phases, so time roughly follows token count, and 1,024 of every
  1,280 tokens are prompt.
- The H100 measurement got +82% (1.82×) at a similar hit rate, which gives *f* ≈ 0.5: in that setup
  (Qwen3-30B-A3B, a mixture-of-experts model) generating output tokens takes a larger share of the time.
- Short outputs raise *f* (5.4× with 16 output tokens). Long outputs at one user lower it to almost
  nothing (1.006×). Time to first token improves either way, because it is mostly prefill.
- Hit rates sit a little below the shared share (88.7% for 90%) because vLLM caches only full
  16-token blocks, and each prefix's first request misses.

## Method and limits

- `run.py` starts a fresh `vllm serve` for each config, so the cache starts empty. It warms the
  server up with a different random seed, so the warm-up does not fill the cache with the measured
  prefixes. It then runs `vllm bench serve --dataset-name prefix_repetition --ignore-eos` and reads
  the hit rate from the `vllm:prefix_cache_hits_total` and `vllm:prefix_cache_queries_total`
  counters on `/metrics`.
- **One run per config.** Treat differences under ~5% as noise.
- These are **saturation throughput** numbers, not a capacity search under a latency target like
  tokenwatt's. The cache-off runs here would miss a 2 s p99 first-token target.
- One small model, synthetic random-token prefixes (a best case for hit rate), one GPU.

Reproduce (needs vLLM 0.32.0; the model downloads from Hugging Face):

    python run.py e1   # latency
    python run.py e2   # throughput sweep

`results.txt` is the summary; `results/` holds the raw `vllm bench serve` JSON for every config.
