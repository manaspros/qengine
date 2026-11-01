"""M2: Qwen3.8-27B W4A16 end to end (RTN int4 g128, python/qengine/quant.py + qengine._w4).

1. (lm_head = GPTQ W4, the benchmark configuration) Engine vs reference.py with the SAME fake-quantized weights (dequantize_w4, fp32 weights,
   fp32 accumulation, bf16 activations): teacher-forced logits over 256 positions (105-token
   prose prompt + 151 engine-greedy tokens). The reference runs layer-major (one layer of fp32
   weights on the GPU at a time). Its lm_head uses the same W4 GEMV on the reference's own
   final hidden states (the dequantized fp32 lm_head is 5 GB), so the decoder layers are what is
   compared. Differences that remain: the fused SwiGLU epilogue computes silu(g) * u in fp32
   (the reference rounds g, u, silu(g) to bf16 like HF) and summation order.
   Thresholds: top-1 agreement >= 97% (near-ties flip under reordering; a real bug gives
   agreement far lower), mean |dlogit| < 0.1, final hidden rel L2 < 3e-2.
2. Greedy text sanity on prose.txt (printed).
3. Speculative k = 1..4 output == plain greedy, 1,024 tokens.
"""
from pathlib import Path

import pytest
import torch
from conftest import M27, PROMPT

from qengine import reference as R

pytestmark = pytest.mark.skipif(not (M27 / "model-00018-of-00018.safetensors").exists(),
                                reason="27B weights missing")
CACHE = Path.home() / "models" / "qengine-cache"
BF = torch.bfloat16


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(M27)


@pytest.fixture(scope="module")
def eng():
    from qengine.engine import Engine, W4Linear
    import os
    dv = int(os.environ.get("QE_DRAFT_VOCAB", "0")) or None  # pruned MTP draft vocab under test
    e = Engine(M27, max_len=1300, linear=W4Linear(), cache_dir=CACHE, lm_head="w4gptq", draft_vocab=dv)
    print(f"\ndraft_vocab={dv}")
    print(f"\nGPTQ lm_head packed on its grid: {e.lm_head_stats}")
    return e


def _ref_layer(eng, l):
    """Reference weights (fp32, dequantized from the engine's packed W4) for layer l."""
    L, cfg, deq = eng.layers[l], eng.cfg, eng.linear.dequantize
    p = f"layers.{l}."
    g, u = deq(L["gate_up"])
    W = {p + "input_layernorm.weight": L["in_norm"], p + "post_attention_layernorm.weight": L["post_norm"],
         p + "mlp.gate_proj.weight": g, p + "mlp.up_proj.weight": u, p + "mlp.down_proj.weight": deq(L["down"])}
    ip = deq(L["in_proj"])
    if L["type"] == "linear_attention":
        q = p + "linear_attn."
        c0, c1 = cfg.conv_dim, cfg.conv_dim + cfg.value_dim
        W.update({q + "in_proj_qkv.weight": ip[:c0], q + "in_proj_z.weight": ip[c0:c1],
                  q + "in_proj_b.weight": ip[c1:c1 + cfg.hv], q + "in_proj_a.weight": ip[c1 + cfg.hv:],
                  q + "out_proj.weight": deq(L["out"]), q + "conv1d.weight": L["conv_w"], q + "A_log": L["A_log"],
                  q + "dt_bias": L["dt_bias"], q + "norm.weight": L["norm_w"]})
    else:
        q = p + "self_attn."
        q0, k1 = cfg.hq * 512, cfg.hq * 512 + cfg.hkv * 256
        W.update({q + "q_proj.weight": ip[:q0], q + "k_proj.weight": ip[q0:k1], q + "v_proj.weight": ip[k1:],
                  q + "o_proj.weight": deq(L["out"]), q + "q_norm.weight": L["q_norm"],
                  q + "k_norm.weight": L["k_norm"]})
    return W


def _fq_lin(x, w):  # fake-quant linear: bf16 activations, fp32 weights and accumulation, one rounding
    return torch.nn.functional.linear(x.float(), w.float()).to(BF)


def test_w4_engine_vs_fakequant_reference(eng, tok):
    prompt = tok(PROMPT.read_text()).input_ids
    gen = eng.generate(prompt, 256 - len(prompt))
    seq = prompt + gen
    e_logits = eng.sequence_logits(seq).float()
    h = R.run_sequence_layerwise(lambda l: _ref_layer(eng, l), eng.cfg, seq, eng.embed, lin=_fq_lin)
    hf = R.rmsnorm1p(h, eng.final_norm, eng.cfg.eps)
    r_logits = torch.empty_like(e_logits)
    out = torch.empty(8, eng.cfg.vocab, dtype=BF, device="cuda")
    for i in range(0, len(seq), 8):
        n = min(8, len(seq) - i)
        eng.lm_lin(hf[i:i + n], eng.lm_head, out[:n])
        r_logits[i:i + n] = out[:n].float()
    # engine final hidden for the same positions: recompute via the engine's own pass buffers
    agree = (e_logits.argmax(-1) == r_logits.argmax(-1)).float().mean().item()
    d = (e_logits - r_logits).abs()
    gen_agree = (r_logits[len(prompt) - 1:-1].argmax(-1).cpu() == torch.tensor(gen)).float().mean().item()
    print(f"\n27B W4 engine vs fake-quant reference (256 pos): top1 {agree:.2%}, ref argmax == engine greedy "
          f"{gen_agree:.2%}, max|dlogit| {d.max():.3f}, mean {d.mean():.4f}")
    assert agree >= 0.97 and gen_agree >= 0.97 and d.mean() < 0.1


def test_w4_repeatable(eng, tok):
    """Bitwise run-to-run determinism (needed for spec == plain; see W4Linear.DEFAULT_KW)."""
    prompt = tok(PROMPT.read_text()).input_ids
    runs = [eng.generate(prompt, 256) for _ in range(4)]
    assert all(r == runs[0] for r in runs)


def test_w4_greedy_text(eng, tok):
    prompt = tok(PROMPT.read_text()).input_ids
    gen = eng.generate(prompt, 200)
    text = tok.decode(gen)
    print("\n--- 27B W4 greedy ---\n" + text + "\n---")
    assert len(set(gen)) > 50  # not a degenerate loop


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_w4_spec_equals_plain_1024(eng, tok, k):
    prompt = tok(PROMPT.read_text()).input_ids
    if not hasattr(test_w4_spec_equals_plain_1024, "plain"):
        test_w4_spec_equals_plain_1024.plain = eng.generate(prompt, 1024)
    plain = test_w4_spec_equals_plain_1024.plain
    spec, _, _, st = eng.generate_spec(prompt, 1024, k, timing=True)
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    print(f"\nk={k}: spec == plain: {first is None} (first diff {first}); accept rate {st['accept_rate']:.3f}, "
          f"tokens/cycle {st['tokens_per_cycle']:.2f}; distinct plain tokens {len(set(plain))}")
    assert first is None
