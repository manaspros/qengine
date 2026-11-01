"""M4 groundwork on Qwen3.5-0.8B BF16: multi-row passes, MTP head, greedy speculative decoding.

Hard requirement: speculative greedy output == plain greedy output token for token (1,024 tokens,
prose.txt) for k = 1..4. Exactness comes from row-independent kernels (tests/test_kernels.py
checks each one bit for bit), so the comparison is exact equality, no tolerance.
"""
import pytest
import torch
from conftest import M08
from hf_data import hf_greedy

from qengine import reference as R

pytestmark = pytest.mark.skipif(not M08.exists(), reason="0.8B weights missing")


@pytest.fixture(scope="module")
def engine():
    from qengine.engine import Engine
    return Engine(M08, max_len=2304)


def test_prefill_chunked_equals_one_by_one(engine):
    """prefill in passes of 8 (and 3) == passes of 1, logits bit for bit (300 positions)."""
    seq = hf_greedy()["seq"][:300]
    a = engine.sequence_logits(seq, chunk=1)
    for c in (8, 3):
        b = engine.sequence_logits(seq, chunk=c)
        assert torch.equal(a, b), c


def test_mtp_logits_vs_reference(engine):
    """Engine MTP rows (hidden after final norm + next token, vLLM convention) vs reference.mtp_step
    fed the same hidden states. Tolerance: like the main model, top-1 >= 98% and mean |dlogit| small
    (bf16 roundings reordered); a convention bug (wrong concat order / norm) breaks it entirely."""
    seq = hf_greedy()["seq"][:96]
    hid, lg = engine.mtp_rows_logits(seq)
    cfg = R.Config.from_dir(M08)
    W = R.load_weights(M08)
    # main hidden vs reference post-norm hidden
    st = R.State(cfg, len(seq))
    ref_h = torch.stack([R.decode_step(W, st, t, cfg, return_hidden=True)[1] for t in seq])
    hrel = ((hid.float() - ref_h.float()).norm() / ref_h.float().norm()).item()
    mst = R.MTPState(cfg, len(seq))
    ref = torch.stack([R.mtp_step(W, mst, hid[t], seq[t + 1], t, cfg)[0] for t in range(len(seq) - 1)])
    e = lg[:-1].float()
    agree = (e.argmax(-1) == ref.float().argmax(-1)).float().mean().item()
    d = (e - ref.float()).abs()
    # how often does the MTP draft equal the actual next-next token (sanity of the convention)
    hit = (e[:-1].argmax(-1).cpu() == torch.tensor(seq[2:])).float().mean().item()
    print(f"\nmain hidden rel err {hrel:.2e}; MTP vs ref top1 {agree:.2%}, max|dlogit| {d.max():.3f} "
          f"mean {d.mean():.4f}; MTP predicts token t+2 on the HF greedy text: {hit:.1%}")
    assert hrel < 2e-2 and agree >= 0.98 and d.mean() < 0.05
    assert hit > 0.3


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_spec_equals_plain_1024(engine, k):
    d = hf_greedy()
    prompt = d["prompt"]
    plain = engine.generate(prompt, 1024)
    assert plain == d["seq"][len(prompt):len(prompt) + 1024]  # plain greedy == HF greedy
    spec, _, _, st = engine.generate_spec(prompt, 1024, k, timing=True)
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    print(f"\nk={k}: spec == plain: {first is None} (first diff {first}); cycles {st['cycles']}, "
          f"accept rate {st['accept_rate']:.3f}, tokens/cycle {st['tokens_per_cycle']:.2f}")
    assert first is None


@pytest.mark.parametrize("k", [2, 4])
def test_w4_spec_equals_plain_0p8b(k):
    """W4 backend: verify rows must be bit-identical to 1-row passes (split-K <= 2)."""
    from qengine.engine import Engine, W4Linear
    eng = Engine(M08, max_len=1300, linear=W4Linear())
    prompt = hf_greedy()["prompt"]
    plain = eng.generate(prompt, 512)
    spec, _, _, st = eng.generate_spec(prompt, 512, k, timing=True)
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    print(f"\n0.8B W4 k={k}: spec == plain {first is None} (first diff {first}), accept {st['accept_rate']:.3f}")
    assert first is None


@pytest.mark.parametrize("k", [2, 4])
def test_spec_draft_vocab_equals_plain_0p8b(k):
    """Pruned MTP draft vocabulary (FR-Spec): output must still equal plain greedy."""
    from qengine.engine import Engine
    eng = Engine(M08, max_len=1300, draft_vocab=16384)
    prompt = hf_greedy()["prompt"]
    plain = eng.generate(prompt, 512)
    spec, _, _, st = eng.generate_spec(prompt, 512, k, timing=True)
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    print(f"\n0.8B draft vocab 16K k={k}: spec == plain {first is None}, accept {st['accept_rate']:.3f}")
    assert first is None
