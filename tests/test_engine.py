"""Engine end-to-end on Qwen3.5-0.8B BF16 vs reference.py and HF transformers (criterion 1)."""
import pytest
import torch
from conftest import M08
from hf_data import hf_greedy, hf_model, hf_teacher_forced_logits

from qengine import reference as R

pytestmark = pytest.mark.skipif(not M08.exists(), reason="0.8B weights missing")


@pytest.fixture(scope="module")
def engine():
    from qengine.engine import Engine
    return Engine(M08, max_len=2304)


def test_graph_steps_match_reference(engine):
    """First 64 prompt+generated positions: graph-replayed logits vs the torch reference."""
    seq = hf_greedy()["seq"][:64]
    eng = engine.sequence_logits(seq).float()
    W = R.load_weights(M08)
    ref = R.run_sequence(W, R.Config.from_dir(M08), seq).float()
    d = (eng - ref).abs()
    agree = (eng.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"\nengine vs reference, 64 pos: top1 {agree:.2%} max|dlogit| {d.max():.3f} mean {d.mean():.4f}")
    assert agree >= 0.98 and d.mean() < 0.05


def test_greedy_1024_matches_hf(engine):
    d = hf_greedy()
    P, seq = len(d["prompt"]), d["seq"]
    gen = engine.generate(d["prompt"], 1024)
    hf = seq[P:P + 1024]
    first = next((i for i, (a, b) in enumerate(zip(gen, hf)) if a != b), None)
    if first is not None:
        # logit gap at the divergence, from the engine's own teacher-forced logits on HF's sequence
        lg = engine.sequence_logits(seq[: P + first])[-1].float()
        print(f"\nfirst divergence at generated token {first}: engine {gen[first]} "
              f"(logit {lg[gen[first]]:.3f}) vs HF {hf[first]} (logit {lg[hf[first]]:.3f})")
    else:
        print("\nengine greedy == HF greedy for all 1024 tokens")
    assert first is None


def test_top1_agreement_2048_vs_hf(engine):
    d = hf_greedy()
    seq = d["seq"]
    model = hf_model()
    hf = hf_teacher_forced_logits(model, seq).float()
    del model
    eng = engine.sequence_logits(seq).float()
    agree = (eng.argmax(-1) == hf.argmax(-1)).float().mean().item()
    diff = (eng - hf).abs()
    P = len(d["prompt"])
    gen_agree = (eng[P - 1:-1].argmax(-1).cpu() == torch.tensor(seq[P:])).float().mean().item()
    print(f"\nengine top1 vs HF teacher-forced (2048 pos): {agree:.4%}; vs HF greedy tokens: {gen_agree:.4%}; "
          f"max|dlogit| {diff.max():.3f} mean {diff.mean():.4f}")
    assert agree >= 0.99 and gen_agree >= 0.99
