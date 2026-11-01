"""Criterion 1 for reference.py: step-by-step reference vs HF teacher-forced logits, 2,048 positions."""
import pytest
import torch
from conftest import M08
from hf_data import hf_greedy, hf_model, hf_teacher_forced_logits

from qengine import reference as R


@pytest.mark.skipif(not M08.exists(), reason="0.8B weights missing")
def test_reference_matches_hf_2048():
    model = hf_model()
    d = hf_greedy(model)
    seq = d["seq"]
    assert len(seq) == 2048
    hf = hf_teacher_forced_logits(model, seq).float()
    del model
    torch.cuda.empty_cache()
    cfg = R.Config.from_dir(M08)
    W = R.load_weights(M08)
    ref = R.run_sequence(W, cfg, seq).float()
    agree = (ref.argmax(-1) == hf.argmax(-1)).float().mean().item()
    diff = (ref - hf).abs()
    # HF greedy tokens came from HF's decode path: compare our argmax with them too
    P = len(d["prompt"])
    gen_agree = (ref[P - 1:-1].argmax(-1).cpu() == torch.tensor(seq[P:])).float().mean().item()
    print(f"\ntop1 agree vs HF teacher-forced: {agree:.4%}  vs HF greedy tokens: {gen_agree:.4%}  "
          f"max|dlogit| {diff.max():.3f}  mean|dlogit| {diff.mean():.4f}")
    assert agree >= 0.99
    assert gen_agree >= 0.99
