"""HF transformers 5.20.1 ground truth for the 0.8B: greedy tokens (decode path) cached on disk."""
import torch
from conftest import CACHE, M08, PROMPT

N_POS = 2048


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(M08)


def prompt_ids():
    return tokenizer()(PROMPT.read_text()).input_ids


def hf_model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(M08, dtype=torch.bfloat16).cuda().eval()


def hf_greedy(model=None):
    """prompt ids + HF greedy continuation up to N_POS tokens total (EOS ignored)."""
    f = CACHE / "hf_greedy_0.8b.pt"
    if f.exists():
        return torch.load(f)
    CACHE.mkdir(parents=True, exist_ok=True)
    model = model or hf_model()
    ids = prompt_ids()
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=N_POS - len(ids), min_new_tokens=N_POS - len(ids),
                             do_sample=False, eos_token_id=None, pad_token_id=0,
                             temperature=None, top_p=None, top_k=None)
    d = {"prompt": ids, "seq": out[0].tolist()}
    torch.save(d, f)
    return d


@torch.no_grad()
def hf_teacher_forced_logits(model, seq):
    return model(torch.tensor([seq], device="cuda")).logits[0]  # [T, V] bf16
