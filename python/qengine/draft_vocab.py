"""Pruned draft vocabulary for the MTP draft head (FR-Spec, arXiv 2502.14856; ideas.md #4).

Default rule (wiki_frac = 0): the `size` lowest token ids (earliest BPE merges) plus all added
special tokens. Chosen by coverage on held-out text (73,780 tokens of technical prose), not on the
benchmark: 32K lowest ids 94.5% vs 75.3% for wikitext-2 frequency alone, 93.3% for a 50/50 mix;
48K: 96.8% / 93.6% / 96.0%.

The draft lm_head only scores the `size` most frequent tokens; verify still uses the full lm_head,
so greedy output is unchanged (only acceptance can drop). Frequencies come from the wikitext-2
TRAIN split (raw text, `~/models/ppl-cache/data/wikitext2_train.txt`, tokenized here with the
model's tokenizer; the benchmark prompt is not in it). Unseen tokens are ranked by id (earlier
BPE merges first). All added/special tokens
(<think>, <|im_end|>, ...) are always included because wikitext has none of them.
Returns sorted int32 ids (ascending, so argmax ties resolve to the lowest vocabulary id like the
full head).
"""
import json
from pathlib import Path

import torch

FREQ_SRC = Path.home() / "models/ppl-cache/data/wikitext2_train.txt"


def token_counts(model_dir, vocab: int) -> torch.Tensor:
    """Token counts of the raw wikitext-2 train text (tokenized here, line structure kept),
    cached in qengine/build/."""
    cache = Path(__file__).resolve().parents[2] / "build" / f"wikitext2_train.{Path(model_dir).name}.counts.pt"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return torch.load(cache)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = FREQ_SRC.read_text()
    ids = []
    for i in range(0, len(text), 1 << 20):  # 1 MB pieces (piece boundaries cost a few tokens)
        ids.extend(tok(text[i:i + (1 << 20)]).input_ids)
    counts = torch.bincount(torch.tensor(ids), minlength=vocab)[:vocab].float()
    torch.save(counts, cache)
    return counts


def draft_vocab_ids(model_dir, size: int, vocab: int, wiki_frac: float = 0.0) -> torch.Tensor:
    """wiki_frac of the slots go to the most frequent wikitext tokens; the rest to the lowest ids
    not already chosen (earliest BPE merges: frequent pieces of the tokenizer's own training
    data, which covers markdown/code/chat text that wikitext lacks)."""
    counts = token_counts(model_dir, vocab)
    tj = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    special = sorted({t["id"] for t in tj.get("added_tokens", []) if t["id"] < vocab})
    chosen = torch.zeros(vocab, dtype=torch.bool)
    chosen[special] = True
    n_wiki = int(size * wiki_frac)
    score = counts - torch.arange(vocab, dtype=torch.float32) * 1e-9
    score[chosen] = -1
    seen = (counts > 0).sum().item()
    top = torch.topk(score, min(n_wiki, seen)).indices
    chosen[top] = True
    for i in range(vocab):  # fill by lowest id
        if chosen.sum() >= size:
            break
        chosen[i] = True
    ids = chosen.nonzero().flatten()[:size] if chosen.sum() > size else chosen.nonzero().flatten()
    return ids.to(torch.int32)

