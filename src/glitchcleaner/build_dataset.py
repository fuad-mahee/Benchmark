"""Build the glitch-token QA dataset with a leakage-controlled train/held-out split.

The paper trains the LoRA on the identified glitch tokens and evaluates repair on
that same population. The held-out split is OUR addition - the test that
distinguishes generalisable repair from memorisation of the training set (RQ4).

CORRECTION HISTORY (documented for the thesis)
----------------------------------------------
v1 built prompts with a stripped token string and a bare-space answer. The
authors' script uses the UNSTRIPPED decoded token inside single quotes, both in
the prompt and as the target (third_party/GlitchCleaner/Fine-tuning/
fine-tuning.py:128-136). We now match that, because a different target string
teaches a different output format and would confound any repair-rate comparison.

v1 also called a token "held out" if its id was not in the training id list, but
a token id can still appear inside a training SEQUENCE (as context, or because
another token's decoded text re-encodes through it). Those tokens are now
excluded from the held-out set, so "the adapter never saw this token" is true as
stated rather than true only of the label.
"""
import json
import random
from pathlib import Path

from ..common.model_utils import token_str
from ..common.prompts import _TEMPLATES


def split_glitch_tokens(glitch_ids: list[int], holdout_fraction: float, seed: int):
    rng = random.Random(seed)
    ids = sorted(glitch_ids)
    rng.shuffle(ids)
    n_hold = int(len(ids) * holdout_fraction)
    return ids[n_hold:], ids[:n_hold]  # train, heldout


def build_examples(tok, token_ids: list[int], task: str = "repetition",
                   style: str = "decoded") -> list[dict]:
    """Prompt + target answer for each token.

    style="decoded" reproduces the authors' construction: the decoded token is
    inserted verbatim (no stripping) and the target is that same string in single
    quotes.
    """
    prefix, suffix = _TEMPLATES[task]
    ex = []
    for tid in token_ids:
        s = tok.decode([tid]) if style == "decoded" else token_str(tok, tid)
        prompt = f"{prefix}{s}{suffix}"
        answer = f"'{s}'" if style == "decoded" else f" {s.strip() or s}"
        ex.append({"token_id": tid, "prompt": prompt, "answer": answer})
    return ex


def training_token_ids(tok, examples: list[dict]) -> set[int]:
    """Every vocabulary id the adapter actually sees during training."""
    seen = set()
    for e in examples:
        seen.update(tok(e["prompt"], add_special_tokens=True).input_ids)
        seen.update(tok(e["answer"], add_special_tokens=False).input_ids)
    return seen


def filter_leaked(heldout_ids: list[int], seen_ids: set[int]) -> tuple[list[int], list[int]]:
    """Split held-out ids into (clean, leaked-and-therefore-dropped)."""
    clean = [t for t in heldout_ids if t not in seen_ids]
    leaked = [t for t in heldout_ids if t in seen_ids]
    return clean, leaked


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(examples)} examples)")
