"""Build the glitch-token QA dataset with a train/held-out split.

The paper trains the LoRA on the identified glitch tokens and (apparently) measures
repair on those same tokens. The held-out split is OUR addition - the decisive test
of whether GlitchCleaner generalizes or memorizes (RQ4).
"""
import json
import random
from pathlib import Path

from ..common.model_utils import token_str
from ..common.prompts import REPETITION_PREFIX, REPETITION_SUFFIX


def split_glitch_tokens(glitch_ids: list[int], holdout_fraction: float, seed: int):
    rng = random.Random(seed)
    ids = sorted(glitch_ids)
    rng.shuffle(ids)
    n_hold = int(len(ids) * holdout_fraction)
    return ids[n_hold:], ids[:n_hold]  # train, heldout


def build_examples(tok, token_ids: list[int]) -> list[dict]:
    """Prompt + expected answer, with prompt char-length recorded for loss masking."""
    ex = []
    for tid in token_ids:
        s = token_str(tok, tid)
        prompt = f"{REPETITION_PREFIX}{s}{REPETITION_SUFFIX}"
        ex.append({"token_id": tid, "prompt": prompt, "answer": f" {s.strip() or s}"})
    return ex


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(examples)} examples)")
