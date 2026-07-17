"""Batched repetition-task sweep: the operational glitch-token test of both papers.

Greedy decoding (temperature 0), constant-length spliced prompts, batched generate.
Reused by: RQ1 census, GlitchProber sample labeling + post-validation,
GlitchProber repair evaluation (with hooks active), GlitchCleaner evaluation
(with adapter active).
"""
import torch
from tqdm import tqdm

from ..common.model_utils import token_str
from ..common.prompts import spliced_batch, is_repetition_correct


@torch.no_grad()
def repetition_sweep(
    model,
    tok,
    token_ids: list[int],
    batch_size: int = 32,
    max_new_tokens: int = 24,
    task: str = "repetition",
    desc: str = "sweep",
) -> dict[int, tuple[bool, str]]:
    """Return {token_id: (is_correct, generated_text)}. is_correct=False => glitch behavior."""
    device = next(model.parameters()).device
    results: dict[int, tuple[bool, str]] = {}
    for i in tqdm(range(0, len(token_ids), batch_size), desc=desc):
        chunk = token_ids[i : i + batch_size]
        input_ids, attention_mask = spliced_batch(tok, chunk, device, task)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, input_ids.shape[1] :]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for tid, text in zip(chunk, texts):
            ok = is_repetition_correct(token_str(tok, tid), text)
            results[tid] = (ok, text)
    return results


def split_glitch_normal(results: dict[int, tuple[bool, str]]) -> tuple[list[int], list[int]]:
    glitch = [t for t, (ok, _) in results.items() if not ok]
    normal = [t for t, (ok, _) in results.items() if ok]
    return glitch, normal
