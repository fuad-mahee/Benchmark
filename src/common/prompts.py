"""Repetition-task prompt construction and answer checking.

Template (GlitchCleaner's exact wording, also matching GlitchProber's task):
    "Can you repeat the string '{token}' and return it back to me? Answer: Here is the repeated string:"

The token under test is spliced in as its RAW VOCABULARY ID between the pre-tokenized
template halves. Consequences (all desirable):
  - every prompt in a sweep has identical length  -> trivially batchable, no padding
  - the exact token id is guaranteed to be what the model sees (no re-encoding drift)
  - attention rows have a fixed width               -> fixed-size feature vectors for the SVM
"""
import torch

REPETITION_PREFIX = "Can you repeat the string '"
REPETITION_SUFFIX = "' and return it back to me? Answer: Here is the repeated string:"

SPELLING_PREFIX = "Please can you spell out the string '"
SPELLING_SUFFIX = "' with hyphens between each letter? Answer: Of course! The spelling of the string is:"

LENGTH_PREFIX = "What is the length of this string '"
LENGTH_SUFFIX = "'? Answer: The length of this string is:"

_TEMPLATES = {
    "repetition": (REPETITION_PREFIX, REPETITION_SUFFIX),
    "spelling": (SPELLING_PREFIX, SPELLING_SUFFIX),
    "length": (LENGTH_PREFIX, LENGTH_SUFFIX),
    # GlitchCleaner's RELEASED CODE (count_passed_glitchtokens in GlitchCleaner.py)
    # differs from their paper's stated template: "Question:" prefix, "return back"
    # (not "return it back"), and newlines. Used by the gccode census protocol.
    "repetition_gccode": (
        "Question: Can you repeat the string '",
        "' and return back to me?\nAnswer: Here is the repeated string:\n",
    ),
}


def template_ids(tok, task: str = "repetition") -> tuple[list[int], list[int]]:
    prefix, suffix = _TEMPLATES[task]
    pre = tok(prefix, add_special_tokens=True).input_ids
    suf = tok(suffix, add_special_tokens=False).input_ids
    return pre, suf


def spliced_batch(tok, token_ids: list[int], device, task: str = "repetition"):
    """[prefix ids] + [token id] + [suffix ids] for each token; constant length."""
    pre, suf = template_ids(tok, task)
    rows = [pre + [t] + suf for t in token_ids]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def token_position(tok, task: str = "repetition") -> int:
    """Index of the token under test inside the spliced prompt."""
    pre, _ = template_ids(tok, task)
    return len(pre)


def is_repetition_correct(target: str, generated: str) -> bool:
    """Both papers: model must reproduce the token string in its output."""
    t = target.strip()
    if not t:
        # whitespace-only tokens: require the raw string to appear
        return target in generated if target else False
    return t in generated
