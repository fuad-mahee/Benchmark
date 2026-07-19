"""Batched repetition-task sweep: the operational glitch-token test of both papers.

Greedy decoding (temperature 0), constant-length spliced prompts, batched generate.
Reused by: RQ1 census, GlitchProber sample labeling + post-validation,
GlitchProber repair evaluation (with hooks active), GlitchCleaner evaluation
(with adapter active).

Checkpointing: pass `checkpoint=<csv path>` and every processed batch is appended
to that CSV immediately. On restart the CSV is loaded and already-done tokens are
skipped, so an interrupted sweep resumes at the exact batch it stopped. The
checkpoint also preserves the FULL generated text per token (tokens.csv only keeps
a snippet), which is useful as qualitative examples in the thesis.
"""
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ..common.model_utils import token_str
from ..common.prompts import spliced_batch, is_repetition_correct


def _load_checkpoint(path: Path) -> dict[int, tuple[bool, str]]:
    done = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done[int(row["token_id"])] = (row["ok"] == "1", row["text"])
    return done


@torch.no_grad()
def repetition_sweep(
    model,
    tok,
    token_ids: list[int],
    batch_size: int = 32,
    max_new_tokens: int = 24,
    task: str = "repetition",
    desc: str = "sweep",
    checkpoint: Path | str | None = None,
    correct_fn=None,
) -> dict[int, tuple[bool, str]]:
    """Return {token_id: (is_correct, generated_text)}. is_correct=False => glitch behavior.

    correct_fn(tok, token_id, generated_text) -> bool overrides the default check
    (used by the gccode protocol to replicate GlitchCleaner's exact judgment)."""
    device = next(model.parameters()).device
    results: dict[int, tuple[bool, str]] = {}
    todo = list(token_ids)

    writer, fh = None, None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            results = _load_checkpoint(checkpoint)
            todo = [t for t in token_ids if t not in results]
            if len(todo) < len(token_ids):
                print(f"{desc}: resumed {len(token_ids) - len(todo)} tokens from "
                      f"checkpoint, {len(todo)} remaining")
        is_new = not checkpoint.exists()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fh = open(checkpoint, "a", newline="", encoding="utf-8")
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        if is_new:
            writer.writerow(["token_id", "ok", "text"])

    try:
        for i in tqdm(range(0, len(todo), batch_size), desc=desc):
            chunk = todo[i : i + batch_size]
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
                if correct_fn is not None:
                    ok = correct_fn(tok, tid, text)
                else:
                    ok = is_repetition_correct(token_str(tok, tid), text)
                results[tid] = (ok, text)
                if writer:
                    writer.writerow([tid, "1" if ok else "0", text])
            if fh:
                fh.flush()
    finally:
        if fh:
            fh.close()

    return {t: results[t] for t in token_ids if t in results}


def split_glitch_normal(results: dict[int, tuple[bool, str]]) -> tuple[list[int], list[int]]:
    glitch = [t for t, (ok, _) in results.items() if not ok]
    normal = [t for t, (ok, _) in results.items() if ok]
    return glitch, normal
