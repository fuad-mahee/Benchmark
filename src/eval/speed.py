"""RQ5: inference speed (tokens/sec) for base model vs GP repair hooks vs GC adapter.

Verifies GlitchCleaner's Table 6 claim (base 66.30 / GC 62.83 / GP 11.82 tok/s)
as RATIOS on our hardware.
"""
import time

import torch

PROMPT = "Write a short story about a robot who learns to paint."


@torch.no_grad()
def tokens_per_second(model, tok, n_new: int = 256, reps: int = 5) -> float:
    device = next(model.parameters()).device
    ids = tok(PROMPT, return_tensors="pt").to(device)
    # warmup
    model.generate(**ids, do_sample=False, max_new_tokens=8, pad_token_id=tok.pad_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    rates = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model.generate(**ids, do_sample=False, max_new_tokens=n_new,
                             pad_token_id=tok.pad_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rates.append((out.shape[1] - ids["input_ids"].shape[1]) / dt)
    return sum(rates) / len(rates)
