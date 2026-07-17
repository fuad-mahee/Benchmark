"""Evaluate GlitchCleaner: repair on TRAIN split vs HELD-OUT split (the paper only
reports the former population). Also collateral checks with the adapter forced on.
"""
from ..ground_truth.sweep import repetition_sweep


def _rate(results):
    ok = sum(1 for k, _ in results.values() if k)
    return ok, ok / len(results) if results else 0.0


def evaluate(peft_model, tok, train_glitch, heldout_glitch, normal_sample,
             batch_size, max_new_tokens) -> dict:
    out = {}

    # lambda = 1 (adapter on): the repair path
    tr = repetition_sweep(peft_model, tok, train_glitch, batch_size, max_new_tokens, desc="GC train split")
    ho = repetition_sweep(peft_model, tok, heldout_glitch, batch_size, max_new_tokens, desc="GC heldout split")
    n_on = repetition_sweep(peft_model, tok, normal_sample, batch_size, max_new_tokens, desc="GC normal (adapter on)")

    out["train_repaired"], out["train_repair_rate"] = _rate(tr)
    out["heldout_repaired"], out["heldout_repair_rate"] = _rate(ho)
    ok, rate = _rate(n_on)
    out["normal_ok_adapter_on"], out["normal_ok_rate_adapter_on"] = ok, rate

    # lambda = 0 (adapter off): sanity - glitches should stay broken, normals fine
    with peft_model.disable_adapter():
        ho_off = repetition_sweep(peft_model, tok, heldout_glitch, batch_size, max_new_tokens,
                                  desc="GC heldout (adapter off)")
    out["heldout_repaired_adapter_off"], out["heldout_repair_rate_adapter_off"] = _rate(ho_off)

    out["n_train"], out["n_heldout"], out["n_normal"] = len(train_glitch), len(heldout_glitch), len(normal_sample)
    out["paper_claim_repair_rate_avg"] = 0.8688
    return out
