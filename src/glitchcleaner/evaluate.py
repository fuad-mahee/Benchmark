"""Evaluate GlitchCleaner: repair on the TRAIN split vs the HELD-OUT split.

The paper reports repair only on the population the adapter was trained on. The
held-out column is the generalisation test. Three controls accompany it:

  * adapter disabled on held-out glitch tokens  -> must be ~0, else the "repair"
    is not coming from the LoRA at all;
  * clean (normal) tokens WITH the paper's lambda gate active -> tests the
    "lossless" claim as the method is actually specified;
  * clean tokens with the adapter forced on (gate bypassed) -> shows what the
    gate is protecting against, i.e. how much of losslessness is the gate's doing
    rather than the adapter's benignity.

The distinction in the last two matters: v1 of this benchmark reported only the
forced-on number and described it as GlitchCleaner's behaviour, which overstated
the collateral damage of the method as published.
"""
from ..ground_truth.sweep import repetition_sweep


def _rate(results):
    ok = sum(1 for k, _ in results.values() if k)
    return ok, (ok / len(results) if results else 0.0)


def evaluate(peft_model, tok, train_glitch, heldout_glitch, normal_sample,
             batch_size, max_new_tokens, task="repetition", correct_fn=None,
             glitch_ids_for_gate=None) -> dict:
    """glitch_ids_for_gate: the set G the lambda gate consults. If None, the gate
    is not applied (adapter always on) and only the forced-on numbers are valid."""
    out = {}

    def sweep(model, ids, desc):
        return repetition_sweep(model, tok, ids, batch_size, max_new_tokens,
                                task=task, desc=desc, correct_fn=correct_fn)

    # --- repair path: prompts containing a known glitch token => lambda = 1 ---
    tr = sweep(peft_model, train_glitch, "GC train split")
    ho = sweep(peft_model, heldout_glitch, "GC heldout split")
    out["train_repaired"], out["train_repair_rate"] = _rate(tr)
    out["heldout_repaired"], out["heldout_repair_rate"] = _rate(ho)

    # --- control: adapter off => repair must collapse ---
    with peft_model.disable_adapter():
        ho_off = sweep(peft_model, heldout_glitch, "GC heldout (adapter off)")
    out["heldout_repaired_adapter_off"], out["heldout_repair_rate_adapter_off"] = _rate(ho_off)

    # --- clean inputs, adapter forced on (gate bypassed) ---
    n_forced = sweep(peft_model, normal_sample, "GC normal (adapter forced on)")
    out["normal_ok_adapter_forced_on"], out["normal_ok_rate_adapter_forced_on"] = _rate(n_forced)

    # --- clean inputs WITH the paper's gate: lambda = 0, so the base model runs ---
    if glitch_ids_for_gate is not None:
        from .gate import GatedModelAdapter
        gated = GatedModelAdapter(peft_model, glitch_ids_for_gate)
        with gated:
            n_gated = sweep(gated, normal_sample, "GC normal (gated, lambda=0)")
            g_gated = sweep(gated, heldout_glitch, "GC heldout (gated, lambda=1)")
        out["normal_ok_gated"], out["normal_ok_rate_gated"] = _rate(n_gated)
        out["heldout_repaired_gated"], out["heldout_repair_rate_gated"] = _rate(g_gated)
        out["gate_stats"] = gated.gate.stats()

    out["n_train"] = len(train_glitch)
    out["n_heldout"] = len(heldout_glitch)
    out["n_normal"] = len(normal_sample)
    out["paper_claim_repair_rate_mistral"] = 0.9480
    out["paper_claim_repair_rate_avg"] = 0.8688
    return out
