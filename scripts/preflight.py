"""PREFLIGHT: validate a model's configuration BEFORE spending GPU hours on it.

Why this exists
---------------
`src/glitchprober/features.py` and `repair.py` resolve MLP submodules by the
dotted names in configs/models.yaml (`module_map`). When a name does not resolve
the code catches AttributeError and CONTINUES with that stream missing. A wrong
module_map therefore does not crash - it silently produces a run whose numbers
look plausible and mean nothing. The same is true of `layers_path`, and of
`n_layers`/`key_layers` that do not match the real architecture.

This script instantiates the model on the META DEVICE (structure only, no weights,
no VRAM, no download beyond config.json) and asserts every structural assumption
the pipeline will make. Run it for every new model before any experiment.

    python scripts/preflight.py --model qwen25-7b-instruct
    python scripts/preflight.py --all-enabled

Exit code 0 = safe to run. Non-zero = do not run; fix the config first.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import get_model_cfg, setup_env


def check(name: str, verbose: bool = True) -> tuple[bool, list[str]]:
    """Return (ok, problems). Prints a structural report when verbose."""
    problems: list[str] = []
    mcfg = get_model_cfg(name)
    hf_id = mcfg["hf_id"]

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print(f"\n{'=' * 78}\n{name}  ->  {hf_id}\n{'=' * 78}")

    # ---- 1. config: gated access, layer count, vocab -------------------------
    try:
        hfcfg = AutoConfig.from_pretrained(
            hf_id, trust_remote_code=mcfg.get("trust_remote_code", False))
    except Exception as e:
        problems.append(f"cannot load config ({type(e).__name__}): "
                        f"{str(e).splitlines()[0][:120]}")
        if verbose:
            print(f"  FAIL  {problems[-1]}")
        return False, problems

    real_layers = getattr(hfcfg, "num_hidden_layers", None)
    real_vocab = getattr(hfcfg, "vocab_size", None)
    arch = getattr(hfcfg, "architectures", ["?"])[0]
    if verbose:
        print(f"  architecture      : {arch}")
        print(f"  hidden layers     : {real_layers}")
        print(f"  intermediate size : {getattr(hfcfg, 'intermediate_size', '?')}")
        print(f"  config vocab_size : {real_vocab}")

    if mcfg.get("n_layers") != real_layers:
        problems.append(f"n_layers in models.yaml is {mcfg.get('n_layers')} but the "
                        f"model has {real_layers}")

    # ---- 2. key layers in range ---------------------------------------------
    kl = mcfg.get("key_layers") or []
    if not kl:
        problems.append("key_layers is empty")
    elif real_layers is not None and (min(kl) < 0 or max(kl) >= real_layers):
        problems.append(f"key_layers {min(kl)}-{max(kl)} out of range for "
                        f"{real_layers} layers")
    elif verbose:
        frac_lo, frac_hi = min(kl) / real_layers, (max(kl) + 1) / real_layers
        print(f"  key layers        : {min(kl)}-{max(kl)} "
              f"({frac_lo:.0%}-{frac_hi:.0%} depth, {len(kl)} layers)")

    # ---- 3. tokenizer: real vocab length (what the sweep iterates) ----------
    try:
        tok = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=mcfg.get("trust_remote_code", False))
        tok_len = len(tok)
        if verbose:
            print(f"  len(tokenizer)    : {tok_len}"
                  + ("" if tok_len == real_vocab else
                     f"   (differs from config vocab_size {real_vocab})"))
        # the census iterates range(len(tok)); ids >= config vocab would be invalid
        if real_vocab is not None and tok_len > real_vocab:
            problems.append(f"len(tokenizer)={tok_len} exceeds config vocab_size="
                            f"{real_vocab}; census would probe invalid ids")
        # prefix-anchored filter requires the anchor to be a single token
        anchor_ids = tok.encode("«", add_special_tokens=False)
        if len(anchor_ids) != 1:
            problems.append(f"filter anchor '<<' is {len(anchor_ids)} tokens for this "
                            f"tokenizer, not 1; ContextualCodec assumes 1")
        elif verbose:
            print(f"  filter anchor     : single token (id {anchor_ids[0]}) OK")
    except Exception as e:
        problems.append(f"tokenizer failed ({type(e).__name__}): "
                        f"{str(e).splitlines()[0][:120]}")

    # ---- 4. structure on meta device: resolve every module the code needs ----
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                hfcfg, trust_remote_code=mcfg.get("trust_remote_code", False))
    except Exception as e:
        problems.append(f"cannot instantiate structure ({type(e).__name__}): "
                        f"{str(e).splitlines()[0][:120]}")
        if verbose:
            print(f"  FAIL  {problems[-1]}")
        return False, problems

    from src.common.model_utils import decoder_layers, mlp_module

    try:
        layers = decoder_layers(model, mcfg)
        n_found = len(layers)
        if verbose:
            print(f"  layers_path       : '{mcfg.get('layers_path')}' -> {n_found} layers OK")
        if real_layers is not None and n_found != real_layers:
            problems.append(f"layers_path resolves {n_found} layers, config says {real_layers}")
    except AttributeError as e:
        problems.append(f"layers_path '{mcfg.get('layers_path')}' does not resolve: {e}")
        if verbose:
            print(f"  FAIL  {problems[-1]}")
            sample = [n for n, _ in model.named_modules()][:40]
            print(f"        module names begin: {sample[:12]}")
        return False, problems

    # every module_map entry must resolve on a key layer
    probe_layer = layers[kl[0]] if kl else layers[0]
    for key in ("gate", "up", "act", "down"):
        dotted = mcfg.get("module_map", {}).get(key)
        if dotted is None:
            problems.append(f"module_map has no '{key}' entry")
            continue
        try:
            mod = mlp_module(probe_layer, mcfg, key)
            shape = ""
            if hasattr(mod, "in_features"):
                shape = f"  [{mod.in_features} -> {mod.out_features}]"
            if verbose:
                print(f"  module_map.{key:<5}: '{dotted}' -> {type(mod).__name__}{shape} OK")
        except AttributeError:
            problems.append(f"module_map.{key} = '{dotted}' does NOT resolve "
                            f"(this would be SILENTLY SKIPPED at runtime)")
            if verbose:
                mlp = getattr(probe_layer, "mlp", None)
                avail = [n for n, _ in mlp.named_children()] if mlp is not None else \
                        [n for n, _ in probe_layer.named_children()]
                print(f"  FAIL  module_map.{key}='{dotted}' unresolved; "
                      f"available: {avail}")

    # gate/up must have matching output width (they are multiplied elementwise)
    try:
        g = mlp_module(probe_layer, mcfg, "gate")
        u = mlp_module(probe_layer, mcfg, "up")
        if hasattr(g, "out_features") and hasattr(u, "out_features"):
            if g.out_features != u.out_features:
                problems.append(f"gate out_features {g.out_features} != up "
                                f"{u.out_features}; they are multiplied elementwise")
    except AttributeError:
        pass

    # ---- 5. attention-pattern extraction (detection needs eager) ------------
    if verbose:
        heads = getattr(hfcfg, "num_attention_heads", "?")
        kv = getattr(hfcfg, "num_key_value_heads", heads)
        print(f"  attention heads   : {heads} (kv {kv})")

    ok = not problems
    if verbose:
        print(f"\n  RESULT: {'PASS - safe to run' if ok else 'FAIL - do NOT run'}")
        for p in problems:
            print(f"    - {p}")
    return ok, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all-enabled", action="store_true")
    args = ap.parse_args()

    cfg = setup_env()
    if args.all_enabled:
        names = [n for n, m in cfg["models"].items() if m.get("enabled")]
    elif args.model:
        names = [args.model]
    else:
        ap.error("give --model NAME or --all-enabled")

    results = {n: check(n)[0] for n in names}
    print(f"\n{'=' * 78}\nSUMMARY")
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
