"""RQ5: tokens/sec - base vs GP repair hooks vs GC adapter (ratios, not absolutes).

  python scripts/run_speed.py --model smoke-test
GP hooks need gp_repair stats; GC needs a trained adapter. Missing pieces are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import get_model_cfg, load_yaml, results_dir
from src.common.io_utils import run_metadata, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")["repair"]

    from src.common.model_utils import load_model, load_tokenizer
    from src.eval.speed import tokens_per_second

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)
    rates = {"base": tokens_per_second(model, tok)}
    print(f"base: {rates['base']:.2f} tok/s")

    # GP hooks active (recompute lightweight stats if repair results absent)
    gt_path = results_dir("ground_truth", args.model) / "tokens.csv"
    if gt_path.exists():
        from src.glitchprober.repair import RepairHooks, compute_neuron_stats
        df = pd.read_csv(gt_path)
        normal = df[df["category"] == "normal"]["token_id"].tolist()
        rng = np.random.default_rng(0)
        sample = list(rng.choice(normal, size=min(200, len(normal)), replace=False))
        stats = compute_neuron_stats(model, tok, sample, mcfg, gp["m"],
                                     gp["neun_up_quantile"], mcfg["batch_size"])
        with RepairHooks(model, mcfg, stats, gp["rule_based"]):
            rates["gp_hooks"] = tokens_per_second(model, tok)
        print(f"gp_hooks: {rates['gp_hooks']:.2f} tok/s")

    adapter = results_dir("gc", args.model) / "adapter"
    if (adapter / "adapter_config.json").exists():
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model, str(adapter))
        peft_model.eval()
        rates["gc_adapter"] = tokens_per_second(peft_model, tok)
        print(f"gc_adapter: {rates['gc_adapter']:.2f} tok/s")

    rel = {k: v / rates["base"] for k, v in rates.items()}
    save_json(run_metadata(model=args.model, tokens_per_second=rates, relative=rel,
                           paper_claims_tok_s={"base": 66.30, "gc": 62.83, "gp": 11.82}),
              results_dir("side_effects", args.model) / "speed.json")


if __name__ == "__main__":
    main()
