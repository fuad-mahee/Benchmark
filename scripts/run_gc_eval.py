"""RQ4: evaluate GlitchCleaner - train-split vs HELD-OUT repair rate (circularity test).

  python scripts/run_gc_eval.py --model smoke-test
Requires run_gc_train.py first.
"""
import argparse
import json
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
    ap.add_argument("--normal-sample", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--protocol", choices=["paper", "gccode"], default="paper")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gc = load_yaml("glitchcleaner.yaml")
    batch = args.batch_size or mcfg["batch_size"]
    out = (results_dir("gc", args.model) if args.protocol == "paper"
           else results_dir("gc", args.model, "gccode"))

    if args.protocol == "gccode":
        task, max_new = "repetition_gccode", 10

        def correct_fn(tok, tid, text):
            return tok.decode([tid]).lstrip() in text
    else:
        task, max_new, correct_fn = "repetition", gc["eval"]["max_new_tokens"], None

    with open(out / "split.json", encoding="utf-8") as f:
        split = json.load(f)
    gt_base = results_dir("ground_truth", args.model)
    gt_path = (gt_base / "tokens.csv") if args.protocol == "paper" else (gt_base / "gccode" / "tokens.csv")
    df = pd.read_csv(gt_path)
    normal = df[df["category"] == "normal"]["token_id"].tolist()
    rng = np.random.default_rng(gc["seed"])
    normal_sample = list(rng.choice(normal, size=min(args.normal_sample, len(normal)), replace=False))

    from peft import PeftModel

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchcleaner.evaluate import evaluate

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)
    peft_model = PeftModel.from_pretrained(model, str(out / "adapter"))
    peft_model.eval()

    results = evaluate(peft_model, tok, split["train_ids"], split["heldout_ids"],
                       normal_sample, batch, max_new, task, correct_fn)

    print(f"\ntrain-split repair rate:   {results['train_repair_rate']:.3f}  (paper population)")
    print(f"HELD-OUT repair rate:      {results['heldout_repair_rate']:.3f}  (the honest number)")
    print(f"normal ok w/ adapter on:   {results['normal_ok_rate_adapter_on']:.3f}")
    save_json(run_metadata(model=args.model, config=gc, protocol=args.protocol,
                           results=results), out / "eval.json")


if __name__ == "__main__":
    main()
