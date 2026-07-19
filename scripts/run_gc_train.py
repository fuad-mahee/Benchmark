"""RQ4: train GlitchCleaner's gated LoRA on the TRAIN split of glitch tokens.

  python scripts/run_gc_train.py --model smoke-test
Requires ground truth. Saves adapter + splits to results/gc/<model>/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import get_model_cfg, load_yaml, results_dir
from src.common.io_utils import Timer, run_metadata, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=["paper", "gccode"], default="paper")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gc = load_yaml("glitchcleaner.yaml")
    task = "repetition" if args.protocol == "paper" else "repetition_gccode"

    gt_base = results_dir("ground_truth", args.model)
    gt_path = (gt_base / "tokens.csv") if args.protocol == "paper" else (gt_base / "gccode" / "tokens.csv")
    df = pd.read_csv(gt_path)
    glitch = df[df["category"] == "glitch"]["token_id"].tolist()
    if len(glitch) < 5:
        sys.exit(f"only {len(glitch)} glitch tokens - not enough to train on")

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchcleaner.build_dataset import build_examples, save_jsonl, split_glitch_tokens
    from src.glitchcleaner.train_lora import train

    tok = load_tokenizer(mcfg)
    train_ids, heldout_ids = split_glitch_tokens(glitch, gc["holdout_fraction"], gc["seed"])
    out = (results_dir("gc", args.model) if args.protocol == "paper"
           else results_dir("gc", args.model, "gccode"))
    save_jsonl(build_examples(tok, train_ids, task), out / "train.jsonl")
    save_jsonl(build_examples(tok, heldout_ids, task), out / "heldout.jsonl")
    save_json({"train_ids": train_ids, "heldout_ids": heldout_ids}, out / "split.json")

    model = load_model(mcfg, attn_impl=None, for_training=True)
    examples = build_examples(tok, train_ids, task)
    with Timer() as t:
        train(model, tok, examples, mcfg, gc, out / "adapter")

    save_json(run_metadata(model=args.model, config=gc, protocol=args.protocol,
                           n_train=len(train_ids),
                           n_heldout=len(heldout_ids), train_seconds=t.seconds),
              out / "train_meta.json")


if __name__ == "__main__":
    main()
