"""RQ4: train GlitchCleaner's gated LoRA on the TRAIN split of glitch tokens.

  python scripts/run_gc_train.py --model mistral-7b-instruct-v01 [--protocol gccode]

Requires ground truth. Saves adapter + splits (including the leakage report) to
results/gc/<model>[/gccode]/.
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
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--tag", default=None, help="subfolder (e.g. for extra seeds)")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gc = load_yaml("glitchcleaner.yaml")
    if args.seed is not None:
        gc["seed"] = args.seed
    task = "repetition" if args.protocol == "paper" else "repetition_gccode"
    style = gc.get("prompt_style", "decoded")

    gt_base = results_dir("ground_truth", args.model)
    gt_path = (gt_base / "tokens.csv") if args.protocol == "paper" else (gt_base / "gccode" / "tokens.csv")
    df = pd.read_csv(gt_path)
    glitch = df[df["category"] == "glitch"]["token_id"].tolist()
    if len(glitch) < 5:
        sys.exit(f"only {len(glitch)} glitch tokens - not enough to train on")

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchcleaner.build_dataset import (build_examples, filter_leaked, save_jsonl,
                                                 split_glitch_tokens, training_token_ids)
    from src.glitchcleaner.train_lora import train

    tok = load_tokenizer(mcfg)
    train_ids, heldout_ids = split_glitch_tokens(glitch, gc["holdout_fraction"], gc["seed"])
    train_ex = build_examples(tok, train_ids, task, style)

    # Leakage control: a token whose id appears anywhere in a training sequence
    # has been seen by the adapter, so it cannot serve as an "unseen" test case.
    leaked = []
    if gc.get("exclude_leaked_from_holdout", True):
        seen = training_token_ids(tok, train_ex)
        heldout_clean, leaked = filter_leaked(heldout_ids, seen)
        print(f"held-out: {len(heldout_ids)} -> {len(heldout_clean)} clean "
              f"({len(leaked)} dropped because their id occurs in training sequences)")
    else:
        heldout_clean = heldout_ids

    out = results_dir("gc", args.model)
    if args.protocol == "gccode":
        out = out / "gccode"
    if args.tag:
        out = out / args.tag
    out.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_ex, out / "train.jsonl")
    save_jsonl(build_examples(tok, heldout_clean, task, style), out / "heldout.jsonl")
    save_json({"train_ids": train_ids, "heldout_ids": heldout_clean,
               "heldout_dropped_leaked": leaked,
               "heldout_ids_before_leak_filter": heldout_ids}, out / "split.json")

    model = load_model(mcfg, attn_impl=None, for_training=True)
    with Timer() as t:
        train(model, tok, train_ex, mcfg, gc, out / "adapter")

    save_json(run_metadata(model=args.model, config=gc, protocol=args.protocol,
                           prompt_style=style,
                           n_glitch_total=len(glitch),
                           n_train=len(train_ids), n_heldout=len(heldout_clean),
                           n_heldout_dropped_leaked=len(leaked),
                           train_seconds=t.seconds),
              out / "train_meta.json")


if __name__ == "__main__":
    main()
