"""RQ1: full-vocabulary glitch census.

  python scripts/run_ground_truth.py --model smoke-test --limit 2000   (pipeline check)
  python scripts/run_ground_truth.py --model llama2-7b-chat            (real run)

Outputs results/ground_truth/<model>/tokens.csv + summary.json.
Every later experiment reads tokens.csv, so run this first per model.
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
    ap.add_argument("--limit", type=int, default=None, help="only first N vocab ids (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]

    from src.common.model_utils import load_model, load_tokenizer
    from src.ground_truth.filter_tokens import classify_vocab
    from src.ground_truth.sweep import repetition_sweep

    tok = load_tokenizer(mcfg)
    print(f"vocab size: {len(tok)}")
    classified = classify_vocab(tok, limit=args.limit)
    candidates = [tid for tid, _, cat in classified if cat == "candidate"]
    print(f"candidates after filtering: {len(candidates)}")

    out = results_dir("ground_truth", args.model)
    model = load_model(mcfg, attn_impl=None)  # fastest available attn for generation
    with Timer() as t:
        results = repetition_sweep(
            model, tok, candidates, batch, gt_cfg["max_new_tokens"], desc="RQ1 sweep",
            checkpoint=out / "sweep_checkpoint.csv",  # batch-level resume + full raw outputs
        )

    rows = []
    for tid, s, cat in classified:
        if cat == "candidate":
            ok, text = results[tid]
            cat = "normal" if ok else "glitch"
            rows.append({"token_id": tid, "token": s, "category": cat,
                         "output_snippet": text[:80]})
        else:
            rows.append({"token_id": tid, "token": s, "category": cat, "output_snippet": ""})
    df = pd.DataFrame(rows)

    df.to_csv(out / "tokens.csv", index=False, encoding="utf-8")
    counts = df["category"].value_counts().to_dict()
    save_json(
        run_metadata(model=args.model, hf_id=mcfg["hf_id"], limit=args.limit,
                     counts=counts, sweep_seconds=t.seconds,
                     paper_reference={"glitchprober_llama2": 6425, "glitchcleaner_llama2": 4743}),
        out / "summary.json",
    )
    print(f"\ncategory counts: {counts}")


if __name__ == "__main__":
    main()
