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
    ap.add_argument("--protocol", choices=["paper", "gccode"], default="paper",
                    help="paper = template as written in the papers, 24 new tokens; "
                         "gccode = GlitchCleaner's released code verbatim (their "
                         "template, 10 new tokens, decode-lstrip containment check)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the protocol's generation budget. Used to "
                         "separate the effect of prompt wording from the effect "
                         "of generation length (the 2x2 factorial in RQ1).")
    ap.add_argument("--tag", default=None,
                    help="subfolder for factorial/ablation cells")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None,
                    help="attention kernel. The census defaulted to SDPA while "
                         "detection requires eager (for attention patterns); the "
                         "two kernels disagree on a small fraction of greedy "
                         "generations, which showed up as spurious detection "
                         "false positives. Set eager to match detection exactly.")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]

    if args.protocol == "gccode":
        task = "repetition_gccode"
        max_new = 10

        def correct_fn(tok, tid, text):  # replicates count_passed_glitchtokens
            return tok.decode([tid]).lstrip() in text
    else:
        task, max_new, correct_fn = "repetition", gt_cfg["max_new_tokens"], None
    if args.max_new_tokens is not None:
        max_new = args.max_new_tokens

    from src.common.model_utils import load_model, load_tokenizer
    from src.ground_truth.filter_tokens import classify_vocab
    from src.ground_truth.sweep import repetition_sweep

    tok = load_tokenizer(mcfg)
    print(f"vocab size: {len(tok)}")
    classified = classify_vocab(tok, limit=args.limit)
    candidates = [tid for tid, _, cat in classified if cat == "candidate"]
    print(f"candidates after filtering: {len(candidates)}")

    out = results_dir("ground_truth", args.model)
    if args.protocol == "gccode":
        out = out / "gccode"
    if args.tag:
        out = out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(mcfg, attn_impl=args.attn_impl)
    with Timer() as t:
        results = repetition_sweep(
            model, tok, candidates, batch, max_new, task=task, desc="RQ1 sweep",
            checkpoint=out / "sweep_checkpoint.csv",  # batch-level resume + full raw outputs
            correct_fn=correct_fn,
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
                     protocol=args.protocol, max_new_tokens=max_new,
                     attn_impl=args.attn_impl or "default(sdpa)",
                     counts=counts, sweep_seconds=t.seconds,
                     paper_reference={"glitchprober_llama2": 6425, "glitchcleaner_llama2": 4743}),
        out / "summary.json",
    )
    print(f"\ncategory counts: {counts}")


if __name__ == "__main__":
    main()
