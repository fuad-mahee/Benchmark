"""RQ2: GlitchProber detection - recall/precision/F1/time vs our ground truth.

  python scripts/run_gp_detect.py --model smoke-test [--seed 0]
Requires run_ground_truth.py to have been run for the model first.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import get_model_cfg, load_yaml, results_dir
from src.common.io_utils import run_metadata, save_json


def load_ground_truth(model_name: str):
    p = results_dir("ground_truth", model_name) / "tokens.csv"
    if not p.exists():
        sys.exit(f"missing {p} - run run_ground_truth.py --model {model_name} first")
    df = pd.read_csv(p)
    candidates = df[df["category"].isin(["normal", "glitch"])]["token_id"].tolist()
    true_glitch = set(df[df["category"] == "glitch"]["token_id"].tolist())
    return candidates, true_glitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: all seeds in config")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")
    gt = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]
    seeds = [args.seed] if args.seed is not None else gp["seeds"]

    candidates, true_glitch = load_ground_truth(args.model)
    print(f"{len(candidates)} candidates, {len(true_glitch)} true glitch tokens")

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchprober.detect import run_detection

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl="eager")  # eager: attention patterns needed

    out = results_dir("gp_detect", args.model)
    all_runs = []
    for seed in seeds:
        r = run_detection(model, tok, mcfg, candidates, true_glitch,
                          gp["detection"], seed, batch, gt["max_new_tokens"])
        all_runs.append(r)
        print(f"seed {seed}: P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} time={r['time_seconds']:.0f}s")

    df = pd.DataFrame(all_runs)
    df.to_csv(out / "runs.csv", index=False)
    save_json(
        run_metadata(model=args.model, config=gp["detection"],
                     mean={"precision": df["precision"].mean(), "recall": df["recall"].mean(),
                           "f1": df["f1"].mean(), "time_seconds": df["time_seconds"].mean()},
                     std={"recall": df["recall"].std(), "f1": df["f1"].std()}),
        out / "summary.json",
    )


if __name__ == "__main__":
    main()
