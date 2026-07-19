"""RQ3: GlitchProber repair - adaptive vs rule-based alpha/beta, with collateral checks.

  python scripts/run_gp_repair.py --model smoke-test [--mode both|adaptive|rule]
Requires ground truth. Use --sample to cap glitch/normal set sizes for quick runs.
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
    ap.add_argument("--mode", choices=["both", "adaptive", "rule"], default="both")
    ap.add_argument("--sample", type=int, default=None, help="cap eval set sizes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--protocol", choices=["paper", "gccode"], default="paper")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")["repair"]
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]

    if args.protocol == "gccode":
        task, max_new = "repetition_gccode", 10

        def correct_fn(tok, tid, text):
            return tok.decode([tid]).lstrip() in text
    else:
        task, max_new, correct_fn = "repetition", gt_cfg["max_new_tokens"], None

    gt_base = results_dir("ground_truth", args.model)
    gt_path = (gt_base / "tokens.csv") if args.protocol == "paper" else (gt_base / "gccode" / "tokens.csv")
    df = pd.read_csv(gt_path)
    glitch = df[df["category"] == "glitch"]["token_id"].tolist()
    normal = df[df["category"] == "normal"]["token_id"].tolist()
    rng = np.random.default_rng(args.seed)
    normal_stats_sample = list(rng.choice(normal, size=max(2, int(len(normal) * gp["gamma"])), replace=False))
    glitch_eval = glitch if not args.sample else list(rng.choice(glitch, size=min(args.sample, len(glitch)), replace=False))
    normal_eval = list(rng.choice(normal, size=min(args.sample or 500, len(normal)), replace=False))

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchprober.repair import (compute_adjustments, compute_neuron_stats,
                                         evaluate_repair)

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)

    stats = compute_neuron_stats(model, tok, normal_stats_sample, mcfg,
                                 gp["m"], gp["neun_up_quantile"], batch, task)
    for li, s in stats.items():
        print(f"layer {li}: |Neun_up|={len(s['neun_up'])} |Neun_down|={len(s['neun_down'])}")

    out = (results_dir("gp_repair", args.model) if args.protocol == "paper"
           else results_dir("gp_repair", args.model, "gccode"))
    results = {}
    if args.mode in ("both", "adaptive"):
        glitch_stats_sample = list(rng.choice(glitch, size=min(len(glitch), len(normal_stats_sample)), replace=False))
        adj = compute_adjustments(model, tok, glitch_stats_sample, mcfg, stats,
                                  gp["adaptive"], batch, task)
        print("adaptive per-layer alpha/beta:", {k: {x: round(y, 3) for x, y in v.items()} for k, v in adj.items()})
        results["adaptive"] = evaluate_repair(model, tok, glitch_eval, normal_eval, mcfg,
                                              stats, adj, batch, max_new, task, correct_fn)
    if args.mode in ("both", "rule"):
        rb = gp["rule_based"]
        results["rule_based"] = evaluate_repair(model, tok, glitch_eval, normal_eval, mcfg,
                                                stats, rb, batch, max_new, task, correct_fn)

    for k, v in results.items():
        print(f"{k}: repair_rate={v['repair_rate']:.3f} normal_break_rate={v['normal_break_rate']:.3f}")

    save_json(run_metadata(model=args.model, seed=args.seed, config=gp,
                           protocol=args.protocol, sample_cap=args.sample, results=results,
                           paper_claims={"adaptive_avg": 0.5006, "rule_based_avg": 0.3679}),
              out / "summary.json")


if __name__ == "__main__":
    main()
