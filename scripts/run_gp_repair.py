"""RQ3: GlitchProber repair - adaptive vs rule-based alpha/beta, with collateral checks.

  python scripts/run_gp_repair.py --model mistral-7b-instruct-v01
  python scripts/run_gp_repair.py --model mistral-7b-instruct-v01 --stream-mode product

Requires ground truth. Both variants are scored on the SAME held-out half of the
glitch set (the other half fits the adaptive factors), so the comparison is fair:
without this the adaptive variant would be fitted on the tokens it is scored on
while the fixed-value variant is not.
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
    ap.add_argument("--stream-mode", choices=["separate", "product"], default=None,
                    help="override configs/glitchprober.yaml (product = the v1 ablation)")
    ap.add_argument("--position", choices=["token", "last"], default=None)
    ap.add_argument("--tag", default=None, help="subfolder for ablation runs")
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")["repair"]
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]
    stream_mode = args.stream_mode or gp.get("stream_mode", "separate")
    position = args.position or gp.get("position", "token")

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

    # Disjoint fit / eval split of the glitch set (see module docstring).
    shuffled = list(rng.permutation(glitch))
    n_fit = int(len(shuffled) * gp.get("adaptive_fit_fraction", 0.5))
    fit_glitch, eval_glitch = shuffled[:n_fit], shuffled[n_fit:]
    if args.sample:
        eval_glitch = eval_glitch[: args.sample]
    normal_eval = list(rng.choice(normal, size=min(args.sample or 500, len(normal)), replace=False))
    print(f"glitch: {len(glitch)} total -> {len(fit_glitch)} fit / {len(eval_glitch)} eval; "
          f"normal collateral sample: {len(normal_eval)}")

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchprober.repair import (compute_adjustments, compute_neuron_stats,
                                         evaluate_repair, neuron_stat_summary)

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)

    stats = compute_neuron_stats(model, tok, normal_stats_sample, mcfg, gp["m"],
                                 gp["neun_up_quantile"], batch, task, position, stream_mode)
    neuron_summary = neuron_stat_summary(stats)
    for k, v in neuron_summary.items():
        print(f"{k}: |Neun_up|={v['n_neun_up']} |Neun_down|={v['n_neun_down']} d_mlp={v['d_mlp']}")

    out = results_dir("gp_repair", args.model)
    if args.protocol == "gccode":
        out = out / "gccode"
    if args.tag:
        out = out / args.tag
    out.mkdir(parents=True, exist_ok=True)

    results, adaptive_factors = {}, None
    if args.mode in ("both", "adaptive"):
        adj = compute_adjustments(model, tok, fit_glitch, mcfg, stats, gp["adaptive"],
                                  batch, task, position, stream_mode)
        adaptive_factors = {f"L{li}.{s}": {kk: round(vv, 6) for kk, vv in v.items()}
                            for (li, s), v in adj.items()}
        print("adaptive factors:", adaptive_factors)
        results["adaptive"] = evaluate_repair(
            model, tok, eval_glitch, normal_eval, mcfg, stats, adj, batch, max_new,
            task, correct_fn, position, stream_mode,
            record_path=out / "per_token_adaptive.csv")
    if args.mode in ("both", "rule"):
        results["rule_based"] = evaluate_repair(
            model, tok, eval_glitch, normal_eval, mcfg, stats, gp["rule_based"], batch,
            max_new, task, correct_fn, position, stream_mode,
            record_path=out / "per_token_rule_based.csv")

    for k, v in results.items():
        print(f"{k}: repair_rate={v['repair_rate']:.4f} ({v['repaired']}/{v['n_glitch']}) "
              f"normal_break_rate={v['normal_break_rate']:.4f} "
              f"({v['normal_broken']}/{v['n_normal_checked']})")

    save_json(run_metadata(model=args.model, seed=args.seed, config=gp,
                           protocol=args.protocol, stream_mode=stream_mode,
                           position=position, sample_cap=args.sample,
                           n_glitch_total=len(glitch), n_fit=len(fit_glitch),
                           n_eval=len(eval_glitch),
                           neuron_selection=neuron_summary,
                           adaptive_factors=adaptive_factors,
                           results=results,
                           paper_claims_mistral={"adaptive": 0.3760, "rule_based": 0.1292},
                           paper_claims_avg={"adaptive": 0.5006, "rule_based": 0.3679}),
              out / "summary.json")


if __name__ == "__main__":
    main()
