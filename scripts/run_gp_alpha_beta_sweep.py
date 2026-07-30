"""RQ3 sensitivity: the alpha/beta grid (and optionally the m threshold) + heatmaps.

  python scripts/run_gp_alpha_beta_sweep.py --model mistral-7b-instruct-v01
  python scripts/run_gp_alpha_beta_sweep.py --model mistral-7b-instruct-v01 --sweep m

This is the "why these values and not others?" experiment: the paper fixes
alpha=4, beta=1.5 for its rule-based baseline and m=1 for neuron selection
without justifying any of them.
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--protocol", choices=["paper", "gccode"], default="paper")
    ap.add_argument("--sweep", choices=["alpha_beta", "m"], default="alpha_beta")
    ap.add_argument("--stream-mode", choices=["separate", "product"], default=None)
    ap.add_argument("--position", choices=["token", "last"], default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")
    sw, rp = gp["alpha_beta_sweep"], gp["repair"]
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]
    stream_mode = args.stream_mode or rp.get("stream_mode", "separate")
    position = args.position or rp.get("position", "token")

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
    glitch_sample = list(rng.choice(glitch, size=min(sw["n_glitch_sample"], len(glitch)), replace=False))
    normal_sample = list(rng.choice(normal, size=min(sw["n_normal_sample"], len(normal)), replace=False))
    stats_sample = list(rng.choice(normal, size=max(2, int(len(normal) * rp["gamma"])), replace=False))

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchprober.repair import (compute_neuron_stats, evaluate_repair,
                                         neuron_stat_summary)
    from src.glitchprober.sweep_alpha_beta import grid_sweep, save_heatmap

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)

    out = results_dir("gp_repair", args.model)
    if args.protocol == "gccode":
        out = out / "gccode"
    if args.tag:
        out = out / args.tag
    out.mkdir(parents=True, exist_ok=True)

    if args.sweep == "alpha_beta":
        stats = compute_neuron_stats(model, tok, stats_sample, mcfg, rp["m"],
                                     rp["neun_up_quantile"], batch, task, position, stream_mode)
        print("neuron selection:", neuron_stat_summary(stats))
        grid = grid_sweep(model, tok, glitch_sample, normal_sample, mcfg, stats,
                          sw["alphas"], sw["betas"], batch, max_new,
                          csv_path=out / "alpha_beta_grid.csv",
                          task=task, correct_fn=correct_fn,
                          position=position, stream_mode=stream_mode)
        save_heatmap(grid, "repair_rate", out / "heatmap_repair_rate.png")
        save_heatmap(grid, "normal_break_rate", out / "heatmap_normal_break_rate.png")
        best = grid.loc[grid["repair_rate"].idxmax()]
        print(f"best cell: alpha={best['alpha']} beta={best['beta']} "
              f"repair={best['repair_rate']:.4f}")
        save_json(run_metadata(model=args.model, seed=args.seed, sweep=sw,
                               protocol=args.protocol, stream_mode=stream_mode,
                               position=position,
                               neuron_selection=neuron_stat_summary(stats),
                               n_glitch=len(glitch_sample), n_normal=len(normal_sample),
                               best_cell={k: float(best[k]) for k in
                                          ("alpha", "beta", "repair_rate", "normal_break_rate")}),
                  out / "alpha_beta_grid_meta.json")
    else:
        # m controls WHICH neurons are eligible for correction at all; the paper
        # fixes m=1 with no justification. alpha/beta held at the paper's values.
        rows = []
        for m in sw["m_sweep"]:
            stats = compute_neuron_stats(model, tok, stats_sample, mcfg, m,
                                         rp["neun_up_quantile"], batch, task, position, stream_mode)
            sel = neuron_stat_summary(stats)
            n_up = sum(v["n_neun_up"] for v in sel.values())
            n_down = sum(v["n_neun_down"] for v in sel.values())
            r = evaluate_repair(model, tok, glitch_sample, normal_sample, mcfg, stats,
                                rp["rule_based"], batch, max_new, task, correct_fn,
                                position, stream_mode)
            rows.append({"m": m, "total_neun_up": n_up, "total_neun_down": n_down,
                         "repair_rate": r["repair_rate"], "repaired": r["repaired"],
                         "n_glitch": r["n_glitch"],
                         "normal_break_rate": r["normal_break_rate"]})
            print(f"m={m}: Neun_up={n_up} Neun_down={n_down} "
                  f"repair={r['repair_rate']:.4f} break={r['normal_break_rate']:.4f}")
            pd.DataFrame(rows).to_csv(out / "m_sweep.csv", index=False)
        save_json(run_metadata(model=args.model, seed=args.seed, m_sweep=sw["m_sweep"],
                               protocol=args.protocol, stream_mode=stream_mode,
                               position=position, alpha_beta=rp["rule_based"],
                               rows=rows),
                  out / "m_sweep_meta.json")


if __name__ == "__main__":
    main()
