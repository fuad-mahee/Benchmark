"""RQ3 sensitivity: the alpha/beta grid sweep + heatmaps.

  python scripts/run_gp_alpha_beta_sweep.py --model smoke-test
Requires ground truth. This is the "why these values and not others?" experiment.
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
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    gp = load_yaml("glitchprober.yaml")
    sw = gp["alpha_beta_sweep"]
    rp = gp["repair"]
    gt_cfg = load_yaml("ground_truth.yaml")
    batch = args.batch_size or mcfg["batch_size"]

    df = pd.read_csv(results_dir("ground_truth", args.model) / "tokens.csv")
    glitch = df[df["category"] == "glitch"]["token_id"].tolist()
    normal = df[df["category"] == "normal"]["token_id"].tolist()
    rng = np.random.default_rng(args.seed)
    glitch_sample = list(rng.choice(glitch, size=min(sw["n_glitch_sample"], len(glitch)), replace=False))
    normal_sample = list(rng.choice(normal, size=min(sw["n_normal_sample"], len(normal)), replace=False))
    stats_sample = list(rng.choice(normal, size=max(2, int(len(normal) * rp["gamma"])), replace=False))

    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchprober.repair import compute_neuron_stats
    from src.glitchprober.sweep_alpha_beta import grid_sweep, save_heatmap

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)
    stats = compute_neuron_stats(model, tok, stats_sample, mcfg, rp["m"],
                                 rp["neun_up_quantile"], batch)

    out = results_dir("gp_repair", args.model)
    grid = grid_sweep(model, tok, glitch_sample, normal_sample, mcfg, stats,
                      sw["alphas"], sw["betas"], batch, gt_cfg["max_new_tokens"],
                      csv_path=out / "alpha_beta_grid.csv")  # cell-level resume
    save_heatmap(grid, "repair_rate", out / "heatmap_repair_rate.png")
    save_heatmap(grid, "normal_break_rate", out / "heatmap_normal_break_rate.png")
    save_json(run_metadata(model=args.model, seed=args.seed, sweep=sw,
                           n_glitch=len(glitch_sample), n_normal=len(normal_sample)),
              out / "alpha_beta_grid_meta.json")


if __name__ == "__main__":
    main()
