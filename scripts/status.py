"""Pipeline status board: which step is DONE / PARTIAL / pending, per model.

  python scripts/status.py

This is the resume map after any interruption: run the first non-DONE step for a
model and it continues from its own checkpoint files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import RESULTS_DIR, setup_env

STEPS = [
    ("1 ground_truth", "ground_truth/{m}/tokens.csv", "ground_truth/{m}/sweep_checkpoint.csv"),
    ("2 gp_detect", "gp_detect/{m}/summary.json", "gp_detect/{m}/checkpoints"),
    ("3 gp_repair", "gp_repair/{m}/summary.json", None),
    ("3b alpha_beta", "gp_repair/{m}/heatmap_repair_rate.png", "gp_repair/{m}/alpha_beta_grid.csv"),
    ("4 gc_train", "gc/{m}/train_meta.json", None),
    ("4b gc_eval", "gc/{m}/eval.json", None),
    ("5 speed", "side_effects/{m}/speed.json", None),
]


def main():
    cfg = setup_env()
    models = list(cfg["models"])
    width = max(len(m) for m in models) + 2
    header = "model".ljust(width) + "".join(name.ljust(16) for name, *_ in STEPS)
    print(header)
    print("-" * len(header))
    for m in models:
        row = m.ljust(width)
        for _name, done_rel, partial_rel in STEPS:
            done = (RESULTS_DIR / done_rel.format(m=m)).exists()
            partial = partial_rel and (RESULTS_DIR / partial_rel.format(m=m)).exists()
            row += ("DONE" if done else "PARTIAL" if partial else "-").ljust(16)
        print(row)
    print("\nPARTIAL = checkpoint exists; rerunning that step resumes from it.")


if __name__ == "__main__":
    main()
