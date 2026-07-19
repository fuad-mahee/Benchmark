"""RQ3 sensitivity analysis: grid-sweep fixed alpha/beta (and optionally m).

Answers: is the paper's rule-based alpha=4/beta=1.5 special? Is the landscape flat
(mechanism works, exact values don't matter) or peaked elsewhere (values suboptimal)?
Each cell reports repair rate AND collateral damage on normal tokens.
"""
from pathlib import Path

import pandas as pd

from .repair import evaluate_repair


def grid_sweep(model, tok, glitch_sample, normal_sample, mcfg, stats,
               alphas, betas, batch_size, max_new_tokens,
               csv_path: Path | str | None = None) -> pd.DataFrame:
    """Cell-level checkpointing: each completed (alpha, beta) cell is written to
    csv_path immediately; on restart, already-computed cells are skipped."""
    rows, done = [], set()
    if csv_path and Path(csv_path).exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = {(r["alpha"], r["beta"]) for r in rows}
        if done:
            print(f"grid sweep: resumed {len(done)} cells from checkpoint")
    for a in alphas:
        for b in betas:
            if (float(a), float(b)) in done:
                continue
            r = evaluate_repair(
                model, tok, glitch_sample, normal_sample, mcfg, stats,
                {"alpha": a, "beta": b}, batch_size, max_new_tokens,
            )
            rows.append({"alpha": float(a), "beta": float(b),
                         "repair_rate": r["repair_rate"],
                         "normal_break_rate": r["normal_break_rate"]})
            if csv_path:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"alpha={a} beta={b}: repair={r['repair_rate']:.3f} "
                  f"break={r['normal_break_rate']:.3f}")
    return pd.DataFrame(rows)


def save_heatmap(df: pd.DataFrame, value: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="alpha", columns="beta", values=value)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(value)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
