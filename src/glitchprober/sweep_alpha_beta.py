"""RQ3 sensitivity analysis: grid-sweep fixed alpha/beta (and optionally m).

Answers: is the paper's rule-based alpha=4/beta=1.5 special? Is the landscape flat
(mechanism works, exact values don't matter) or peaked elsewhere (values suboptimal)?
Each cell reports repair rate AND collateral damage on normal tokens.
"""
import pandas as pd

from .repair import evaluate_repair


def grid_sweep(model, tok, glitch_sample, normal_sample, mcfg, stats,
               alphas, betas, batch_size, max_new_tokens) -> pd.DataFrame:
    rows = []
    for a in alphas:
        for b in betas:
            r = evaluate_repair(
                model, tok, glitch_sample, normal_sample, mcfg, stats,
                {"alpha": a, "beta": b}, batch_size, max_new_tokens,
            )
            rows.append({"alpha": a, "beta": b,
                         "repair_rate": r["repair_rate"],
                         "normal_break_rate": r["normal_break_rate"]})
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
