## 1. Quantitative evidence

**Artifacts examined**
- `E:/user3/FYS_SWK/Benchmark/results/gp_repair/mistral-7b-instruct-v01/alpha_beta_grid.csv` (30 cells, n=500 glitch + 500 normal each)
- `E:/user3/FYS_SWK/Benchmark/results/gp_repair/mistral-7b-instruct-v01/alpha_beta_grid_meta.json`
- `E:/user3/FYS_SWK/Benchmark/src/glitchprober/repair.py`, `E:/user3/FYS_SWK/Benchmark/src/glitchprober/sweep_alpha_beta.py`, `E:/user3/FYS_SWK/Benchmark/scripts/run_gp_alpha_beta_sweep.py`
- `E:/user3/FYS_SWK/Benchmark/docs/RUNLOG.md` lines 134-142 (Finding 6)
- `E:/user3/FYS_SWK/Benchmark/configs/glitchprober.yaml`, `E:/user3/FYS_SWK/Benchmark/configs/models.yaml`

**Repair counts (/500)**

| alpha \ beta | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 4.0 | row range |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 8 | 17 | 19 | 34 | **34 (6.8pp)** |
| 2 | 80 | 80 | 79 | 79 | 77 | 77 | 3 (0.6pp) |
| 4 | 112 | 107 | 106 | 104 | 102 | 101 | 11 (2.2pp) |
| 8 | 116 | 116 | 114 | 116 | 112 | 102 | 14 (2.8pp) |
| 16 | 119 | 120 | 115 | 116 | 111 | 102 | 18 (3.6pp) |

**Beta variation at fixed alpha** (relative = range / row max): alpha=1 → 100% (0 → 34); alpha=2 → 3.8%; alpha=4 → 9.8%; alpha=8 → 12.1%; alpha=16 → 15.0%.

**The auditor's two factual claims both verify exactly.**
- alpha=1: 0/500 at beta 0.25 and 0.5 → 34/500 at beta=4. Fisher **p = 6.5e-11**. Wilson CIs [0.000, 0.008] vs [0.049, 0.094] — disjoint. Note alpha=1 makes the alpha channel a literal identity (`x /= 1.0`), so this row is the **beta-only arm**, and the 0/500 cells are an effective no-op null control.
- Monotonicity in alpha: 5/6 beta-columns are only *weakly* monotone (ties at beta=1.5 and 4.0); beta=2.0 has an inversion, 112 → 111 at alpha 8 → 16. So "strictly monotonic" is false. The inversion is 1 token (Fisher p = 1.0) — real as a strict-monotonicity counterexample, meaningless as an effect.

**Beta's effect is not zero — it flips sign.** OLS slope of repaired-count on log2(beta): alpha=1 **+8.69** tokens/doubling; alpha=2 **−0.86**; alpha=4 **−2.70**; alpha=8 **−2.92**; alpha=16 **−4.06**. Within-row Spearman rho vs beta: **+0.986** at alpha=1, and **−0.956, −1.000, −0.820, −0.886** at alpha=2/4/8/16 (alpha=4 is *perfectly* rank-monotone decreasing across all six beta levels). Permutation test on the summed within-row rho for the alpha>=2 block: **p < 1e-5**. Pooled over alpha>=2, beta<=0.5 gives 850/4000 = 21.25% vs beta=4 gives 382/2000 = 19.10%, diff −2.15pp, Fisher p = 0.053 (and this is the *conservative* unpaired test — see §4).

**But the magnitude ordering is overwhelming.** Marginal spread: alpha **20.17pp**, beta **0.64pp** — a 31.5x ratio. Two-way variance decomposition on the full grid: alpha **97.1%**, beta main effect **0.1%**, interaction/residual 2.9%. Excluding the alpha=1 row: alpha 91.4%, beta 6.3%, residual 2.3%.

**No individual beta contrast at alpha>=2 is significant.** alpha=4: 112 vs 101, p=0.44. alpha=8: 116 vs 102, p=0.32. alpha=16: 119 vs 102, p=0.22. The unpaired 95% LSD at p≈0.22, n=500 is 5.14pp; observed beta effects are 0.6–3.6pp.

**Other Finding-6 sub-claims:** "Best cell 24.0% (alpha=16, beta=0.5)" — verified exactly. "No cell approaches 37.6%" — verified (max 0.240). "alpha=4 yields 20-22%" — verified (0.202–0.224). "Saturates at 23-24% at alpha=8-16" — true only at low beta; at beta=4 both alpha=8 and alpha=16 give 20.4%. "Collateral 1.6-2.4%" — true for alpha>=4 only; alpha=1 row is 0.0-0.4%, alpha=2 row is 1.0%.

## 2. Two provenance problems that matter more than the beta dispute

**(a) The mechanistic explanation has no artifact.** The "0-4 Neun_up neurons per layer vs ~10k Neun_down" figure is **not persisted anywhere** — `alpha_beta_grid_meta.json` contains no `neuron_selection` key, and the sweep script *as committed at the grid's commit* (706e27d) neither printed nor saved neuron counts. `neuron_stat_summary()` and the `neuron_selection=` metadata field exist only in **uncommitted working-tree code**. The number is currently a stdout recollection. It is at least *plausible* — the criterion is act > 1.0 in >3045 of 3075 sampled normal tokens over 10 key layers × d_mlp 14336 = 143,360 slots, and it does not degenerate (sample is large enough that ">99%" is not just "all") — but it is unverified.

**(b) The grid was produced by an implementation the repo has since repudiated.** `git status` is **not clean** (9 modified files, incl. `src/glitchprober/repair.py`, 148 committed lines → 283 working-tree lines). The committed code that produced the grid hooks **only `down_proj`'s input — the already-multiplied SwiGLU product** — at the **last sequence position**, and fires the correction on **every forward pass including all decode steps**. The current `configs/glitchprober.yaml` sets `stream_mode: separate` and comments that `product` was "our v1 error" and that `separate` (gate and data adjusted independently) is "the paper". Since Neun_up sparsity on the *product* stream is exactly the mechanism Finding 6 invokes, and the gate/up streams pre-multiplication are far less sparse, the beta-inertness mechanism may not survive the corrected implementation at all.

## 3. Ruling: **partially supported — the conclusion survives, the wording does not**

- "**beta is inert**" — **refuted as written.** "Inert" asserts no effect. The data show a sign-flipping, alpha-dependent effect: a large, highly significant *positive* effect when alpha is disabled (0/500 → 34/500, p=6.5e-11, from a verified zero baseline), and a small, systematic *negative* effect whenever alpha is active (permutation p < 1e-5). That is an interaction, not inertness. The auditor is right.
- "**repair is flat across beta everywhere**" — **refuted.** "Everywhere" is falsified by the alpha=1 row.
- "**rises monotonically in alpha**" — **overstated.** Weakly monotone in 5/6 columns; one 1-token inversion. The auditor is technically right and substantively nitpicking.
- "**alpha does all the work**" — **supported** as a statement about effect magnitude (31.5x marginal spread; 97.1% vs 0.1% of variance; beta cannot reach even a third of alpha's best cell on its own).
- The **downstream thesis conclusion** — that the effective ingredient is coarse suppression of Neun_down, not the calibrated two-factor scheme — **survives**, and is arguably *strengthened*: in the paper's own operating regime, increasing beta makes repair slightly *worse*.

The auditor has not overturned the finding; they have overturned three words of it. The bigger exposure is §2, which neither party raised.

## 4. Recommended thesis wording

> Across the 30-cell grid (alpha ∈ {1,2,4,8,16} × beta ∈ {0.25,...,4}, 500 glitch and 500 normal tokens per cell, single seed), the two parameters contribute very unequally. Alpha's marginal spread is 20.2 percentage points against beta's 0.6 (a 31.5-fold difference); a two-way decomposition attributes 97.1% of grid variance to alpha and 0.1% to beta's main effect.
>
> Beta is not, however, without effect — its effect is conditional on alpha and changes sign. At alpha=1, where the suppression term is the identity, beta alone lifts repair from 0/500 (beta<=0.5) to 34/500 at beta=4 (+6.8pp, Fisher p = 6.5e-11): the promotion term does something, but at best a third of what suppression achieves (best overall cell 24.0%, alpha=16/beta=0.5). Once alpha >= 2, the relationship inverts: repair declines with beta in every row (Spearman rho −0.82 to −1.00; permutation test on the pooled within-row ordering, p < 1e-5), though no individual contrast reaches significance at n=500 (largest, alpha=16: 23.8% → 20.4%, p = 0.22) and the pooled contrast is borderline (21.3% vs 19.1%, p = 0.053).
>
> Repair is non-decreasing in alpha in five of six beta-columns and saturates at 23-24% by alpha=8; the single exception (beta=2.0, 112 → 111 tokens from alpha 8 to 16) is a one-token inversion well inside sampling noise. Saturation is itself beta-dependent: at beta=4 both alpha=8 and alpha=16 plateau at 20.4% rather than 23-24%.
>
> The defensible reading is therefore not that beta is inert, but that **within the parameter regime the method actually operates in, the promotion term contributes nothing positive and mildly antagonises suppression; the mechanism's effective ingredient is coarse division of the Neun_down set.** No cell approaches the paper's claimed 37.6%.

Two sentences that must be added rather than dropped:

> This grid was produced by an implementation that applies the correction to the multiplied SwiGLU product rather than to the gate and data streams independently; the latter is the reading of the paper we adopt elsewhere, and the sensitivity result has not yet been reproduced under it. The Neun_up/Neun_down cardinalities that motivate the mechanistic interpretation were observed at run time but not recorded in the run's metadata, and are reported here as unverified.

Delete the phrase "beta is inert; alpha does all the work" from `E:/user3/FYS_SWK/Benchmark/docs/RUNLOG.md` line 134 and the words "flat across beta everywhere" and "monotonically" on lines 138. A hostile reviewer with the CSV can falsify all three in about ninety seconds.

## 5. Experiments needed for a stronger claim, and cost

Anchor: the existing 30-cell grid is 30 × 1000 prompts × 24 new tokens ≈ 720k generated tokens; at batch 32 on the A6000 this is roughly **0.3-0.6 GPU-hours** per grid (consistent with the ~11 minutes between the `summary.json` and grid timestamps).

| # | Experiment | Why it changes the claim | Cost |
|---|---|---|---|
| 1 | Persist `neuron_stat_summary()` into the grid metadata | Turns the mechanistic explanation from prose into evidence. Already implemented in the working tree; just needs committing and one re-run. | **Free** (CPU, rides along with any re-run) |
| 2 | Save per-token outcomes (`record_path=`, already written) and run **McNemar** on the beta contrasts | Cells are paired — same 500 tokens, same neuron stats, greedy decoding — so the unpaired Fisher tests above are badly conservative. Paired tests will likely make the 2-3pp negative beta effects significant, converting "not inert, but not detectable" into "not inert, and measurably harmful". This is the single highest-value item. | **~0.5 GPU-h** (one grid re-run) |
| 3 | Add a true **beta=0** column and a finer beta ladder at alpha=1 (0, 8, 16, 32) | The grid has no zero-beta control, and the alpha=1 row is unsaturated at its top end — the beta-only ceiling is currently unknown. | **~0.15 GPU-h** (9 extra cells) |
| 4 | Re-run the grid under `stream_mode: separate` (the paper-faithful reading) | Finding 6's mechanism depends on Neun_up sparsity in the *product* stream. This is the experiment that could actually overturn the finding, and the repo's own config already calls the current basis an error. | **~0.5 GPU-h** |
| 5 | Seeds 0/1/2 on the corrected implementation | Every number above is single-seed. Needed for any "flat/saturating" language. | **~1.5 GPU-h** |
| 6 | Repeat on a second model (Llama-2-7b, key layers 19-28 per both papers) | Prevents "true of Mistral only". | **~1.5-3 GPU-h** |

Items 1-5 make the claim defensible: **~2.5-3 GPU-hours on one A6000, no new code beyond committing the working tree.** Adding item 6 for generality: **~5-6 GPU-hours.** Items 1-3 alone (~0.7 GPU-h) would resolve the specific dispute with the auditor.