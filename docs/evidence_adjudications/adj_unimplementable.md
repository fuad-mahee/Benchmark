# Adjudication: "GlitchProber's adaptive α/β is unimplementable as published"

## 1. Quantitative evidence

### 1a. What the paper actually specifies (`papers/extracted/Glitchprober_text.txt`)

| Item | Paper location | Content |
|---|---|---|
| Eq. 7 | line 615 | `Neun↑ = {i \| Act[i] > m for over 99% of tokens in N′}` |
| Eq. 8 | line 616 | `Neun↓ = {i \| Act[i] ≤ m for all tokens in N′}` |
| Eq. 9 | lines 499–501 | `ΔAct↑ = (1/\|Neun↑\|) Σ (Act_normal[i] − Act_glitch[i])` |
| Eq. 10 | lines 593–603 | `ΔAct↓ = (1/\|Neun↓\|) Σ (Act_glitch[i] / Act_normal[i])` — **signed**, no abs, no guard |
| Eq. 11/12 | lines 611, 613 | `β = k1·ΔAct↑ + b1` ; `α = k2·ΔAct↓ + b2` |
| Range restriction | lines 605–610 | "through linear transformation **and range restriction**, the algorithm maps ΔAct↑ and ΔAct↓ to **appropriate numerical intervals**" |
| Constants | lines 614–617 | "derived through an adaptive process tailored to the specific dynamics of each model. **A set of default values is provided**" |
| Algorithm 2 | lines 652–669 | β, α computed **once at lines 3–4, before** the token and layer loops; both take **only N′** as data |
| Settings | line 832 | m = 1, γ = 0.1, same key layers as detection |
| Rule-based baseline | line 850 | fixed **α = 4, β = 1.5** |
| Table 5 (Mistral) | lines 867–869 | rule-based 359 tokens / 12.92%; GlitchProber 1,045 / **37.60%** |
| Key layers | lines 688, 699–701 | Llama-2 = 19–28 only; **Mistral never stated** |

Implied Mistral glitch denominator from Table 5: 359/0.1292 = **2,778.6** and 1045/0.3760 = **2,779.3** → the paper's census is ~2,779 tokens (comparable to our gccode census of 2,552, not our paper-protocol 988).

**Undefined in the paper:** (i) numeric values of k1,b1,k2,b2 — a full-text grep returns only the symbols at lines 611–614, no table, no appendix; the "set of default values is provided" has **no referent anywhere in the paper**, and no code was released; (ii) the "adaptive process" that derives them; (iii) the intervals of the range restriction, for either factor; (iv) whether α,β are **global scalars** (Algorithm 2) or **per-layer** (Sec. 4.2.2 text: "For the MLP module in each layer, we calculate…") — the paper says both; (v) Algorithm 2's `statisticsBeta(N′,·)`/`statisticsAlpha(N′,·)` take only the normal sample although Eq. 9–10 require `Act_glitch` — an internal inconsistency; (vi) which tensor `Act[i]` is (gate σ(Z1), data Z2, or their product) — Fig. 5 shows *both* a "Modified MLP gate" and a "Modified MLP data"; (vii) sequence position for statistics and for correction; (viii) Mistral key layers.

### 1b. Where our implementation deviated (v1 = commit `706e27d`, which produced **every archived number**)

| # | Deviation | Status |
|---|---|---|
| D1 | Eq. 10 denominator `\|Act_normal[i]\| + 1e-6` instead of signed `Act_normal[i]` | **Alters a published equation.** Neun↓ is by construction the low/negative-activation set, so the sign flip affects a large fraction of terms |
| D2 | `alpha = max(α, 1.0)` hard-coded | Invented range restriction; **binding** under identity constants |
| D3 | No range restriction on β at all | Paper applies it to both |
| D4 | Hooked the **already-multiplied product** at `down_proj` input | One stream instead of the paper's two (Fig. 5). The repo now states this itself: *"adding beta to a product is not the same as adding beta to its factors. All v1 repair numbers therefore measured a variant, not Algorithm 2."* |
| D5 | Statistics at last prompt position; correction applied at **all positions and every decoding step** | Paper silent, but this is a broader intervention than "correcting the glitch token" |
| D6 | k1=k2=1, b1=b2=0 | Our choice, explicitly flagged in config |
| D7 | Per-layer α,β vs Algorithm 2's global scalar | Ambiguity resolved by us |

### 1c. Measured numbers (all v1)

- **Adaptive, identity constants:** paper protocol 5/988 = **0.51%** (95% CI 0.06–0.95%); gccode 97/2,552 = **3.80%** (3.06–4.54%).
- **Why:** with k2=1,b2=0, α = ΔAct↓ ≈ a ratio near 1, then clamped by `max(α,1.0)` → α ≈ 1.0 → division by ~1 → **a near-null intervention**. The grid confirms this: the α=1.0 row spans **0.0%–6.8%**, exactly the adaptive regime. **The 0.5%/3.8% figures measure "no suppression", not the adaptive method.**
- **Rule-based at the paper's own α=4, β=1.5:** 213/988 = **21.6%** (paper protocol), 666/2,552 = **26.1%** (gccode) — vs the paper's **12.92%**. Our reimplementation is **1.67×–2.02× more effective than theirs at identical parameters**.
- **30-cell grid** (α ∈ {1,2,4,8,16} × β ∈ {0.25,0.5,1,1.5,2,4}; 500 glitch + 500 normal per cell, paper protocol):
  - max **24.0%** at (α=16, β=0.5), 95% CI **[20.3%, 27.7%]**; z = **−6.28** vs 37.6%
  - monotone and saturating in α: per-α means 2.6% → 15.7% → 21.1% → 22.5% → **22.8%**; α 8→16 buys **+0.6 pt**
  - flat in β: per-β means span **0.64 pt** across the entire axis (17.1%, 16.9%, 16.9%, 17.3%, 16.8%, 16.6%)
  - collateral rises with α: 0.23% → 1.0% → 1.7% → 1.87% → 1.90% (max 2.4%); the paper reports no collateral metric at all
- **The ratio argument (protocol-independent):** the paper's adaptive-over-rule-based gain on Mistral is **2.91×**. Within our measured landscape the best cell beats the (α=4, β=1.5) cell by **1.15×**. For the paper's ratio to hold on our landscape, some cell would have to reach **60.5%** repair; the observed maximum is 24.0%.

### 1d. Artifact integrity problems the audit surfaced

- `results/gp_repair/**` are all dated **2026-07-19**, `git_commit: 706e27d`. `src/glitchprober/repair.py`, `scripts/run_gp_repair.py`, `src/glitchprober/sweep_alpha_beta.py` and `configs/glitchprober.yaml` are **modified and uncommitted** (269 lines changed in repair.py alone), switching to `stream_mode: separate`, removing the abs() from Eq. 10, restricting the correction to the token position/prefill, and adding a disjoint `adaptive_fit_fraction: 0.5` fit/eval split. **None of it has been re-run.** Every number in §1c belongs to an implementation the repo itself now labels "a variant, not Algorithm 2".
- RUNLOG Finding 6's neuron counts ("0–4 Neun_up per layer vs ~10k Neun_down") and the adaptive per-layer factors ("betas in [−0.6, 0.8], alphas ~1.0") were only **printed to stdout**, never persisted. No artifact in `results/` backs them. (The new `neuron_stat_summary()` / `d_act_up` / `d_act_down` plumbing fixes this going forward.)

## 2. Ruling: **partially supported — the word "unimplementable" is refuted**

The auditor is right, and on stronger grounds than they stated.

**Supported:** the *specification* claim. k1,b1,k2,b2 are never given, the "adaptive process" is never described, the range-restriction intervals are never stated, the promised "default values" do not appear in the paper, Algorithm 2 contradicts Eq. 9–10 on its inputs and contradicts Sec. 4.2.2 on scope, and no code exists. The published adaptive configuration **cannot be reconstructed**. That is under-specification / non-reproducibility.

**Not supported:** the *impossibility* claim. Three independent defects:
1. **The tested point is degenerate, not representative.** Identity constants plus our own `max(α,1)` clamp pin α≈1, which the grid independently identifies as the null-intervention regime. Reporting 0.5%/3.8% as "the adaptive method's performance" is indefensible; it is the performance of doing nothing.
2. **Eq. 10 was altered.** Under the published *signed* form, ΔAct↓ has an unknown sign and magnitude distribution, so we cannot even assert that the identity mapping yields α≈1 under the paper's equation.
3. **The ceiling was measured with the wrong intervention.** The 24% grid ran on the product stream at all positions. The repo now agrees this is not Algorithm 2. Until re-run, the 24% number cannot be attributed to the paper's method at all — and it is the sole basis for "the claim cannot be approached".

The grid *would* be a legitimate ceiling argument under the global-scalar reading of Algorithm 2 (the constants act only through α and β, so sweeping (α,β) subsumes sweeping (k1,b1,k2,b2)) — but not under the per-layer reading, where the grid explores only the diagonal of a 20-dimensional space (10 key layers × 2 streams).

## 3. Recommended thesis wording

> GlitchProber's adaptive adjustment (Eq. 9–12) is not reproducible from the paper. The paper gives the functional form β = k1·ΔAct↑ + b1 and α = k2·ΔAct↓ + b2, but never states k1, b1, k2 or b2; never describes the "adaptive process" by which they are said to be derived; and never gives the intervals to which the "range restriction" maps the two factors. The "set of default values" the text says "is provided" does not appear anywhere in the paper, and no implementation was released. Algorithm 2 compounds this: it computes β and α once, from the normal sample N′ alone, although Eq. 9–10 require glitch-token activations, and it treats them as global scalars where Sec. 4.2.2 derives them per MLP module. The reported 37.60% repair rate for Mistral-7B-Instruct-v0.1 therefore cannot be verified against, or reconstructed from, the paper's description. We do not claim the method is unimplementable — only that the published configuration is unrecoverable, so any reimplementation reports its own calibration, not the authors'.
>
> Because the calibration is unavailable, we evaluated the *mechanism* directly, sweeping the two factors over a 5×6 grid (α ∈ {1,2,4,8,16}, β ∈ {0.25…4}; 500 glitch and 500 normal tokens per cell). Repair rate rises monotonically in α and saturates: per-α means 2.6%, 15.7%, 21.1%, 22.5%, 22.8%, with the best cell at 24.0% (95% CI 20.3–27.7%) and only +0.6 points bought by doubling α from 8 to 16. It is flat in β — the per-β means span 0.64 points across the entire axis. Collateral damage on normal tokens rises with α, reaching 2.4%, a cost the paper does not report. At the paper's own fixed values (α = 4, β = 1.5) we obtain 21.6% under our census and 26.1% under GlitchCleaner's, against the 12.92% the paper reports for that same configuration — our reimplementation is, if anything, more effective than theirs at identical settings. The claim that is not supported by the landscape we measure is therefore not the absolute number but the *gain attributed to calibration*: the paper reports a 2.91× improvement of the adaptive variant over its own fixed-value baseline, whereas within our grid the best attainable point improves on (α = 4, β = 1.5) by 1.15×. For the paper's ratio to hold on our landscape, some (α, β) would have to deliver 60.5% repair; the observed maximum is 24.0%.
>
> Threats to validity: the sweep covers global scalar (α, β), which matches Algorithm 2 but not the per-layer reading of Sec. 4.2.2; key layers for Mistral are our assumption (the paper states them only for Llama-2); and GlitchProber has no released code, so any divergence between our reimplementation and the published claim cannot be attributed with certainty.

Delete "unimplementable as published" from RUNLOG Finding 5 and replace it with "the published adaptive calibration is unrecoverable". Do **not** cite the 0.5%/3.8% adaptive numbers as a performance measurement of the method — cite them only as a demonstration that the identity mapping is degenerate, and say so explicitly.

## 4. Experiments needed for a stronger claim, and cost

**E0 — mandatory before anything above is publishable.** Re-run repair + grid under the working-tree implementation (`stream_mode=separate`, signed Eq. 10, token-position/prefill correction, disjoint fit/eval split), both protocols, and keep `--stream-mode product` as a labelled ablation so v1 vs v2 is a reported comparison rather than a silent replacement. Persist `neuron_stat_summary()` and the per-layer `d_act_up`/`d_act_down` so Findings 5–6 have artifacts. *Cost: ~2–3 GPU-hours on the A6000 (the 32k-token full sweep took 5m24s at batch 128; a 30-cell × 1,000-token grid is the long pole).* Note the two-stream change may materially move the β result — Neun↑ under m=1 on σ(Z1) is a different set than on the product, so "β is inert" must be re-established per stream.

**E1 — decisive on the ceiling, cheapest high-value run.** Evaluate the α→∞ limit by setting `act[Neun↓] = 0` outright. This is the exact supremum of the entire divide-by-α family. If it still caps near 24%, then **no** (k2,b2) can reach 37.6% under the global-scalar reading, converting "our grid did not find it" into "the mechanism's own limit is below the claim". Pair it with β ∈ {8,16,32} and β<0 to close the β axis for sign as well as magnitude. *Cost: ~8 extra cells, ~20 minutes.*

**E2 — closes the "you tested one arbitrary constant mapping" objection directly.** Sweep (k2,b2) — and (k1,b1) — rather than (α,β), with per-layer/per-stream factors enabled, logging the induced α vector alongside repair rate. First run the free version: one call to `compute_adjustments` now records `d_act_up`/`d_act_down` per (layer, stream), which immediately tells you the *reachable* range of α as (k2,b2) vary and whether α≈16 is attainable by any linear map with plausible constants. *Cost: minutes for the diagnostic; ~2 GPU-hours per protocol for a 50-draw random search.*

**E3 — removes a confound in the ceiling.** Key-layer ablation for Mistral (the 19–28 set is our assumption). At minimum, contrast 19–28 against 19–31 and a mid-depth set at the best grid point. *Cost: ~30 minutes.*

**E4 — supports the "β is structurally inert" claim.** Run the `m_sweep` already in the config (m ∈ {0.25,0.5,1,2,4}) per stream and report |Neun↑| at each. If |Neun↑| is near zero at the paper's own m=1, that is a criticism of the paper's Eq. 7, not of our implementation — but it must be shown per stream, not on the product. *Cost: ~30–45 minutes.*