# Run Log

Chronological record of every benchmark run: what was executed, what came out,
how it was verified. Newest entries at the bottom.

---

## 2026-07-17 - Skeleton built and smoke-tested (commit 8824319)

- Environment: native Windows, Python 3.12 venv, torch 2.6.0+cu124, RTX A6000 recognized.
- Full pipeline (steps 1, 2, 3, 4, 4b) executed end-to-end on `smoke-test`
  (SmolLM2-135M-Instruct, first 1,500 vocab ids) purely to validate the machinery.
- Verification: each step produced its expected artifacts; GP detection precision
  came out exactly 1.0, confirming the post-validation construction; GC LoRA showed
  0.11% trainable params (paper's <0.1% scale) and decreasing loss.
- Note: smoke-test METRICS are meaningless for the thesis (135M model, tiny slice
  of vocab, 3 epochs) - the run only proves the pipeline works.

## 2026-07-17 - Checkpointing + reporting infrastructure added

- Batch-level checkpoint/resume in the repetition sweep (step 1 and everywhere
  sweeps are reused); chunk-level checkpoints in GP detection; cell-level in the
  alpha/beta grid.
- `scripts/status.py` (resume map) and `scripts/aggregate_results.py`
  (thesis tables: results/benchmark_report.{json,md}) added.
- Verification: smoke-test step 1 rerun was interrupted and resumed from its
  checkpoint (see next entry).

## 2026-07-19 - Step 1 (RQ1) Mistral-7B-Instruct-v0.1: census + first finding

**Attempt 1 - FAILED verification.** Full 32k-vocab sweep completed in 5m24s
(batch 128, A6000), but the token filter classified 12,385 tokens (39% of vocab)
as UNREACHABLE vs the papers' ~230-scale, and glitch count (544) was far below
GlitchCleaner's 2,539. Root cause: our decode->re-encode reachability check
ignored SentencePiece leading-space semantics.

**Fix.** Reimplemented the filter to mirror GlitchCleaner's own released code
(third_party/GlitchCleaner/Fine-tuning/tokenfilter.py - the context-anchored
"«"-prefix roundtrip from Land & Bartolo's Magikarp protocol). Rerun resumed from
checkpoint: 19,483 tokens reused, only the 12,260 newly-eligible swept.

**Attempt 2 - verified.** Counts: 30,755 normal / 988 glitch / 253 unreachable /
3 special / 1 undecodable (total 32,000 ✓; filtered 257 ≈ papers' scale ✓).

**FINDING 1 (protocol sensitivity of the glitch census).** Comparing our 988
glitch tokens against GlitchCleaner's published list for this model
(third_party/GlitchCleaner/Glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens.csv,
2,539 tokens):
  - overlap (both glitch): 841
  - their-glitch but repeats fine under our protocol: 1,698 (67% of their list)
  - our-glitch not in their list: 147
  - filtering disagreements: 0
Their released evaluation code also deviates from their paper's template
("Question:" prefix, "return back", newlines, max_new_tokens=10 vs paper-implied
longer generation). Hypothesis: most of the census gap is template wording +
generation-length (10 vs 24 tokens). **CONFIRMED (same day):** the `--protocol gccode` census (their template verbatim,
max_new_tokens=10, their lstrip-containment check) yields **2,552 glitch tokens,
2,537 of which are in their published 2,539-token list** (99.9% coverage, Jaccard
0.993). Artifacts: results/ground_truth/mistral-7b-instruct-v01/gccode/.
Conclusion: our implementation reproduces theirs essentially exactly under their
protocol; the 988-vs-2,539 gap is pure protocol sensitivity. ~2/3 of this model's
"glitch tokens" (per their list) stop being glitchy when the model may generate
24 tokens instead of 10 with the paper-text template. Thesis implication: glitch
censuses - and therefore all published repair-rate denominators - are protocol
artifacts to a first approximation. Both protocols are now first-class in the
pipeline; downstream steps use the paper protocol as primary ground truth, with
gccode as the comparability baseline against GlitchCleaner's claims.

## 2026-07-19 - Step 2 (RQ2) Mistral: GlitchProber detection, paper-protocol census

3 seeds, gamma=0.1, PCA-75, SVM(poly, C=1, deg=3), post-validation on.
Results vs our paper-protocol census (988 glitch / 31,743 candidates):

| seed | precision | recall | F1 | time |
|---|---|---|---|---|
| 0 | 0.994 | 0.174 | 0.296 | 209s |
| 1 | 1.000 | 0.335 | 0.502 | 212s |
| 2 | 0.997 | 0.328 | 0.494 | 213s |
| mean | 0.997 | 0.279 +/- 0.091 | 0.431 +/- 0.116 | 211s |

Paper claims for Mistral: precision 100%, recall 67.41%, F1 0.8053, 42m39s.

**FINDING 2 (preliminary): massive seed variance.** Recall doubles between
identical runs that differ only in the random 10% sample (0.174 vs 0.335). The
paper reports single numbers with no variance.

**FINDING 3 (preliminary): post-validation does not guarantee 100% precision.**
1-2 false positives per seed vs the census: temperature-0 batched fp16 inference
is not bit-deterministic, so borderline tokens flip between runs. The paper's
100%-by-construction only holds relative to its own within-run judgments.

Confound to rule out before comparing recall to the paper: our census has a 3.1%
glitch base rate (~98 positives in the training sample) vs ~8% under the paper's
own protocol. Rerunning detection against the gccode census (2,552 glitch) for a
base-rate-matched comparison - in progress.

**Base-rate control (gccode census, 2,552 glitch, ~8% base rate):**

| seed | precision | recall | F1 |
|---|---|---|---|
| 0 | 0.997 | 0.413 | 0.584 |
| 1 | 0.996 | 0.382 | 0.553 |
| 2 | 0.999 | 0.417 | 0.588 |
| mean | 0.997 | 0.404 +/- 0.019 | 0.575 +/- 0.019 |

**FINDING 4: the recall shortfall survives the base-rate control.** Under the
paper's own census conditions, detection recall reproduces at ~40%, not the
claimed 67.4% (F1 0.575 vs 0.805). The base rate explains part of the earlier
gap (28% -> 40%) and nearly all of the seed variance (std 0.091 -> 0.019), but
the headline claim does not reproduce from the paper's published description.
Caveat for thesis: GlitchProber has no released code, so reimplementation-vs-
claim divergence cannot be fully attributed (threats to validity). Finding 3
(precision 99.7% != 100%) replicates under both protocols.

## 2026-07-19 - Step 3 (RQ3) Mistral: GlitchProber repair + alpha/beta sensitivity

Repair evaluated on the full glitch set per protocol; collateral on 500 normal
tokens; m=1, gamma=0.1, seed 0.

| protocol | adaptive repair | rule-based repair | rule collateral |
|---|---|---|---|
| paper (988 glitch) | 0.5% | 21.6% | 1.0% |
| gccode (2,552 glitch) | 3.8% | 26.1% | 6.2% |

Paper claims (Mistral): adaptive 37.60%, rule-based 12.92%.

**FINDING 5: the adaptive alpha/beta method is unimplementable as published.**
Eq. 9-12's constants (k1,b1,k2,b2) are undisclosed; with identity constants the
adaptive adjustments are near-zero (computed betas in [-0.6, 0.8], alphas ~1.0)
and repair is 0.5-3.8%. The claimed 37.6% cannot be approached from the paper's
own description. Meanwhile our rule-based runs (21.6-26.1%) EXCEED their claimed
rule baseline (12.92%).

**FINDING 6: beta is inert; alpha does all the work.** The paper's key-neuron
criterion (act > m=1 in >99% of normal tokens) selects only 0-4 Neun_up neurons
per layer on Mistral (vs ~10k Neun_down), so beta amplification has nothing to
act on. 30-cell grid (alphas 1-16 x betas 0.25-4, 500 glitch + 500 normal per
cell): repair is flat across beta everywhere, rises monotonically in alpha, and
saturates at 23-24% around alpha=8-16 - above the paper's chosen alpha=4 (which
yields 20-22%). Best cell 24.0% (alpha=16, beta=0.5). No cell approaches the
claimed 37.6%. Collateral rises with alpha (1.6-2.4% on the grid; unreported in
the paper). Heatmaps: results/gp_repair/mistral-7b-instruct-v01/heatmap_*.png.

Interpretation for thesis: the repair mechanism has real but modest effect
(~quarter of glitch tokens), its published "precise calculation" of alpha/beta is
not reproducible, and its effective ingredient is coarse suppression of silent
neurons, not the calibrated two-factor scheme the paper describes.

## 2026-07-19 - Step 4 (RQ4) Mistral: GlitchCleaner gated LoRA, train vs HELD-OUT

LoRA r=4/alpha=4 on MLP gate+up of layers 19-28 (0.0204% trainable params - the
"<0.1%" claim holds). 80/20 split, 3 epochs, lr 2e-4, seed 0. Loss converged
(1.90->0.34 paper-protocol; 1.12->0.15 gccode).

| protocol | n_train / n_heldout | train-split repair | HELD-OUT repair | normal ok (adapter forced on) | heldout repair (adapter off) |
|---|---|---|---|---|---|
| paper | 791 / 197 | 81.4% | 55.3% | 93.0% | 0.0% |
| gccode | 2,042 / 510 | 89.9% | 74.5% | 96.6% | 0.0% |

Paper claims: 86.88% avg across models, 94.80% for Mistral.

**FINDING 7: GlitchCleaner's headline number roughly reproduces on its own
population.** Under their protocol, train-split repair is 89.9% vs their claimed
94.8% (gap plausibly hyperparameter detail). The claim is real - for the tokens
the adapter was trained on.

**FINDING 8: ~15-26 points of the headline is memorization.** On glitch tokens
the adapter never saw, repair drops to 74.5% (their protocol) / 55.3% (paper
protocol). The adapter-off control (0.0%) confirms all repair comes from the
LoRA. Still, held-out GlitchCleaner (55-75%) far exceeds GlitchProber's best
grid cell (24%): the ranking of the two methods survives honest evaluation,
but both absolute claims shrink.

Also notable: with the adapter forced on, 3.4-7.0% of normal tokens break -
the "lossless" property rests entirely on the lambda gate being right, which in
turn requires a detector at inference time (the coupling the paper leaves
implicit).

## 2026-07-19 - Step 5 (RQ5) Mistral: inference speed

Greedy 256-token generation, 5 reps after warmup, single prompt, A6000 fp16:

| variant | tok/s | relative to base | GlitchCleaner Table 6 (relative) |
|---|---|---|---|
| base | 31.76 | 1.00 | 1.00 |
| GP repair hooks | 19.99 | 0.63 | 0.18 (11.82/66.30) |
| GC adapter (unmerged, gated) | 17.93 | 0.56 | 0.95 (62.83/66.30) |

**FINDING 9: both speed claims fail to reproduce as ratios.** GlitchCleaner's
"negligible impact" is a 44% slowdown in our unmerged gated-LoRA setup (their
gate precludes weight-merging, so extra matmuls are inherent); GlitchProber's
alleged 5.6x catastrophe is only a 1.6x penalty when the hooks are implemented
with plain tensor ops. The paper's comparison likely reflects implementation
quality, not method cost. Caveats for thesis: different hardware (H200 vs
A6000), single-prompt setting, and their GP reimplementation is unpublished.

Absolute numbers are hardware-bound; only ratios are compared.
