# Run Log

Chronological record of every benchmark run: what was executed, what came out,
how it was verified. Newest entries at the bottom.

> **Read the last entry first.** Entries are appended, never rewritten, so an
> earlier section may state a conclusion a later one withdraws. The final entry
> (2026-07-31, independent verification pass) lists every claim in this file that
> did not survive recomputation from `results/`. Sections it supersedes carry an
> inline pointer.

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

---

## 2026-07-30 - EXTERNAL AUDIT + CORRECTIONS (read this before citing anything above)

The implementation was submitted to an independent adversarial audit. It found two
errors that invalidate specific results above, several correctness defects, and
several conclusions stated more strongly than the evidence supports. Everything
below supersedes the corresponding claims above. The authoritative write-up is
`docs/benchmark_study.tex`.

### Material errors (results above are affected)

* **E1 - GP repair modified the wrong tensor.** Findings 5/6 and all RQ3 numbers
  above adjusted the already-multiplied gate x data product instead of the two
  streams separately (paper Eq. 3-5, Fig. 5). Corrected in `repair.py`; the old
  behaviour is retained as `--stream-mode product` for the ablation.
* **E2 - GC's lambda gate was absent.** Findings 8/9's clean-input and speed
  numbers used an always-on adapter, so they were not measuring GlitchCleaner.
  Gate implemented in `src/glitchcleaner/gate.py`.

### Correctness defects fixed
Unseeded GC training; dropped final gradient-accumulation window each epoch;
adaptive alpha/beta fitted on their own evaluation set; abs() inserted into the
published signed Eq. 10; 18/197 held-out tokens leaked into training sequences;
training targets stripped/unquoted where upstream uses unstripped+quoted; census
ran SDPA while detection ran eager; stale checkpoint silently resumed by the
"corrected" grid; m-sweep configured but never executed. All addressed; see
commits 0c03490 and a3d1042.

### Findings re-adjudicated against the stored artefacts

* **Finding 1 - umbrella claim SURVIVES, stated cause REFUTED.** A post-hoc
  factorial over the archived generations (6 of 8 cells recoverable, since greedy
  decoding makes prefix truncation exact) attributes the +1,564 gap as:
  prompt wording **+1,549 (99.0%)**, match rule +14 (0.9%), generation length
  **+1 (0.06%)**. The run log above blamed generation length - the one factor that
  turned out to be inert. Mechanism: 42.7% of gccode-glitch tokens emit pure
  whitespace (newline attractor from the template's trailing newline). The
  disagreement is also bidirectional: 146 tokens go the other way.
* **Finding 3 - measurement stands, cause REFUTED, framing was unfair.** Precision
  99.7% is real, but the residual comes from OUR OWN SDPA-vs-eager kernel
  mismatch (0.37-0.42% of generations differ in text, 0.021% in verdict,
  deterministically), not fp16 nondeterminism. Under the paper protocol ALL false
  positives entered via sample labelling; post-validation admitted 0 in 1,047
  confirmations. Counts are 1,0,1 and 3,4,1 - not "1-2 per seed". Also: the
  paper's 100% is definitional (post-validation defines membership by the same
  test), so the honest claim is narrower - a self-referential 100% does not
  survive contact with an independent census, and the residual is governed by the
  inference stack, which neither paper specifies.
* **Finding 5 - "unimplementable" WITHDRAWN.** Correct claim: the published
  adaptive calibration is *unrecoverable* (k1,b1,k2,b2 never given; "adaptive
  process" never described; range-restriction intervals never stated; promised
  default values absent; Algorithm 2 contradicts Eq. 9-10 on inputs and Sec. 4.2.2
  on scope; no code). Our identity-constant result must NOT be cited as the
  method's performance - it pins alpha~1, which the grid independently identifies
  as the null-intervention regime. The defensible quantitative claim is about the
  *gain attributed to calibration*: the paper reports 2.91x (37.60/12.92), which
  from our measured (alpha=4,beta=1.5) point would require ~60% repair; the grid
  maximum is 24.0%.
* **Finding 6 - "beta is inert" WITHDRAWN.** Alpha dominates (marginal spread
  20.2pp vs 0.6pp), but beta's effect is conditional and sign-flipping: at
  alpha=1 it lifts repair 0/500 -> 34/500; at alpha>=2 repair declines slightly
  as beta grows. "Alpha does all the work" survives as a magnitude statement.
* **NEW - the unstated implementation choices dominate.** Tensor (product vs
  separate) and positions (token-only vs all) are both unspecified in the paper
  and together move repair rate by roughly an order of magnitude at fixed
  alpha/beta. For this method they matter more than the withheld constants.

## 2026-07-31 - CORRECTED RERUNS COMPLETE (these supersede ALL numbers above)

Mistral-7B-Instruct-v0.1. Every number below comes from the corrected
implementation (separate gate/data streams, token position, real lambda gate,
upstream GC hyperparameters, seeded, leakage-filtered, fixed accumulation).

### RQ1 - factorial CLOSED by measurement
gccode prompt @ 24 tokens: **2,550 glitch** (vs 2,552 at 10 tokens; paper prompt
@ 24 tokens = 988). Generation budget moves the count by **2 tokens**; the prompt
wording moves it by ~1,560. The post-hoc attribution (prompt 99.0% / oracle 0.9%
/ length 0.06%) is now confirmed by a direct run, not extrapolation.

### RQ3 - GP repair, corrected
| config (alpha=4, beta=1.5) | repair | collateral |
|---|---|---|
| separate streams / token position (paper reading) | 6.48% (32/494) | 0.00% |
| product / token | 2.23% (11/494) | 0.00% |
| separate / all positions | 17.41% (86/494) | 1.20% |
| product / all positions (= our v1) | 15.18% (75/494) | 1.00% |

**7.8x spread across four defensible readings of the same paper.** Position
matters more than which tensor. Paper claims 12.92% for this configuration -
which sits INSIDE our range, so the claim is neither confirmed nor refuted; it is
under-determined by the paper's own description.

Adaptive (identity constants): 1.01% paper / 0.24% gccode - degenerate, do NOT
cite as the method's performance.

Corrected grid best cell: alpha=8, beta=1.0 -> **7.60%** (v1 grid said 24.0%; that
grid was product/all-positions AND had silently resumed a stale checkpoint).

**m-sweep (new, and the most informative RQ3 result):**
| m | Neun_up | Neun_down | repair | collateral |
|---|---|---|---|---|
| 0.25 | 376 | 530 | 5.0% | 0.2% |
| 0.5 | 195 | 2,222 | 4.8% | 0.2% |
| 1.0 (paper) | 63 | 17,041 | 6.8% | 0.2% |
| 2.0 | 5 | 150,755 | 8.0% | 0.2% |
| 4.0 | 1 | 270,882 | **12.2%** | 0.4% |
Repair RISES monotonically as the promotion set shrinks to a single neuron. This
is direct evidence that suppression of persistently-silent neurons carries the
entire mechanism and promotion contributes nothing measurable - a cleaner form of
the withdrawn "beta is inert" claim, established by a different experiment.

### RQ4 - GC repair, corrected: THE MEMORISATION GAP LARGELY DISAPPEARS
| protocol | n_train | n_heldout | train | **held-out** | adapter off | clean (gated) |
|---|---|---|---|---|---|---|
| paper | 791 | 187 (10 leaked, dropped) | 35.15% | **37.43%** | 0.00% | 100.00% |
| gccode | 2,042 | 475 (35 dropped) | 82.96% | **78.11%** | 0.00% | 99.60% |

**Finding 8 is substantially WITHDRAWN.** The 15-26 point gap reported earlier was
mostly an artefact of our own broken training (3 epochs, wrong LR, unseeded,
dropped gradients). With the authors' own hyperparameters the gap is 4.85pp under
their protocol, and under the paper protocol held-out EXCEEDS train. GlitchCleaner
generalises to unseen glitch tokens far better than we previously reported.

What remains: held-out 78.11% vs the paper's claimed 94.80% for this model. And
the losslessness claim is now SUPPORTED - with the real gate, clean tokens are
unaffected (100% / 99.6%).

> **SUPERSEDED - see V1 in the 2026-07-31 verification entry.** The losslessness
> sentence above is withdrawn. Under the gccode protocol the gate never closed
> (`gate_stats.examples_gate_off == 0`), so 99.6% is the always-on number, not a
> gate measurement. The repair figures in the table above are unaffected.

### RQ5 - speed, corrected: BOTH SLOWDOWN CLAIMS WITHDRAWN
base 31.97 / GP hooks 31.55 / GC gated adapter 31.57 tok/s -> ~1% overhead each.
The earlier 37-44% penalties were artefacts of our v1 configuration (repair hooks
firing at every decode step rather than only at the token's position).
**CAVEAT: this run has not been repeated with controlled ordering and the change
from the v1 measurement is large. Re-verify before publishing (see HANDOVER.md).**

> **SUPERSEDED - see V2 in the 2026-07-31 verification entry.** The stated cause
> is contradicted by the code: `run_speed.py` never calls `bind_position()`, so
> both runs measured the every-decode-step configuration. "GC gated adapter" is
> also mislabelled - no gate is applied in that script. Do not cite either number.

---

## 2026-07-31 - INDEPENDENT VERIFICATION PASS (new author; supersedes specific claims above)

Every headline number in this log and in `docs/benchmark_study.tex` was recomputed
from the artefacts under `results/` rather than accepted from the text. Most
reproduce exactly. The exceptions are recorded below, each with the file the
correct value comes from. Verification script: `scripts/verify_claims.py`.

### Reproduced exactly (no change)
RQ1 census 988 / 2,552 and filtering 253+3+1; budget-24 cell 2,550; anchoring
2,537 of 2,539 (Jaccard 0.9933) against
`third_party/GlitchCleaner/Glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens.csv`;
bidirectional count 146; gap 1,564; whitespace mechanism 1,089/2,552 = 42.7%;
all RQ2 figures including per-seed false positives 1,0,1 and 3,4,1; the RQ3 2x2
(6.48 / 2.23 / 17.41 / 15.18 %) and the m-sweep; all RQ4 figures. Both papers'
quoted claims were checked against `papers/extracted/` and are quoted correctly.

### V1 - The losslessness conclusion above is WRONG. The gate does not close.
`results/gc/mistral-7b-instruct-v01/gccode/eval.json` records
`gate_stats: {examples_gate_on: 975, examples_gate_off: 0}`. Under the gccode
protocol the gate was open for **every** example, including all 500 normal
tokens. The 99.60% "clean (gated)" figure is therefore not a measurement of the
gate at all - it is byte-identical to the forced-on figure (498/500) because the
adapter was active in both.

Cause: the gate is a set-membership test over the whole input, and the prompt
template contains ids **304 (`and`)** and **464 (`'`)**, both of which are in
GlitchCleaner's own published glitch list. Confirmed against upstream:
`GlitchCleaner.py` reads `glitchtokens = df['index'].tolist()` and sets the flag
via `any(token.item() in self.glitchtokens for token in token_ids[b])`.
Their published G for Mistral also contains `for`, `with`, `by`, `about`,
`after`, `because`, `here`, `using`, `=` and `!`. Measured: the gate opens on
**57.4% (197/343)** of sentences drawn from the two papers themselves.

This was already established in `docs/evidence_adjudications/upstream_code.md`
(sections 1.6 and D2, with GSM8K 0-shot 92.27% and MMLU 89.77%), but never
reached this log, HANDOVER.md or the study document, all of which assert the
opposite. The 2026-07-31 entry above ("the losslessness claim is now SUPPORTED")
is **WITHDRAWN**. The claim holds only for our artificial single-token probes
under the paper-protocol census, which happens to exclude `and` and `'`.

The corrected statement is stronger and is about the paper, not about us:
GlitchCleaner's stated mechanism for losslessness - clean inputs bypass the
adapter - does not operate on ordinary text. What its Table 4 actually shows is
that the adapter is *empirically* benign with the gate open. Their Table 4 also
proves the gate leaked: a truly gated-off adapter would give bit-identical
scores, and theirs differ (Gemma-2b-it MMLU 38.14 -> 40.27).

### V2 - RQ5 does not measure the configuration it is described as measuring.
`scripts/run_speed.py` constructs `RepairHooks(...)` but never calls
`.bind_position(tok)`. In `src/glitchprober/repair.py` that leaves
`_tok_pos = None`, so `__enter__` falls back to `pos = -1` - the hooks fire at
the last position of every forward pass, including every decode step. That is
the **v1** configuration. `git log -- scripts/run_speed.py` shows the file is
unmodified since commit 8824319, so BOTH speed runs used it.

Consequence: the explanation given above for the 37-44% -> ~1% reversal ("repair
hooks firing at every decode step rather than only at the token's position") is
contradicted by the code, since the corrected run fires at every decode step too.
The cause of the reversal is unexplained and the result must not be published
until the defect is fixed and the run repeated. Separately, no gate is applied to
the GC variant in `run_speed.py`, so "GC gated adapter 31.57" above is mislabelled.

### V3 - Neuron totals cited in HANDOVER appear in no artefact.
HANDOVER (and `docs/review_panel/committee.md` R2, from which it was copied)
state "63 promotion neurons against 16,928 suppression neurons". Summing
`neuron_selection` in `results/gp_repair/mistral-7b-instruct-v01/summary.json`
gives **66 / 17,201**. `m_sweep.csv` at m=1.0 gives **63 / 17,041** - a different
run (500-token glitch sample). 16,928 matches nothing on disk. The substantive
claim is unaffected and is cleaner when stated correctly: ~66 promotion against
~17,201 suppression out of 286,720 candidate neurons (10 layers x 2 streams x
14,336).

Related: "product mode gives Neun_up = 0 in all ten layers" is true of
product/**token** (`ablation_product/`), but the v1 configuration was
product/**last**, which gives 6,2,2,5,0,0,1,0,2,4 (`abl_product_last/`). The
study document's "0-4 per layer" was approximately right for the config it named.

### V4 - The study's beta paragraph quotes v1 numbers as if corrected.
"alpha's marginal spread is 20.2pp against beta's 0.6" and "beta lifts repair
from 0/500 to 34/500" both come from `alpha_beta_grid.v1_product_last.csv`.
Recomputed from the corrected `alpha_beta_grid.csv`: marginal spreads are
**6.13pp (alpha)** and **1.20pp (beta)**, and the alpha=1 row runs
**1/500 -> 13/500** across beta = 0.25 ... 4.0. The corrected grid maximum is
**7.60%**, attained at a **three-way tie**: (8, 1.0), (8, 4.0) and (16, 1.0).
Value at the paper's (4, 1.5) is 6.80% (34/500) on the grid's 500-token sample,
against 6.48% (32/494) on the held-out half in `summary.json` - two different
denominators for the same configuration, which is why both must be labelled.

### V5 - Smaller corrections, all recomputed
* Held-out leakage is **10 of 197** (`train_meta.json: n_heldout_dropped_leaked`),
  not the 18 of 197 stated twice in the study document. gccode: 35 of 510.
* Newline comparison mixes predicates: newline-only is 1,088 vs **0**;
  whitespace-or-empty is 1,115 vs 3. "1,088 against 3" compares two predicates.
* Short-token oracle: **32** whitespace-only candidates and **3,535** with a
  stripped length <= 1 (3,503 excluding the whitespace-only ones), not 31 and
  3,504. The figure "3,253 of 30,755 normal verdicts rest on <= 1 stripped
  character" is confirmed.
* `train_seconds` is **148.9** for one GC adapter, so "training cost being the
  constraint" for a single seed is not supportable; three seeds is ~7.5 minutes.
* `alpha_beta_grid_meta.json` no longer exists, so the HANDOVER item asking for
  its deletion is stale.
* Table 1's caption ("reproduces the reported value to within rounding in all
  five cases") does not hold for Mistral: 954.7 predicted against 956 reported.
  A stronger form of the same finding is available - GlitchCleaner's Mistral rate
  (37.65%) and Yi rate (53.27%) DIFFER from GlitchProber's published 37.60% and
  53.26%, and GlitchCleaner's GlitchProber-column average is 50.08% against
  GlitchProber's own 50.06%. That is the signature of `count = round(rate x |G|)`
  followed by recomputing the rate from the rounded count, and it is independent
  confirmation that the counts were derived rather than quoted.

### Not an error, but the largest unused asset
`third_party/GlitchCleaner/LoRA-Parameter/Mistral-7B-Instruct-v0.1.pt` is the
authors' own trained adapter. No experiment has used it. Evaluating their weights
directly would anchor RQ4 the way the census anchors RQ1, and would remove the
"our reconstruction may differ" alternative explanation from the one result where
it currently carries the most weight.
