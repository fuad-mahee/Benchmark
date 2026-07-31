## 1. Quantitative evidence

**Provenance of `G` in `src/glitchprober/detect.py`** — three paths, only one of which is post-validated:

| # | Line | Path | Post-validated? |
|---|---|---|---|
| 1 | 80 | `G = set(t for t in sample if not sample_results[t][0])` — sampled tokens labeled glitch during training-set construction | **No** |
| 2 | 87 | `G \|= {t for t,(ok,_) in val.items() if not ok}` — SVM positives that fail the re-run test | Yes |
| 3 | 89 | `G \|= set(predicted_glitch)` — only when `post_validation: false` (not used; config is `true`) | No |

Paths 1 and 2 are disjoint by construction (`rest` excludes `sample_set`); measured overlap = 0 in all 6 runs. **The auditor's mechanism is real: path 1 bypasses post-validation entirely.**

**Reconstruction.** I rebuilt `G` per seed from the checkpoints and reproduced `TP/FP/FN` and `n_predicted_glitch_by_svm` **exactly** for all 6 runs, so the attribution below is not an approximation.

| protocol | seed | TP | **FP** | precision | FP via sampled-label | FP via post-validation |
|---|---|---|---|---|---|---|
| paper | 0 | 172 | **1** | 0.9942 | 1 | 0 |
| paper | 1 | 331 | **0** | 1.0000 | 0 | 0 |
| paper | 2 | 324 | **1** | 0.9969 | 1 | 0 |
| gccode | 0 | 1054 | **3** | 0.9972 | 0 | 3 |
| gccode | 1 | 976 | **4** | 0.9959 | 2 | 2 |
| gccode | 2 | 1063 | **1** | 0.9991 | 0 | 1 |

- Primary (paper) protocol FPs are **1, 0, 1** — the auditor is correct; "1–2 per seed" is wrong and contradicts the table printed directly above it in `docs/RUNLOG.md` (seed 1 shows precision 1.000).
- **Post-validation produced 0 false positives in 1,047 post-validated positives under the primary protocol.** Both primary-protocol FPs are the *same token*, id 28012 `'inement'`, entering via sampled labeling.
- Under gccode, post-validation *is* implicated: 6 of 8 FPs.

**Mechanism.** For every FP, the generated text differs between the census sweep and the detection sweep, so this is genuine generation divergence, not a scoring-function bug. But it is **not** batch nondeterminism:

- Token 28012: seed 0 at batch 5/slot 3, seed 2 at batch 12/slot 30, in two *different* random samples → **byte-identical** output `"'iment '"`, both ≠ census `"'inement '"`.
- Token 27952: three post-validate sweeps of sizes 1316/1244/1260, slots 27/6/9 → **byte-identical** `'\n'×10`, ≠ census.

Divergence is stable across batch composition, so composition is not the driver. The actual cause is a **systematic attention-kernel mismatch between scripts**:

- `E:/user3/FYS_SWK/Benchmark/scripts/run_ground_truth.py:57` → `load_model(mcfg, attn_impl=None)` → SDPA
- `E:/user3/FYS_SWK/Benchmark/scripts/run_gp_detect.py:60` → `load_model(mcfg, attn_impl="eager")`

Template, task, `max_new_tokens`, `batch_size=32`, and fp16 dtype are identical across the two scripts; the attention implementation is the only difference. Population disagreement (same tokens, same prompts, SDPA vs eager): **verdict flips 2/9,522 = 0.021%** on sampled tokens, with **raw text differing on 0.37–0.42%** of tokens — i.e. ~18× more tokens diverge in text than flip the pass/fail verdict.

## 2. Ruling: **partially supported — headline number correct, both stated causes refuted**

| Component | Verdict |
|---|---|
| "precision 99.7%, not 100%" | **Supported.** 0.99705 (paper), 0.99738 (gccode). |
| "post-validation yields" this | **Refuted for the primary protocol** (0/1047 post-validated FPs). Holds only under gccode. |
| "fp16 batch nondeterminism" | **Refuted.** Divergences are byte-stable across batch compositions; cause is SDPA-vs-eager kernel mismatch between our own scripts. |
| "1–2 false positives per seed" | **Refuted.** 1, 0, 1 (mean 0.67). |

The deeper issue: this was never evidence that post-validation is broken. GlitchProber's 100%-precision claim is **definitional** — post-validation defines membership in `G` by test failure, so scoring against the same oracle is tautologically 100%. Our sub-100% arises only because the census and the detector ran under different inference stacks. As written, Finding 3 attributes to the paper a defect that is actually an artifact of our own configuration inconsistency.

## 3. Recommended thesis wording

> GlitchProber's reported 100% detection precision is true by construction rather than empirically: post-validation admits a token to the glitch set precisely when it fails the repetition test, so any evaluation against that same oracle returns 100% by definition. To test whether the claim carries any information beyond its own definition, we scored the detector against an independently generated full-vocabulary census. Precision was 99.7% under both protocols (mean 0.9971 paper, 0.9974 gccode; 2 and 8 false positives respectively over three seeds).
>
> The residual is a measurement artifact, and its provenance matters. Our census was generated with SDPA attention and our detection sweeps with eager attention (required for attention-pattern feature extraction); prompts, batch size, dtype and decoding were otherwise identical. Under greedy decoding these two kernels disagree on the generated string for 0.37–0.42% of tokens and on the pass/fail verdict for 0.021% (2/9,522). We verified the divergence is systematic rather than stochastic: token 28012 produced byte-identical divergent output at two different batch positions in two different random samples, and token 27952 across three post-validate sweeps of different size and composition. Batched fp16 inference was therefore *not* a source of run-to-run nondeterminism here.
>
> We further note that in Algorithm 1 the glitch set has two entry points — tokens labeled during construction of the SVM training sample, and SVM positives confirmed by post-validation — and only the second is re-validated. Under the primary protocol all false positives entered by the first path; post-validation itself admitted no false positives in 1,047 confirmations. Under the gccode protocol, six of eight entered through post-validation.
>
> We therefore do not claim that post-validation fails. We claim something narrower and, for reproducibility, more consequential: a "100% precision" figure defined relative to a system's own oracle does not survive contact with an independently produced ground truth, and the gap is governed by the inference stack — attention kernel, precision, decoding path — none of which either paper specifies.

Delete "1-2 false positives per seed" and "temperature-0 batched fp16 inference is not bit-deterministic" from `E:/user3/FYS_SWK/Benchmark/docs/RUNLOG.md:87-90`; the table at lines 74-79 is already correct.

## 4. Experiments needed for a stronger claim

**(a) Decisive attribution — rerun the census under eager attention.** Add an `--attn-impl` flag to `run_ground_truth.py` and re-sweep both protocols with `eager`. Existing detection checkpoints already contain the eager verdicts, so re-scoring is offline CPU work. Prediction if the kernel is the whole story: FPs → 0 for both protocols. Cost: 206 s + 314 s GPU plus ~2 model loads ≈ **15 min wall-clock**; re-scoring seconds. This is cheap enough that it should be run before the claim is written up at all.

**(b) Separating kernel from composition, with an effect size.** 2×2 on the ~75 text-divergent tokens: {eager, SDPA} × {two disjoint batch compositions}, k=3 repeats. Yields a direct estimate of each factor rather than the two anecdotes above. ~900 generations ≈ seconds of compute; **~5 min** including model loads.

**(c) Population-level oracle-instability rate.** The 0.021% flip rate is estimated from 9,522 sampled tokens with only 2 events — the CI is enormous (roughly 0.003–0.08%). A full 31,743-token census under both kernels gives the exact population number and turns an anecdote into a citable reproducibility statistic. Cost **~10 min** GPU for both kernels, one protocol; ~20 min for both protocols.

**(d) Optional generalization.** Repeat (c) at bf16 and fp32 to show whether oracle instability is a precision artifact or survives at higher precision. ~2× the (c) cost in time, but fp32 will not fit alongside fp16 on a single A6000 for a 7B model without offload — budget **~30–45 min** and check VRAM.

One caveat worth stating in threats-to-validity regardless: whether GlitchProber's Algorithm 1 re-validates sampled-and-labeled tokens is not specified in the paper, so path 1 in `detect.py` is our reimplementation choice, not an established property of their method.