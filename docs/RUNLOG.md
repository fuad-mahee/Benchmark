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
