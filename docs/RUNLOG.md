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
