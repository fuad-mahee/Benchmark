# Pipeline Documentation

Read this to understand what the benchmark does, what each artifact means, and how
to resume after any interruption. The chronological record of actual runs lives in
[RUNLOG.md](RUNLOG.md).

## What is being benchmarked

Two papers claim to mitigate **glitch tokens** (vocabulary entries so undertrained
that they derail the model — ask the model to repeat one and it outputs garbage):

- **GlitchProber** (Zhang et al., ASE 2024): detects glitch tokens with an SVM over
  PCA-reduced internal activations; repairs by nudging MLP neuron activations
  (amplify by β / suppress by α). Claims F1 0.7835, repair rate 50.06%.
- **GlitchCleaner** (Fan, Li & Li, AAAI-26): repairs with gated LoRA adapters (r=4)
  on the MLP gate/data projections of the same layers. Claims repair rate 86.88%.

Their numbers are not comparable as published (different glitch counts, no variance,
undisclosed constants, apparent train-on-test). We rerun both under one protocol.

## The five steps (run in this order per model)

| # | Script | What it does | Key outputs |
|---|--------|--------------|-------------|
| 1 | `run_ground_truth.py` | Filter untestable tokens, then ask the model to repeat every remaining vocab token (temp 0). Failures = glitch tokens. | `results/ground_truth/<m>/tokens.csv` (every token labeled), `summary.json` (counts), `sweep_checkpoint.csv` (full raw generations) |
| 2 | `run_gp_detect.py` | GlitchProber detection: 10% sample -> activations -> PCA(75) -> SVM -> classify rest -> re-validate positives. Scored against step 1. | `results/gp_detect/<m>/runs.csv` (per seed), `summary.json` (mean±std) |
| 3 | `run_gp_repair.py` | GlitchProber repair: build normal-token neuron profile, adjust glitch activations via α/β (adaptive AND rule-based modes). Also measures how many normal tokens the fix breaks. | `results/gp_repair/<m>/summary.json` |
| 3b | `run_gp_alpha_beta_sweep.py` | Grid-sweep fixed α×β values -> sensitivity heatmaps. Tests whether the paper's values are justified. | `alpha_beta_grid.csv`, `heatmap_*.png` |
| 4 | `run_gc_train.py` | Split glitch tokens 80/20. Train GlitchCleaner's gated LoRA on the 80%. | `results/gc/<m>/adapter/`, `split.json`, `train_meta.json` |
| 4b | `run_gc_eval.py` | Repair rate on the trained 80% (paper's population) vs the unseen 20% (the honest number), + collateral checks. | `results/gc/<m>/eval.json` |
| 5 | `run_speed.py` | tokens/sec: base vs GP-hooks vs GC-adapter (ratios). | `results/side_effects/<m>/speed.json` |

Then `aggregate_results.py` folds everything into `results/benchmark_report.{json,md}` —
the five thesis tables, pre-formatted, with paper claims alongside for comparison.

## Checkpointing / resuming after interruption

- **Where am I?** -> `python scripts/status.py` prints DONE / PARTIAL / pending per
  model per step. Rerun the first non-DONE step; it resumes automatically.
- Step 1 checkpoints per batch (`sweep_checkpoint.csv`): a killed sweep restarts at
  the exact batch it stopped.
- Step 2 checkpoints its three expensive phases per seed (`checkpoints/*_seed<n>.csv`):
  sample labeling, SVM predictions (per 2048-token chunk), post-validation.
- Step 3b checkpoints per grid cell (`alpha_beta_grid.csv` is written after every cell).
- Steps 3/4/4b/5 are short (minutes); interruption just means rerunning that step.
- Determinism: all sampling is seeded (configs), so a resumed run uses the same
  sample sets as the interrupted one.

## Conventions

- Every summary JSON embeds `timestamp` and `git_commit` (code state that produced it).
- Every config knob lives in `configs/*.yaml` - nothing experiment-relevant is
  hardcoded. Paper claims are recorded next to our numbers in the outputs.
- Repair rate always = repaired / |our ground-truth glitch set for that model|.
