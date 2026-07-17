# Glitch-Token Mitigation Benchmark

Independent verification of the claims of two glitch-token mitigation papers, for thesis use:

- **GlitchProber** — Zhang et al., ASE 2024 (`papers/Glitchprober.pdf`): SVM-based detection on PCA-reduced
  intermediate activations + repair by adjusting MLP neuron activations (α/β).
- **GlitchCleaner** — Fan, Li & Li, AAAI-26 (`papers/Glitchcleaner.pdf`): repair via gated LoRA branches
  (r=4) on the MLP gate/data projections of key layers.

Everything runs on **native Windows** (no WSL) on a single RTX A6000 (48 GB).

## Research questions → scripts → results

| RQ | Question | Script | Output |
|----|----------|--------|--------|
| RQ1 | How many glitch tokens does each model really have? | `scripts/run_ground_truth.py` | `results/ground_truth/<model>/` |
| RQ2 | Does GlitchProber detect with claimed recall/time? | `scripts/run_gp_detect.py` | `results/gp_detect/<model>/` |
| RQ3 | Does GP repair reach ~50%? How sensitive to α/β/m? | `scripts/run_gp_repair.py`, `scripts/run_gp_alpha_beta_sweep.py` | `results/gp_repair/<model>/` |
| RQ4 | Does GC reach ~87% — and on *held-out* tokens? | `scripts/run_gc_train.py`, `scripts/run_gc_eval.py` | `results/gc/<model>/` |
| RQ5 | Capability (GSM8K/MMLU) & speed side effects | `scripts/run_speed.py`, `src/eval/capabilities.py` | `results/side_effects/<model>/` |

## Quickstart

```powershell
# 1. One-time environment setup (Python 3.12 venv + PyTorch CUDA + deps)
.\setup_env.ps1

# 2. Activate the environment
.\.venv\Scripts\Activate.ps1

# 3. Smoke-test the full pipeline on a 135M model (minutes, ~300 MB download)
python scripts\run_ground_truth.py --model smoke-test --limit 2000
python scripts\run_gp_detect.py    --model smoke-test
python scripts\run_gp_repair.py    --model smoke-test
python scripts\run_gc_train.py     --model smoke-test
python scripts\run_gc_eval.py      --model smoke-test

# 4. Real runs (enable models in configs/models.yaml first; gated models need `hf auth login`)
python scripts\run_ground_truth.py --model llama2-7b-chat
...
```

## Repo layout

```
configs/         one YAML per concern (models, ground truth, GP, GC) — every number the
                 thesis reports traces back to a value here plus a seed
src/common/      model/tokenizer loading, prompt construction, config, result I/O
src/ground_truth token filtering (special/undecodable/unreachable) + full-vocab repetition sweep
src/glitchprober reimplementation from the paper's pseudocode (Alg. 1 & 2)
src/glitchcleaner gated-LoRA training + held-out evaluation (the circularity test)
src/eval/        shared metrics, lm-eval-harness wrapper, inference-speed timing
scripts/         thin argparse entry points, one per experiment
results/         CSV/JSON outputs, one folder per RQ — thesis tables generate from these
papers/          the two PDFs + clean text extractions
third_party/     upstream GlitchCleaner clone (reference; not imported directly)
```

## Method notes / deviations (report these in the thesis)

- **Constant-length prompts:** the target token id is spliced directly between pre-tokenized
  template halves, so every prompt has identical length and the token under test is guaranteed
  to be the exact vocabulary id (re-encoding drift is impossible).
- **GlitchProber's α/β constants (k1,b1,k2,b2) are undisclosed** in the paper. Our default
  reimplementation exposes them in `configs/glitchprober.yaml`; the α/β grid sweep quantifies
  how much the choice matters. This is itself a reproducibility finding.
- **GlitchProber code** is only "available" via a Google Sites page; we reimplement from the
  paper. GlitchCleaner has a real repo (`third_party/GlitchCleaner`) used as reference.
- **Timing claims** are verified as *ratios* on our hardware (1×A6000), never as absolute
  minutes (papers used 2×A100).
- Attention is run with `attn_implementation="eager"` when attention patterns are extracted
  (required for `output_attentions=True`). No FlashAttention on native Windows; both tools
  are affected equally, so comparisons remain fair.
