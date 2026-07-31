# HANDOVER — Glitch-Token Mitigation Benchmark

**Read this first.** It tells you what the project is, what has been done, what the
current numbers are, exactly where work stopped, and what to do next in priority
order. Written 2026-07-31.

---

## 1. What this project is

An **independent reproduction study** of two published papers that claim to detect
and repair *glitch tokens* (vocabulary entries so under-trained that the model
produces garbage when they appear in a prompt):

| Paper | What it does | Headline claim | Code released? |
|---|---|---|---|
| **GlitchProber** (Zhang et al., ASE 2024) | Detects glitch tokens via SVM on internal activations; repairs by editing MLP neuron activations (factors α, β) | avg repair 50.06%; Mistral detection recall 67.41%, repair 37.60% | **No** — project page only |
| **GlitchCleaner** (Fan, Li & Li, AAAI 2026) | Repairs via gated LoRA adapters; gate λ=1 only when input contains a known glitch token | avg repair 86.88%; **Mistral 94.80%** | **Yes** — cloned to `third_party/` |

**We are not building a new method.** We re-run both under one controlled protocol
and check whether the published claims hold. The papers are in `papers/`, with
clean text extractions in `papers/extracted/`.

### Why it's worth doing (the motivation, in one paragraph)

GlitchCleaner's central claim is a comparison against GlitchProber, but it makes
that comparison by importing numbers across two different experimental setups. The
two papers used *different glitch-token populations* for the same model (2,779 vs
2,539). A "repair rate" is a fraction of that population, so the rates measure
different things. We verified that GlitchCleaner's reported GlitchProber **token
counts** are GlitchProber's *rate* × GlitchCleaner's *own denominator* — true for
all five shared models — despite the paper stating those figures are "taken from
their original paper." Fixing the denominator is the study's core contribution.

---

## 2. Where everything is

| What | Where |
|---|---|
| Repo (identical path on every machine — **load-bearing**) | `E:\user3\FYS_SWK\Benchmark` |
| GitHub (private) | `https://github.com/fuad-mahee/Benchmark` |
| Python env | `.venv\Scripts\python.exe` (Python 3.12, torch 2.6.0+cu124) |
| Model weights cache | `F:\hf_cache` (~15 GB for Mistral) |
| **Main deliverable** | `docs/benchmark_study.tex` (standalone LaTeX study) |
| Run narrative (chronological, with corrections) | `docs/RUNLOG.md` |
| Pipeline reference (what each step does) | `docs/PIPELINE.md` |
| All results | `results/` |
| Upstream GlitchCleaner code | `third_party/GlitchCleaner/` (gitignored — re-clone if missing) |

```powershell
# get set up on a new machine
git clone https://github.com/fuad-mahee/Benchmark.git E:\user3\FYS_SWK\Benchmark
cd E:\user3\FYS_SWK\Benchmark
.\setup_env.ps1
git clone --depth 1 https://github.com/FAVENO/GlitchCleaner third_party/GlitchCleaner
python scripts\status.py     # shows which stage each model has completed
```

Hardware used: RTX A6000 48 GB, fp16, native Windows (no WSL). One A6000 is enough
for everything at 7B scale.

---

## 3. How the benchmark actually works

Three ingredients:

1. **Real models** — the same open-weights models the papers used, downloaded and
   run locally. Identical to what the authors had.
2. **Reimplemented methods** — GlitchProber rebuilt from the paper's pseudocode
   (no code exists); GlitchCleaner rebuilt from its paper, with the authors'
   released code used as the specification of record.
3. **Their published claims** — read from the PDFs, plus GlitchCleaner's published
   glitch-token CSVs (real data we compare against directly).

**We never take their numbers on faith and never re-run their numbers.** We
regenerate everything: probe the model for its own glitch tokens → run our
reimplementation → compare to their claims → add tests they never ran.

### The one structural caveat you must keep in mind

When our number disagrees with a paper's number there are always two explanations:
*the claim is wrong*, or *our reconstruction differs from theirs*. Two instruments
distinguish them:

- **Anchoring** — under GlitchCleaner's own protocol our census recovers **2,537 of
  their 2,539** published tokens (Jaccard 0.993). Our machinery is therefore
  demonstrably faithful, so remaining disagreements are informative.
- **Refusal to attribute** — GlitchProber released no code, so any gap is reported
  as *"we could not reproduce X from the published description"*, **never** as
  *"X is false."* Keep this discipline; reviewers check for it.

### The five research questions

| RQ | Question | Script |
|---|---|---|
| RQ1 | How many glitch tokens exist, and how much does that depend on the probe? | `run_ground_truth.py` |
| RQ2 | Does GP detection reproduce its recall/F1? | `run_gp_detect.py` |
| RQ3 | Does GP repair reproduce, and is α/β/m justified? | `run_gp_repair.py`, `run_gp_alpha_beta_sweep.py` |
| RQ4 | Does GC repair reproduce, and does it hold on **unseen** tokens? | `run_gc_train.py`, `run_gc_eval.py` |
| RQ5 | Inference-speed cost | `run_speed.py` |

Two probing protocols exist and both matter — because **GlitchCleaner's released
code does not use the prompt its paper describes**:

- `--protocol paper` — the template as written in the papers, 24-token budget
- `--protocol gccode` — their released code verbatim, 10-token budget

`paper` is what a reader would implement; `gccode` is what anchors us to their
published artefacts. Every script takes the flag.

---

## 4. Status: what is DONE

Everything below is complete for **Mistral-7B-Instruct-v0.1** and committed.

```
model                    RQ1    RQ2    RQ3    grid   m-sweep  RQ4    RQ5
mistral-7b-instruct-v01  DONE   DONE   DONE   DONE   DONE     DONE   DONE
```

### Current results (these SUPERSEDE everything in older commits)

**RQ1 — the census is a property of the prompt, not the model**

| | paper protocol | gccode protocol |
|---|---|---|
| Glitch | **988** | **2,552** |
| Normal | 30,755 | 29,191 |
| Filtered (unreachable/special/undecodable) | 257 | 257 |

Factorial decomposition of the 1,564-token gap (now **measured**, not extrapolated
— gccode prompt at 24 tokens gives 2,550, i.e. the budget moves it by 2 tokens):

| Factor | Δ glitch | Share |
|---|---|---|
| Generation budget 24→10 | +1 | 0.06% |
| Match rule (strip vs lstrip) | +14 | 0.9% |
| **Prompt wording** | **+1,549** | **99.0%** |

Mechanism: under the gccode template, **42.7%** of "glitch" tokens emit pure
whitespace (trailing-newline repetition attractor). Disagreement is bidirectional
(146 tokens go the other way).

**RQ2 — GP detection** (3 seeds, mean ± s.d.)

| | Precision | Recall | F1 |
|---|---|---|---|
| Ours, paper protocol | 0.997 | 0.279 ± 0.091 | 0.431 ± 0.116 |
| Ours, gccode protocol | 0.997 | 0.404 ± 0.019 | 0.575 ± 0.019 |
| **GP claims (Mistral)** | 1.000 | **0.674** | 0.805 |

Recall does not reproduce; the base-rate control (gccode) closes part of the gap
but not most of it. The "100% precision" is definitional (post-validation defines
membership by the same test).

**RQ3 — GP repair. The paper's unstated choices dominate the result.**

At the paper's own α=4, β=1.5, four defensible readings of the same paper:

| Tensor adjusted | Positions | Repair | Collateral |
|---|---|---|---|
| separate gate/data (Fig. 5) | token only | 6.48% | 0.00% |
| product | token only | 2.23% | 0.00% |
| separate | all positions | 17.41% | 1.20% |
| product | all positions | 15.18% | 1.00% |

**7.8× spread.** The paper's claimed 12.92% for this configuration falls *inside*
our range — under-determined, neither confirmed nor refuted.

**m-sweep — the strongest RQ3 result:** repair *rises* (5.0% → 12.2%) as the
promotion set shrinks from 376 neurons to **1**. Suppression carries the entire
mechanism; promotion contributes nothing measurable.

Adaptive α/β: **unrecoverable** — k₁,b₁,k₂,b₂ are never given in the paper. Our
identity-constant run (1.01%) is *degenerate* (pins α≈1 = no intervention) and
**must not be cited as the method's performance.**

**RQ4 — GC repair. Major self-correction.**

| protocol | n_train | n_heldout | train | **held-out** | adapter off | clean (gated) |
|---|---|---|---|---|---|---|
| paper | 791 | 187 | 35.15% | **37.43%** | 0.00% | 100.0% |
| gccode | 2,042 | 475 | 82.96% | **78.11%** | 0.00% | 99.6% |

An earlier version of this study reported a 15–26 point "memorisation gap." **That
was mostly our own broken training** (3 epochs instead of 15, wrong LR, unseeded,
dropped gradients). With the authors' own hyperparameters the gap is 4.85 points,
and under the paper protocol held-out *exceeds* train. GlitchCleaner generalises
much better than we previously reported. What remains: 78.11% held-out vs the
claimed 94.80%. Losslessness is **supported** with the real gate.

**RQ5 — speed:** base 31.97 / GP hooks 31.55 / GC gated 31.57 tok/s → ~1% overhead
each. This **withdraws** an earlier claim of 37–44% penalties. ⚠ See §6.1 — this
needs re-verification before publishing.

### Infrastructure built

- **Checkpoint + resume** on every long sweep (survives interruption exactly)
- **Configuration fingerprint guard** — refuses to resume a checkpoint written
  under different settings (added after one silently poisoned a result)
- **Provenance** — every result JSON records commit, **dirty-tree flag**, and
  library/GPU versions
- **Raw evidence retention** — full generations kept; this is what allowed the RQ1
  factorial to be computed *after the fact* without re-running the model
- `scripts/status.py` (resume map), `scripts/aggregate_results.py` (thesis tables)

---

## 5. WHERE WORK STOPPED

Work stopped when the **Claude session usage limit was hit** mid-way through an
automated review of the study document.

- ✅ All Mistral experiments finished (chain completed, exit 0)
- ✅ Four of six review agents finished — reviews saved at
  `C:\Users\user3\.claude\projects\E--user3-FYS-SWK-Benchmark\<session>\subagents\workflows\wf_f5f1a561-87c\extracted\*.md`
- ❌ **Two agents failed on the limit: the fact-checker and the synthesis editor.**
  So there is no consolidated revision list — you have four raw reviews instead.
- ❌ `docs/benchmark_study.tex` still contains `[pending]` placeholders in RQ3/RQ4/RQ5
  because the final numbers landed after it was written.

**Nothing is corrupted and nothing needs re-running.** The gap is purely in the
write-up.

---

## 6. WHAT TO DO NEXT — priority order

### 6.1 MUST DO — verification before anything is published

1. **Re-verify the RQ5 speed result.** It reversed a large earlier finding
   (37–44% → ~1%) in a single unreplicated run, and the three variants were
   measured in one process where ordering/thermal effects could matter.
   ```powershell
   python scripts\run_speed.py --model mistral-7b-instruct-v01     # repeat 3x
   ```
   Measure each variant in a *fresh process* and report mean ± s.d. If it holds,
   it is a clean result; if not, do not publish either number.

2. **Run the attention-kernel A/B** (~15 min). The claim that detection false
   positives come from an SDPA-vs-eager mismatch is currently an *inference* from
   the fact that two scripts used different kernels — there is no controlled run.
   ```powershell
   python scripts\run_ground_truth.py --model mistral-7b-instruct-v01 --attn-impl eager --tag eager
   python scripts\run_ground_truth.py --model mistral-7b-instruct-v01 --protocol gccode --attn-impl eager --tag eager
   ```
   Then re-score detection against the eager census. Prediction: false positives → 0.
   Either report the direct contrast, or state plainly that the A/B was not run.

3. **Delete the poisoned artefact.** `results/gp_repair/mistral-7b-instruct-v01/alpha_beta_grid_meta.json`
   was written by the run that silently resumed a stale checkpoint. Its
   `neuron_selection` block is fresh but its `best_cell` is stale — the worst case.
   Delete it and regenerate, or the document will point readers at a half-valid file.

4. **Re-run the corrected GP repair from a clean tree.** Every post-audit artefact
   carries `"git_dirty": true`, which violates the study's own stated provenance
   standard. These runs take minutes. Commit first, then re-run.

### 6.2 MUST DO — finish the study document

`docs/benchmark_study.tex` compiles with `pdflatex` (run twice). Remaining work:

5. **Fill the three `[pending]` blocks** with the §4 numbers above: the RQ3
   implementation 2×2 + corrected grid + m-sweep, the RQ4 table, the RQ5 table.

6. **Apply the review panel's required changes.** The four reviews are in the
   `extracted/` folder named in §5. The highest-value ones, already verified:

   - **The RQ3 paragraph "The claim the landscape does not support" is refuted by
     our own corrected data.** It argues from the v1 number (20.8% at α=4,β=1.5),
     which came from the configuration we classify as our own error. Corrected
     value is 6.48% (separate/token) — *below* the paper's 12.92%, not above. The
     whole "would require ~60% repair" inference collapses. **Rewrite it** using
     the 2×2 range, and make the honest point instead: the paper's figure sits
     inside the range spanned by its own unstated choices.
   - **The E1 narrative mis-states the neuron asymmetry.** Product mode gives
     `Neun_up = 0` in *all ten* layers (not "0–4"), and the corrected separate mode
     gives **63** promotion neurons against **16,928** suppression neurons. So β is
     structurally starved in *both* — that is a property of the paper's own ">99%
     of normal tokens" criterion, not of our bug. This supports a *better* claim
     than the one withdrawn, and the m-sweep confirms it independently.
   - **Disclose the three denominators.** Repair rates use 988 (census), 494
     (held-out half), and 500 (grid sample) in different places. For a study whose
     thesis is "fix the denominator," every reported rate must state which one.
   - **"Prompt wording" is a residual, not a measured factor.** Label it as such,
     or isolate it (the trailing-newline ablation is one sweep, ~5 min).
   - **Soften RQ2 claims to what 3 seeds support**, and add CIs to the grid.
   - **Add the concrete oracle number:** 3,253 of 30,755 "normal" verdicts
     (10.6%) rest on single-character matches that could occur incidentally.

7. **Re-run the review panel's fact-check + synthesis** — the two agents that died.
   Re-invoke with the cached script; completed agents replay from cache:
   ```
   Workflow({scriptPath: "...\\workflows\\scripts\\study-review-panel-wf_f5f1a561-87c.js",
             resumeFromRunId: "wf_f5f1a561-87c"})
   ```

### 6.3 SHOULD DO — extend coverage

8. **More models.** Everything is model-agnostic; enable in `configs/models.yaml`.
   - Ungated, ready now: **Yi-6B-Chat**, **Qwen-7B-Chat** (⚠ Qwen has a custom
     architecture — verify its `module_map` before trusting results)
   - Gated (needs HF licence + `hf auth login`): **Llama-2-7b-chat** (both papers'
     primary model — highest value), **Gemma-2b-it** (⚠ use layers 6–15 from the
     authors' released checkpoint, not the guessed 11–16 currently in the config)
   ```powershell
   python scripts\run_ground_truth.py --model yi-6b-chat --batch-size 128
   python scripts\run_ground_truth.py --model yi-6b-chat --protocol gccode --batch-size 128
   # then run_gp_detect / run_gp_repair / run_gc_train / run_gc_eval / run_speed
   ```
   A single model is ~1 GPU-hour for the ungated 7Bs; Gemma's 256k vocabulary is
   ~8× the sweep time.

9. **Three seeds for the GC adapter** (currently one). Training is the constraint.
   `python scripts\run_gc_train.py --model mistral-7b-instruct-v01 --seed 1 --tag seed1`

10. **RQ5 capability evals** (GSM8K/MMLU) — `lm_eval 0.4.12` is installed and
    `src/eval/capabilities.py` wraps it, but no run has been done. Note the
    limitation documented in that file: GP is hook-based, so lm-eval cannot
    measure it with hooks active.

### 6.4 KNOWN GOTCHAS

- **Never trust a resumed checkpoint across a code change.** The guard now blocks
  it, but if you see `RuntimeError: ... produced by a different configuration`,
  that is the guard working — move the old file aside deliberately.
- **The correctness oracle is weak for short tokens.** 31 whitespace-only
  candidates pass trivially; 3,504 have ≤1 stripped character. Shared with the
  authors' code, but still a real construct-validity threat. Do not report
  short-token results without this caveat.
- **`--protocol` must match across steps.** A detector scored against the wrong
  census is meaningless.
- **GlitchProber has no released code.** Never write "the paper is wrong" — write
  "we could not reproduce X from the published description."
- **Sync discipline:** `git pull` when you sit down, `git add -A && git commit && git push`
  before you leave. Chats sync separately via Google Drive (`~/.claude/projects`
  is a junction into `My Drive\ClaudeProjects`); wait for "Up to date" before
  switching machines, and use `claude --resume` from the repo directory.

---

## 7. The findings, as they currently stand

Ordered by how much they'd survive a hostile reviewer.

1. **The census is a property of the probe.** 988 vs 2,552 glitch tokens on the
   same model, same filtering; 99.0% of the gap is prompt *wording*. Anchored by
   recovering 2,537/2,539 of GlitchCleaner's published list. **Strongest result.**
2. **Competitor counts are derived, not measured.** GlitchCleaner's reported
   GlitchProber token counts = GP's rate × GC's denominator, for all five models,
   while stating they are "taken from their original paper." Verified arithmetic.
3. **GlitchProber's unstated implementation choices move repair 7.8×**, and its
   adaptive calibration is unrecoverable (four constants never published).
4. **Promotion is vacuous; suppression carries the mechanism.** m-sweep: repair
   *rises* as the promotion set shrinks to one neuron.
5. **GP detection recall does not reproduce** (0.404 vs claimed 0.674) even under a
   base-rate control — reported as a reproduction failure, not a refutation.
6. **GlitchCleaner mostly generalises** (held-out 78.11%), losslessness holds with
   the real gate, but falls short of the claimed 94.80%.
7. **Neither paper reports variance or collateral damage.** We report both.

**Be careful with #6** — an earlier version of this study claimed a large
memorisation gap that turned out to be our own training bug. That correction is
documented in `docs/RUNLOG.md` and should stay documented; it is evidence the
process works, not something to hide.

---

## 8. One-paragraph summary for your supervisor

We rebuilt both glitch-token mitigation methods from scratch and re-ran them on a
model both papers use, under one controlled protocol, regenerating every number
rather than citing published ones. The core finding is that the glitch-token
population — the denominator of every published repair rate — is set by the prompt
used to find it: 988 tokens under the papers' stated probe versus 2,552 under
GlitchCleaner's actual code, with 99% of that gap attributable to prompt wording
alone. We also verified that GlitchCleaner's reported GlitchProber figures are
rescaled onto GlitchCleaner's own denominator rather than quoted. Our own
implementation was adversarially audited before reporting; two errors were found,
corrected, and the affected experiments re-run — one correction reversed a finding
of ours in the original authors' favour. All results, raw generations, and the
correction history are version-controlled and reproducible.
