# Benchmark Results — Independent Evaluation of Two Glitch-Token Repair Methods

**Status: interim report. Run paused 2026-08-04 after 10.8 hours of GPU time,
28 completed pipeline steps, zero failures. Everything below is measured, not
projected. Section 8 states exactly what is still outstanding.**

---

## 1. What this study is about, in plain terms

A language model does not read letters. It reads **tokens** — chunks of text drawn
from a fixed dictionary the model was built with. A typical model has 30,000 to
250,000 of them.

Some of those tokens almost never appeared in the model's training data. The model
therefore never learned what they mean, and when one shows up in a prompt the model
behaves erratically: ask it to simply repeat the token back to you and it produces
something unrelated. These are called **glitch tokens**. They matter because a user
can hit one by accident and get nonsense with no warning, and because attackers can
use them deliberately to push a model into unsafe behaviour.

Two research papers propose ways to find and fix these tokens:

| Paper | What it does | What it claims |
|---|---|---|
| **GlitchProber** (ASE 2024) | Finds glitch tokens by inspecting the model's internal signals, then repairs them by adjusting those signals during use | Fixes 50.06% of glitch tokens on average |
| **GlitchCleaner** (AAAI 2026) | Repairs them by attaching a small trained add-on ("adapter") to the model | Fixes 86.88% on average — and claims to beat GlitchProber by more than 30 points |

**Our study does not propose a new method.** It asks a simpler question: *do these
published numbers hold up when someone else runs the experiments?*

## 2. Why the question is worth asking

Before running anything, four problems were visible in the published material.

**a) The two papers counted different numbers of glitch tokens for the same model.**
For the model `Mistral-7B-Instruct-v0.1`, GlitchProber's own tables imply it worked
with 2,779 glitch tokens; GlitchCleaner says 2,539. This matters enormously,
because a "repair rate" is a *fraction* — repaired tokens divided by total glitch
tokens. If the denominators differ, the two percentages are not measuring the same
thing and cannot be compared.

**b) One paper's figures for the other were calculated, not measured.**
GlitchCleaner states its GlitchProber figures were "taken from their original
paper." The *percentages* were. The *token counts* were not. For all five models
they share, GlitchCleaner's reported GlitchProber count equals GlitchProber's
percentage multiplied by GlitchCleaner's own denominator:

| Model | GlitchProber's own count | GlitchCleaner reports | GP's rate × GC's total |
|---|---|---|---|
| Llama-2-7b-chat | 4,021 | 2,968 | 2,968.2 |
| Gemma-2b-it | 13,638 | 14,548 | 14,548.6 |
| Mistral-7B-Instruct | 1,045 | 956 | 954.7 |
| Qwen-7B-Chat | 14,765 | 13,320 | 13,319.7 |
| Yi-6B-Chat | 4,317 | 3,188 | 3,187.6 |

The number 956 appears nowhere in GlitchProber's paper, which reports 1,045.
Rescaling is a defensible way to put two percentages on a common axis, but the
result is presented as a measured count with a citation, and it silently assumes
one paper's success rate would carry over to a *different set* of tokens.

**c) Neither paper reports variation between runs.** Both methods involve
randomness. Both report a single number with no spread, so a reader cannot tell a
real difference from run-to-run noise.

**d) GlitchCleaner appears to test on the same tokens it trained on.** The paper
describes building a training set from the glitch tokens it found, training on it,
and then reporting the repair rate over the glitch tokens. No held-out set is
described.

## 3. What we actually did

We rebuilt both methods from scratch and re-ran them on five models under one
consistent procedure, regenerating every number rather than quoting any.

The key step is that we **do not trust either paper's list of glitch tokens.** We
generate our own, by testing the model's entire dictionary one token at a time:
put the token in a prompt, ask the model to repeat it, and record whether it
succeeds. This takes 30 minutes to nearly 2 hours per model depending on size.

**A complication we discovered and had to handle.** GlitchCleaner's published
software does not use the prompt its paper describes — the wording differs, and it
allows the model only 10 words of output instead of the longer budget implied by
the paper. Rather than choose one, we run **both**:

- **"paper" procedure** — the prompt exactly as written in the papers
- **"code" procedure** — the prompt exactly as their released software uses it

This turned out to be the single most consequential decision in the study
(Section 6, Finding 1).

**Models tested.** Five, in two groups:

- **Anchored group** — `Mistral-7B-Instruct-v0.1` and `gemma-2b-it`. Both papers
  used these, and GlitchCleaner published its glitch-token lists and trained
  adapters for them. This lets us check our setup against their published output.
- **Generalisation group** — `Qwen2.5-7B-Instruct`, `Qwen2.5-14B-Instruct`,
  `gemma-7b-it`. Neither paper used these. They test whether the findings hold on
  models that did not exist when the papers were written, and — since each family
  has two sizes sharing a dictionary — whether model size changes anything.

## 4. Evidence that our setup is correct

This is the most important section, because every criticism we make is worthless
if our reimplementation is simply wrong.

We ran our procedure under GlitchCleaner's own conditions and compared against the
data they published.

**Test 1 — do we find the same glitch tokens they did?**

| Model | They published | We found | Tokens in common |
|---|---|---|---|
| Mistral-7B-Instruct | 2,539 | 2,552 | **2,537 (99.9%)** |
| gemma-2b-it | 29,831 | 29,835 | **29,785 (99.85%)** |

**Test 2 — does their own released adapter produce their published repair rate
inside our test harness?**

| Model | They published | We measure | Difference |
|---|---|---|---|
| Mistral-7B-Instruct | 2,407 repaired (94.80%) | 2,408 (94.84%) | 1 token |
| gemma-2b-it | 20,697 repaired (69.38%) | 20,657 (69.25%) | 0.13 points |

Two independent models, two independent checks each. Our harness reproduces both
their token lists and their repair rates almost exactly. **This means the
differences reported below are not artefacts of our setup**, and it also confirms
that GlitchCleaner's method genuinely works as published. Our criticism is about
how the field *measures*, not about whether their adapter functions.

For GlitchProber no such check is possible — the authors released no code. We
therefore report any gap as *"we could not reproduce this from the published
description,"* never as *"the claim is false."*

## 5. An audit of our own work

Before reporting anything we had our implementation reviewed by an independent
adversarial process instructed to treat all our code and conclusions as unverified.
It found errors serious enough to invalidate specific conclusions. All were fixed,
the affected experiments re-run, and the corrections documented in
`docs/RUNLOG.md`. The most significant:

- We had been adjusting the wrong internal signal in GlitchProber's repair step.
- We had omitted GlitchCleaner's input-dependent switch, which invalidated every
  measurement about its effect on *normal* tokens.
- Our training of GlitchCleaner's adapter used our own settings rather than the
  authors'. When corrected, a "memorisation gap" we had reported **largely
  disappeared** — the shortfall was ours, not theirs. That conclusion was withdrawn.

We report this because a study that only audits other people's work is not
credible, and because two of our own retracted findings were in the original
authors' favour.

## 6. Findings

### Finding 1 — The number of glitch tokens is a property of the question you ask, not of the model

Same model. Same dictionary. Same filtering. Only the wording of the test prompt
differs:

| Model | "paper" prompt | "code" prompt | inflation |
|---|---|---|---|
| Mistral-7B-Instruct | 988 (3.11%) | 2,552 (8.04%) | **×2.58** |
| Qwen2.5-14B | 5,672 (4.37%) | 8,185 (6.31%) | ×1.44 |
| Qwen2.5-7B | 11,190 (8.63%) | 14,715 (11.35%) | ×1.32 |
| gemma-2b-it | 24,959 (9.77%) | 29,835 (11.68%) | ×1.20 |
| gemma-7b-it | 49,670 (19.44%) | 53,245 (20.84%) | ×1.07 |

For Mistral, simply changing the wording of the question **more than doubles** the
apparent number of broken tokens.

We isolated the cause rather than guessing. Because we kept every model response,
we could re-test the recorded outputs under each variation separately. Of the
1,564-token difference on Mistral:

| Cause | Effect |
|---|---|
| Shorter output budget (10 vs 24) | 1 token (0.06%) |
| Stricter matching rule | 14 tokens (0.9%) |
| **Prompt wording** | **1,549 tokens (99.0%)** |

The mechanism is visible in the raw outputs: under the second prompt, **42.7%** of
the tokens labelled "glitchy" produce nothing but blank lines. The prompt ends in a
newline, and that appears to push the model into repeating newlines rather than
answering. It is not that those tokens are broken — it is that the question invites
a blank answer.

**The effect follows a rule.** The distortion is largest exactly where the model is
cleanest. Ranking the five models by their intrinsic glitch rate and by how much
the prompt inflates it gives a perfect inverse ordering (Spearman ρ = −1.000, n = 5,
p ≈ 0.008). The reason is that the prompt converts a roughly similar *slice* of
otherwise-normal tokens — between 1.7% and 5.1% — in every model. When a model's
true glitch rate is low, that slice doubles the count; when it is already high, it
barely moves it.

**Why this matters:** every published glitch-token count, and therefore the
denominator of every published repair rate, contains an additive measurement
artefact of roughly 2–5% of the dictionary. Because both papers report *averages
across models*, they are averaging numbers with very different artefact loads.

### Finding 2 — Bigger models do not reliably have fewer glitch tokens

GlitchCleaner states that glitch tokens become less common "as the scale and
coverage of training data increase." Within a single family, holding the dictionary
and training recipe fixed and changing only size:

| Family | Smaller | Larger | Direction |
|---|---|---|---|
| Qwen2.5 | 7B: 8.63% | 14B: 4.37% | halves — consistent with the claim |
| Gemma | 2B: 9.77% | 7B: **19.44%** | **doubles — contradicts the claim** |

`gemma-7b-it` has glitch behaviour on nearly **one token in five**. This finding
involves no reimplementation of either method — it is just the model and a prompt —
so it cannot be explained away as our error.

### Finding 3 — GlitchProber's repair step is described too vaguely to reproduce, and the ambiguity is expensive

The paper leaves two things unstated: *which* internal signal is adjusted, and *at
which points* in the text. Both are reasonable readings of the same paper. Using
the paper's own recommended settings, the four combinations give:

| Model | reading A | reading B | reading C | reading D | spread |
|---|---|---|---|---|---|
| Mistral-7B | 6.48% | 2.23% | 15.18% | 17.41% | **7.8×** |
| Qwen2.5-7B | 12.89% | 12.62% | 25.77% | 19.52% | 2.0× |
| Qwen2.5-14B | 9.31% | 5.54% | 29.27% | 30.68% | 5.5× |
| gemma-2b-it | 16.83% | 12.00% | 12.34% | 23.45% | 2.0× |

A reader implementing the paper faithfully could land anywhere in that range. The
paper's claimed figure for Mistral (12.92% for this configuration) falls *inside*
our range — so it is neither confirmed nor refuted; it is **undetermined by the
paper's own description.**

Separately, four constants that the repair formula depends on are never given
numerically anywhere in the paper, and no code was released.

### Finding 4 — The repair method's active ingredient is not the one the paper emphasises

GlitchProber's repair has two halves: *boost* neurons that should be active, and
*suppress* neurons that should be silent. We varied the threshold that decides
which neurons belong to each group. As the "boost" group shrinks toward nothing,
repair does not degrade — **it improves**:

| Model | boost group: many → almost none | repair rate |
|---|---|---|
| Mistral-7B | 376 → 1 neuron | 5.0% → **12.2%** |
| Qwen2.5-7B | 1,602 → 9 neurons | 9.2% → **19.2%** |
| Qwen2.5-14B | 619 → 1 neuron | 9.0% → **20.4%** |

Replicated on three models across two families. The suppression half does
essentially all the work; the boosting half contributes nothing measurable. The
same conclusion appears in the parameter sweep, where the suppression strength
moves results 5–6 percentage points while the boost strength moves them 1–4.

### Finding 5 — Neither paper reports what its repair does to healthy tokens

A repair that fixes broken tokens by damaging working ones is not obviously an
improvement. The quantity is cheap to measure and neither paper reports it. We
measure it in every configuration; it ranges from 0% to 7% of normal tokens
depending on model and settings, and it rises as the repair is made more
aggressive.

## 7. What this adds up to

The contribution is about **measurement**, not about a new method.

The central metric of this research area — "repair rate" — is a fraction whose
denominator is not a stable property of the model being studied. It depends on the
wording of the prompt used to construct it, by a factor that ranges from 1.07× to
2.58× across the models we tested, in a direction that systematically distorts
results most for the best-behaved models. On top of that, the two papers' rates are
computed over different token populations, and one paper's figures for the other
were arithmetically derived rather than measured.

The practical recommendation that follows is concrete: papers in this area should
publish their glitch-token census and their probe prompt verbatim, not just the
resulting percentage; should evaluate learned repair on tokens held out from
training; and should report collateral damage alongside repair rate.

We emphasise that this is not a finding that the methods do not work. GlitchCleaner
reproduces its published results almost exactly in our hands on two independent
models. What does not survive is the *comparability* of the published numbers.

## 8. What is not yet finished

The run was paused with roughly 26 hours of computation outstanding. Everything
completed is listed above; nothing below is estimated or assumed.

| Work | Status |
|---|---|
| Glitch-token census, 5 models × 2 procedures | **complete** |
| Verification against GlitchCleaner's published data | **complete** (2 models) |
| GlitchProber repair + settings sweeps | complete for Mistral, Qwen 7B, Qwen 14B; partial for gemma-2b-it; not started for gemma-7b-it |
| GlitchCleaner retraining on the new models | not started |
| GlitchProber detection accuracy on the new models | not started |
| Inference-speed measurements on the new models | not started |

The pipeline resumes from where it stopped; completed steps are recorded and will
not be repeated.

**Specific limitations to keep in mind when reading the above:**

1. Finding 1's statistical strength rests on five models. A perfect rank
   correlation over five points is unlikely by chance (p ≈ 0.008) but is not a
   large sample. Two further models would strengthen it materially.
2. Findings about GlitchProber cannot be firmly attributed, because it released no
   code. They are statements about what can be reproduced from a published
   description.
3. The layer band our repair experiments target keeps the same *number* of layers
   the original authors used, which means it covers a different *proportion* of a
   48-layer model than a 28-layer one. This affects the size comparison in the
   repair findings, though not the census findings.
4. The test for "did the model repeat the token" is a text-containment check, as in
   both papers. It is weak for very short tokens, where a match can occur by
   accident. This affects our numbers and the published ones equally.

---

*All figures in this document are produced by code in this repository and stored
under `results/`. Every result file records the software versions and the exact
code revision that produced it. `scripts/verify_claims.py` recomputes the headline
numbers directly from the stored raw data.*
