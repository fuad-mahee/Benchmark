## 1. Quantitative evidence

### Variables that differ between the two protocols (from code)

`scripts/run_ground_truth.py:36-43` + `src/common/prompts.py:23-34, 59-65` — the `gccode` branch changes **four** things at once:

| # | Variable | `paper` | `gccode` |
|---|---|---|---|
| 1 | Prompt prefix | `"Can you repeat the string '"` | `"Question: Can you repeat the string '"` |
| 2 | Prompt suffix | `"' and return it back to me? Answer: Here is the repeated string:"` | `"' and return back to me?\nAnswer: Here is the repeated string:\n"` (three sub-edits: *it* dropped, `" Answer:"`→`"\nAnswer:"`, **trailing newline appended**) |
| 3 | `max_new_tokens` | 24 (`configs/ground_truth.yaml`) | 10 |
| 4a | Target-string derivation | `token_str()` = `convert_tokens_to_string(convert_ids_to_tokens([id]))` | `tok.decode([id]).lstrip()` |
| 4b | Match rule | `target.strip() in generated` | `target.lstrip() in generated` (trailing whitespace retained ⇒ stricter) |

Variables 4a/4b are one code change but two distinct semantic effects; they yield different effective targets for **48 / 31,743** tokens (all trailing-`\r` tokens, e.g. id 1271 `';\r'` → paper `';'`, gccode `';\r'`).

**Held constant and verified:** model, dtype, greedy decoding, and the vocab filter. The candidate sets are identical — 31,743 shared token ids, **0 exclusive to either side**. So "output extraction" did *not* change: both protocols decode `gen[:, prompt_len:]` with `skip_special_tokens=True` (`src/ground_truth/sweep.py:79-80`). The auditor's list should be 3 loci (prompt / length / oracle), not 4.

### Confusion matrix (n = 31,743 shared candidates)

|  | gccode normal | gccode glitch | total |
|---|---|---|---|
| **paper normal** | 29,045 | **1,710** | 30,755 |
| **paper glitch** | 146 | 842 | 988 |
| total | 29,191 | 2,552 | 31,743 |

Reproduces `summary.json` exactly (988 / 2,552). Note the disagreement is **bidirectional**: 146 tokens are glitch under paper but normal under gccode — the run log's narrative only describes one direction.

### The specific question asked: does the paper generation contain the target beyond token 10?

For the 1,710 focus tokens (gccode-glitch, paper-normal), I re-encoded the stored paper generation and found the smallest prefix length *k* at which the paper oracle is satisfied:

| k | 1 | 2 | 3 | 4 | 5 | 15 |
|---|---|---|---|---|---|---|
| tokens | 51 | 1,627 | 25 | 5 | 1 | **1** |

- **1,709 / 1,710 (99.94%) succeed at k ≤ 10.** 1,678 (98.1%) succeed by k = 2.
- **Exactly 1 token (0.06%) required more than 10 generated tokens.**
- Only 273 / 1,710 (16.0%) of the paper generations were even *longer* than 10 tokens, so for 84% the length variable is structurally incapable of mattering.

Symmetrically, across all 30,755 paper-protocol successes, only **1** first succeeds at k > 10. The 24-token budget bought the paper protocol exactly one extra "normal" label.

### Post-hoc factorial (6 of 8 cells recoverable from archived generations)

Greedy decoding makes prefix-truncation an exact simulation of a shorter `max_new_tokens`, and both oracles can be re-applied to either text. Glitch counts on the shared set:

| prompt | max_new | oracle | glitch |
|---|---|---|---|
| paper | 24 | paper | 989 |
| paper | 24 | gccode | 1,003 |
| paper | 10 | paper | 990 |
| paper | 10 | gccode | 1,004 |
| gccode | 10 | paper | 2,558 |
| gccode | 10 | gccode | 2,553 |
| **gccode** | **24** | **either** | **NOT COMPUTABLE** |

Sequential attribution of the +1,564 gap, **identical in both orderings** (no measurable interaction among the observable factors):

| factor | Δ glitch | share of gap |
|---|---|---|
| generation length 24→10 | **+1** | **0.06%** |
| correctness oracle | **+14** | **0.9%** |
| prompt text (residual) | **+1,549** | **99.0%** |

*Fidelity caveat:* recomputed cells are 989/2,553 vs stored 988/2,552. Exactly one row differs on each protocol — token id 3, the `<0x00>` byte-fallback token, whose generation loses its NUL bytes on CSV round-trip. 31,742/31,743 labels reproduce exactly. All cells above are ±1.

*Truncation-validity caveat:* the checkpoint stores decoded text, not ids, so truncation goes through a decode→re-encode. Drift is bounded: 12/31,743 (0.04%) of paper texts re-encode to >24 tokens; 81.7% re-encode to ≤10, where truncation is a no-op.

### Mechanism behind the 99% residual

Median re-encoded generation length: **paper 3 tokens** (EOS-terminated), **gccode 11** (cap-limited). Under the gccode template the model frequently emits *nothing but newlines*:

- 1,088 / 31,743 gccode generations (3.4%) are newlines only; 1,115 (3.5%) are entirely whitespace vs **3** (0.009%) under the paper template.
- Within the 1,710 focus set, **955 (55.8%)** of gccode generations are entirely whitespace.
- Of all 2,552 gccode-glitch tokens, **1,089 (42.7%)** produced pure whitespace.

Example (id 304, target `and`): paper → `"' and '"` (3 tokens, then EOS); gccode → `"\n\n\n\n\n\n\n\n\n\n"`. This is consistent with the template's **trailing `"\n"`** putting the model in a newline-repetition attractor — but that sub-cause is *not* isolated, since three prompt edits moved together.

### Empirical bound on the one untestable cell

Among the 29,191 gccode successes, first-success index: k=3 → 15,534; k=4 → 11,128; k=5 → 1,388; k=6 → 192; k=7 → 36; k=8 → 5; k=9 → 3; k=10 → 19. Only **22 / 29,191 (0.075%)** first succeed at k ≥ 9. The hazard of a late first-success has essentially collapsed by k≈6. Extrapolating that survival curve, a 24-token gccode run would be expected to reclassify **order-of-tens**, not thousands, of the 2,552. This is an extrapolation, not a measurement — and 1,565 / 1,710 (91.5%) of focus tokens were still generating at the cap, in a degenerate attractor whose escape probability is genuinely unmeasured.

### Published-list check
Published GlitchCleaner list = 2,539. Our gccode = 2,552, overlap 2,537, **Jaccard 0.9933**. Our paper = 988, overlap 841, Jaccard 0.3131. The reproduction claim is solid.

## 2. Ruling: **partially supported** — umbrella claim stands, stated cause is refuted

- **"Protocol sensitivity" is supported**, and more strongly than the run log argued: model, dtype, decoding and vocab filter were held fixed, the candidate sets are byte-identical (0 exclusive ids either way), and both label sets regenerate from the stored text at 31,742/31,743. Nothing but harness configuration differs. The word *pure* is defensible for this meaning.
- **The auditor is right on process.** The gccode run changed three loci simultaneously and tested none individually; "**CONFIRMED**" was unearned at the time it was written.
- **The run log's actual causal claim is refuted, not merely unproven.** "~2/3 … stop being glitchy when the model may generate 24 tokens instead of 10" attributes the effect to generation length. Length accounts for **1 of 1,564 flips (0.06%)**. The oracle accounts for 0.9%. **99.0% is prompt text.** The run log named the one variable that turned out to be inert.
- **The auditor's implied remedy is too pessimistic.** "No single cause is isolated" was true of the *original argument*, but the archived raw generations support a post-hoc 6-of-8-cell factorial that isolates three of the three labeling loci, because greedy decoding makes truncation exact and both oracles are re-applicable offline. Only the prompt × length interaction remains open.
- Also unstated in the run log: the disagreement is bidirectional (146 tokens go the other way), and the "~2/3" framing hides that these are not the same 2/3 as the published-list comparison's 1,698.

## 3. Recommended thesis wording

> Under identical model weights, greedy decoding and vocabulary filter, the repetition-task census of Mistral-7B-Instruct-v0.1 yields 988 glitch tokens under the template as written in the papers and 2,552 under GlitchCleaner's released evaluation code — a 2.6× difference produced entirely by harness configuration. Our gccode-protocol census recovers 2,537 of the authors' 2,539 published tokens (Jaccard 0.993), so the gap is not an implementation error on our side.
>
> The two protocols differ in three respects: prompt text, generation budget (24 vs 10 new tokens), and the correctness oracle. Because decoding is greedy, prefix-truncating a stored 24-token generation exactly simulates a 10-token budget under the same prompt, and either oracle can be re-applied offline to either set of generations; six of the eight cells of the 2×2×2 design are therefore recoverable post hoc from the archived generations without new inference. The resulting attribution of the 1,564-token gap is order-independent: generation length +1 (0.06%), correctness oracle +14 (0.9%), prompt text +1,549 (99.0%). Concretely, for the 1,710 tokens labelled glitch under gccode but normal under the paper protocol, 1,709 (99.94%) already satisfy the paper oracle within the first 10 generated tokens — 98.1% within the first two. The 10-vs-24-token budget is essentially inert; the census gap is a prompt-wording effect.
>
> The mechanism is visible in the raw generations. Under the gccode template the model produces output that is entirely whitespace for 1,115 of 31,743 tokens (3.5%), against 3 (0.009%) under the paper template; within the 1,710 disputed tokens, 55.8% of gccode generations are pure whitespace, typically an unbroken run of newlines. Median generation length is 3 tokens under the paper template (the model answers and emits EOS) versus 11 under gccode (the cap). This is consistent with the trailing newline that terminates GlitchCleaner's template, but we did not ablate the three prompt edits ("Question:" prefix, "return it back"→"return back", newline placement) separately, so we attribute the effect to the prompt as a whole and not to any single edit.
>
> Two limits are stated explicitly. First, the cell (gccode prompt × 24 tokens) cannot be recovered from the archived data and was not run; 91.5% of the disputed tokens were still generating at the 10-token cap, so a longer budget under the gccode prompt could in principle reclassify some of them. An empirical bound is available: among the 29,191 gccode successes, only 22 (0.075%) first satisfy the oracle at generated-token index ≥ 9, so the expected reclassification is of order tens rather than thousands — an extrapolation from the observed first-success hazard, not a measurement. Second, the disagreement is bidirectional: 146 tokens are glitch under the paper protocol but normal under gccode, so neither census is a subset of the other.
>
> The implication for the literature is unchanged and, if anything, sharpened: the glitch-token census — and therefore the denominator of every published repair rate — is determined mainly by the exact prompt string used to probe the model, a detail that neither paper specifies precisely enough to reproduce, and which differs between GlitchCleaner's paper text and its own released code.

Edits to make in `docs/RUNLOG.md:53-67`: delete "**CONFIRMED (same day)**", delete the sentence beginning "~2/3 of this model's...", and soften "are protocol artifacts to a first approximation" to the measured attribution above.

## 4. Experiments needed for a stronger claim, and cost

Baseline: the paper sweep took 205.7 s and the gccode sweep 314.4 s on the A6000 (`summary.json`), generation time roughly linear in `max_new_tokens`.

1. **Close the factorial (highest value, near-zero cost).** One sweep: gccode prompt at `max_new_tokens=24`. Both oracles apply post hoc, so a single run fills both missing cells and converts the hazard extrapolation into a measurement. ≈ 314 s × 2.4 ≈ **12–13 min GPU + ~1 min load**. There is no defensible reason not to run this before the thesis is submitted.
2. **Decompose the 99% prompt residual.** One-at-a-time ablation of the three prompt edits at `max_new_tokens=10`: **3 sweeps ≈ 16 min**. A full 2³ sub-factorial (7 more cells) ≈ **37 min**. Given the whitespace evidence, run the trailing-`"\n"` ablation first — it is a single-character change and is the most likely single cause.
3. **Generalization.** Repeat items 1–2 on a second model to show the effect is not Mistral-specific. Llama-2-7b-chat is the papers' primary model but is gated; per-model cost is **< 1 GPU-hour**. A 256k-vocab model (gemma-2b-it) would cost roughly 8× the sweep time.
4. **Robustness of the oracle finding (optional, free).** The +14 oracle effect is driven by 48 trailing-`\r` tokens; report it as a footnote rather than a finding — it is below the noise floor of the census.

Total to make every claim above a measurement rather than an extrapolation: **roughly one GPU-hour on the existing hardware**, no new data, no new code beyond a `--max-new-tokens` override on the existing `--protocol gccode` path.