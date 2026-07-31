# Committee Review — *An Independent Reproduction and Differential Evaluation of Two Glitch-Token Mitigation Methods*

*Reviewed against `docs/benchmark_study.tex` @ commit `b14e7d4` (995 lines). The file changed under me mid-review — the five-model rescaling table was added while I was verifying — so I cite both line numbers and phrases. All numbers below were recomputed from `results/` with the repo venv.*

---

## (a) Overall assessment

There is a real contribution here, and it is not the table of numbers: it is the demonstration that the *denominator* of every published repair rate in this literature is an artefact of the probe, anchored credibly by recovering 2,537 of GlitchCleaner's 2,539 published tokens (Jaccard 0.9933 — verified), plus the forensic result that GlitchCleaner's competitor counts are its own denominator times GlitchProber's rate in all five shared models (verified: 2968.2/14548.6/13319.7/3187.6 against reported 2968/14548/13320/3188). That is a contribution a committee should want published, and the held-out split for GlitchCleaner is a second, smaller one. But the document is currently undermined by a specific and serious defect: **a live argument in RQ3 reasons from the very configuration the study classifies as its own error (E1), and is contradicted by the study's own corrected artefact already sitting in `results/`** — and separately, the audit section's flagship withdrawal ("β is inert") is retracted on a causal story the corrected data does not support. The self-audit reads as a strength in principle and as an admission in practice, for reasons of placement and proportion that are fixable. I would not sign off in the present state.

---

## (b) Required changes, most important first

### R1. The RQ3 paragraph "The claim the landscape does not support" (lines 794–806) argues from the erroneous configuration and is refuted by your own corrected data

Line 798–800 states: *"On our measured landscape, at the paper's own fixed values we obtain a repair rate **higher** than the 12.92% it reports for that configuration."* The only source for that is the v1 grid — 20.8% at (α,β) = (4, 1.5) — which is **product-stream, all-positions**, i.e. the configuration you label "our v1 error" in Table 4 (line 757) and E1 (line 458).

Your corrected artefacts say the opposite:

| artefact | config | rule-based (α=4, β=1.5) |
|---|---|---|
| `results/gp_repair/mistral-7b-instruct-v01/summary.json` | separate / token, held-out half | **32/494 = 6.48 %** |
| `.../gccode/summary.json` | separate / token, gccode | **63/1276 = 4.94 %** |
| `.../ablation_product/summary.json` | product / token | 11/494 = 2.23 % |

Both corrected values are **below** 12.92%, not above. The entire downstream inference collapses with it: your "roughly 60% repair" figure is 20.8 × 2.91; from 6.48% the same ratio demands 18.9%, which is *inside* your grid's observed range (max 24.0%). As written the paragraph makes GlitchProber's calibration gain look unattainable using a number your own audit invalidated.

**Fix:** rewrite the paragraph against `summary.json` (2026-07-30T18:36:24). If the honest corrected statement is "at the paper's fixed values we obtain *less than half* what the paper reports for that configuration, and the 2.91× gain is therefore not excluded by our landscape," say that. It is a weaker claim and it is the one you have.

### R2. The E1 narrative mis-attributes the Neun↑ asymmetry (lines 458–467); the corrected data supports a *sharper* claim than the one you withdrew

Line 461–465 asserts that the neuron criterion *"applied to the product, selected 0–4 neurons per layer, leaving the amplification factor β with almost nothing to act on. Our pre-correction conclusion that 'β is inert' was therefore partly an artefact of our own error."*

Both halves are wrong against your post-audit artefacts:

- `ablation_product/summary.json` (product): `n_neun_up = 0` in **all ten** layers — not 0–4.
- `summary.json` (separate, corrected): per-module `n_neun_up` = 0–21, **total 63** across 20 modules, against `n_neun_down` **total 16,928**.

So β acts on ≤63 of 286,720 candidate neurons *in the corrected implementation too*. The asymmetry is a property of the paper's own ">99% of normal tokens" criterion on this model, not of your bug. You withdrew a genuine finding on a false premise, and the corrected numbers — which appear nowhere in the document — support something better: **the promotion branch of GlitchProber's published two-factor scheme is near-vacuous by construction on Mistral, at a ratio of 63:16,928.** That is a substantive claim about the method.

**Fix:** correct the E1 paragraph, report both neuron totals in RQ3, and reinstate a properly scoped version of the β finding (β is not *inert* — see the corrected grid — but it is structurally starved).

Related: line 787–788's *"β alone lifts repair from 0/500 to 34/500"* is a v1 number. The corrected grid now on disk (`alpha_beta_grid.csv`, α=1 row: 1, 1, 4, 9 of 500 for β = 0.25/0.5/1.0/1.5) already contradicts the "0/500" anchor. Restate from the corrected file.

### R3. The metric definitions do not match the denominators actually used (line 392)

Line 392 defines repair rate over *"our glitch census for this model under this protocol"*. In practice three different denominators are in play:

- 988 (census, `summary.json: n_glitch_total`)
- 494 (held-out half, `n_eval` — what Table 4 actually reports, per its own caption at line 746)
- 500 (`n_glitch_sample`, the α/β grid)

For a document whose stated purpose is *"Ambiguity about denominators is the specific defect this study exists to correct"* (line 389), this is the one inconsistency you cannot afford.

**Fix:** define all three explicitly in §4.6, and annotate every reported repair rate in the Results with which denominator it uses.

### R4. Quarantine `alpha_beta_grid_meta.json`, and disclose that every post-audit artefact is dirty-tree

Two provenance problems, both self-inflicted against your own stated standard:

1. `results/gp_repair/mistral-7b-instruct-v01/alpha_beta_grid_meta.json` (2026-07-30T18:39:13) declares `"stream_mode": "separate"` but its `best_cell` is `{α:16.0, β:0.5, repair_rate:0.24, normal_break_rate:0.018}` — **byte-identical to the v1 product-stream grid's best cell**, which is not possible for two runs with disjoint neuron sets (Neun↑ = 0 vs 63). This is precisely the incident you narrate at lines 429–431 ("a 'corrected' parameter grid that had silently resumed its own pre-correction checkpoint... reporting stale numbers under fresh metadata"). The fingerprint guard now works — `alpha_beta_grid.csv.fingerprint.json` was created 00:51 and forced a clean restart — but **the poisoned file was never removed**, its `neuron_selection` block *is* fresh (so it is half-valid, the worst case), and the artefact map (line 985) still points readers at it.
2. Every post-audit artefact carries `"git_dirty": true` (commit `0c03490`: `summary.json`, `gccode/summary.json`, `ablation_product/summary.json`, `alpha_beta_grid_meta.json`). By your own sentence at line 421 — *"a commit hash alone is not provenance if the run was launched from an edited tree"* — none of the corrected GlitchProber repair numbers currently has clean provenance.

**Fix:** delete or rename the stale meta file; re-run the corrected GP repair evaluations from a clean tree (they cost minutes), or state in §Threats that they are dirty-tree runs and why that is tolerable. Do not describe the safeguard as having worked while its casualty is still in the artefact map.

### R5. "Prompt wording" is a residual, not a measured factor — and the isolating experiment costs about five minutes

Table 3 (lines 593–609) labels the 99.0% row *"Prompt wording (residual)"*, and the abstract (line 58–59) upgrades this to *"attributes 99.0% of that gap to the prompt wording **alone**"*. A residual absorbs everything the two measured factors do not; it is evidence that budget and match rule are not the cause, not evidence that wording is. You concede this at line 618–620, which is to your credit — and then leave it there.

You state a specific mechanism (trailing newline → newline-repetition attractor), you have `--tag` and `--max-new-tokens` plumbing, and a full-vocabulary census costs **314 seconds** (`gccode/summary.json: sweep_seconds`). Two or three additional censuses — P_code minus the trailing newline; P_code minus the `Question:` prefix — convert your headline causal claim from a residual into a measurement and directly test the stated mechanism.

**Fix:** run them. For a study whose central recommendation is "specify the probe," leaving your own central causal claim unisolated when isolating it costs 15 GPU-minutes is the place a committee will push hardest. Until then, delete "alone" from the abstract.

### R6. There is no related work and no bibliography

The document cites exactly one prior work (Land & Bartolo 2024, line 266), inline, with no `\cite` and no `\bibliography`. A committee will require, at minimum:

- The Magikarp / glitch-token detection line the token filter is inherited from, properly cited.
- **GlitchHunter** — it is a baseline column in GlitchProber's own Table 3, and you *use* its numbers implicitly (TP 1,233 / 44.37% is a fourth independent derivation of the 2,779 population), yet it is never named.
- The reproducibility-study genre you are working in, so the reader knows what standard you are holding yourself to.
- Positioning of your own construct: "the measured population is a property of the elicitation prompt" has clear analogues in the prompt-sensitivity and benchmark-construction literature. Name them, or a reader will assume you don't know they exist.

### R7. The reader is never told what to do differently in deployment, and the one number that would tell them is never computed

§7.5 (lines 826–836) contains the most decision-relevant observation in the study: GlitchCleaner's λ is a membership test, so coverage is bounded by whatever detector built G, and your RQ2 says that detector recovers ~40%. **You never multiply.** 0.40 × (0.55–0.75) ≈ **22–30% end-to-end coverage**, against a headline of 94.80%. That composition is the single most useful thing this study produces for a practitioner, and it is buried in a paragraph inside a subsection whose numbers are pending.

There is also no Conclusion section at all — the document ends on "Recommendations for this literature" and "On our own errors," both addressed to *authors*, never to a *reader deploying this*.

**Fix:** compute the composed bound explicitly, put it in the abstract, and add a short Conclusion that answers "which method, under what conditions, and what should I not believe."

### R8. The stated resource constraints are contradicted by your own artefacts (line 891, line 405)

Line 891: *"one adapter seed for \gc{} (three are planned, **training cost being the constraint**)."* `results/gc/mistral-7b-instruct-v01/train_meta.json` records `"train_seconds": 31.44`. Thirty-one seconds. Likewise, "three seeds" (line 405) for a detection run that costs 211 s each — ten seeds is 35 minutes.

The same objection applies to the 80/20 split (line 373), which is never justified: with 197 held-out tokens the binomial CI on 55.3% is roughly ±7 points, and 5-fold cross-validation costs 5 × 31 s.

**Fix:** either raise the seed count and cross-validate the held-out estimate, or replace the false constraint with the real one. A committee will not accept an unreal budget excuse in a study whose thesis is that other people under-report variance.

### R9. Unjustified protocol constants presented as properties of the papers

- **Line 301, "Generation budget: 24 tokens" for P_paper.** Why 24? Neither paper specifies it; this is your choice, presented in a column headed "as written in the papers." A reader asks immediately. (Your +1 factorial result is the answer — say so.)
- **Line 302, match rule `strip()` vs `lstrip()`.** `strip()` is *your* reading, not a published rule. Attribute it.
- **Lines 731–733, "identity constants with an explicit clamp."** The clamp is never given numerically. `configs/glitchprober.yaml` has `k1=b1=k2=b2` and `alpha_min: 1.0` — and `alpha_min = 1.0` is *exactly* what pins α ≈ 1 and makes the configuration degenerate. State those five numbers in the text.
- **Which protocol is primary downstream, and why.** Table 4, the grid, and GC training all default to P_paper; the document never says so or defends it.

### R10. Table 1's caption over-claims on the one model the study is about (lines 145–147, 156)

The caption says the rescaling *"reproduces the reported value to within rounding in all five cases."* Four cases land within 0.6 of a token. Mistral does not: GP-rate × GC-|G| = **954.7**, reported **956**. To hit 956 the rate would have to lie in [37.63%, 37.67%]; GlitchProber prints 37.60%. It is off by more than rounding, and it is the row your whole study rests on.

**Fix:** say it. "Four of five reproduce to within 0.6 of a token; Mistral is off by 1.3, which the printed two-decimal rate does not explain." That is *more* convincing than an overclaim, because the pattern across the other four is already conclusive. While you are there: note that GlitchCleaner's average for the GlitchProber column (50.08%) is the mean of the *recomputed* rates, not GlitchProber's own 50.06% — independent confirmation that the counts came first and the rates were derived from them.

### R11. The mechanism paragraph compares two different quantities (lines 611–614)

*"1,088 of 31,743 generations (3.4%) consist of newlines only, against 3 under P_paper."* Recomputed from `sweep_checkpoint.csv`: P_code newline-only = 1,088 ✓; **P_paper newline-only = 0**. The "3" is whitespace-only (2) plus empty (1). One definition, both sides: 1,088 vs 0, or 1,115 vs 3. In this document of all documents, an apples-to-oranges comparison inside the mechanism argument is costly.

### R12. §4.5 reports GlitchCleaner settings no stored run used, and omits a disclosed deviation (lines 355–359)

*"We also adopt their optimisation settings — 15 epochs, learning rate 1×10⁻⁴, effective batch 512..."* — present indicative. The only stored GC artefacts (`train_meta.json`, commit `706e27d`) record **3 epochs, lr 2e-4, batch 8 × accum 2 = 16**. `configs/glitchcleaner.yaml` does carry the 15/1e-4/8×64 values, so this is a tense problem, not a fabrication — but it must read as a specification, cite the config, and disclose the deviation the config itself flags: `batch_size: 8  # DEVIATION: upstream 64`.

### R13. The factorial caption asserts a basis and an absence of interaction that are not yet supported (lines 595–598)

The caption claims the decomposition is *"over the 31,743 shared candidates"* and that *"the ordering of factors does not change the result, indicating no measurable interaction."* The budget cell requires `gccode/budget24/`, which is still filling (it covered 31,488 of 31,743 ids when I sampled it, and its glitch count moved between two reads seconds apart). And "no measurable interaction" is nowhere shown — the eight cells are never tabulated. Line 590 says *"six of the eight factorial cells are recoverable from stored data"* without saying what happened to the other two.

**Fix:** print the 8-cell table, state that two cells required a fresh census, and drop the interaction claim unless the table shows it.

### R14. The self-audit needs restructuring — see (c)/(3) below for the diagnosis; the required change is concrete

Move the D1–D9 inventory to an appendix; keep E1, E2 and the withdrawal list in place (~15 lines). Then add the table the section promises but never provides. Line 453 asserts *"the size of that difference is itself a result"* — so make it a result:

| quantity | pre-audit | post-audit | Δ |
|---|---|---|---|

Right now the only pre/post pair a reader can find is the two product rows of Table 4, and they are not labelled as the contrast. An audit section that quantifies its own effect is evidence; one that lists nine defects and quantifies none is a confession.

---

## (c) Optional improvements

1. **Five derivations, not three** (line 130–131). GlitchProber's own Table 3 gives two more: 1288/0.4635 = 2778.8 (rule-based) and 1233/0.4437 = 2779.0 (GlitchHunter). Free.
2. **The population disagreement is far worse than 9%, and you already know it** (line 582). Across models, GP's implied |G| vs GC's stated |G|: Llama 6,425 vs 4,743 (35%), Yi 8,105 vs 5,985 (35%), Qwen 30,690 vs 27,686 (11%), Gemma 27,964 vs 29,831 (7%). Mistral's 9% is the *second smallest*. Your own `ground_truth/.../summary.json` already carries a `paper_reference` block with the Llama pair. Adding one column to Table 1 makes RQ1's headline a five-model result costing zero GPU — and partly answers your own external-validity threat (line 899–905).
3. **Table 2 gives precision with no spread** (line 650–651) while giving ± on recall and F1. Either add it or say why the per-seed counts (1,0,1 and 3,4,1) are reported instead.
4. **Line 687, "2 of 9,522 sampled"** — 9,522 is 3 × 3,174 = three seeds × 10% of 31,743 (verified against `checkpoints/label_sample_seed*.csv`). Say so; a reader cannot reconstruct it.
5. **Line 881, "3,504 have a stripped length of at most one character."** I recompute 3,535 at ≤1, of which 31 are whitespace-only. Your 3,504 is the ≤1 count *excluding* those 31. Write "a further 3,504" — a one-word fix in a document that promises every number is recomputable.
6. **Artefact-map drift** (lines 968, 984). The appendix claims *"Every number in this document is produced by code in this repository"* — Table 1's numbers are read out of two PDFs. Add a row (ideally a small extraction CSV so the arithmetic is checkable). And line 984 points to `.../abl_*/`; the directory on disk is `ablation_product/`.
7. **The 0.5-page abstract is doing too little.** It currently omits the held-out result, the under-specification-dominates-parameters result, and the composed deployment bound — three of your five real contributions.

---

## (d) Verdict

**Major revisions.**

The intellectual contribution is real — the denominator-is-the-probe result is anchored, the five-model rescaling forensics is decisive and free, and the held-out split is a genuine addition — but a headline argument in RQ3 currently rests on a configuration the study itself declares erroneous and that its own corrected artefacts contradict (R1), the flagship withdrawal is justified by a causal story the data refutes (R2), the metric denominators are inconsistent with the ones actually computed (R3), and the study has no related work or bibliography (R6); none of these is cosmetic, and R1–R3 are correctable from data already on disk.