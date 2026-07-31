CANONICAL RECOMPUTED BENCHMARK TABLE — mistral-7b-instruct-v01
Repo: E:/user3/FYS_SWK/Benchmark  |  Recomputed with E:/user3/FYS_SWK/Benchmark/.venv/Scripts/python.exe
Every number below was recomputed from the rawest artifact that exists; summary files were used only as the thing being checked, never as a source.

================================================================================
0. RECONSTRUCTION-STATUS SUMMARY (read this first)
================================================================================
EXACT (recomputed from raw per-token evidence, bit-for-bit):
  - Census counts, both protocols (from tokens.csv, cross-checked against sweep_checkpoint.csv raw generations, and re-judged from scratch with the tokenizer)
  - Set comparisons vs the authors' published list
  - Detection TP/FP/FN/precision/recall/F1 for all 3 seeds x 2 protocols (from label_sample/post_validate/svm_predictions checkpoints)
  - Detection means and stds
  - GlitchCleaner n_train / n_heldout / split disjointness / partition property
ARITHMETIC-ONLY (integers stored, ratio verified, but the per-token evidence behind the integers was NOT saved):
  - GlitchProber repair: adaptive and rule-based repaired/normal_broken counts
  - Alpha/beta grid: all 30 cells (rates are exactly k/500, so counts are recoverable, but not which tokens)
  - GlitchCleaner repair rates: train / held-out / adapter-off / normal-ok
  - Speed: three tok/s means and their ratios
CANNOT BE RECONSTRUCTED (raw evidence never written):
  - Which tokens were repaired by GP (evaluate_repair has a `record_path` parameter; scripts/run_gp_repair.py never passes it)
  - Which tokens the GC adapter repaired (src/glitchcleaner/evaluate.py returns counts only)
  - Which 500 glitch / 500 normal tokens each alpha/beta grid cell used, and per-cell generations
  - The adaptive alpha/beta values actually derived at run time (printed to stdout, never serialised)
  - Per-repetition speed timings (src/eval/speed.py returns the mean of 5 reps only)
  - Wall-clock for the ground-truth-gccode sweep is stored, but no per-batch timing anywhere
  - GC training loss curves / any training telemetry beyond a single train_seconds scalar

================================================================================
1. CENSUS (RQ1) — EXACT
================================================================================
Source: results/ground_truth/mistral-7b-instruct-v01/tokens.csv (+ gccode/tokens.csv)
Both files: 32000 rows, token_id exactly contiguous 0..31999, no duplicates.

PROTOCOL "paper" (template as written in the papers, max_new_tokens=24, containment check on token_str):
  normal         30755
  glitch           988
  unreachable      253
  special            3
  undecodable        1
  SUM            32000   <- verified == vocabulary size
  candidates (normal+glitch) = 31743
  glitch rate among candidates = 988/31743 = 0.03112culated → 0.031125 (3.1125%)
  sweep wall-clock (stored) = 205.7225649000029 s

PROTOCOL "gccode" (GlitchCleaner's released-code template, max_new_tokens=10, decode-lstrip containment):
  normal         29191
  glitch          2552
  unreachable      253
  special            3
  undecodable        1
  SUM            32000   <- verified == vocabulary size
  candidates (normal+glitch) = 31743
  glitch rate among candidates = 2552/31743 = 0.080396 (8.0396%)
  sweep wall-clock (stored) = 314.38567519999924 s

The 257 non-candidate ids (253 unreachable + 3 special + 1 undecodable) are the IDENTICAL id set under both protocols — verified True. The protocols differ only in the repetition judgement, as intended.

Both summary.json count dicts match the recomputed value_counts exactly.

VERIFICATION OF tokens.csv AGAINST RAW GENERATIONS (sweep_checkpoint.csv)
  paper : checkpoint holds 31743 unique ids; id set identical to the candidate set: True
          category("normal") <=> ok==1 mismatches: 0
          ok=1 -> 30755, ok=0 -> 988 (exactly the census split)
  gccode: checkpoint holds 31743 unique ids; id set identical: True
          mismatches: 0 ; ok=1 -> 29191, ok=0 -> 2552

INDEPENDENT RE-JUDGEMENT (loaded mistralai/Mistral-7B-Instruct-v0.1 tokenizer, re-applied the correctness rule to every stored generation, ignoring the stored ok flag):
  paper  : 0 disagreements / 31743
  gccode : 0 disagreements / 31743
So the labels are not merely internally consistent — the judgement function reproduces them from raw text.

ONE COSMETIC DEFECT (does not affect any label): token_id 3 (the <0x00> byte token) has output_snippet in tokens.csv that is NOT equal to checkpoint text[:80] in both protocols. Cause: the generation contains literal NUL characters, which are lost on the tokens.csv round-trip.
  paper : tokens.csv snippet = ''            vs checkpoint = '\x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00'
  gccode: tokens.csv snippet = '```\n'       vs checkpoint = '```\n\x00 \x00 \x00 \x00'
  Token 3 is category "normal" under both protocols; its ok flag is unaffected. All other 31742 snippets match exactly.

================================================================================
2. SET COMPARISON vs THE AUTHORS' PUBLISHED GLITCH LIST — EXACT
================================================================================
Authors' file: third_party/GlitchCleaner/Glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens.csv
  single column "index"; 2539 data rows; 2539 unique ids; 0 duplicates; min=304, max=31993; none >= 32000.
  All 2539 ids fall inside our candidate set (none is classified unreachable/special/undecodable by our filter) — verified.

OURS(paper) vs THEIRS
  |ours| = 988      |theirs| = 2539
  overlap      =  841
  ours-only    =  147
  theirs-only  = 1698
  union        = 2686
  Jaccard      = 841/2686 = 0.313105
  overlap/|theirs| = 841/2539 = 0.331233
  overlap/|ours|   = 841/988  = 0.851215
  Category of all 1698 theirs-only ids in our paper census: 1698 "normal" (0 unreachable/special/undecodable).

OURS(gccode) vs THEIRS
  |ours| = 2552     |theirs| = 2539
  overlap      = 2537
  ours-only    =   15
  theirs-only  =    2
  union        = 2554
  Jaccard      = 2537/2554 = 0.993344
  overlap/|theirs| = 2537/2539 = 0.999212
  overlap/|ours|   = 2537/2552 = 0.994122
  The 2 theirs-only ids: 16884, 28335 (both "normal" in our gccode census).
  The 15 ours-only ids: 714, 1200, 1696, 6928, 7520, 11202, 13324, 21733, 22325, 22360, 24455, 24754, 26620, 30770, 31749.

OURS(paper) vs OURS(gccode)
  overlap = 842 ; paper-only = 146 ; gccode-only = 1710 ; union = 2698 ; Jaccard = 0.312083
  paper-glitch is NOT a subset of gccode-glitch (146 tokens are glitch under the paper protocol but normal under the code protocol).
  Of the 146 paper-only tokens, 0 appear in the authors' list.
  Of the 1710 gccode-only tokens, 1696 appear in the authors' list.
  Three-way intersection (paper ∩ gccode ∩ authors) = 841. So exactly one token is glitch under both our protocols yet absent from the authors' list.

Headline: the authors' published list is reproduced at 99.9% recall ONLY under their released-code protocol; under the protocol as described in the papers it is reproduced at 33.1%.

================================================================================
3. DETECTION (RQ2) — EXACT
================================================================================
Reconstructed by rebuilding G = {ids labelled glitch in label_sample_seed{S}.csv} ∪ {ids failing post_validate_seed{S}.csv}, then scoring against the census glitch set. Sources: results/gp_detect/mistral-7b-instruct-v01/checkpoints/ (+ gccode/checkpoints/).

--- PROTOCOL "paper" (true_glitch = 988, candidates = 31743) ---
seed n_sampled svm_pos postval_fail sample_glitch |G|  TP  FP  FN   precision      recall        F1           time_s
0    3174      163     87           86            173  172   1  816  0.9942196532  0.1740890688  0.2962962963  209.3022154000064
1    3174      444     219          112           331  331   0  657  1.0000000000  0.3350202429  0.5018953753  212.3325809000234
2    3174      440     226          99            325  324   1  664  0.9969230769  0.3279352227  0.4935262757  212.5035465999972
  Every stored TP/FP/FN/precision/recall/F1 matched the recomputed value exactly (integers identical, floats to <1e-12).
  n_candidates (31743), n_sampled (3174), n_predicted_glitch_by_svm, and post_validate row counts all matched. svm_predictions_seed*.csv holds 28569 rows = 31743 - 3174 (the unsampled remainder) in all six runs.

  MEANS (recomputed vs summary.json):
    precision   0.9970475767007558  == 0.9970475767007558  MATCH
    recall      0.27901484480431843 vs 0.2790148448043185  MATCH (last-ulp float difference only)
    f1          0.43057264909503196 vs 0.43057264909503207 MATCH (last-ulp)
    time_s      211.37944763334235  == 211.37944763334235  MATCH
  STDS:
    recall  sample(ddof=1) = 0.09093741378807656 ; population(ddof=0) = 0.07425008743637436 ; stored = 0.09093741378807654
    f1      sample(ddof=1) = 0.11636199829687575 ; population(ddof=0) = 0.09500917375931693 ; stored = 0.11636199829687573
    => The SAMPLE standard deviation (ddof=1, pandas Series.std default) reproduces. The population std does NOT.
    Not stored but computed here: precision std(ddof=1) = 0.002892183861408422 ; time_seconds std(ddof=1) = 1.8009657495491924

--- PROTOCOL "gccode" (true_glitch = 2552, candidates = 31743) ---
seed n_sampled svm_pos postval_fail sample_glitch |G|   TP    FP  FN    precision      recall        F1           time_s
0    3174      1316    799          258           1057  1054   3  1498  0.9971617786  0.4130094044  0.5840953173  221.5056889000116
1    3174      1244    714          266           980    976   4  1576  0.9959183673  0.3824451411  0.5526613817  221.7129216999747
2    3174      1260    787          277           1064  1063   1  1489  0.9990601504  0.4165360502  0.5879424779  228.6687632999965
  All stored values matched the recomputed values exactly.

  MEANS (recomputed == summary.json, all MATCH):
    precision 0.9973800987805369 ; recall 0.40399686520376177 ; f1 0.5748997255973199 ; time_s 223.96245796666093
  STDS:
    recall  sample(ddof=1) = 0.0187474509622346  ; population = 0.015307229611774758 ; stored = 0.01874745096223463  -> ddof=1
    f1      sample(ddof=1) = 0.01935479571633785 ; population = 0.01580312452694445  ; stored = 0.01935479571633785  -> ddof=1
    Not stored: precision std(ddof=1) = 0.0015822287824830365 ; time_seconds std(ddof=1) = 4.077096856377376

STD CONVENTION VERDICT: sample standard deviation, ddof=1 (n-1), on n=3 seeds. This is pandas' default and it reproduces both stored stds in both protocols to <1e-12. With only 3 seeds the ddof choice inflates the reported spread by a factor sqrt(3/2) = 1.2247 relative to the population convention — worth stating explicitly in the chapter.

FALSE POSITIVES ARE NOT CLASSIFIER ERRORS — they are run-to-run nondeterminism (BONUS FINDING, exact):
Every FP token was labelled "normal" in the census but "glitch" during detection, and in every case the two runs produced DIFFERENT generations for the same greedy-decoded prompt (batch composition differs between runs).
  paper  seed 0 & 2: id 28012 — census text "'inement '" vs detection text "'iment '"
  paper  seed 1: none
  gccode seed 0: ids 10157, 27952, 28335 ; seed 1: 19416, 27952, 29290, 30951 ; seed 2: 27952
    e.g. id 27952 census "\n' долж '\n\nIs there anything else" vs detection "\n\n\n\n\n\n\n\n\n\n"
  Note id 28335 is also one of the two "theirs-only" ids in section 2 — the same borderline token.
So the reported precision ~0.997 is a floor: true classifier precision on this run is 1.000 and the deficit is decode nondeterminism.

INTERPRETIVE NOTE ON RECALL: FN counts every census glitch token not in G, including the ~90% of the vocabulary never sampled. The recall ceiling is therefore set by how many SVM positives survive post-validation, not by the SVM's separability alone.

================================================================================
4. GLITCHPROBER REPAIR (RQ3) — ARITHMETIC-ONLY (raw per-token evidence NOT saved)
================================================================================
Sources: results/gp_repair/mistral-7b-instruct-v01/summary.json and gccode/summary.json
Config recorded in both files: m=1.0, neun_up_quantile=0.99, gamma=0.1, adaptive{k1=1.0,b1=0.0,k2=1.0,b2=0.0}, rule_based{alpha=4.0, beta=1.5}, seed=0.

PROTOCOL "paper" (n_glitch = 988 = the full census glitch set — verified equal):
  adaptive    repair_rate       =   5/988 = 0.005060728744939271   (0.506%)
              normal_break_rate =   1/500 = 0.002                  (0.2%)
  rule_based  repair_rate       = 213/988 = 0.21558704453441296    (21.559%)
              normal_break_rate =   5/500 = 0.01                   (1.0%)
  Paper claims for comparison: adaptive_avg 0.5006, rule_based_avg 0.3679.

PROTOCOL "gccode" (n_glitch = 2552 = the full census glitch set — verified equal):
  adaptive    repair_rate       =  97/2552 = 0.03800940438871473   (3.801%)
              normal_break_rate =   1/500  = 0.002                 (0.2%)
  rule_based  repair_rate       = 666/2552 = 0.2609717868338558    (26.097%)
              normal_break_rate =  31/500  = 0.062                 (6.2%)

All four repair rates and four break rates are arithmetically exact given the stored numerators/denominators (division reproduces the stored float to <1e-15).

WHAT CANNOT BE RECONSTRUCTED HERE (state this in the write-up):
  a) No per-token record file exists in either gp_repair directory (glob for *record* returns nothing). src/glitchprober/repair.py:evaluate_repair accepts `record_path=` but scripts/run_gp_repair.py never passes it. So WHICH 5 / 213 / 97 / 666 tokens were repaired, and what they generated, is unrecoverable.
  b) The adaptive alpha/beta actually derived at run time are printed to stdout only and never serialised. The single most load-bearing number of the adaptive variant (the derived alpha, reported elsewhere as saturating near the clamp) is therefore not in the artifacts.
  c) The 500 "normal" collateral tokens are drawn by np.random.default_rng(seed=0) after two prior draws from the same generator, so the exact id list is in principle re-derivable by re-running the RNG sequence — but only against the code version that produced the file (see the provenance warning below), and the ids themselves were not written out.

PROVENANCE WARNING (critical for the chapter):
  The stored gp_repair artifacts carry git_commit 706e27d. The committed src/glitchprober/repair.py at that commit (and still at HEAD) implements the intervention on the ALREADY-MULTIPLIED gated product captured at the input of down_proj, at the LAST prompt position, and scripts/run_gp_repair.py at that commit fits the adaptive alpha/beta on a glitch sample of size min(len(glitch), len(normal_stats_sample)) = all 988 (resp. all 2552) glitch tokens and then EVALUATES on that same full set — the adaptive variant is fitted and scored on identical tokens, the fixed variants are not.
  The working tree now contains an uncommitted rewrite of repair.py (separate MLP-gate and MLP-data streams, position="token", alpha_min clamp) and of run_gp_repair.py (disjoint adaptive_fit_fraction=0.5 fit/eval split). Re-running today will NOT reproduce the numbers above. The stored numbers measure the v1 "product-stream, last-position, fitted-on-eval-set" variant.

================================================================================
5. ALPHA/BETA GRID (RQ3 sensitivity) — ARITHMETIC-ONLY
================================================================================
Source: results/gp_repair/mistral-7b-instruct-v01/alpha_beta_grid.csv (+ _meta.json)
30 cells = 5 alphas x 6 betas, complete. Each cell: n_glitch = 500, n_normal = 500 (subsamples of the PAPER-protocol census; there is no gccode grid).
Every rate in the file is exactly k/500 (max |rate*500 - round| = 0.000e+00), so integer counts are exactly recoverable. The CSV itself stores only rates — the columns for integer counts exist in the working-tree code but not in this file.

REPAIR RATE (rows = alpha, cols = beta), with counts out of 500 in brackets:
alpha\beta   0.25          0.50          1.00          1.50          2.00          4.00
 1.0     0.000 [  0]   0.000 [  0]   0.016 [  8]   0.034 [ 17]   0.038 [ 19]   0.068 [ 34]
 2.0     0.160 [ 80]   0.160 [ 80]   0.158 [ 79]   0.158 [ 79]   0.154 [ 77]   0.154 [ 77]
 4.0     0.224 [112]   0.214 [107]   0.212 [106]   0.208 [104]   0.204 [102]   0.202 [101]
 8.0     0.232 [116]   0.232 [116]   0.228 [114]   0.232 [116]   0.224 [112]   0.204 [102]
16.0     0.238 [119]   0.240 [120]   0.230 [115]   0.232 [116]   0.222 [111]   0.204 [102]

NORMAL-BREAK RATE, with counts out of 500 in brackets:
alpha\beta   0.25         0.50         1.00         1.50         2.00         4.00
 1.0     0.000 [ 0]   0.000 [ 0]   0.002 [ 1]   0.004 [ 2]   0.004 [ 2]   0.004 [ 2]
 2.0     0.010 [ 5]   0.010 [ 5]   0.010 [ 5]   0.010 [ 5]   0.010 [ 5]   0.010 [ 5]
 4.0     0.016 [ 8]   0.016 [ 8]   0.016 [ 8]   0.016 [ 8]   0.016 [ 8]   0.022 [11]
 8.0     0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.022 [11]
16.0     0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.018 [ 9]   0.024 [12]

MAXIMUM REPAIR CELL: alpha = 16.0, beta = 0.50 -> repair_rate 0.240 = 120/500, normal_break_rate 0.018 = 9/500.
  This maximum is UNIQUE — exactly 1 cell attains it, no ties.
THE PAPER'S CELL: alpha = 4.0, beta = 1.5 -> repair_rate 0.208 = 104/500, normal_break_rate 0.016 = 8/500.
  Gap to the grid optimum: 0.032 absolute = 16 tokens out of 500. The paper's fixed values are neither optimal nor meaningfully suboptimal — they sit 3.2 points below the best cell on a landscape whose entire alpha>=4 region spans only 0.202..0.240.
  (Note the grid's 0.208 at (4, 1.5) is measured on a 500-token subsample; the full-set rule_based number in section 4 is 0.21559 on all 988. These are two different measurements, not a discrepancy.)

HOW REPAIR RATE VARIES WITH BETA AT EACH FIXED ALPHA (exact numbers, betas in order 0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
  alpha = 1.0 : 0.000, 0.000, 0.016, 0.034, 0.038, 0.068   [0, 0, 8, 17, 19, 34]
      deltas: +0.000, +0.016, +0.018, +0.004, +0.030 ; range 0.068 (0 -> 34 tokens)
      SHAPE: MONOTONE INCREASING. This is the only row where beta does real work, and only because alpha=1 means the Neun_down division is the identity (x/1), so beta is the sole active mechanism. Even so it tops out at 6.8%.
  alpha = 2.0 : 0.160, 0.160, 0.158, 0.158, 0.154, 0.154   [80, 80, 79, 79, 77, 77]
      deltas: 0.000, -0.002, 0.000, -0.004, 0.000 ; range 0.006 (77..80 tokens, spread 3 tokens)
      SHAPE: MONOTONE NON-INCREASING, and flat to within 3 tokens out of 500 — statistically indistinguishable from flat.
  alpha = 4.0 : 0.224, 0.214, 0.212, 0.208, 0.204, 0.202   [112, 107, 106, 104, 102, 101]
      deltas: -0.010, -0.002, -0.004, -0.004, -0.002 ; range 0.022 (101..112, spread 11 tokens)
      SHAPE: MONOTONE DECREASING. Increasing beta 16-fold costs 11 tokens.
  alpha = 8.0 : 0.232, 0.232, 0.228, 0.232, 0.224, 0.204   [116, 116, 114, 116, 112, 102]
      deltas: 0.000, -0.004, +0.004, -0.008, -0.020 ; range 0.028 (102..116, spread 14 tokens)
      SHAPE: NON-MONOTONE (dips at beta=1.0, recovers at 1.5, then falls). The +/-0.004 wiggles are 2 tokens — noise.
  alpha = 16.0: 0.238, 0.240, 0.230, 0.232, 0.222, 0.204   [119, 120, 115, 116, 111, 102]
      deltas: +0.002, -0.010, +0.002, -0.010, -0.018 ; range 0.036 (102..120, spread 18 tokens)
      SHAPE: NON-MONOTONE (rises 1 token to the global max at beta=0.5, then declines with two 1-token upticks).

  PRECISE VERDICT ON BETA: beta is inert-to-mildly-harmful everywhere except the degenerate alpha=1 row. For alpha >= 2 the rate never increases with beta by more than 2 tokens out of 500 (0.004), while it decreases by up to 18 tokens (0.036). No row for alpha >= 2 is flat in the strict sense (every row has at least one nonzero delta), but rows alpha=2 and alpha=4 are monotone decreasing and rows alpha=8 and alpha=16 are non-monotone with noise-scale fluctuations. Column means across alpha: beta=0.25 -> 0.1708, 0.5 -> 0.1692, 1.0 -> 0.1688, 1.5 -> 0.1728, 2.0 -> 0.1684, 4.0 -> 0.1664 (total spread 0.0064 = 3.2 tokens). The claim "beta does nothing, and if anything hurts" is supported.

HOW REPAIR RATE VARIES WITH ALPHA (the mechanism that actually works, then saturates):
  beta=0.25: 0.000, 0.160, 0.224, 0.232, 0.238   [0, 80, 112, 116, 119]
  beta=0.50: 0.000, 0.160, 0.214, 0.232, 0.240   [0, 80, 107, 116, 120]
  beta=1.00: 0.016, 0.158, 0.212, 0.228, 0.230   [8, 79, 106, 114, 115]
  beta=1.50: 0.034, 0.158, 0.208, 0.232, 0.232   [17, 79, 104, 116, 116]
  beta=2.00: 0.038, 0.154, 0.204, 0.224, 0.222   [19, 77, 102, 112, 111]
  beta=4.00: 0.068, 0.154, 0.202, 0.204, 0.204   [34, 77, 101, 102, 102]
  Row means across beta: alpha=1 -> 0.026000, alpha=2 -> 0.157333, alpha=4 -> 0.210667, alpha=8 -> 0.225333, alpha=16 -> 0.227667.
  Alpha is monotone increasing at every beta, with sharply diminishing returns: 1->2 gains ~13 points, 2->4 gains ~5, 4->8 gains ~1.5, 8->16 gains ~0.2. A 16x alpha buys ~24% repair and no more; collateral breakage saturates in parallel at 9-12/500.

NOT RECONSTRUCTABLE: which 500 glitch and which 500 normal tokens were used, and the generations per cell. Only the two rate columns were written.

================================================================================
6. GLITCHCLEANER (RQ4)
================================================================================
Sources: results/gc/mistral-7b-instruct-v01/{split.json, train.jsonl, heldout.jsonl, eval.json, train_meta.json} and gccode/ equivalents.
(Parsing note for anyone re-running this: train.jsonl must be split on "\n" only — str.splitlines() also splits on U+2028/U+2029/U+0085, which json.dumps(ensure_ascii=False) leaves unescaped, and several glitch tokens decode to exactly those characters.)

--- SPLIT INTEGRITY — EXACT ---
PROTOCOL "paper":
  n_train   = 791 (791 unique ids)   n_heldout = 197 (197 unique ids)
  train.jsonl ids == split.json train_ids: True ; heldout.jsonl ids == split.json heldout_ids: True
  DISJOINT: |train ∩ heldout| = 0  -> True
  PARTITION: train ∪ heldout (988 ids) == census glitch set (988 ids) -> True. 0 missing, 0 extra.
  Actual holdout fraction = 197/988 = 0.199393 (config 0.2; int(988*0.2) = 197 — consistent with floor semantics)
PROTOCOL "gccode":
  n_train   = 2042 (2042 unique)     n_heldout = 510 (510 unique)
  train.jsonl ids == split.json: True ; heldout.jsonl ids == split.json: True
  DISJOINT: |train ∩ heldout| = 0 -> True
  PARTITION: train ∪ heldout (2552) == census glitch set (2552) -> True. 0 missing, 0 extra.
  Actual holdout fraction = 510/2552 = 0.199843 (int(2552*0.2) = 510)

--- REPAIR NUMBERS — ARITHMETIC-ONLY (per-token outcomes not saved) ---
PROTOCOL "paper":
  TRAIN-split repair       = 644/791 = 0.8141592920353983  (81.416%)
  HELD-OUT repair          = 109/197 = 0.5532994923857868  (55.330%)
  ADAPTER-OFF held-out     =   0/197 = 0.0                 (0.000%)   <- control passes: repair is the LoRA's doing
  normal OK (adapter on)   = 465/500 = 0.93                (93.0%)
  normal BROKEN by adapter =  35/500 = 0.07                (7.0%)
  MEMORISATION GAP (train - heldout) = 0.8141592920 - 0.5532994924 = 0.260860 (26.09 points)
  train_seconds = 31.441829699993832
PROTOCOL "gccode":
  TRAIN-split repair       = 1836/2042 = 0.8991185112634672 (89.912%)
  HELD-OUT repair          =  380/510  = 0.7450980392156863 (74.510%)
  ADAPTER-OFF held-out     =    0/510  = 0.0                (0.000%)  <- control passes
  normal OK (adapter on)   =  483/500  = 0.966              (96.6%)
  normal BROKEN by adapter =   17/500  = 0.034              (3.4%)
  MEMORISATION GAP = 0.8991185113 - 0.7450980392 = 0.154020 (15.40 points)
  train_seconds = 81.67975239999942
  Paper claim for comparison (both files): repair_rate_avg = 0.8688.

--- TWO CAVEATS ON THE HELD-OUT NUMBER (recomputed here, EXACT) ---
(a) LEAKAGE. I re-tokenised every training prompt+answer and counted which held-out ids nonetheless appear as token ids inside training SEQUENCES:
      paper : 1377 distinct vocab ids seen during training; 18/197 held-out ids leaked (9.14%); 179 clean.
      gccode: 3128 distinct vocab ids seen during training; 67/510 held-out ids leaked (13.14%); 443 clean.
    The stored held-out rates are therefore mildly optimistic. Because per-token outcomes were not saved, a leakage-corrected held-out rate CANNOT be computed from the artifacts — it requires a re-run.
(b) GATE BYPASSED. The stored field is named normal_ok_rate_adapter_on and was produced with the adapter FORCED ON for all 500 normal tokens, i.e. GlitchCleaner's lambda gate was not applied. The 93.0% / 96.6% figures are gate-bypassed numbers and overstate collateral damage relative to the method as published. The working-tree src/glitchcleaner/evaluate.py renames this to normal_ok_adapter_forced_on and adds a gated control — that control does not exist in these artifacts.

--- PROMPT/TARGET FORMAT ACTUALLY USED (EXACT) ---
  All 791 / 2042 training prompts reconstruct exactly as prefix + token_str(tok, id) + suffix for the protocol's template (and, for this tokenizer, identically under raw decode). Answers are the v1 form " {stripped token}" for 791/791 and 2042/2042 records — NOT the authors' quoted-unstripped form. Example: {"token_id": 29240, "prompt": "Can you repeat the string 'ː' and return it back to me? Answer: Here is the repeated string:", "answer": " ː"}.
  Working-tree build_dataset.py has since switched to the authors' construction, so these artifacts predate that correction.

--- NOT RECONSTRUCTABLE ---
  Which train/held-out/normal tokens the adapter fixed or broke, and their generations. No record file exists in either gc directory (only adapter/, eval.json, split.json, train.jsonl, heldout.jsonl, train_meta.json). Nothing beyond a scalar train_seconds records the training run.

================================================================================
7. SPEED (RQ5) — ARITHMETIC-ONLY
================================================================================
Source: results/side_effects/mistral-7b-instruct-v01/speed.json (git_commit 36a3094, 2026-07-20T11:55:18)
Protocol: 256 new tokens, greedy, mean of 5 repetitions after 1 warm-up, single prompt.

  base        = 31.75744315158763 tok/s   ratio to base = 1.0
  gp_hooks    = 19.98954963228586 tok/s   ratio to base = 0.6294445537340602
  gc_adapter  = 17.92915214263845 tok/s   ratio to base = 0.5645653542401801
  All three stored "relative" values reproduce exactly from tokens_per_second (to <1e-15).

  Expressed as slowdown factors (base / variant):
    gp_hooks : 1.588702x slower
    gc_adapter: 1.771274x slower

  PAPER CLAIMS (stored alongside): base 66.30, gc 62.83, gp 11.82 tok/s
    paper ratios: gc/base = 0.9476621417797888 ; gp/base = 0.17828054298642534
    paper slowdowns: gc 1.055228x ; gp 5.609137x
  RATIO COMPARISON (the only fair comparison across hardware):
    GC: claimed 0.948 of base, measured 0.565 — the "nearly lossless" claim fails as a ratio.
    GP: claimed 0.178 of base, measured 0.629 — GP is 3.5x LESS costly than claimed here, i.e. the paper's headline that GP is catastrophically slow does not reproduce either.
    ORDERING REVERSAL: the papers place gc_adapter faster than gp_hooks (62.83 vs 11.82). Our measurement reverses it — gc_adapter (17.93) is SLOWER than gp_hooks (19.99). This ordering, not the absolute rates, is the reproducible claim being falsified.

  NOT RECONSTRUCTABLE: the 5 individual per-repetition rates (src/eval/speed.py returns only their mean), hence no variance/CI on any of the three numbers. There is no record of GPU, driver, dtype, or batch settings in speed.json beyond model id.

================================================================================
8. ALL TIMING / WALL-CLOCK IN THE ARTIFACTS (complete inventory)
================================================================================
GROUND TRUTH (sweep_seconds, whole-vocabulary repetition sweep, wall-clock):
  paper  protocol : 205.7225649000029 s   (2026-07-19T15:36:42, git c269d8f)
  gccode protocol : 314.38567519999924 s  (2026-07-19T15:44:36, git e4006f3)
  smoke-test      : 6.1441459999768995 s  (2026-07-19T15:21:23)
DETECTION (time_seconds, per seed, covers sample+label+features+PCA+SVM+classify+post-validate):
  paper  : [209.3022154000064, 212.33258090002343, 212.5035465999972] ; sum 634.138342900027 ; mean 211.37944763334235 ; sd(ddof=1) 1.8009657495491924
  gccode : [221.5056889000116, 221.7129216999747, 228.6687632999965]  ; sum 671.8873738999828 ; mean 223.96245796666093 ; sd(ddof=1) 4.077096856377376
  smoke-test: [3.733224199997494]
GLITCHCLEANER TRAINING (train_seconds, LoRA fine-tune only, excludes eval):
  paper  : 31.441829699993832 s  (2026-07-19T16:36:52)
  gccode : 81.67975239999942 s   (2026-07-19T16:38:55)
  smoke-test: 3.9071410999968066 s
RUN TIMESTAMPS (ISO, from run_metadata; these bound each stage's wall-clock by difference):
  ground_truth paper   2026-07-19T15:36:42   ground_truth gccode  2026-07-19T15:44:36
  gp_detect paper      2026-07-19T15:58:33   gp_detect gccode     2026-07-19T16:14:12
  gp_repair paper      2026-07-19T16:19:22   gp_repair gccode     2026-07-19T16:21:05
  alpha_beta_grid      2026-07-19T16:30:11
  gc train paper       2026-07-19T16:36:52   gc eval paper        2026-07-19T16:37:19
  gc train gccode      2026-07-19T16:38:55   gc eval gccode       2026-07-19T16:39:37
  speed                2026-07-20T11:55:18   benchmark_report     2026-07-20T11:55:54
  Derived (timestamp deltas, upper bounds including model load): gc eval paper <= 27 s ; gc eval gccode <= 42 s.
NO TIMING EXISTS FOR: the GlitchProber repair runs (gp_repair summary.json has a timestamp but no elapsed field), any individual alpha/beta cell (only one timestamp for all 30), the speed repetitions individually, and any per-batch/per-stage breakdown of the detection total.

================================================================================
9. CROSS-FILE CONSISTENCY OF results/benchmark_report.json
================================================================================
Every headline number in results/benchmark_report.json for mistral-7b-instruct-v01 (rq1 counts, rq2 mean/std, rq3 repair dict, rq4 gc dict, rq5 speed dict) is byte-identical to the corresponding stage summary.json. The aggregator introduces no new numbers and no drift. It reports only the PAPER-protocol variants — the gccode census, gccode detection, gccode repair, and gccode GC results are absent from it and must be pulled from their own summary files.

================================================================================
10. ARTIFACT/CODE PROVENANCE — FLAGS FOR THE WRITE-UP
================================================================================
The working tree is NOT clean; `git status --porcelain` shows modified configs/glitchcleaner.yaml, configs/glitchprober.yaml, scripts/run_gp_alpha_beta_sweep.py, scripts/run_gp_repair.py, src/common/io_utils.py, src/glitchcleaner/train_lora.py, src/glitchprober/repair.py, src/glitchprober/sweep_alpha_beta.py, plus an untracked src/glitchcleaner/gate.py.
Consequence, stage by stage:
  RQ1 census (git c269d8f / e4006f3): code paths unchanged since. Artifacts are current and fully re-derivable.
  RQ2 detection (git 3dec6a7 / 32ba28a): code paths unchanged since. Artifacts current.
  RQ3 repair + alpha/beta grid (git 706e27d): the committed repair.py implements the "product at down_proj input, last position" intervention and fits the adaptive parameters on the full evaluation set. The working tree replaces this with separate gate/data streams at the token position and a disjoint 50/50 adaptive fit/eval split. THESE ARTIFACTS ARE STALE — re-running reproduces neither the adaptive nor the rule-based numbers.
  RQ4 GlitchCleaner (git 706e27d): the working tree changes the prompt/target construction to the authors' quoted-unstripped form, adds leakage filtering of the held-out set, and adds the lambda-gate control. THESE ARTIFACTS ARE STALE for the same reason; the stored normal-ok figures are gate-bypassed and the held-out set is 9.1%/13.1% leaked.
  RQ5 speed (git 36a3094): produced against the committed (stale) repair hooks and the committed GC adapter, so the gp_hooks rate measures the v1 intervention.

FILES USED (all absolute):
  E:/user3/FYS_SWK/Benchmark/results/ground_truth/mistral-7b-instruct-v01/{tokens.csv,sweep_checkpoint.csv,summary.json}
  E:/user3/FYS_SWK/Benchmark/results/ground_truth/mistral-7b-instruct-v01/gccode/{tokens.csv,sweep_checkpoint.csv,summary.json}
  E:/user3/FYS_SWK/Benchmark/results/gp_detect/mistral-7b-instruct-v01/{runs.csv,summary.json,checkpoints/*.csv}
  E:/user3/FYS_SWK/Benchmark/results/gp_detect/mistral-7b-instruct-v01/gccode/{runs.csv,summary.json,checkpoints/*.csv}
  E:/user3/FYS_SWK/Benchmark/results/gp_repair/mistral-7b-instruct-v01/{summary.json,alpha_beta_grid.csv,alpha_beta_grid_meta.json}
  E:/user3/FYS_SWK/Benchmark/results/gp_repair/mistral-7b-instruct-v01/gccode/summary.json
  E:/user3/FYS_SWK/Benchmark/results/gc/mistral-7b-instruct-v01/{split.json,train.jsonl,heldout.jsonl,eval.json,train_meta.json} (+ gccode/)
  E:/user3/FYS_SWK/Benchmark/results/side_effects/mistral-7b-instruct-v01/speed.json
  E:/user3/FYS_SWK/Benchmark/results/benchmark_report.json
  E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/Glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens.csv
RECOMPUTATION SCRIPTS (kept for audit):
  C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/recompute.py
  C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/followup.py
  C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/leak.py