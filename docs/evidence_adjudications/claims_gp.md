{
 "claims": [
  {
   "metric": "Detection True Positives (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "1,873",
   "scope": "mistral-specific",
   "location": "p.9, Table 3 (Performance comparison of GlitchProber and other baselines on different LLMs)",
   "quote": "TP 1,288 1,233 1,873"
  },
  {
   "metric": "Detection Precision (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "100.00%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "Mistral-7B-Instruct-v0.1 Precision 11.44% 100.00% 100.00%"
  },
  {
   "metric": "Detection Recall (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "67.41%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "Recall 46.35% 44.37% 67.41%"
  },
  {
   "metric": "Detection F1-score (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "0.8053",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "F1-score 0.1836 0.6147 0.8053"
  },
  {
   "metric": "Detection True Positives (Mistral, GlitchHunter baseline)",
   "value": "1,233",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "TP 1,288 1,233 1,873"
  },
  {
   "metric": "Detection Precision (Mistral, GlitchHunter baseline)",
   "value": "100.00%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Recall (Mistral, GlitchHunter baseline)",
   "value": "44.37%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "Recall 46.35% 44.37% 67.41%"
  },
  {
   "metric": "Detection F1-score (Mistral, GlitchHunter baseline)",
   "value": "0.6147",
   "scope": "mistral-specific",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection True Positives (Mistral, Rule-based Random Sampling baseline)",
   "value": "1,288",
   "scope": "mistral-specific",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Precision (Mistral, Rule-based Random Sampling baseline)",
   "value": "11.44%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3",
   "quote": "Precision 11.44% 100.00% 100.00%"
  },
  {
   "metric": "Detection Recall (Mistral, Rule-based Random Sampling baseline)",
   "value": "46.35%",
   "scope": "mistral-specific",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection F1-score (Mistral, Rule-based Random Sampling baseline)",
   "value": "0.1836",
   "scope": "mistral-specific",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Precision (average over 5 LLMs, GlitchProber)",
   "value": "100.00%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3, 'Average Performance' row",
   "quote": "Precision 16.32% 100.00% 100.00%"
  },
  {
   "metric": "Detection Recall (average over 5 LLMs, GlitchProber)",
   "value": "64.47%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3, 'Average Performance' row; also stated in Sec. 5.2 text",
   "quote": "GlitchProberachievesarecallrateof64.47%,surpassingGlitchHunter'26.52%"
  },
  {
   "metric": "Detection F1-score (average over 5 LLMs, GlitchProber)",
   "value": "0.7835",
   "scope": "cross-model-average",
   "location": "p.9, Table 3, 'Average Performance' row",
   "quote": "F1-score 0.2364 0.4049 0.7835"
  },
  {
   "metric": "Detection Precision (average over 5 LLMs, GlitchHunter)",
   "value": "100.00%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Recall (average over 5 LLMs, GlitchHunter)",
   "value": "26.52%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3",
   "quote": "Recall 46.24% 26.52% 64.47%"
  },
  {
   "metric": "Detection F1-score (average over 5 LLMs, GlitchHunter)",
   "value": "0.4049",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Precision (average over 5 LLMs, Rule-based Random Sampling)",
   "value": "16.32%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection Recall (average over 5 LLMs, Rule-based Random Sampling)",
   "value": "46.24%",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection F1-score (average over 5 LLMs, Rule-based Random Sampling)",
   "value": "0.2364",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Average True Positives across models",
   "value": "NOT REPORTED \u2014 Table 3's 'Average Performance' block lists only Precision/Recall/F1, no averaged TP row",
   "scope": "cross-model-average",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Abstract-level average F1 score of GlitchProber",
   "value": "0.86",
   "scope": "cross-model-average",
   "location": "p.1, Abstract",
   "quote": "withanaverageF1scoreof0.86andanaveragerepairrateof50.06%"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Llama-2-7b-chat, GlitchProber)",
   "value": "4,446 / 100.00% / 69.22% / 0.8181",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Llama-2-7b-chat, GlitchHunter)",
   "value": "1,955 / 100.00% / 30.43% / 0.4724",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Llama-2-7b-chat, Rule-based)",
   "value": "2,936 / 24.74% / 45.70% / 0.3210",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Qwen-7B-Chat, GlitchProber)",
   "value": "19,366 / 100.00% / 63.08% / 0.7736",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Qwen-7B-Chat, GlitchHunter)",
   "value": "4,031 / 100.00% / 14.42% / 0.2521",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Qwen-7B-Chat, Rule-based)",
   "value": "15,419 / 21.04% / 50.24% / 0.2966",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Gemma-2b-it, GlitchProber)",
   "value": "17,387 / 100.00% / 62.18% / 0.7668",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Gemma-2b-it, GlitchHunter)",
   "value": "3,240 / 100.00% / 10.56% / 0.1910",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Gemma-2b-it, Rule-based)",
   "value": "13,777 / 11.30% / 49.27% / 0.1838",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Yi-6B-Chat, GlitchProber)",
   "value": "4,900 / 100.00% / 60.45% / 0.7535",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Yi-6B-Chat, GlitchHunter)",
   "value": "2,662 / 100.00% / 32.84% / 0.4944",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Detection TP / Precision / Recall / F1 (Yi-6B-Chat, Rule-based)",
   "value": "3,215 / 13.10% / 39.67% / 0.1969",
   "scope": "other-model",
   "location": "p.9, Table 3"
  },
  {
   "metric": "Time cost (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "42 min 39 s",
   "scope": "mistral-specific",
   "location": "p.9, Table 2 (Time cost comparison)",
   "quote": "Mistral-7B-Instruct-v0.1 651min17s 64min26s 42min39s"
  },
  {
   "metric": "Time cost (Mistral-7B-Instruct-v0.1, GlitchHunter)",
   "value": "64 min 26 s",
   "scope": "mistral-specific",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Time cost (Mistral-7B-Instruct-v0.1, Exhaustive Search)",
   "value": "651 min 17 s",
   "scope": "mistral-specific",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Average Time Cost (5 LLMs, GlitchProber)",
   "value": "89 min 9 s",
   "scope": "cross-model-average",
   "location": "p.9, Table 2, 'Average Time Cost' row",
   "quote": "AverageTimeCost 1,609min42s 473min11s 89min9s"
  },
  {
   "metric": "Average Time Cost (5 LLMs, GlitchHunter)",
   "value": "473 min 11 s",
   "scope": "cross-model-average",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Average Time Cost (5 LLMs, Exhaustive Search)",
   "value": "1,609 min 42 s",
   "scope": "cross-model-average",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Claimed time saving vs state-of-the-art (GlitchHunter)",
   "value": "approximately 40%",
   "scope": "cross-model-average",
   "location": "p.2 Contributions ('EffectiveGlitchTokenDetection') and p.11 Conclusion",
   "quote": "cansaveapproximately40%oftimeinglitchtokendetectioncomparedtothestate-of-the-artapproaches"
  },
  {
   "metric": "Claimed reduction of redundant operations vs exhaustive search",
   "value": "95% (exhaustive search needs ~20 generated tokens per token at temperature 0; GlitchProber needs a single forward pass)",
   "scope": "general",
   "location": "p.10, Sec. 7.2 'Rationale of GlitchProber versus Exhaustive Search'",
   "quote": "Thisapproachsubstantiallyreduces95%ofredundantoperationsofthoserequiredbythetraditionalexhaustivesearchmethod"
  },
  {
   "metric": "Time cost (Llama-2-7b-chat): Exhaustive / GlitchHunter / GlitchProber",
   "value": "619 min 43 s / 74 min 11 s / 61 min 38 s",
   "scope": "other-model",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Time cost (Qwen-7B-Chat): Exhaustive / GlitchHunter / GlitchProber",
   "value": "2,228 min 23 s / 720 min 42 s / 92 min 48 s",
   "scope": "other-model",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Time cost (Gemma-2b-it): Exhaustive / GlitchHunter / GlitchProber",
   "value": "3,575 min 9 s / 681 min 16 s / 96 min 43 s",
   "scope": "other-model",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Time cost (Yi-6B-Chat): Exhaustive / GlitchHunter / GlitchProber",
   "value": "974 min 4 s / 825 min 25 s / 140 min 57 s",
   "scope": "other-model",
   "location": "p.9, Table 2"
  },
  {
   "metric": "Repaired Tokens (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "1,045",
   "scope": "mistral-specific",
   "location": "p.10, Table 5 (Performance comparison of GlitchProber and Rule-based method)",
   "quote": "RepairedTokens 359 1,045"
  },
  {
   "metric": "Repair Rate (Mistral-7B-Instruct-v0.1, GlitchProber)",
   "value": "37.60%",
   "scope": "mistral-specific",
   "location": "p.10, Table 5",
   "quote": "RepairRate 12.92% 37.60%"
  },
  {
   "metric": "Repaired Tokens (Mistral-7B-Instruct-v0.1, Rule-based Fix)",
   "value": "359",
   "scope": "mistral-specific",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Repair Rate (Mistral-7B-Instruct-v0.1, Rule-based Fix)",
   "value": "12.92%",
   "scope": "mistral-specific",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Average Repaired Tokens (5 LLMs, GlitchProber)",
   "value": "7,758",
   "scope": "cross-model-average",
   "location": "p.10, Table 5 'Average' row; also p.2 Contributions",
   "quote": "GlitchProbersuccessfullyrepairsanaverageof7,758tokensacrossthefiveLLMs"
  },
  {
   "metric": "Average Repair Rate (5 LLMs, GlitchProber)",
   "value": "50.06%",
   "scope": "cross-model-average",
   "location": "p.1 Abstract, p.2 Contributions, p.10 Table 5 'Average' row and Sec. 6.2 text",
   "quote": "achievinganaveragerepairrateof50.06%acrossthefivemodels"
  },
  {
   "metric": "Average Repaired Tokens (5 LLMs, Rule-based Fix)",
   "value": "5,613",
   "scope": "cross-model-average",
   "location": "p.10, Table 5 'Average' row",
   "quote": "RepairedTokens 5,613 7,758"
  },
  {
   "metric": "Average Repair Rate (5 LLMs, Rule-based Fix)",
   "value": "36.79%",
   "scope": "cross-model-average",
   "location": "p.10, Table 5 'Average' row",
   "quote": "RepairRate 36.79% 50.06%"
  },
  {
   "metric": "Improvement in average repair rate of GlitchProber over rule-based baseline",
   "value": "13.27% (percentage points: 50.06% - 36.79%)",
   "scope": "cross-model-average",
   "location": "p.11, Conclusion",
   "quote": "theaveragerepairrateofGlitchProberwasimprovedby13.27%comparedwiththebaselinemethod"
  },
  {
   "metric": "Repaired Tokens / Repair Rate (Llama-2-7b-chat): Rule-based vs GlitchProber",
   "value": "3,805 / 59.22% vs 4,021 / 62.58%",
   "scope": "other-model",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Repaired Tokens / Repair Rate (Qwen-7B-Chat): Rule-based vs GlitchProber",
   "value": "10,645 / 34.68% vs 14,765 / 48.11%",
   "scope": "other-model",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Repaired Tokens / Repair Rate (Gemma-2b-it): Rule-based vs GlitchProber",
   "value": "9,865 / 35.28% vs 13,638 / 48.77%",
   "scope": "other-model",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Repaired Tokens / Repair Rate (Yi-6B-Chat): Rule-based vs GlitchProber",
   "value": "3,390 / 41.83% vs 4,317 / 53.26%",
   "scope": "other-model",
   "location": "p.10, Table 5"
  },
  {
   "metric": "Rule-based fix baseline: fixed alpha and beta",
   "value": "alpha = 4, beta = 1.5",
   "scope": "general",
   "location": "p.9-10, Sec. 6.2 RQ5 text",
   "quote": "therule-basedfixmethodusesfixed\ud835\udefc and\ud835\udefdvalues(\ud835\udefc =4and\ud835\udefd =1.5)andlacksflexibility"
  },
  {
   "metric": "Glitch token count (Llama-2-7b-chat) \u2014 only per-model count explicitly stated in the paper",
   "value": "6,425 (out of 32,000 vocabulary tokens traversed)",
   "scope": "other-model",
   "location": "p.3, Sec. 3.1 Experiment Setup",
   "quote": "Wetraverseallthe32,000tokensoftheLlama2modelandeventuallyidentify6,425glitchtokensfromtheentirevocabulary"
  },
  {
   "metric": "Glitch token count (Mistral-7B-Instruct-v0.1) \u2014 NOT stated; derived from Tables 3 and 5",
   "value": "~2,779 (derived: 1,873 / 0.6741 = 2,778.9; and 1,045 / 0.3760 = 2,779.3)",
   "scope": "mistral-specific",
   "location": "derived from p.9 Table 3 and p.10 Table 5 (never printed as a number in the paper)"
  },
  {
   "metric": "Glitch token count (Qwen-7B-Chat) \u2014 NOT stated; derived",
   "value": "~30,690-30,701 (19,366 / 0.6308 = 30,700.7; 14,765 / 0.4811 = 30,690.1)",
   "scope": "other-model",
   "location": "derived from p.9 Table 3 and p.10 Table 5"
  },
  {
   "metric": "Glitch token count (Gemma-2b-it) \u2014 NOT stated; derived",
   "value": "~27,963 (17,387 / 0.6218 = 27,962.4; 13,638 / 0.4877 = 27,963.9)",
   "scope": "other-model",
   "location": "derived from p.9 Table 3 and p.10 Table 5"
  },
  {
   "metric": "Glitch token count (Yi-6B-Chat) \u2014 NOT stated; derived",
   "value": "~8,106 (4,900 / 0.6045 = 8,105.9; 4,317 / 0.5326 = 8,105.1)",
   "scope": "other-model",
   "location": "derived from p.9 Table 3 and p.10 Table 5"
  },
  {
   "metric": "Empirical-study prompt set sizes (Llama2, RQ1)",
   "value": "6,425 prompts with glitch tokens and 6,425 prompts with normal tokens",
   "scope": "other-model",
   "location": "p.4, Sec. 3.2 RQ1",
   "quote": "onecontaining6,425promptswithglitchtokensandanothercontaining6,425promptswithnormaltokens"
  },
  {
   "metric": "Ablation F1-score (Mistral): GlitchProber / No-Post / No-PCA",
   "value": "0.8612 / 0.5429 / \u2014 (No-PCA incomplete, exceeded 250 GB server memory)",
   "scope": "mistral-specific",
   "location": "p.9, Table 4 (Comparison of performance and memory usage)"
  },
  {
   "metric": "Ablation memory usage (Mistral): GlitchProber / No-Post / No-PCA",
   "value": "107.15 GB / 102.67 GB / 250.00 GB (exceeded max)",
   "scope": "mistral-specific",
   "location": "p.9, Table 4"
  },
  {
   "metric": "Ablation F1-score / Memory (Llama-2-7b-chat)",
   "value": "F1 0.8529 / 0.6097 / \u2014; Memory 103.71 GB / 101.22 GB / 250.00 GB",
   "scope": "other-model",
   "location": "p.9, Table 4"
  },
  {
   "metric": "Ablation F1-score / Memory (Qwen-7B-Chat)",
   "value": "F1 0.8854 / 0.5510 / \u2014; Memory 109.03 GB / 101.20 GB / 250.00 GB",
   "scope": "other-model",
   "location": "p.9, Table 4"
  },
  {
   "metric": "Ablation F1-score / Memory (Gemma-2b-it)",
   "value": "F1 0.8143 / 0.4226 / \u2014; Memory 131.04 GB / 127.96 GB / 250.00 GB",
   "scope": "other-model",
   "location": "p.9, Table 4"
  },
  {
   "metric": "Ablation F1-score / Memory (Yi-6B-Chat)",
   "value": "F1 0.8718 / 0.4507 / \u2014; Memory 83.32 GB / 82.76 GB / 250.00 GB",
   "scope": "other-model",
   "location": "p.9, Table 4"
  },
  {
   "metric": "Average F1 improvement of GlitchProber over GlitchProber-No-Post",
   "value": "0.32",
   "scope": "cross-model-average",
   "location": "p.9, Sec. 5.3 RQ4 text",
   "quote": "markedlysurpassesGlitchProber-No-Post,demonstratingaverageimprovementsof0.32"
  },
  {
   "metric": "Best hyperparameter-group average F1 without post-processing (Figure 6 selection experiment)",
   "value": "0.6117 (at C = 1, degree = 3, all three features)",
   "scope": "cross-model-average",
   "location": "p.10, Sec. 7.1 Hyperparameters Choice",
   "quote": "thehyperparametergroupinthelowerrightcornerachievesthehighestF1-scoreof0.6117withoutpost-processing"
  },
  {
   "metric": "Post-process time, chosen config (Attn_pattern+MLP_gate+MLP_data, C=1, degree=3)",
   "value": "1,500.23 s",
   "scope": "general",
   "location": "p.10, Table 6 (Post process time in seconds)"
  },
  {
   "metric": "Post-process time range across all 28 SVM/feature configurations",
   "value": "1,357.70 s (min: Attn_pattern, C=1 degree=3) to 1,581.00 s (max: MLP_data, C=1 degree=3)",
   "scope": "general",
   "location": "p.10, Table 6"
  },
  {
   "metric": "Model spec (Mistral-7B-Instruct-v0.1): params / vocab / hidden layers / intermediate size / attention heads",
   "value": "7.24B / 32,000 / 32 / 14,336 / 32",
   "scope": "mistral-specific",
   "location": "p.9, Table 1 (Summary of models in evaluation)",
   "quote": "Mistral-7B-Instruct-v0.1 7.24B 32,000 32 14,336 32"
  },
  {
   "metric": "Model spec (Llama-2-7b-chat)",
   "value": "6.74B / 32,000 / 32 / 11,008 / 32",
   "scope": "other-model",
   "location": "p.9, Table 1"
  },
  {
   "metric": "Model spec (Qwen-7B-Chat)",
   "value": "7.72B / 151,936 / 32 / 22,016 / 32",
   "scope": "other-model",
   "location": "p.9, Table 1"
  },
  {
   "metric": "Model spec (Gemma-2b-it)",
   "value": "2.51B / 256,000 / 18 / 16,384 / 8",
   "scope": "other-model",
   "location": "p.9, Table 1"
  },
  {
   "metric": "Model spec (Yi-6B-Chat)",
   "value": "6.06B / 64,000 / 32 / 11,008 / 32",
   "scope": "other-model",
   "location": "p.9, Table 1"
  },
  {
   "metric": "Mistral attention-pattern value modes (normal vs glitch tokens), RQ2 qualitative-quantitative claim",
   "value": "normal tokens approximately 0.5; glitch tokens approximately 0.7",
   "scope": "mistral-specific",
   "location": "p.4, Sec. 3.3 RQ2 Ubiquity",
   "quote": "theattentionvaluesofnormaltokensandglitchtokensprimarilyapproximatingaround0.5and0.7,respectively"
  },
  {
   "metric": "Qwen attention-pattern value ranges (normal vs glitch tokens), RQ2",
   "value": "normal tokens mainly in [0, 0.2]; glitch tokens concentrate in [0.8, 1]",
   "scope": "other-model",
   "location": "p.4, Sec. 3.3 RQ2 Ubiquity"
  },
  {
   "metric": "Layers exhibiting greatest Wasserstein-distance divergence (Llama2)",
   "value": "layers 19-31 (of 32)",
   "scope": "other-model",
   "location": "p.4, Sec. 3.2 RQ1",
   "quote": "exhibitgreaterdifferencesinthedownstreamlayersclosertotheoutput,e.g.,layers19-31"
  },
  {
   "metric": "Fine-tuning mitigation experiment dataset size (Llama-2-7b-chat)",
   "value": "3,000 Q&A pairs of repetition tasks; evaluated on GSM8K, HumanEval, MMLU (numeric results only on external website, not in paper)",
   "scope": "other-model",
   "location": "p.10-11, Sec. 7.3 Glitch Token Mitigation By Fine-tuning"
  },
  {
   "metric": "Rule-based Random Sampling baseline construction",
   "value": "randomly select half (50%) of model tokens as candidate set, then remove high-frequency English words via NLTK; 100 independent experiments averaged",
   "scope": "general",
   "location": "p.8, Sec. 5.1 Evaluation of Detection Baselines / Evaluation Settings",
   "quote": "weconducted100independentexperimentsandaveragedtheresultstoobtainstatisticallysignificantconclusions"
  }
 ],
 "hyperparameters": [
  {
   "name": "Sampling rate gamma (detection)",
   "value": "gamma = 0.1 (used in all experiments); paper states range [0.1, 0.3] gives a good balance",
   "disclosed": true,
   "location": "p.5 Sec. 4.1.1 (range) and p.8 Sec. 5.1 Evaluation Settings (value used)"
  },
  {
   "name": "Sampling rate gamma (fix phase)",
   "value": "same gamma = 0.1, applied to the normal token set N to draw N' subset",
   "disclosed": true,
   "location": "p.7 Sec. 4.2.1 and p.9 Sec. 6.1 Evaluation Settings"
  },
  {
   "name": "PCA dimension P",
   "value": "P = 75 (default); paper states range [50, 200] gives a good balance",
   "disclosed": true,
   "location": "p.5 Sec. 4.1.1 and p.8 Sec. 5.1 Evaluation Settings"
  },
  {
   "name": "SVM regularization parameter C",
   "value": "C = 1",
   "disclosed": true,
   "location": "p.8 Sec. 5.1 Evaluation Settings; justified in p.10 Sec. 7.1 / Figure 6"
  },
  {
   "name": "SVM polynomial kernel degree",
   "value": "degree = 3",
   "disclosed": true,
   "location": "p.8 Sec. 5.1 Evaluation Settings; justified in p.10 Sec. 7.1 / Figure 6"
  },
  {
   "name": "SVM kernel type",
   "value": "polynomial kernel (implied by 'the parameters C and degree in the SVM's polynomial kernel'); other SVM settings (gamma/coef0, class weighting, train/test split) never given",
   "disclosed": true,
   "location": "p.10 Sec. 7.1"
  },
  {
   "name": "Activation threshold m (fix phase)",
   "value": "m = 1",
   "disclosed": true,
   "location": "p.9 Sec. 6.1 Evaluation Settings \u2014 'For the threshold, we set m=1 to determine Neun-up and Neun-down'"
  },
  {
   "name": "Neun-up (highly-activated key neuron) criterion",
   "value": "Neun-up = {i | Act[i] > m for over 99% of tokens in N'} (Eq. 7). Rationale: 'we consider a neuron to be a key neuron if it is activated in over 99% of the tokens'",
   "disclosed": true,
   "location": "p.7 Sec. 4.2.1, Eq. 7"
  },
  {
   "name": "Neun-down (consistently-silent key neuron) criterion",
   "value": "Neun-down = {i | Act[i] <= m for ALL tokens in N'} (Eq. 8) \u2014 note: 'all', not 99%",
   "disclosed": true,
   "location": "p.7 Sec. 4.2.1, Eq. 8"
  },
  {
   "name": "Key layers (Llama-2-7b-chat)",
   "value": "layers 19 to 28 designated as key layers (from the layers 19-31 divergence region; layers very close to the output were excluded because modifying them degraded the fix)",
   "disclosed": true,
   "location": "p.8 Sec. 4.3 Key Layers Selection"
  },
  {
   "name": "Key layers (Mistral-7B-Instruct-v0.1)",
   "value": "never specified \u2014 only Llama2's 19-28 is given as 'for instance'; no per-model key-layer list or selection algorithm is provided for Mistral, Qwen, Gemma, or Yi",
   "disclosed": false,
   "location": "p.8 Sec. 4.3; Algorithm 1/2 take 'KeyLayers[]' as an undefined input"
  },
  {
   "name": "Constant k1 (Eq. 11, beta = k1 * DeltaAct-up + b1)",
   "value": "never given; paper says only 'derived through an adaptive process tailored to the specific dynamics of each model' and 'A set of default values is provided' \u2014 the default values are never printed",
   "disclosed": false,
   "location": "p.7 Sec. 4.2.2, Eq. 11 and following paragraph"
  },
  {
   "name": "Constant b1 (Eq. 11)",
   "value": "never given (same 'adaptive process' / unstated 'default values' language)",
   "disclosed": false,
   "location": "p.7 Sec. 4.2.2, Eq. 11"
  },
  {
   "name": "Constant k2 (Eq. 12, alpha = k2 * DeltaAct-down + b2)",
   "value": "never given",
   "disclosed": false,
   "location": "p.7 Sec. 4.2.2, Eq. 12"
  },
  {
   "name": "Constant b2 (Eq. 12)",
   "value": "never given",
   "disclosed": false,
   "location": "p.7 Sec. 4.2.2, Eq. 12"
  },
  {
   "name": "Numerical intervals for the 'range restriction' mapping of beta and alpha",
   "value": "never given \u2014 text says DeltaAct values are mapped 'to appropriate numerical intervals' but no interval bounds are stated",
   "disclosed": false,
   "location": "p.7 Sec. 4.2.2"
  },
  {
   "name": "Adjustment operators (fix)",
   "value": "Neun-up neurons: Act[n] <- Act[n] + beta (additive). Neun-down neurons: Act[n] <- Act[n] / alpha (division). Rule-based baseline uses fixed alpha = 4, beta = 1.5.",
   "disclosed": true,
   "location": "p.8 Algorithm 2 lines 8-13; p.9-10 Sec. 6.2"
  },
  {
   "name": "Decoding temperature",
   "value": "temperature = 0 (to eliminate randomness); 'All other configurations are default'",
   "disclosed": true,
   "location": "p.3 Sec. 3.1 Experiment Setup"
  },
  {
   "name": "Glitch-token oracle / labeling task",
   "value": "single repetition task, prompt: \"Can you repeat the token '{token}' and return it back to me?\" \u2014 token is glitch iff the model fails to reproduce it. (Li et al.'s length and spelling tasks are NOT used.)",
   "disclosed": true,
   "location": "p.3 Sec. 3.1"
  },
  {
   "name": "Exhaustive-search baseline task set",
   "value": "paraphrasing, spelling, and length calculation per token; ~20 generated tokens per token at temperature 0",
   "disclosed": true,
   "location": "p.8 Sec. 5.1 and p.10 Sec. 7.2"
  },
  {
   "name": "Instrumentation tool",
   "value": "TransformerLens (hooks inserted into all intermediate layers during the first forward pass)",
   "disclosed": true,
   "location": "p.3 Sec. 3.1"
  },
  {
   "name": "Features used for detection",
   "value": "all three: attention pattern + MLP gate + MLP data",
   "disclosed": true,
   "location": "p.10 Sec. 7.1"
  },
  {
   "name": "Features used for fix",
   "value": "two MLP-based features only (MLP gate and MLP data); attention patterns explicitly NOT modified",
   "disclosed": true,
   "location": "p.7 Sec. 4.2.1 note and p.10 Sec. 7.1"
  },
  {
   "name": "Hardware / experiment environment",
   "value": "Workstation, Ubuntu 22.04.3 LTS, 250 GB system memory, 2x NVIDIA A100 GPU with 80 GB memory each (160 GB total GPU memory). The 250.00 GB figure is the memory ceiling cited in Table 4 for the failed No-PCA runs.",
   "disclosed": true,
   "location": "p.8 Sec. 5.1 Experiment Environment; ceiling repeated in Table 4 footnote"
  },
  {
   "name": "Random seeds / number of repeated runs for GlitchProber",
   "value": "never given \u2014 only the rule-based random-sampling baseline is said to be run 100 times and averaged; GlitchProber's own random sampling (gamma=0.1) is reported from what appears to be a single run per model",
   "disclosed": false,
   "location": "p.8 Sec. 5.1 Evaluation Settings"
  },
  {
   "name": "Train/validation split of the sampled set S",
   "value": "never specified \u2014 S is used to train the SVM and the classifier is then applied to V \\ S; no held-out validation protocol described",
   "disclosed": false,
   "location": "p.5-6 Sec. 4.1.2, Algorithm 1"
  }
 ],
 "notes": "(a) WHICH TENSOR THE REPAIR MODIFIES \u2014 UNDERSPECIFIED. The paper defines MLP status as two distinct things: the MLP gate sigma(Z1) and the MLP data Z2, whose element-wise product is the gated output Z-tilde = sigma(Z1) (*) Z2 (Eq. 5, p.2). Section 4.2.1 says \"For the MLP module in each layer, we calculate the activation statistics of the tokens in the normal token set N'\", and Eqs. 7-10 use a single generic Act[i] = \"the activation value of the i-th neuron in the MLP module\". Algorithm 2 (p.8) is equally generic: Activation <- hookModel(Token, Layer); Act[Neuron] <- Act[Neuron] + beta; Act[Neuron] <- Act[Neuron] / alpha; hookModel(Token, Layer) <- Activation. Nowhere does the paper state whether the hook writes to sigma(Z1), to Z2, or to the product Z-tilde. Figure 5's caption/labels show BOTH \"Modified MLP gate\" and \"Modified MLP data\" as outputs, and Sec. 7.1 says \"two MLP-based features are chosen for the fix process\", which suggests both tensors are edited separately \u2014 but if both are edited, a single Neun-up/Neun-down set and a single (alpha, beta) pair cannot be well-defined, since gate and data have different value distributions (gate is post-activation, data is pre-multiplication linear output). The one thing stated unambiguously is the negative: \"we only adjust the activation values of the MLP module and not the attention patterns\" (p.7). This is a genuine reproducibility hole: an implementer must guess among gate / data / product. Note also that with m = 1, the Neun-up condition (Act[i] > 1 for >99% of normal tokens) is very hard to satisfy for a post-SiLU/GELU gate tensor and much easier for a raw Z2 tensor, so the threshold value is only meaningful once the tensor is fixed \u2014 the paper never resolves this.\n\n(b) SEQUENCE POSITION \u2014 SPECIFIED ONLY FOR ATTENTION, UNSPECIFIED FOR MLP. For attention the paper is explicit: attention patterns \"can be extracted from the corresponding row A[n] of the attention scores matrix A\" (p.2, Sec. 2.1), i.e. the last/current position's row, containing only weights to previously generated tokens. Extraction happens \"in the model's first forward process\" (p.5, Sec. 4.1.1) because \"the first forward comprehensively reflects the model's understanding of the input sequence\" (p.3). But the input is a fixed template \u2014 \"Can you repeat the token '{token}' and return it back to me?\" \u2014 so the last position is a template token, NOT the candidate glitch token itself. The paper never says whether the MLP gate/MLP data features are taken at the glitch token's own position, at the final prompt position, or pooled/averaged over positions; nor how features from multiple key layers and 32 attention heads are concatenated into the vector fed to PCA; nor how the variable-length attention row (which grows with prompt length) is made fixed-dimensional. Same gap in the fix phase: the statistics over N' and the runtime adjustment are described per-layer with no position indexing.\n\n(c) VARIANCE / ERROR BARS \u2014 NONE ANYWHERE. Every number in Tables 1-6 and Figures 2 and 6 is a single point estimate. No standard deviations, confidence intervals, error bars, or significance tests appear in the paper. The only nod to repetition is for a baseline, not for the proposed tool: \"For the rule-based random sampling methods, we conducted 100 independent experiments and averaged the results to obtain statistically significant conclusions\" (p.8) \u2014 and even there only the mean is reported. GlitchProber itself depends on a random sample of the vocabulary (gamma = 0.1) and on an SVM trained on that sample, yet no run-to-run variability is quantified. Temperature is set to 0 \"to eliminate randomness\", which removes decoding variance but not sampling variance. No seed is reported.\n\nARITHMETIC AND SELF-CONSISTENCY ISSUES WORTH CHECKING:\n1. Two different \"average F1\" values for the same tool. The Abstract and Conclusion claim an average F1 of 0.86; Table 3's own \"Average Performance\" row gives 0.7835. The 0.86 figure matches the mean of the GlitchProber column in Table 4 (0.8529, 0.8612, 0.8854, 0.8143, 0.8718 -> 0.8571). Table 3 and Table 4 disagree per model too (Llama2: 0.8181 vs 0.8529; Mistral: 0.8053 vs 0.8612) with no explanation of why the same configuration yields different F1 in two tables.\n2. Table 5's average repaired-token count does not match its own rows. GlitchProber column: (4,021 + 1,045 + 14,765 + 13,638 + 4,317)/5 = 7,557.2, but the table and the Contributions section both say 7,758. (The rule-based average 5,613 does check out, as do both average repair rates: 50.064% and 36.786%.) The headline \"repairs an average of 7,758 tokens\" is therefore ~200 tokens higher than the data supports.\n3. The \"approximately 40% time saving vs state-of-the-art\" claim does not match Table 2. Average GlitchProber 89 min 9 s vs GlitchHunter 473 min 11 s is an 81% saving, not 40%. Per model the savings are Llama2 17%, Mistral 34%, Qwen 87%, Gemma 86%, Yi 83%. No individual number is 40%; Mistral (34%) is the closest.\n4. The claimed 0.32 average F1 gain over No-Post does not match Table 4: (0.8571 - 0.5154) = 0.342.\n5. Section 7.1's \"highest F1-score of 0.6117 without post-processing\" is inconsistent with Table 4's No-Post column mean of 0.5154 for the same C=1, degree=3, all-three-features setting.\n6. Table 3 is internally consistent in the sense that F1 = 2PR/(P+R) holds for every row, and the implied glitch-token totals from Table 3 and Table 5 agree per model (Llama2 6,425; Mistral ~2,779; Qwen ~30,690-30,701; Gemma ~27,963; Yi ~8,106). Only the Llama2 count (6,425) is ever printed; the other four must be back-derived.\n7. Precision of exactly 100.00% for GlitchProber is a definitional artifact, not a result: Algorithm 1 line 14 re-runs the repetition-task oracle on every token the SVM flags as glitchy and discards it if the model repeats it correctly. Any method with that post-check has 100% precision by construction, so the precision comparison against GlitchHunter (also 100%) carries no information, and the F1 numbers are a monotone function of recall alone.\n8. Threats-to-validity admits the fix formulation fails: \"In the fixing phase, the linear computation of alpha and beta proves untenable\" (p.11) \u2014 i.e. the authors themselves state Eqs. 11-12 do not work, while still reporting the 50.06% repair rate obtained from them.\n9. Mistral is the weakest repair case by a wide margin (37.60% vs the 50.06% average, and 12.92% for the rule-based baseline vs 36.79% average), so quoting the cross-model averages for Mistral would overstate repair by roughly 12.5 percentage points. Conversely Mistral's detection recall (67.41%) is above the 64.47% average, and its GlitchHunter recall (44.37%) is far above GlitchHunter's 26.52% average \u2014 so Mistral flatters the baseline on detection and penalises the tool on repair.\n10. Sample-size asymmetry: Mistral has the smallest glitch-token population of the five (~2,779, about 8.7% of a 32,000 vocabulary), so its per-model percentages rest on a much smaller denominator than Qwen's (~30,700) or Gemma's (~27,963), which dominate any unweighted \"average\" that is computed over token counts rather than over models."
}