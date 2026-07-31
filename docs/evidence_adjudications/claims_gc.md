{
 "claims": [
  {
   "metric": "Glitch token count, Mistral-7B-Instruct-v0.1",
   "value": "2539",
   "scope": "mistral-specific",
   "location": "Table 2, 'Glitch tokens' column (p.6); restated in Extended Evaluation text (p.6)",
   "quote": "Mistral-7B-Instruct-v0.1 contains 2,539 glitch tokens, whereas Mistral-7B-Instruct-v0.3 contains only 1,020"
  },
  {
   "metric": "GlitchCleaner repaired-token count, Mistral-7B-Instruct-v0.1 (repeat task)",
   "value": "2407",
   "scope": "mistral-specific",
   "location": "Table 2, p.6",
   "quote": "RepairedTokens 956 2407 / Mistral-7B-Instruct-v0.1 2539 / RepairRate 37.65% 94.80%"
  },
  {
   "metric": "GlitchCleaner repair rate, Mistral-7B-Instruct-v0.1 (repeat task)",
   "value": "94.80%",
   "scope": "mistral-specific",
   "location": "Table 2, p.6",
   "quote": "RepairRate 37.65% 94.80%"
  },
  {
   "metric": "GlitchProber repaired-token count, Mistral-7B-Instruct-v0.1 (quoted/derived, not re-run)",
   "value": "956",
   "scope": "mistral-specific",
   "location": "Table 2, p.6; provenance stated on p.6 body text",
   "quote": "where the GlitchProber results are taken from their original paper"
  },
  {
   "metric": "GlitchProber repair rate, Mistral-7B-Instruct-v0.1 (imported from Zhang et al. 2024, ASE'24)",
   "value": "37.65%",
   "scope": "mistral-specific",
   "location": "Table 2, p.6",
   "quote": "the GlitchProber results are taken from their original paper"
  },
  {
   "metric": "GSM8K, Mistral-7B-Instruct-v0.1, original model",
   "value": "34.57",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K, Mistral-7B-Instruct-v0.1, GlitchCleaner",
   "value": "34.04",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K, Mistral-7B-Instruct-v0.1, GlitchProber",
   "value": "32.98",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K, Mistral-7B-Instruct-v0.1, full fine-tuning baseline",
   "value": "25.54",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "MMLU, Mistral-7B-Instruct-v0.1, original model",
   "value": "53.45",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "MMLU, Mistral-7B-Instruct-v0.1, GlitchCleaner",
   "value": "53.38",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "MMLU, Mistral-7B-Instruct-v0.1, GlitchProber",
   "value": "53.37",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "MMLU, Mistral-7B-Instruct-v0.1, full fine-tuning baseline",
   "value": "53.38",
   "scope": "mistral-specific",
   "location": "Table 4, p.7"
  },
  {
   "metric": "Model used for the Wasserstein-distance activation analysis (component/layer selection evidence)",
   "value": "Mistral-7B-Instruct-v0.1 (all layers analyzed; equal number of normal tokens sampled as control; no numeric distances given in main text)",
   "scope": "mistral-specific",
   "location": "Figure 1 caption + 'Gated LoRA Architecture' text, pp.3-4",
   "quote": "Wasserstein distance between the activations of glitch tokens and normal tokens across different components within Transformer blocks in the Mistral-7B-Instruct-v0.1 model"
  },
  {
   "metric": "Token-filtering counts for Mistral-7B-Instruct-v0.1 (special / undecodable / unreachable)",
   "value": "NOT REPORTED in main text - deferred to Appendix B",
   "scope": "mistral-specific",
   "location": "Token Filter section, p.3",
   "quote": "The results for other models are provided in Appendix B."
  },
  {
   "metric": "Key-layer indices used for Mistral-7B-Instruct-v0.1",
   "value": "NOT REPORTED - only the Llama-2-7b-chat example (layers 19-28) is given",
   "scope": "mistral-specific",
   "location": "Experimental Setup, p.5",
   "quote": "For example, in Llama-2-7b-chat, we select layers 19 through 28 as the key layers."
  },
  {
   "metric": "Glitch token count, Mistral-7B-Instruct-v0.3",
   "value": "1020",
   "scope": "other-model",
   "location": "Extended Evaluation, p.6",
   "quote": "Mistral-7B-Instruct-v0.3 contains only 1,020"
  },
  {
   "metric": "GlitchCleaner repair rate, Mistral-7B-Instruct-v0.3",
   "value": "93.73%",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Mistral-7B-Instruct-v0.3 benchmarks before GlitchCleaner (GSM8K / MMLU / C-Eval / MetaBench / PIQA / AGIEval)",
   "value": "50.11 / 59.73 / 44.42 / 61.57 / 81.55 / 36.62",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Mistral-7B-Instruct-v0.3 benchmarks after GlitchCleaner",
   "value": "48.75 / 59.89 / 43.61 / 62.00 / 81.66 / 36.80",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "GlitchCleaner average repair rate (headline claim)",
   "value": "86.88%",
   "scope": "cross-model-average",
   "location": "Abstract, Contribution (3), Table 2 'Average' row",
   "quote": "our method achieves an average repair rate of 86.88%"
  },
  {
   "metric": "GlitchProber average repair rate (headline comparison, imported)",
   "value": "50.08%",
   "scope": "cross-model-average",
   "location": "Abstract/Introduction and Table 2 'Average' row",
   "quote": "their repair success rate remains only 50.08%, still a long way from being usable"
  },
  {
   "metric": "Claimed improvement of GlitchCleaner over GlitchProber",
   "value": "\"over 30%\" (absolute gap is 36.80 percentage points; relative gain is 73.5%)",
   "scope": "cross-model-average",
   "location": "Abstract; Contribution (3); p.2",
   "quote": "it improves repair accuracy by more than 30% with no degradation in performance"
  },
  {
   "metric": "Average GSM8K (5 models): original / GlitchCleaner / GlitchProber / full fine-tuning",
   "value": "31.43 / 31.04 / 24.96 / 25.46",
   "scope": "cross-model-average",
   "location": "Table 4 'Average' rows, p.7"
  },
  {
   "metric": "Average MMLU (5 models): original / GlitchCleaner / GlitchProber / full fine-tuning",
   "value": "50.77 / 51.06 / 48.93 / 49.03",
   "scope": "cross-model-average",
   "location": "Table 4 'Average' rows, p.7"
  },
  {
   "metric": "Glitch token count, Llama-2-7b-chat",
   "value": "4743 (out of 31,771 traversed after filtering)",
   "scope": "other-model",
   "location": "Glitch Token Identification, p.3; Table 2",
   "quote": "We traverse the 31,771 filtered tokens in the Llama-2-7B-chat model and identified 4,743 glitch tokens."
  },
  {
   "metric": "GlitchCleaner repaired tokens / repair rate, Llama-2-7b-chat",
   "value": "4210 / 88.76%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchProber repaired tokens / repair rate, Llama-2-7b-chat (imported)",
   "value": "2968 / 62.58%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "Glitch token count, Gemma-2b-it",
   "value": "29831",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchCleaner repaired tokens / repair rate, Gemma-2b-it",
   "value": "20697 / 69.38%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchProber repaired tokens / repair rate, Gemma-2b-it (imported)",
   "value": "14548 / 48.77%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "Glitch token count, Qwen-7B-Chat",
   "value": "27686",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchCleaner repaired tokens / repair rate, Qwen-7B-Chat",
   "value": "24582 / 88.79%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchProber repaired tokens / repair rate, Qwen-7B-Chat (imported)",
   "value": "13320 / 48.11%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "Glitch token count, Yi-6B-Chat",
   "value": "5985",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchCleaner repaired tokens / repair rate, Yi-6B-Chat",
   "value": "5547 / 92.68%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GlitchProber repaired tokens / repair rate, Yi-6B-Chat (imported)",
   "value": "3188 / 53.27%",
   "scope": "other-model",
   "location": "Table 2, p.6"
  },
  {
   "metric": "GSM8K / MMLU, Llama-2-7b-chat: original / GlitchCleaner / GlitchProber / finetuning",
   "value": "GSM8K 23.05 / 23.88 / 21.23 / 21.54 ; MMLU 46.38 / 46.23 / 46.31 / 42.34",
   "scope": "other-model",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K / MMLU, Gemma-2b-it: original / GlitchCleaner / GlitchProber / finetuning",
   "value": "GSM8K 10.99 / 9.33 / 10.77 / 2.67 ; MMLU 38.14 / 40.27 / 38.16 / 33.62",
   "scope": "other-model",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K / MMLU, Qwen-7B-Chat: original / GlitchCleaner / GlitchProber / finetuning",
   "value": "GSM8K 47.84 / 47.46 / 48.60 / 40.33 ; MMLU 54.27 / 53.82 / 54.25 / 54.14",
   "scope": "other-model",
   "location": "Table 4, p.7"
  },
  {
   "metric": "GSM8K / MMLU, Yi-6B-Chat: original / GlitchCleaner / GlitchProber / finetuning",
   "value": "GSM8K 40.71 / 40.49 / 11.22 / 37.22 ; MMLU 61.63 / 61.58 / 52.55 / 61.67",
   "scope": "other-model",
   "location": "Table 4, p.7"
  },
  {
   "metric": "Inference speed, original model (Llama-2-7b-chat, H200 GPU)",
   "value": "66.30 tokens/sec",
   "scope": "other-model",
   "location": "Table 6, p.7",
   "quote": "We evaluate the inference speed of the Llama-2-7b-chat model after GlitchCleaner correction on an H200 GPU"
  },
  {
   "metric": "Inference speed, GlitchCleaner (Llama-2-7b-chat, H200 GPU)",
   "value": "62.83 tokens/sec (5.23% slowdown vs original)",
   "scope": "other-model",
   "location": "Table 6, p.7"
  },
  {
   "metric": "Inference speed, GlitchProber (Llama-2-7b-chat, H200 GPU) - authors' own reimplementation",
   "value": "11.82 tokens/sec (5.61x slower than original, 5.32x slower than GlitchCleaner)",
   "scope": "other-model",
   "location": "Table 6, p.7",
   "quote": "compare it with the original model's inference speed and that of a simple GlitchProber implementation"
  },
  {
   "metric": "Model + GPU on which the inference-speed table was measured",
   "value": "Llama-2-7b-chat on a single H200 GPU (NOT Mistral; no batch size, sequence length, precision, or generation length reported)",
   "scope": "other-model",
   "location": "Model Inference Speed Comparison, p.6 + Table 6 caption, p.7"
  },
  {
   "metric": "Repair rate on Spelling-type glitch tokens (LLaMA-2 only)",
   "value": "94.43%",
   "scope": "other-model",
   "location": "Table 3, p.6",
   "quote": "measure the repair rates on the LLaMA-2 model"
  },
  {
   "metric": "Repair rate on Length-type glitch tokens (LLaMA-2 only)",
   "value": "92.45%",
   "scope": "other-model",
   "location": "Table 3, p.6"
  },
  {
   "metric": "Llama-3.1-8B-Instruct repair rate",
   "value": "95.09%",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Llama-3.1-8B-Instruct benchmarks before / after GlitchCleaner (GSM8K, MMLU, C-Eval, MetaBench, PIQA, AGIEval)",
   "value": "before 75.58 / 68.05 / 53.86 / 55.69 / 79.86 / 42.40 ; after 77.33 / 68.29 / 54.08 / 55.06 / 79.49 / 42.44",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Qwen3-8B repair rate",
   "value": "98.44%",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Qwen3-8B benchmarks before / after GlitchCleaner (GSM8K, MMLU, C-Eval, MetaBench, PIQA, AGIEval)",
   "value": "before 88.40 / 72.97 / 79.63 / 57.23 / 76.82 / 56.80 ; after 87.83 / 72.65 / 79.32 / 57.04 / 75.51 / 57.88",
   "scope": "other-model",
   "location": "Table 5, p.7"
  },
  {
   "metric": "Token filtering, Llama-2-7B-chat: total filtered / vocabulary size",
   "value": "229 filtered out of 32,000",
   "scope": "other-model",
   "location": "Token Filter, p.3",
   "quote": "In the Llama-2-7B-chat model (Touvron et al. 2023), we filter 229 tokens out of 32,000 tokens, including 3 special tokens, 2 undecodeable tokens, and 224 unreachable tokens."
  },
  {
   "metric": "Token filtering breakdown, Llama-2-7B-chat: SPECIAL / UNDECODEABLE / UNREACHABLE",
   "value": "3 / 2 / 224",
   "scope": "other-model",
   "location": "Token Filter, p.3"
  },
  {
   "metric": "Filtered vocabulary actually traversed, Llama-2-7B-chat",
   "value": "31771 (= 32,000 - 229)",
   "scope": "other-model",
   "location": "Glitch Token Identification, p.3"
  },
  {
   "metric": "Token filtering counts for all models other than Llama-2-7B-chat",
   "value": "NOT REPORTED in main text - deferred to Appendix B",
   "scope": "general",
   "location": "Token Filter, p.3",
   "quote": "The results for other models are provided in Appendix B."
  },
  {
   "metric": "Additional parameter overhead of the LoRA branches",
   "value": "less than 0.1% of the original model's parameters (no absolute parameter count given; my check on Llama-2-7B with r=4, 2 modules x 10 layers gives ~1.21M params, ~0.018%)",
   "scope": "general",
   "location": "Abstract; Contribution (1); Experimental Setup p.5; Conclusion",
   "quote": "the number of parameters added by the LoRA branches is less than 0.1% of the original model parameters"
  },
  {
   "metric": "Number of models evaluated in the main repair-rate comparison",
   "value": "Text says \"six widely used models\" but only five are named and only five appear in Table 2 (Llama-2-7b-chat, Gemma-2b-it, Mistral-7B-Instruct-v0.1, Qwen-7B-Chat, Yi-6B-Chat)",
   "scope": "general",
   "location": "Repair Rate section, p.5-6 + Table 2",
   "quote": "Specifically, we evaluate six widely used models, including Llama-2-7b-chat ..., Mistral-7B-Instruct-v0.1 ..., Qwen-7B-Chat ..., Gemma-2b-it ... and Yi-6B-Chat"
  },
  {
   "metric": "Definition of repair rate (denominator)",
   "value": "ratio of correct responses to the total number of faulty (glitch) tokens",
   "scope": "general",
   "location": "Repair Rate, p.6",
   "quote": "We calculate the accuracy rate as the ratio of correct responses to the total number of faulty tokens"
  },
  {
   "metric": "GlitchProber key-neuron screening threshold (as described by GlitchCleaner authors)",
   "value": "neurons activated in most cases (99%) or inactive in most cases (99%) for normal tokens",
   "scope": "general",
   "location": "Differences From GlitchProber, p.5",
   "quote": "identifies neurons that are either activated in most cases (99%) or remain inactive in most cases (99%) for normal tokens"
  },
  {
   "metric": "Decoding temperature used for glitch-token identification and evaluation",
   "value": "0",
   "scope": "general",
   "location": "Glitch Token Identification, p.3",
   "quote": "we set the temperature parameter to 0, which ensures that the model produces identical outputs for the same input"
  },
  {
   "metric": "DeepSeek-V3-0324 parameter count (motivating example, Table 1)",
   "value": "685 billion parameters",
   "scope": "general",
   "location": "Table 1 caption, p.2"
  },
  {
   "metric": "Evaluation harness used for GSM8K/MMLU",
   "value": "lm-evaluation-harness (Gao et al. 2024); no few-shot setting, no seed, no n-shot count reported",
   "scope": "general",
   "location": "Unaffected Performance, p.6"
  },
  {
   "metric": "Gating variable lambda at inference",
   "value": "lambda = 1 if the tokenized input intersects the known glitch-token set G, else 0; W' = W + lambda * (alpha/r) * A B",
   "scope": "general",
   "location": "Gated LoRA Architecture p.4 + Algorithm 1 lines 13-18, p.5",
   "quote": "When the model input contains glitch tokens, lambda is set to 1; otherwise, lambda is set to 0."
  },
  {
   "metric": "Gating condition, formal (Algorithm 1)",
   "value": "I <- Tokenize(P); if I intersect G != empty then lambda<-1 else lambda<-0 - i.e. a set-membership lookup against the precomputed glitch token set G, not a learned/online detector",
   "scope": "general",
   "location": "Algorithm 1, lines 13-18, p.5",
   "quote": "13: I <- Tokenize(P) 14: if I\u2229G\u2260\u2205 then 15: lambda<-1 16: else 17: lambda<-0"
  },
  {
   "metric": "Gating during training",
   "value": "lambda is hard-set to 1 for the whole fine-tuning stage (Algorithm 1 line 10: 'Set lambda <- 1')",
   "scope": "general",
   "location": "Algorithm 1, line 10, p.5",
   "quote": "10: Set lambda<-1"
  },
  {
   "metric": "Repair evaluation set vs LoRA training set (train/test identity)",
   "value": "SAME SET - no held-out split is described anywhere in the main text; the LoRA is fine-tuned on a dataset built from the model's own detected glitch tokens and then evaluated on whether it repairs those glitch tokens",
   "scope": "general",
   "location": "Repair Rate, p.5 (first paragraph); corroborated by Method step (4) p.3 and 'Gated LoRA Branches Parameter Fine-tuning' p.4",
   "quote": "We fine-tune the model with added LoRA branches using a dataset built from glitch tokens filtered by the model, then evaluate whether the repaired model could generate correct responses."
  },
  {
   "metric": "Repair evaluation set vs training set - corroborating wording #2",
   "value": "dataset construction and fine-tuning are both defined over 'these glitch tokens' / 'the identified glitch tokens', with no sampling, split, or holdout language",
   "scope": "general",
   "location": "GlitchCleaner method overview step (4), p.3",
   "quote": "(4) constructing a dataset using these glitch tokens and their expected correct responses, then fine-tuning the model using the constructed dataset to train the gated LoRA branches."
  },
  {
   "metric": "Repair evaluation set vs training set - corroborating wording #3",
   "value": "training data uses the identical prompt template as the detection/evaluation stage",
   "scope": "general",
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4",
   "quote": "We construct a question-answering dataset using the identified glitch tokens for subsequent fine-tuning. The dataset includes the same prompt templates described in Glitch Token Identification, paired with their corresponding expected responses."
  },
  {
   "metric": "Repetition-task prompt template (verbatim, method section)",
   "value": "\"Can you repeat the string '{token}' and return it back to me? Answer: Here is the repeated string:\"",
   "scope": "general",
   "location": "Glitch Token Identification, p.3",
   "quote": "Can you repeat the string '{token}' and return it back to me? Answer: Here is the repeated string:"
  },
  {
   "metric": "Repetition-task prompt as shown in the motivating examples (Table 1) - wording differs from the method section",
   "value": "\"Question: Can you repeat the string 'Azalera' and return back to me? Answer: Here is the repeated string:\" (note: 'return back to me', missing the word 'it', and prefixed with 'Question:')",
   "scope": "general",
   "location": "Table 1, p.2",
   "quote": "Question:Can you repeat the string 'Azalera' and return back to me? Answer:Here is the repeated string:"
  },
  {
   "metric": "Spelling-task prompt (as shown in Table 1 examples)",
   "value": "\"Question: Please can you spell out the string '{token}' with hyphens between each letter? Answer: Of course! The spelling of the string is:\"",
   "scope": "general",
   "location": "Table 1, p.2"
  },
  {
   "metric": "Length-task prompt (as shown in Table 1 examples)",
   "value": "\"Question: What is the length of this string '{token}'? Answer: The length of this string is:\"",
   "scope": "general",
   "location": "Table 1, p.2"
  },
  {
   "metric": "Few-shot examples for the spelling/length repair datasets",
   "value": "used but NOT specified in main text - deferred to Appendix G",
   "scope": "general",
   "location": "Repair Rate, p.6",
   "quote": "we include specific few-shot examples in the prompts. The specific details are presented in Appendix G."
  },
  {
   "metric": "Weighted (token-count-weighted) repair rate implied by Table 2, GlitchCleaner - DERIVED BY ME, not printed in paper",
   "value": "57,443 / 70,784 = 81.15% (vs the paper's unweighted 86.88%)",
   "scope": "cross-model-average",
   "location": "derived from Table 2, p.6"
  },
  {
   "metric": "Weighted (token-count-weighted) repair rate implied by Table 2, GlitchProber - DERIVED BY ME, not printed in paper",
   "value": "34,980 / 70,784 = 49.42% (vs the paper's unweighted 50.08%)",
   "scope": "cross-model-average",
   "location": "derived from Table 2, p.6"
  }
 ],
 "hyperparameters": [
  {
   "name": "LoRA rank r",
   "value": "4",
   "disclosed": true,
   "location": "Results > Experimental Setup, p.5 ('We set the LoRA rank parameter r to 4 and the scaling factor alpha to 4.')"
  },
  {
   "name": "LoRA scaling factor alpha",
   "value": "4",
   "disclosed": true,
   "location": "Results > Experimental Setup, p.5"
  },
  {
   "name": "Effective LoRA scale alpha/r",
   "value": "1.0 (derived: 4/4)",
   "disclosed": true,
   "location": "derived from Experimental Setup, p.5 + update rule W' = W + lambda*(alpha/r)*A*B, p.4"
  },
  {
   "name": "Target modules",
   "value": "MLP gate and MLP data components only - i.e. the two parallel projection matrices W1 and W2 of the gated MLP (gate_proj and up_proj). The down-projection W3 and all attention modules are explicitly NOT adapted.",
   "disclosed": true,
   "location": "Gated LoRA Architecture, p.4 ('we integrate LoRA branches into the projection matrices W1 and W2 from Equations (1)'); Algorithm 1 lines 4-5"
  },
  {
   "name": "Key layers (Llama-2-7b-chat)",
   "value": "layers 19 through 28 (10 layers), taken from GlitchProber's configuration",
   "disclosed": true,
   "location": "Experimental Setup, p.5"
  },
  {
   "name": "Key layers (Mistral-7B-Instruct-v0.1 and every other model)",
   "value": "not given; only 'we adopt the configuration from GlitchProber, primarily targeting the posterior layers of the model'",
   "disclosed": false,
   "location": "Experimental Setup, p.5"
  },
  {
   "name": "Gate variable lambda",
   "value": "binary; 1 when tokenized input intersects the precomputed glitch-token set G, else 0; forced to 1 during fine-tuning",
   "disclosed": true,
   "location": "Gated LoRA Architecture p.4; Algorithm 1 lines 10, 13-18, p.5"
  },
  {
   "name": "Decoding temperature (detection + evaluation)",
   "value": "0",
   "disclosed": true,
   "location": "Glitch Token Identification, p.3"
  },
  {
   "name": "Training epochs",
   "value": "not stated anywhere in the main text",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5 (no numbers given)"
  },
  {
   "name": "Learning rate",
   "value": "not stated",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5"
  },
  {
   "name": "Batch size",
   "value": "not stated",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5"
  },
  {
   "name": "Optimizer",
   "value": "not stated",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5"
  },
  {
   "name": "LR scheduler / warmup",
   "value": "not stated",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5"
  },
  {
   "name": "LoRA dropout",
   "value": "never mentioned",
   "disclosed": false,
   "location": "n/a - absent from the paper"
  },
  {
   "name": "LoRA B-matrix initialization",
   "value": "never mentioned (matters for the losslessness claim at lambda=0)",
   "disclosed": false,
   "location": "n/a - absent from the paper"
  },
  {
   "name": "Which parameters are frozen",
   "value": "the full main model body is frozen; only LoRA branch parameters are trained",
   "disclosed": true,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4-5 ('we froze the parameters of the main model body and exclusively train the parameters of the LoRA branches')"
  },
  {
   "name": "Training set size / number of QA pairs",
   "value": "not stated explicitly; implied to be one prompt-response pair per identified glitch token (e.g. 2,539 for Mistral-v0.1, 4,743 for Llama-2)",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning, p.4"
  },
  {
   "name": "Train/validation/test split",
   "value": "none described; evaluation appears to be on the training glitch tokens themselves",
   "disclosed": false,
   "location": "Repair Rate, p.5"
  },
  {
   "name": "Training hardware / training time",
   "value": "not stated (H200 is mentioned only for the inference-speed measurement); training cost only described qualitatively as 'limited computational cost' and 'highly efficient training time and space requirements'",
   "disclosed": false,
   "location": "Gated LoRA Branches Parameter Fine-tuning p.5; Conclusion p.7"
  },
  {
   "name": "Model precision / dtype (fp16, bf16, etc.)",
   "value": "not stated",
   "disclosed": false,
   "location": "n/a - absent from the paper"
  },
  {
   "name": "Max new tokens / generation length for the repetition task",
   "value": "not stated",
   "disclosed": false,
   "location": "n/a - absent from the paper"
  },
  {
   "name": "Answer-matching / correctness criterion for a 'repaired' token",
   "value": "not formally defined; only 'if the model produces unexpected outputs, we classify the corresponding token as a glitch token' - no string-match rule, normalization, or tolerance specified",
   "disclosed": false,
   "location": "Glitch Token Identification, p.3"
  },
  {
   "name": "Number of random seeds / repeated runs / variance",
   "value": "none; every table reports a single point estimate with no std dev or CI",
   "disclosed": false,
   "location": "Tables 2-6"
  },
  {
   "name": "GSM8K/MMLU n-shot configuration",
   "value": "not stated; only 'using the lm-evaluation-harness'",
   "disclosed": false,
   "location": "Unaffected Performance, p.6"
  },
  {
   "name": "Inference-speed measurement setup (batch size, prompt length, generation length, whether lambda was 0 or 1)",
   "value": "not stated; only model (Llama-2-7b-chat) and GPU (H200)",
   "disclosed": false,
   "location": "Model Inference Speed Comparison p.6 + Table 6, p.7"
  },
  {
   "name": "Ablation on r and alpha",
   "value": "referenced but values/results deferred to Appendix F",
   "disclosed": false,
   "location": "Ablation Study, p.7 ('Appendix F analyzes the effects of varying the r and alpha hyperparameters')"
  },
  {
   "name": "Ablation on inserting gated LoRA into other Transformer components",
   "value": "referenced but deferred to Appendix E",
   "disclosed": false,
   "location": "Ablation Study, p.7"
  },
  {
   "name": "Wasserstein distance computation procedure",
   "value": "referenced but deferred to Appendix C; no numeric distances in the main text (Figure 1 is a plot only)",
   "disclosed": false,
   "location": "Gated LoRA Architecture, p.4"
  }
 ],
 "notes": "CRITICAL ITEMS FOR YOUR REPRODUCTION WORK:\n\n(a) GATING MECHANISM. lambda is NOT learned and NOT input-content-inferred. Algorithm 1 lines 13-18 make it an explicit set-membership test: tokenize the prompt, and if the token-id set intersects G (the precomputed glitch-token set), set lambda=1, else lambda=0. This means GlitchCleaner presupposes a complete, correct glitch-token list at inference time - the detection problem is assumed solved, and detection cost/error is entirely outside the reported 86.88%. Consequence: the \"repair rate\" is conditional on oracle detection. Any end-to-end pipeline number would be repair_rate x detection_recall. Also note lambda is hard-set to 1 for all of training (line 10), so the model never sees the lambda=0 path during optimization.\n\n(b) TRAIN == TEST. There is no held-out split anywhere in the main text. The decisive sentence is the first line of the Repair Rate section (p.5): \"We fine-tune the model with added LoRA branches using a dataset built from glitch tokens filtered by the model, then evaluate whether the repaired model could generate correct responses.\" Same-sentence train-then-evaluate, same token set, same prompt template (\"The dataset includes the same prompt templates described in Glitch Token Identification\"), and the repair-rate denominator is \"the total number of faulty tokens\" - i.e. all detected glitch tokens, which is exactly the training set. So Table 2's 94.80% for Mistral is a training-set memorization figure, not generalization. This matches your Finding 7/8 (GC train-split 89.9% vs held-out 55-75%).\n\n(c) PROMPT TEMPLATE INCONSISTENCY. The method section (p.3) gives: \"Can you repeat the string '{token}' and return it back to me? Answer: Here is the repeated string:\". Table 1 (p.2) shows a different string: \"Question: Can you repeat the string 'Azalera' and return back to me? Answer: Here is the repeated string:\" - it adds a \"Question:\" prefix and drops the word \"it\". Detection rates are sensitive to this; pick one and state which. Note the extracted PDF text has whitespace stripped, so the quotes above are re-spaced reconstructions; token-level whitespace (e.g. leading space, quote style) is unverifiable from the extraction.\n\nINTERNAL INCONSISTENCY WORTH EXPLOITING (losslessness claim): If lambda=0 makes W' exactly equal to W, then the GlitchCleaner column of Table 4 must be bit-identical to the Original model column on GSM8K/MMLU (benchmark prompts contain no glitch tokens). It is not. Gemma-2b-it MMLU goes 38.14 -> 40.27 (+2.13), Llama-2 GSM8K goes 23.05 -> 23.88 (+0.83), Gemma GSM8K drops 10.99 -> 9.33 (-1.66). Under a truly gated-off adapter these deltas are mathematically impossible. Either (i) the gate fired on benchmark prompts (i.e. some benchmark tokens are in G, which would mean the \"lossless\" claim is false in practice), (ii) the numbers are re-run with nondeterminism they did not control, or (iii) the eval was run with lambda=1. The paper never states which. This is a clean, checkable contradiction.\n\nGLITCHPROBER NUMBERS ARE IMPORTED, EXCEPT THE SPEED NUMBER. p.6 states explicitly \"where the GlitchProber results are taken from their original paper\" (ASE'24). But Table 6's GlitchProber 11.82 tok/s is the authors' own \"simple GlitchProber implementation\" - self-coded, unspecified, unoptimized, and by their own wording not the original authors' code. So the speed comparison and the repair-rate comparison are not from the same source and the 5.6x slowdown attributed to GlitchProber is not independently sourced.\n\nGLITCHPROBER \"RepairedTokens\" COLUMN IS ALMOST CERTAINLY DERIVED, NOT MEASURED. For every row, GP_repaired == round(GP_rate x GlitchCleaner's own glitch-token count): 0.6258x4743=2968, 0.3765x2539=956, 0.4811x27686=13320, 0.5327x5985=3188, 0.4877x29831~14550 (reported 14548). If GlitchProber's paper detected a different number of glitch tokens per model (very likely, since detection methods differ), then the GP RepairedTokens column describes a quantity that was never measured by anyone. Worth cross-checking against the GlitchProber paper's own token counts.\n\nAVERAGING METHOD. Both the 86.88% and 50.08% are unweighted arithmetic means over the five per-model rates (I verified: 434.41/5 and 250.38/5). Token-count-weighted means are 81.15% (GC) and 49.42% (GP) - so the headline number is inflated ~5.7 points by giving Mistral (2,539 tokens) the same weight as Gemma (29,831 tokens). Mistral is simultaneously GC's best model (94.80%) and GP's worst (37.65%), so the unweighted mean maximizes the headline gap. The abstract's \"improvement of over 30%\" is a 36.80-point absolute gap; stated as a ratio it is 1.735x, and the \"over 30%\" phrasing is ambiguous between the two.\n\nMISTRAL COVERAGE GAPS. For Mistral-7B-Instruct-v0.1 the paper gives ONLY: glitch count 2539, repaired 2407, rate 94.80%, GP rate 37.65%/956, GSM8K 34.57->34.04, MMLU 53.45->53.38, and the Figure 1 Wasserstein plot. It does NOT give: Mistral's token-filter breakdown (Appendix B), Mistral's key-layer indices, Mistral inference speed, or Mistral spelling/length repair rates (Table 3 is Llama-2 only). If you need Mistral key layers you must either fetch Appendix B/the GitHub repo (https://github.com/FAVENO/GlitchCleaner) or infer from GlitchProber's config.\n\nOTHER GAPS: \"six widely used models\" vs five actually reported. Table 3 (spelling 94.43%, length 92.45%) gives no denominators, no GlitchProber baseline, and no per-type token counts, so raw repaired counts cannot be recovered. Table 5 gives no GlitchProber baseline and no token counts except Mistral-v0.3's 1,020. No seeds, no variance, no repeats anywhere. Training hyperparameters (epochs/LR/batch/optimizer/scheduler) are entirely absent from the main text - reproduction requires the repo."
}