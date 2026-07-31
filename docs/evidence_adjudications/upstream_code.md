GLITCHCLEANER RELEASED CODE — PRECISE BEHAVIOURAL AUDIT
Repo: E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/ (AAAI-26, "GlitchCleaner: Lightweight Glitch Tokens Repairing by Lossless Gated LoRA in LLMs")
Paper text: E:/user3/FYS_SWK/Benchmark/papers/extracted/Glitchcleaner_text.txt

Files audited (absolute paths):
- E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/GlitchCleaner.py (296 lines — inference/eval path)
- E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/Fine-tuning/fine-tuning.py (663 lines — training path)
- E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/Fine-tuning/tokenfilter.py (63 lines)
- E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/Tutorials.ipynb (4 cells)
- E:/user3/FYS_SWK/Benchmark/third_party/GlitchCleaner/Glitchtokens/*.csv, LoRA-Parameter/*.pt

Claims below marked [MEASURED] were verified by executing code in E:/user3/FYS_SWK/Benchmark/.venv (torch 2.6.0, transformers 5.14.1) against the authors' own classes and their shipped checkpoints/CSVs, using the locally cached mistralai/Mistral-7B-Instruct-v0.1 tokenizer — which is exactly the model `fine-tuning.py:49` is configured for. Everything else is read directly from source.

================================================================
1. GATING MECHANISM (config_flag / lambda)
================================================================

1.1 How it is computed — GlitchCleaner.py:61-67 (identical logic at fine-tuning.py:318-326)

    def create_config_flag(self, token_ids):
        batch_size = token_ids.shape[0]
        config_flag = torch.zeros(batch_size, device=..., dtype=self.model.dtype)   # :63
        for b in range(batch_size):
            if any(token.item() in self.glitchtokens for token in token_ids[b]):    # :65
                config_flag[b] = 1                                                  # :66
        return config_flag.view(-1, 1, 1)                                           # :67

Properties:
- Granularity is PER SEQUENCE, not per token. Returned shape is (B,1,1) (:67), which broadcasts over
  every sequence position and every hidden channel. If one token anywhere in a 2000-token context is in
  the glitch set, the adapter is applied at ALL positions of that sequence.
  [MEASURED] With a 5-token sequence containing one glitch id, the per-position |LoRA delta| was
  non-zero at all 5 positions: [4.2724, 7.0737, 12.3981, 10.5914, 5.2434].
- It is binary (0.0 or 1.0), cast to the model dtype (:63). There is no soft/learned gate. No parameters.
- `self.glitchtokens` is a PYTHON LIST, not a set — GlitchCleaner.py:159 `glitchtokens = df['index'].tolist()`.
  So `token.item() in self.glitchtokens` (:65) is an O(|G|) linear scan executed once per token, plus one
  GPU->CPU sync per token via `.item()`. Cost is O(seq_len × |G|) per sequence in pure Python.
  [MEASURED] 512-token prompt, |G|=27,686 (Qwen-sized): 55.3 ms as shipped (list) vs 0.5 ms with a set
  — a 113x self-inflicted slowdown, incurred on EVERY forward and EVERY generate call.

1.2 Where it is computed
- GlitchCleaner.py:102 (forward) and :117 (generate).
- In `generate`, the flag is computed ONCE from the prompt ids (:117) and then held fixed for every
  decoding step. Tokens emitted during decoding are never re-checked. (Matches paper Algorithm 1 lines
  13-18, so not a discrepancy, but worth stating.)

1.3 How it is applied to the LoRA branch — GlitchCleaner.py:29-33

    def forward(self, x, config_flag=None):
        if config_flag is None:
            return self.linear(x)                                     # :31  LoRA skipped
        else:
            return self.linear(x) + (config_flag * self.lora(x))      # :33  LoRA COMPUTED, then scaled

Delivery is by monkey-patching the bound `forward` of each wrapped Linear with a functools.partial that
pins config_flag (GlitchCleaner.py:69-87 `_patch_lora_forward`), then restoring it (:89-99
`_restore_lora_forward`). Patch/restore brackets every forward (:103,:107) and generate (:118,:120).
The patch/restore cycle is idempotent — repeated calls do not nest partials.

1.4 ***IS THE LoRA STILL COMPUTED WHEN THE FLAG IS 0? YES.***  (This is the key inference-cost finding.)

Because `_patch_lora_forward` is called unconditionally at :103 and :118, `config_flag` is NEVER None on
the wrapper path. The `config_flag is None` fast path at :30-31 is dead code in practice. Execution always
takes :33, which evaluates `self.lora(x)` — i.e. `alpha/rank * (x @ A @ B)` (:18) — and only then
multiplies by the 0/1 scalar. Nothing is skipped, no branch is elided, no kernel is avoided.

[MEASURED] Instrumenting the authors' own `GlitchCleaner.LoRALayer.forward` with a call counter, using
their real `LinearWithLoRA`/`GlitchCleaner` classes on a 4-layer stub with 2 target layers (4 wrapped
Linears total):
    CLEAN input -> config_flag = [0.0] -> LoRALayer.forward invocations = 4
    DIRTY input -> config_flag = [1.0] -> LoRALayer.forward invocations = 4
So the adapter FLOPs on a clean prompt are identical to those on a glitch prompt. The gate is a numerical
mask, not a control-flow gate. Consequences for the thesis:
  (a) The inference-speed number in paper Table 6 (66.30 -> 62.83 tok/s, -5.2%) is the cost of ALWAYS
      running the adapter; it is not evidence that clean inputs are free. The paper's framing
      ("activated only when ... detected", Fig. 2 caption; "For clean inputs, the model operates in its
      original form", p.1) describes control flow the code does not implement.
  (b) The 62.83 tok/s figure additionally cannot include the Python gate cost at its shipped complexity
      for a model with a large glitch set (see 1.1 measurement).
  (c) A reimplementation that genuinely short-circuits (`if flag==0: return self.linear(x)`) is NOT
      equivalent to this code in cost, only in output. Worth stating explicitly in a comparison section.

1.5 fine-tuning.py has the OPPOSITE default semantics — fine-tuning.py:291-295

    def forward(self, x, config_flag=None):
        base_out = self.linear(x)
        if config_flag is not None:
            return base_out + config_flag * self.lora(x)   # :294
        return base_out + self.lora(x)                     # :295  <-- LoRA UNCONDITIONALLY ON

So in the training file, config_flag=None means "adapter always on"; in the inference file
(GlitchCleaner.py:31) config_flag=None means "adapter off". Two files, same class name, inverted
behaviour on the same code path. This matters at fine-tuning.py:526, where the Yi evaluation branch calls
`model.generate(...)` on the RAW model rather than `wrapped_model` — bypassing the wrapper, hitting the
None path, and therefore evaluating Yi with the LoRA unconditionally enabled. (GlitchCleaner.py:256 does
this correctly, calling the wrapper.)

1.6 ***THE GATE FIRES ON ALMOST ALL ORDINARY TEXT*** (bears directly on the "lossless" claim)

The gate tests set membership of raw ids, and the authors' own glitch lists contain extremely common
tokens. [MEASURED] for Mistral-7B-Instruct-v0.1 with the shipped Glitchtokens CSV (|G| = 2,539 = 7.93%
of the 32,000-token vocab):

- Ordinary English function words that are IN the authors' Mistral glitch list:
      id 304 'and', id 354 'for', id 395 'with', id 486 'by'   (also id 327 '=', id 464 "'", id 340 'de',
      id 432 'com', id 443 '}', id 648 '+', id 414 '\\')
- The repair-evaluation boilerplate ITSELF is made partly of glitch tokens. Tokenizing
  text1+text2[1:] (GlitchCleaner.py:269-270,:276) gives
      [1, 22478, 28747, 2418, 368, 13750, 272, 1423, 464, 464, 304, 604, 852, 298, 528, 28804, 13,
       2820, 16981, 28747, 4003, 349, 272, 9332, 1423, 28747, 13]
  of which 464 ("'") x2 and 304 (" and") are glitch tokens. => config_flag == 1 for EVERY repair-eval
  prompt on this model, irrespective of the target token. The gate is vacuous during the headline
  repair-rate measurement.
- On real benchmark text:
      GSM8K test, question only (zero-shot): gate fires on 1217/1319 = 92.27% of examples
      GSM8K, 5-shot style context (6 blocks): gate fires on 500/500 = 100.0%
      MMLU test, question+choices (zero-shot): gate fires on 12605/14042 = 89.77% of examples
      mean glitch-token density inside a GSM8K question: 4.75% of tokens
  So paper Table 4 ("GlitchCleaner does not affect model performance") was, for Mistral, measured with the
  adapter ACTIVE on ~90-100% of evaluation items. The preserved MMLU/GSM8K scores are evidence that the
  learned delta is small, NOT evidence that the gate isolates clean inputs. The paper's causal story
  ("This gated design improves the reliability ... while preserving the lossless nature of the repair
  process", p.1-2) is not what produced the numbers.
  Note: glitch-list density is even higher for Qwen-7B-Chat (27,686/151,936 = 18.2%) and
  gemma-2b-it (29,831/256,000 = 11.7%), so leakage there will be at least as severe.

1.7 Latent shape constraint
config_flag is (B,1,1). `generate` with num_beams>1 or num_return_sequences>1 expands the batch dim
internally, after the flag is pinned, so the broadcast against (B*k, S, d) will raise. Safe only for the
greedy num_beams=1 setting the code actually uses (max_new_tokens=10, do_sample=False).

================================================================
2. LoRA CONSTRUCTION
================================================================

GlitchCleaner.py:6-19 == fine-tuning.py:266-280 (byte-equivalent maths).

- rank r = 4. fine-tuning.py:29 `lora_r = 4`; GlitchCleaner.py:171 default `config.get('lora_r', 4)`.
- alpha = 4. fine-tuning.py:30 `lora_alpha = 1 * lora_r`; GlitchCleaner.py:172 default `lora_alpha = lora_r`.
- scaling = alpha/rank = 1.0 exactly (GlitchCleaner.py:18, fine-tuning.py:280). The scaling factor is
  therefore a no-op in every shipped configuration; alpha is a free parameter that does nothing at the
  released setting. [MEASURED] all six checkpoints record lora_r=4, lora_alpha=4.
- Initialisation (GlitchCleaner.py:9-11, fine-tuning.py:269-272):
      std_dev = 1 / sqrt(rank)                       -> 0.5 for r=4
      A = randn(in_dim, rank) * std_dev              -> A ~ N(0, 0.25), shape (in_dim, r)
      B = zeros(rank, out_dim)                       -> standard "delta starts at 0"
  A's init scale is INDEPENDENT OF in_dim. This is not the reference LoRA init (kaiming_uniform_(A,
  a=sqrt(5)), std ~ 1/sqrt(3*in_dim) ~ 0.009 for in_dim=4096). Here std = 0.5, ~57x larger. B=0 is standard.
- Forward: `alpha/rank * (x @ A @ B)` (GlitchCleaner.py:18) — right-multiply convention, so A is (in,r)
  and B is (r,out), matching the paper's W' = W + (alpha/r)AB with A in R^{dxr}, B in R^{rxd}.
- [MEASURED, and this is a substantive finding] In all six released checkpoints the elementwise std of A
  is still exactly its init value, and its mean is ~0:
      Deepseek 0.5013 (mean -0.00077) | Llama-2 0.4995 (-0.00160) | Mistral 0.5000 (-0.00096)
      Qwen     0.5000 (+0.00030)      | Yi      0.4997 (+0.00168) | gemma   0.4993 (+0.00101)
  (target 1/sqrt(4) = 0.5000). A is statistically indistinguishable from its random initialisation.
  In effect only B was trained: the learned map is ΔW = A_random @ B_learned, i.e. a fixed random
  projection to r=4 followed by a learned readout. Trained ||(alpha/r)·A@B||_F per wrapped module:
      Mistral 6.38 | gemma 6.88 | Yi 10.27 | Llama-2 12.72 | Deepseek 20.78 | Qwen 40.71
  Per-rank row norms of B are near-uniform across the 4 ranks (e.g. Llama-2 gate_proj_19:
  [0.180, 0.233, 0.210, 0.189]) — no rank collapse.
- Which modules are wrapped:
      Llama-family path (GlitchCleaner.py:205-208, fine-tuning.py:394-397):
          layer.mlp.gate_proj  and  layer.mlp.up_proj
      Qwen path (GlitchCleaner.py:179-183):
          layer.mlp.w1  and  layer.mlp.w2
      layer.mlp.down_proj / mlp.c_proj is NOT wrapped. Attention is NOT wrapped.
  This matches the paper's "W1 and W2" / "MLP gate and MLP data" (paper p.4). `lora_head = False`
  (fine-tuning.py:32) is declared and never read.
- Dispatch is by duck-typing, GlitchCleaner.py:47-48:
      is_qwen  = hasattr(model,'transformer') and hasattr(model.transformer,'h')
      is_llama = hasattr(model,'model')       and hasattr(model.model,'layers')
  gemma/Mistral/Yi/Deepseek all take the is_llama path. Note GlitchCleaner.py:179 selects the wrapping
  branch by string (`"qwen" in model_path.lower()`) while :47-48 select the PATCHING branch by attribute —
  two independent dispatch mechanisms that must agree.
- Adapter size (exact, 10 layers x 2 modules x (in*4 + 4*out)):
      Llama-2-7b-chat            1,208,320 = 0.0179%
      Mistral-7B-Instruct-v0.1   1,474,560 = 0.0204%
      Qwen-7B-Chat               1,208,320 = 0.0156%
      Yi-6B-Chat                 1,208,320 = 0.0199%
      Deepseek-llm-7b-chat       1,208,320 = 0.0175%
      gemma-2b-it                1,474,560 = 0.0588%
  The paper's "<0.1% of parameters" claim (p.1, p.5) HOLDS for all six.

================================================================
3. TRAINING HYPERPARAMETERS (fine-tuning.py) — complete
================================================================

  lora_r                    4                       :29
  lora_alpha                1 * lora_r = 4          :30
  lora_mlp                  True                    :31
  lora_head                 False (never read)      :32
  target_layers             range(19, 29)  HARDCODED, model-independent   :35
  batch_size                64                      :38
  gradient_accumulation     8                       :39   (effective batch 512, recorded at :608)
  num_epochs                15                      :40
  learning_rate             1e-4                    :41
  weight_decay              0.01                    :42
  warmup_ratio              0.1                     :43
  optimizer                 AdamW(trainable, lr, weight_decay)  :414  (default betas 0.9/0.999, eps 1e-8)
  scheduler                 get_scheduler("linear", ...)        :420-425
  num_training_steps        num_epochs * ceil(len(dl)/accum)    :416-417
  num_warmup_steps          int(0.1 * num_training_steps)       :418
  gradient clipping         clip_grad_norm_(trainable_params, 1.0)  :458  (adapter params only)
  trainable selection       params whose name contains "lora"   :402-407 (base frozen at :107, :117-118)
  dtype                     see below                           :112
  max sequence length       384                                 :144 (default, never overridden at :252-257)
  shuffle                   True, NO seed set                   :262
  loss masking              see below                           :169-173
  eval seed                 FIXED_SEED = 42, passed ONLY to lm_eval  :56, :630-635
  lm_eval tasks             ["mmlu","gsm8k"], eval_batch_size 8  :57-58

Dtype: `initialize_model(model_path, device="auto", quant_type="float32")` at :112. "float32" is NOT one of
the handled strings — :89 tests "bfloat16", :95 tests "float16" — so it falls to the else branch :101-105,
which calls `from_pretrained(model_path, device_map=device_map)` with NO torch_dtype. The bf16/fp16 code
paths in the released script are therefore dead. The adapter dtype is inherited: `dtype=model.dtype` (:391).
Consistent with the checkpoints: the five non-Qwen .pt files are float32; Qwen-7B-Chat.pt is bfloat16
(Qwen's remote code forces its own dtype), so the Qwen run used a different numeric regime than the others.
GlitchCleaner.py:151-154 likewise passes no torch_dtype at load time.

Loss masking — fine-tuning.py:155-173:
    instruction = tokenizer(prompt,  add_special_tokens=False)      :155
    response    = tokenizer(answer,  add_special_tokens=False)      :156
    input_ids   = instruction.input_ids + response.input_ids + [eos]     :158-162
    labels      = [-100]*len(instruction.input_ids) + response.input_ids + [eos]  :169-173
Loss is on the answer tokens plus the appended EOS only. Padding labels are -100 (:228-239). NOTE
`add_special_tokens=False` on the instruction (:155) means NO BOS is prepended to training sequences —
whereas the evaluation prompt DOES carry BOS (GlitchCleaner.py:271 `tokenizer.encode(text1)` keeps id 1).
Train and eval sequences therefore differ at position 0 as well.

Other loop details:
- :449 loss divided by accum; :450 undone for logging; :454-457 step on accumulation boundary OR last batch.
- :224 pads attention_mask with `torch.zeros(pad_len)` (float32) and cats it onto an int64 mask;
  [MEASURED] torch.cat type-promotes, so the attention mask reaching the model is float32. Labels stay
  int64 (`torch.full((pad_len,), -100)`). Works, but is unintended.
- No train/validation split anywhere. :583 iterates `for token_id in glitchtokens` — the SAME list used to
  build the training set at :127. The repair rate written to CSV at :615 is training-set accuracy.
- :588 `idx = glitchtokens.index(token_id)` is an O(n) first-index lookup used only for progress printing;
  wrong under duplicate ids.
- :569 `pad_token_id=tokenizer.eos_token_ids` — plural, a typo. [MEASURED] this only resolves under
  transformers 5.x, where PreTrainedTokenizerBase.__getattr__ strips a trailing "_ids"/"_id" suffix
  (returns 2 for Mistral). Under transformers 4.x — the version contemporary with the paper and with
  lm_eval 0.4.x — there is no such __getattr__ and this raises AttributeError, so the released training
  script's own evaluation loop would not run as shipped for the authors' likely environment.

================================================================
4. HOW TRAINING EXAMPLES ARE CONSTRUCTED — fine-tuning.py:127-136
================================================================

    for glitchtoken in glitchtokens:
        token_str = tokenizer.decode([glitchtoken])                                    # :128
        prompt = (f"Question: Can you repeat the string '{token_str}' and return back to me?\n"
                  f"Answer: Here is the repeated string:\n")                           # :129-132
        answer = f"'{token_str}'"                                                      # :135

Exact strings:
  prompt = "Question: Can you repeat the string '" + token_str + "' and return back to me?\nAnswer: Here is the repeated string:\n"
  target = "'" + token_str + "'"

- Token rendering: DECODED TEXT, spliced into a Python f-string, then the whole string is re-tokenized
  (:155). This is NOT raw id splicing.
- Quoting: single ASCII apostrophes on both sides, in both prompt and target.
- Whitespace: token_str is `tokenizer.decode([glitchtoken])` and is NOT lstripped at :128. For a
  SentencePiece token carrying a leading word-boundary marker, decode() as the sole token drops the
  leading space, so the leading-space information is silently lost before the string re-enters the prompt.
- Target/answer string: the token re-quoted, `'{token_str}'`, followed by EOS (:162, :173).
- No chat template is applied for ANY model during training.
- max_length 384, truncation from the right (:176-179) — never triggers for these short prompts.

*** 4.1 DECODE->RE-ENCODE DESTROYS THE TARGET TOKEN IN ~40% OF TRAINING EXAMPLES *** [MEASURED]

Because the token is re-tokenized inside the context `...string '<tok>' and...`, the original glitch token
id often does not reappear. For Mistral-7B-Instruct-v0.1 over the authors' own 2,539-token list:

    TRAIN prompts in which the original glitch id survives decode->re-encode:  1516/2539 = 59.71%
    => 40.29% of training examples do not contain, as a token id, the token they are supposed to repair.
    TRAIN answer/label sequences containing the original glitch id:            1515/2539 = 59.67%

Concrete failures (id, decoded, first prompt ids):
    327 '='    -> [22478,28747,2418,368,13750,272,1423,464,2731,...]   ('=' merged into 2731)
    340 'de'   -> [...,464,450,28742,...]                              (re-tokenized as 450)
    354 'for'  -> [...,464,1392,28742,...]
    395 'with' -> [...,464,3415,28742,...]
    443 '}'    -> [...,1423,464,11339,...]                             ("'}" merged into one token 11339)
    464 "'"    -> [...,1423,23713,...]                                 ("''" merged into 23713)

This is a genuine train/test mismatch, because the EVALUATION path uses raw id splicing (Section 5) and
therefore always contains the true id. The model is trained on one token sequence and tested on another
for ~40% of the corpus. It also means the adapter cannot be learning a token-id-conditioned repair for
those 40% — it can only be learning a surface-string behaviour.

*** 4.2 λ DURING TRAINING IS 1 ONLY BY ACCIDENT ***
The paper's Algorithm 1 line 10 says "Set λ←1" for the fine-tuning stage. The code does not: :358 calls
`create_config_flag(input_ids)` on the training batch like any other input. [MEASURED] for Mistral it
happens to be 1 for 2539/2539 = 100.00% of training examples — but only because the boilerplate contains
glitch ids 464 ("'") and 304 (" and") (Section 1.6). Had it not, the 40.29% of examples whose target id is
destroyed would have produced config_flag = 0, hence `0 * lora(x)`, hence ZERO gradient to the adapter.
This is a latent correctness hazard that a reimplementation on a different glitch list will hit.

*** 4.3 PROMPT STRING DIFFERS FROM THE PAPER ***
Paper body (p.3): "Can you repeat the string '{token}' and return it back to me? Answer: Here is the
repeated string:"  — with "return IT back", no "Question:" prefix, no newline before "Answer:".
Code (both files): "Question: Can you repeat the string '{token}' and return back to me?\nAnswer: Here is
the repeated string:\n" — no "it". The paper is internally inconsistent: its own Table 1 examples (p.2)
use "and return back to me", matching the code, while the body text does not. A reimplementation following
the paper prose verbatim will not reproduce the code.

================================================================
5. GLITCH-TOKEN IDENTIFICATION / REPAIR-CHECK — GlitchCleaner.py:236-296
                                                (mirrored at fine-tuning.py:508-576)
================================================================

Two mutually inconsistent protocols, selected by `if 'Yi' in model_path` (GlitchCleaner.py:245):

5.1 NON-Yi PATH (Llama-2, Mistral, Qwen, gemma, Deepseek) — GlitchCleaner.py:268-283
    text1 = "Question: Can you repeat the string '"                                        :269
    text2 = "' and return back to me?\nAnswer: Here is the repeated string:\n"             :270
    tokens1 = tokenizer.encode(text1);  tokens2 = tokenizer.encode(text2)                  :271-272
    if 'Qw' in model_path:  tokens = cat(tokens1, [token_id], tokens2)                     :274
    else:                   tokens = cat(tokens1, [token_id], tokens2[1:])                 :276
  -> RAW ID SPLICE. `tokens2[1:]` drops the BOS that SentencePiece prepends; the Qwen branch keeps the full
     tokens2 because its tokenizer adds no BOS (comment at :273).
    max_new_tokens = 10, do_sample = False, temperature = None, top_p = None               :280
    all_response = tokenizer.decode(response_tokens, skip_special_tokens=True)             :281
    response     = all_response[k:]                                                        :282
      where k = len(f"Question: Can you repeat the string '{token}' and return back to me?\n"
                    f"Answer: Here is the repeated string:\n")                             :277-278
    return string_to_repeat in response,  string_to_repeat = decode([token_id]).lstrip()   :243, :283

  Output extraction: the FULL sequence (prompt + generation) is decoded to text, then sliced by a CHARACTER
  offset k computed from a separately reconstructed f-string. This is not the length of the actual decoded
  prompt.
  [MEASURED] for Mistral over all 2,539 glitch tokens, len(decoded_prompt) - k is NEVER 0:
        delta = +1 for 1690 tokens, delta = +2 for 849 tokens
  Example (id 304, 'and'):
        decoded(spliced) = "Question: Can you repeat the string ' and ' and return back to me?\n..."
        f-string used    = "Question: Can you repeat the string 'and' and return back to me?\n..."
        len 104 vs k 102, delta 2
  Because delta > 0, `response` begins with the last 1-2 characters of the prompt ("\n" or ":\n") rather
  than dropping generated text. For Mistral this is benign — [MEASURED] no glitch token's lstripped target
  is "\n", ":" or ":\n". But the invariant is unguarded: any tokenizer for which the decoded prompt is
  SHORTER than the f-string would silently truncate the model's answer and under-count repairs.

  Correctness test: bare SUBSTRING CONTAINMENT, `string_to_repeat in response`, over a 10-new-token window.
  No exact match, no normalisation, no case folding, no anchoring.
  [MEASURED] how permissive this is, for Mistral: the searched substring is
        1 char for 663 tokens, 2 chars for 414 tokens  ->  1077/2539 = 42.42% of targets are <= 2 chars.
  A 1-character target such as "e" or "a" is matched by almost any English continuation. 42% of the reported
  repair rate for this model rests on a test that a broken model can pass by accident. (No target is the
  empty string, so the test is never unconditionally true.) This same predicate is what produced the
  glitch-token lists in the first place, which is consistent with 'and', 'for', 'with', 'by' appearing in
  the Mistral list.

5.2 Yi PATH — GlitchCleaner.py:245-267 — a different protocol entirely
    content = f"Can you repeat the CHARACTER '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"   :246
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, ...)  :248-253
    output_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False, ...)       :256-262
    response = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)        :264-265
    return token_without_space in response                                                :267
  Differences from 5.1: "character" not "string"; no "Question:" prefix; token inserted as DECODED TEXT not
  a raw id; a CHAT TEMPLATE is applied; extraction is a correct TOKEN-space slice (prompt_len) rather than
  the broken character offset. So Yi's 92.68% in Table 2 was produced under a materially different and
  strictly better-implemented protocol than the other four models', and the paper's averaged 86.88% mixes
  the two. This is not disclosed in the paper.

  Net result: three different prompt-construction schemes coexist —
     training (all models):     decoded text,  no chat template, no BOS
     eval non-Yi:               raw id splice, no chat template, BOS present
     eval Yi:                   decoded text,  chat template applied

5.3 `count_passed_glitchtokens` never reports its result — GlitchCleaner.py:285-296
    The function accumulates `passed_tokens` and `passed_token_ids`, prints a running percentage every 100
    tokens (:294-296), and then the FILE ENDS at line 296. There is no final print of the repair rate and
    no return statement. Tutorials.ipynb cell 3 calls `count_passed_glitchtokens(loaded_model, model_name)`
    and gets None. The advertised headline metric cannot be read off the released tutorial without editing
    it; the last progress line printed is at the largest multiple of 100 <= |G|.

================================================================
6. THE TOKEN FILTER — Fine-tuning/tokenfilter.py
================================================================

Mechanism (a re-implementation of the Magikarp / Land & Bartolo round-trip test):
    START_PREFIX = "«"                                                          :7
    start_prefix_id = tokenizer.encode("«", add_special_tokens=False)[0]        :5
    embedding_size  = model.get_input_embeddings().weight.size(0)               :6

    encode(s):  prepend "«", encode(add_special_tokens=False);                  :13-20
                if tokens[0] != start_prefix_id: record in self.prefix_error and RETURN [0,1]  :16-18
                else strip the prefix token and return the rest                 :19-20
    decode(t):  prepend start_prefix_id, decode(skip_special_tokens=False);     :22-24
                if decoded[0] == " ": drop it  (comment: "e.g. mistral, but not llama2")  :25-26
                assert decoded.startswith("«"); return decoded[1:]              :27-30
  The "«" prefix exists to neutralise the leading-space/word-boundary behaviour of SentencePiece so the
  round trip is measured in a non-initial position.

Categories and the EXACT test for each — token_classify, :32-49:
  UNDECODEABLE (a) — `try: s = self.decode([token_id]) except: return "UNDECODEABLE"`   :33-36
                     A bare `except:` catching any exception, including the assert at :27.
  Then `id = self.encode(s)` :38 and branch on `id == [token_id]` :39:
  If the round trip reproduces the id exactly:
     SPECIAL       — `len(s) >= 3 and s[0] in "[<" and s[-1] in "]>" and any(c.isalpha() for c in s)` :40-41
                     Purely lexical on the decoded string; the tokenizer's own special-token registry
                     (all_special_ids / added_tokens_decoder) is never consulted. Mis-classifies ordinary
                     text like "<div>" or "[abc]" as SPECIAL, and misses special tokens not bracketed that
                     way. The paper (p.3) names '<s>', '<unk>', '<endoftext>' as the intent.
     UNDECODEABLE (b) — `"\ufffd" in s`  i.e. the U+FFFD replacement character           :42-43
     NORMAL        — everything else                                                     :44-45
  Else (round trip yields a different id or length):
     UNREACHABLE   — :46-47.  Note the [0,1] sentinel returned by encode() on prefix failure (:18) can
                     never equal [token_id], so prefix-encoding failures are silently folded into
                     UNREACHABLE rather than surfaced.

  filter_token() :51-57 — iterates `range(self.embedding_size)` and returns every id whose category != NORMAL.
  Iterating embedding ROWS, not tokenizer vocab: for gemma-2b-it the embedding has 256,128 rows vs
  vocab_size 256,000, so 128 padding rows are classified too.
  get_ith_token() :59-63 — decode, then replace ' ' with '▁' (display helper, unused elsewhere).

*** 6.1 THE FILTER IS NEVER APPLIED IN THE RELEASED PIPELINE ***
`import tokenfilter` at fine-tuning.py:24 is the ONLY reference. `TokenFilter` is never instantiated, and
`filter_token()` is never called, anywhere in the repository. fine-tuning.py:121-122 reads the glitch-token
CSV directly and uses it unfiltered. The filter is dead code with respect to reproducing the paper. The
paper's stage-1 numbers (p.3: Llama-2, 229 of 32,000 removed = 3 SPECIAL + 2 UNDECODEABLE + 224
UNREACHABLE, leaving 31,771) cannot be regenerated by running anything in this repo.

================================================================
7. LAYERS AND CONFIG PER MODEL — from the shipped LoRA-Parameter/*.pt
================================================================

Every checkpoint is a flat dict of 40 tensors plus one 'config' entry (41 keys). Key naming:
  Llama-family: "<gate_proj|up_proj>_<layer>_<A|B>"   parsed at GlitchCleaner.py:212-217 (needs >=4 parts)
  Qwen:         "<w1|w2>_<layer>_<A|B>"               parsed at GlitchCleaner.py:187-192 (needs >=3 parts)

  checkpoint                    lora_r  lora_alpha  target_layers        A shape      B shape        dtype
  Llama-2-7b-chat.pt              4        4        [19..28]             (4096,4)     (4,11008)      fp32
  Mistral-7B-Instruct-v0.1.pt     4        4        [19..28]             (4096,4)     (4,14336)      fp32
  Qwen-7B-Chat.pt                 4        4        [19..28]  (w1/w2)    (4096,4)     (4,11008)      BF16
  Yi-6B-Chat.pt                   4        4        [19..28]             (4096,4)     (4,11008)      fp32
  Deepseek-llm-7b-chat.pt         4        4        [19..28]             (4096,4)     (4,11008)      fp32
  gemma-2b-it.pt                  4        4        ***[6..15]***        (2048,4)     (4,16384)      fp32

Model depths (fetched from the HF configs) for context:
  Llama-2-7b-chat 32 layers | Mistral-7B-v0.1 32 | Qwen-7B-Chat 32 | Yi-6B-Chat 32 | deepseek-llm-7b 30
  gemma-2b-it 18 layers, hidden 2048, intermediate 16384 — consistent with the (2048,4)/(4,16384) shapes.
  (Qwen's config lists intermediate_size 22016; its w1/w2 are each 22016/2 = 11008 wide, matching.)

*** 7.1 gemma-2b-it uses layers 6-15, which the paper never states, and which its own trainer cannot produce ***
  - fine-tuning.py:35 hardcodes `target_layers = range(19, 29)` with no model dispatch. gemma-2b-it has 18
    layers (indices 0-17), so running the released script on gemma wraps NOTHING (the `layer_idx in
    target_layers` guard at :395 never fires), trains nothing, and cannot emit gemma-2b-it.pt. The shipped
    gemma checkpoint is not reproducible from the shipped trainer without an undocumented edit.
  - Layers 6-15 of 18 is the 33%-89% band. The last two layers (16, 17) are excluded. The paper
    (p.5) describes key layers as "primarily targeting the posterior layers of the model" and gives only
    the Llama-2 19-28 example. For gemma the actual choice is the middle of the network. The paper's stated
    rationale ("the difference ... becomes more apparent in the model's later layers", p.4) is not what was
    done for gemma-2b-it.
  - The Llama-2 choice of 19-28 does trace to GlitchProber, which states it "designated layers 19 to 28 as
    key layers, optimally positioned in the middle to lower sections of the model's architecture"
    (Glitchprober_text.txt:701) — so GlitchCleaner's citation is accurate for that one model, but the
    per-model scaling rule for the other five is undocumented in both papers.
  - GlitchCleaner.py:173 has a fallback `target_layers = config.get('target_layers', range(19,29))`; since
    all six checkpoints record target_layers, the fallback is never exercised by the tutorial.

================================================================
8. DISCREPANCIES BETWEEN CODE AND PAPER — consolidated
================================================================

D1. GATING IS NOT CONTROL FLOW. Paper: "a gating mechanism dynamically controls the ACTIVATION of these
    branches" (abstract); "For clean inputs, the model operates in its original form" (p.1); Fig. 2
    caption: "LoRA branches ... only when glitch tokens are detected". Code: GlitchCleaner.py:33 always
    evaluates `self.lora(x)` and multiplies by 0 or 1. [MEASURED] LoRALayer.forward is invoked the same
    number of times (4/4) with flag=0 and flag=1. The clean-input path costs the same as the glitch path.
    Impact: Table 6's 62.83 vs 66.30 tok/s is the always-on cost; the paper's "negligible impact on
    inference speed" is defensible as a number but not as the mechanism described.

D2. THE GATE LEAKS ON ~90-100% OF BENCHMARK INPUTS, so the "lossless" claim is not testing what it says.
    [MEASURED, Mistral] the eval boilerplate itself contains glitch ids 464/304 => flag==1 on 100% of
    repair-eval prompts; GSM8K 0-shot 92.27%, GSM8K 5-shot 100.0%, MMLU 0-shot 89.77%. Paper Table 4's
    preserved scores therefore measure "small learned delta", not "adapter disabled". The paper's stated
    mechanism for losslessness (p.1-2, p.5) is not the mechanism that produced Table 4.

D3. λ IS NOT SET TO 1 DURING TRAINING. Paper Algorithm 1 line 10: "Set λ←1". Code fine-tuning.py:358
    computes it from the batch like any other input. It evaluates to 1 for Mistral only incidentally
    (Section 4.2). On a glitch list not containing the boilerplate tokens, up to 40% of examples would
    receive zero gradient.

D4. TRAIN/EVAL TOKEN-RENDERING MISMATCH. Training splices the DECODED STRING and re-tokenizes
    (fine-tuning.py:128-132, :155); evaluation splices the RAW ID (GlitchCleaner.py:271-276). [MEASURED]
    the original id survives re-encoding in only 59.71% of Mistral training prompts and 59.67% of the
    label sequences. The paper (p.4, "The dataset includes the same prompt templates described in Glitch
    Token Identification") asserts the two are identical. They are not, for ~40% of the corpus.

D5. GLITCH-TOKEN IDENTIFICATION IS A BARE SUBSTRING TEST, not "unexpected output" classification. Paper
    p.3: "if the model produces unexpected outputs, we classify the corresponding token as a glitch token."
    Code: `string_to_repeat in response` over 10 new tokens (GlitchCleaner.py:283). [MEASURED] 42.42% of
    Mistral targets are <=2 characters, so both detection and repair-rate measurement are dominated by a
    predicate that is easy to satisfy by chance. The resulting list contains 'and', 'for', 'with', 'by',
    '=', "'", 'de', 'com', '}', '+'.

D6. THE OUTPUT-EXTRACTION SLICE IS SYSTEMATICALLY MISALIGNED. GlitchCleaner.py:277-282 slices the decoded
    full sequence by len() of a reconstructed f-string. [MEASURED] the offset is wrong by +1 (1690 tokens)
    or +2 (849 tokens) for every one of Mistral's 2,539 glitch tokens; never correct. Benign here, but
    silently truncates answers on any tokenizer where the sign flips. Not mentioned in the paper.

D7. Yi USES A DIFFERENT, INCOMPATIBLE PROTOCOL (chat template + decoded-text insertion + correct token-space
    extraction; prompt says "character" not "string") — GlitchCleaner.py:245-267. The paper presents one
    uniform template (p.3) and averages Yi's 92.68% into the headline 86.88% (Table 2) without disclosure.

D8. REPORTED REPAIR RATES ARE TRAINING-SET ACCURACY. fine-tuning.py:583 evaluates on exactly the
    `glitchtokens` list used to build the training set at :127. No split, no held-out set anywhere in the
    repo. The paper (p.5, "We fine-tune the model with added LoRA branches using a dataset built from
    glitch tokens filtered by the model, then evaluate whether the repaired model could generate correct
    responses") does not disclose that these are the same tokens. Consistent with this repo's own prior
    finding (commit 36a3094: GC train-split 89.9% vs held-out 55-75%).

D9. THE TOKEN FILTER IS NEVER RUN (Section 6.1). Paper devotes a full subsection to it (p.3) with exact
    counts for Llama-2 (229 filtered: 3/2/224). Nothing in the repo invokes it.

D10. gemma-2b-it's ACTUAL LAYERS (6-15 of 18) ARE UNDOCUMENTED AND UNREPRODUCIBLE from the released trainer,
    which hardcodes range(19,29) (fine-tuning.py:35). They also contradict the paper's "posterior layers"
    rationale (p.4-5). See Section 7.1.

D11. `alpha` IS INERT. alpha/rank = 4/4 = 1.0 in every shipped config. The paper presents alpha as a tuned
    scaling factor ("a scaling factor α is applied to the update", p.3; "We set the LoRA rank parameter r
    to 4 and the scaling factor α to 4", p.5) and Appendix F is said to ablate r and α — but at α=r the
    factor does nothing, and [MEASURED] all six checkpoints are α=r=4.

D12. MATRIX A IS EFFECTIVELY UNTRAINED. [MEASURED] elementwise std of A is 0.4993-0.5013 across all six
    checkpoints vs the init value 1/sqrt(r) = 0.5000, mean ~0. The released adapters are
    ΔW = A_random @ B_learned. The paper describes A and B symmetrically as "the learnable low-rank
    matrices" (p.3). Also relevant: A's init scale 1/sqrt(rank) is independent of in_dim and ~57x the
    standard LoRA kaiming init at in_dim=4096 (GlitchCleaner.py:9-10) — a deviation from Hu et al. that
    the paper does not mention.

D13. PROMPT STRING MISMATCH. Paper body p.3 uses "and return IT back to me"; code uses "and return back to
    me" and adds a "Question: " prefix and a newline before "Answer:". The paper's own Table 1 matches the
    code, so the paper is internally inconsistent. (Section 4.3)

D14. A SIXTH MODEL IS SHIPPED BUT UNREPORTED. Glitchtokens/Deepseek-llm-7b-chat-glitch-tokens.csv (8,533
    tokens) and LoRA-Parameter/Deepseek-llm-7b-chat.pt exist and are wired into GlitchCleaner.py:146-147
    and Tutorials.ipynb cell 2, but Deepseek-llm-7b-chat appears nowhere in Table 2. The published average
    is exactly (88.76+69.38+94.80+88.79+92.68)/5 = 86.882%, confirming Deepseek is excluded from the
    headline number. No reason is given.

D15. TWO CLASSES NAMED LinearWithLoRA WITH INVERTED None-SEMANTICS (GlitchCleaner.py:29-33 skips the LoRA;
    fine-tuning.py:291-295 applies it). Combined with fine-tuning.py:526 calling the RAW model for the Yi
    branch, the training script evaluates Yi with the adapter unconditionally on. Not mentioned anywhere.

D16. Glitch-token counts DO match the paper. CSV row counts (minus header): Llama-2 4743, gemma-2b-it 29831,
    Mistral 2539, Qwen 27686, Yi 5985 — all identical to Table 2. Deepseek 8533 (unreported). The
    "<0.1% parameters" claim also holds (Section 2, 0.0156%-0.0588%). These two claims reproduce.

================================================================
9. OPERATIONAL BLOCKERS FOR RUNNING THE RELEASED CODE AS-IS
================================================================
- GlitchCleaner.py:127 hardcodes `hf_model_path = "/root/autodl-tmp/Qwen-7B-Chat"` (the authors' machine),
  so Tutorials.ipynb's Qwen option fails on any other host. The other five models resolve to real HF ids
  (:139-147).
- GlitchCleaner.py:157-158 build relative paths ("LoRA-Parameter/...", "Glitchtokens/..."), so the notebook
  must be launched from the repo root.
- GlitchCleaner.py:151-154 loads with no torch_dtype, i.e. fp32 — ~28 GB for a 7B model.
- fine-tuning.py:46 `os.environ["HF_TOKEN"] = "your_token"` is a placeholder that will actively override a
  valid ambient token.
- fine-tuning.py:49-53 point at /root/autodl-fs/... paths.
- fine-tuning.py:569 `tokenizer.eos_token_ids` (Section 3) raises AttributeError on transformers 4.x.
- fine-tuning.py:35 makes the trainer silently a no-op for any model with <29 layers (gemma-2b-it).
- `count_passed_glitchtokens` returns None and never prints the final rate (Section 5.3).
- fine-tuning.py:262 shuffles with no seed; only lm_eval is seeded (:56, :630-635). Training runs are not
  reproducible.

================================================================
10. SCRATCH ARTEFACTS (reproduce the [MEASURED] numbers)
================================================================
C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/inspect_lora.py           — checkpoint config/keys/shapes/norms
C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/analyze_weights.py        — A-init-preservation, ||dW||_F, per-rank B norms
C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/test_prompt_mismatch.py   — train vs eval tokenization, slice offset, target-length distribution
C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/test_gate.py              — boilerplate glitch ids, gate leakage, eos_token_ids, dtype promotion
C:/Users/user3/AppData/Local/Temp/claude/E--user3-FYS-SWK-Benchmark/1f1c701f-057c-4b66-8b46-7684d6c60f54/scratchpad/test_lora_computed.py     — imports the authors' classes, counts LoRALayer.forward under flag=0/1, per-position delta, list-vs-set gate cost

Two caveats on scope. All [MEASURED] tokenizer-level numbers are for Mistral-7B-Instruct-v0.1, the only one
of the six models cached locally and the one fine-tuning.py is configured for; the same measurements should
be repeated per model before the per-model figures are quoted in the thesis, though the structural findings
(D1, D3, D4, D5, D6, D9, D10, D11, D12) are properties of the source and hold model-independently. The
gate-leakage rates in D2 are computed over benchmark text tokenized with the authors' glitch list and are
exact for that setup; they are not a claim about what the authors' own H200 runs measured.