"""GlitchCleaner's gating mechanism (paper: "W' = W + lambda * (alpha/r) A B").

The paper's defining property is that the LoRA branch is INPUT-DEPENDENT:
lambda = 1 when the input contains a glitch token, lambda = 0 otherwise, so a
clean prompt is processed by the mathematically unmodified base model. The
authors' released code implements this per example in the batch
(third_party/GlitchCleaner/GlitchCleaner.py: create_config_flag / forward).

CORRECTION HISTORY (documented for the thesis)
----------------------------------------------
v1 of this benchmark used a plain always-active PEFT adapter and toggled it
globally for one control run. Repair rates on glitch prompts are unaffected
(lambda would be 1 for those prompts anyway), but any claim about CLEAN-input
behaviour - "losslessness", capability preservation, inference speed - was
measuring an always-on adapter and therefore was not measuring GlitchCleaner.
This module supplies the missing gate.

IMPLEMENTATION NOTE
-------------------
The flag multiplies the output of every LoRA-B projection, which is exactly
equivalent to scaling the branch's contribution, and mirrors the authors'
`config_flag * self.lora(x)`. Like theirs, the branch is still COMPUTED when the
flag is 0 - the gate removes the behavioural effect, not the arithmetic cost.
That distinction matters for the inference-speed comparison and is reported.
"""
import torch


class GatedLoRA:
    """Context manager applying per-example lambda to a PeftModel's LoRA branches."""

    def __init__(self, peft_model, glitch_ids):
        self.model = peft_model
        self.glitch = torch.tensor(sorted(set(int(t) for t in glitch_ids)), dtype=torch.long)
        self._flag = None          # [batch, 1, 1] float tensor, or None => all ones
        self.handles = []
        self.n_gate_on = 0
        self.n_gate_off = 0

    # -- flag computation -------------------------------------------------
    def compute_flag(self, input_ids: torch.Tensor) -> torch.Tensor:
        """lambda_b = 1 iff example b contains at least one known glitch token."""
        g = self.glitch.to(input_ids.device)
        contains = torch.isin(input_ids, g).any(dim=1)          # [batch] bool
        self.n_gate_on += int(contains.sum())
        self.n_gate_off += int((~contains).sum())
        return contains.view(-1, 1, 1)

    def set_flag(self, input_ids: torch.Tensor):
        self._flag = self.compute_flag(input_ids)
        return self._flag

    def clear_flag(self):
        self._flag = None

    # -- hook plumbing ----------------------------------------------------
    def __enter__(self):
        def hook(_module, _inp, out):
            if self._flag is None:
                return out
            # During generation the batch is stable; guard anyway.
            if self._flag.shape[0] != out.shape[0]:
                return out
            return out * self._flag.to(out.dtype)

        for name, module in self.model.named_modules():
            if name.endswith("lora_B.default") or name.endswith("lora_B"):
                if hasattr(module, "weight"):
                    self.handles.append(module.register_forward_hook(hook))
        if not self.handles:
            raise RuntimeError("no lora_B modules found - is this a PeftModel?")
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []
        self._flag = None

    # -- convenience ------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Set lambda from the prompt, then generate. Mirrors upstream's generate()."""
        self.set_flag(input_ids)
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self.clear_flag()

    def stats(self) -> dict:
        return {"examples_gate_on": self.n_gate_on, "examples_gate_off": self.n_gate_off}


class GatedModelAdapter:
    """Minimal nn.Module-like facade so a gated model can be passed to code that
    only calls .generate(), .parameters() and .config (e.g. our sweep helpers)."""

    def __init__(self, peft_model, glitch_ids):
        self.gate = GatedLoRA(peft_model, glitch_ids)
        self.model = peft_model
        self.config = peft_model.config

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
        return self

    def generate(self, input_ids=None, **kwargs):
        # kwargs may carry attention_mask etc.; the gate only needs input_ids.
        self.gate.set_flag(input_ids)
        try:
            return self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self.gate.clear_flag()

    def __enter__(self):
        self.gate.__enter__()
        return self

    def __exit__(self, *a):
        self.gate.__exit__(*a)
