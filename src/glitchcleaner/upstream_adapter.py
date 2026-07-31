"""Load GlitchCleaner's OWN released adapter weights and run them under our harness.

WHY THIS EXISTS
---------------
Everywhere else in this study, RQ4 compares our retrained adapter against the
paper's published rate. That leaves one alternative explanation permanently open:
maybe the gap is our training, not their claim. The authors shipped their trained
weights (`third_party/GlitchCleaner/LoRA-Parameter/<model>.pt`), so the
explanation can simply be removed. This module loads those weights.

WHAT IS REPRODUCED, EXACTLY
---------------------------
Their `LinearWithLoRA` (GlitchCleaner.py:20-32) computes

    y = linear(x) + config_flag * (alpha / rank) * ((x @ A) @ B)

with `A: [in_features, rank]`, `B: [rank, out_features]` -- note this is the
transpose of PEFT's convention, which is why the weights are not loaded through
PEFT. `config_flag` is the per-example lambda gate, 1 when the input contains any
id from the known glitch set G (GlitchCleaner.py:61-67). Both are reproduced here
rather than approximated, so any disagreement with their published number cannot
be attributed to our adapter arithmetic.

Their checkpoint stores `{gate_proj,up_proj}_<layer>_{A,B}` plus a `config` entry
carrying `lora_r`, `lora_alpha` and `target_layers`. We read the configuration
from the checkpoint rather than from our own YAML, so the run is theirs end to end.

third_party/ is a reference copy and is never modified; this file re-implements
their arithmetic against their stored tensors.
"""
import torch

from ..common.model_utils import decoder_layers, mlp_module


class UpstreamLoRALinear(torch.nn.Module):
    """Wraps a base nn.Linear with GlitchCleaner's LoRA branch and its lambda gate.

    The flag is read from a shared holder rather than passed as an argument,
    because it must vary per example while `generate()` owns the forward calls.
    Like upstream, the branch is COMPUTED even when the flag is 0 -- the gate
    removes the behavioural effect, not the arithmetic cost.
    """

    def __init__(self, base, A, B, alpha, rank, flag_holder):
        super().__init__()
        self.base = base
        self.register_buffer("A", A.to(base.weight.dtype))
        self.register_buffer("B", B.to(base.weight.dtype))
        self.scaling = alpha / rank
        self.flag = flag_holder

    def forward(self, x):
        out = self.base(x)
        delta = self.scaling * ((x @ self.A) @ self.B)
        f = self.flag.value
        if f is None:                      # ungated: adapter forced on
            return out + delta
        if f.shape[0] != x.shape[0]:       # batch mismatch: leave the base model alone
            return out
        return out + f.to(delta.dtype) * delta


class FlagHolder:
    """Mutable cell shared by every wrapped Linear, so one write gates all of them."""

    def __init__(self):
        self.value = None                  # None => forced on; else [batch, 1, 1]


class UpstreamGlitchCleaner:
    """Facade with .generate()/.parameters() so our sweep helpers accept it."""

    def __init__(self, model, glitch_ids, flag: FlagHolder, meta: dict):
        self.model = model
        self.config = model.config
        self.flag = flag
        self.meta = meta
        self.glitch = torch.tensor(sorted({int(t) for t in glitch_ids}), dtype=torch.long)
        self.n_on = self.n_off = 0

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
        return self

    def set_gated(self, on: bool):
        """on=True  -> lambda computed per example (the method as specified)
           on=False -> lambda ignored, adapter always active (the ablation)."""
        self._gated = on
        if not on:
            self.flag.value = None
        return self

    @torch.no_grad()
    def generate(self, input_ids=None, **kw):
        if getattr(self, "_gated", True):
            g = self.glitch.to(input_ids.device)
            contains = torch.isin(input_ids, g).any(dim=1)
            self.n_on += int(contains.sum())
            self.n_off += int((~contains).sum())
            self.flag.value = contains.view(-1, 1, 1)
        try:
            return self.model.generate(input_ids=input_ids, **kw)
        finally:
            if getattr(self, "_gated", True):
                self.flag.value = None

    def gate_stats(self):
        return {"examples_gate_on": self.n_on, "examples_gate_off": self.n_off}


def load_upstream_adapter(model, mcfg: dict, ckpt_path, glitch_ids):
    """Attach the authors' released LoRA weights to `model` in place.

    Returns (facade, meta). `meta` records what the checkpoint declared, so the
    run's provenance is the authors' configuration rather than ours.
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    rank = int(cfg.get("lora_r", 4))
    alpha = float(cfg.get("lora_alpha", rank))
    layers_declared = sorted(int(x) for x in cfg.get("target_layers", range(19, 29)))

    flag = FlagHolder()
    layers = decoder_layers(model, mcfg)
    # upstream key prefix -> our architecture-agnostic module_map key
    wanted = {"gate_proj": "gate", "up_proj": "up"}
    attached, seen_layers = [], set()

    for li in layers_declared:
        for up_name, map_key in wanted.items():
            A, B = ck.get(f"{up_name}_{li}_A"), ck.get(f"{up_name}_{li}_B")
            if A is None or B is None:
                continue
            base = mlp_module(layers[li], mcfg, map_key)
            if A.shape != (base.in_features, rank) or B.shape != (rank, base.out_features):
                raise RuntimeError(
                    f"{up_name}_{li}: checkpoint shapes {tuple(A.shape)}/{tuple(B.shape)} "
                    f"do not match this model's {map_key}_proj "
                    f"({base.in_features} -> {base.out_features}, rank {rank}). "
                    "Wrong model for this adapter?")
            wrapped = UpstreamLoRALinear(base, A, B, alpha, rank, flag).to(base.weight.device)
            parent_path = mcfg["module_map"][map_key]
            parent, leaf = parent_path.rsplit(".", 1)
            obj = layers[li]
            for part in parent.split("."):
                obj = getattr(obj, part)
            setattr(obj, leaf, wrapped)
            attached.append(f"L{li}.{map_key}")
            seen_layers.add(li)

    if not attached:
        raise RuntimeError(f"no LoRA tensors matched this model in {ckpt_path}")

    meta = {
        "checkpoint": str(ckpt_path),
        "declared_config": {"lora_r": rank, "lora_alpha": alpha,
                            "target_layers": layers_declared},
        "n_modules_attached": len(attached),
        "layers_attached": sorted(seen_layers),
        "n_tensors_in_checkpoint": len([k for k in ck if k != "config"]),
    }
    return UpstreamGlitchCleaner(model, glitch_ids, flag, meta), meta
