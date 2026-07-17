"""Feature extraction for GlitchProber detection (paper Sec. 4.1.1).

Per token, from each key layer, at the position of the token under test:
  - attention pattern : attention probabilities of the last prompt position, all heads
  - MLP gate          : sigma(gate_proj(x))   (output of the activation module)
  - MLP data          : up_proj(x)

Prompts are constant-length (spliced), so feature vectors have fixed size.
Stored as float16; cast to float32 for PCA.
"""
import numpy as np
import torch
from tqdm import tqdm

from ..common.model_utils import decoder_layers, mlp_module
from ..common.prompts import spliced_batch


class _Catcher:
    """Forward hooks that stash the last-position activation of chosen modules."""

    def __init__(self):
        self.store: dict[str, torch.Tensor] = {}
        self.handles = []

    def add(self, module, name: str):
        def hook(_m, _inp, out, _name=name):
            t = out[0] if isinstance(out, tuple) else out
            self.store[_name] = t[:, -1, :].detach().to(torch.float16).cpu()
        self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def extract_features(
    model,
    tok,
    token_ids: list[int],
    mcfg: dict,
    features: list[str],
    batch_size: int = 32,
    desc: str = "features",
) -> np.ndarray:
    """Return float16 array [n_tokens, feature_dim]."""
    device = next(model.parameters()).device
    layers = decoder_layers(model, mcfg)
    key_layers = mcfg["key_layers"]
    use_attn = "attn_pattern" in features

    catcher = _Catcher()
    for li in key_layers:
        layer = layers[li]
        if "mlp_gate" in features:
            try:
                catcher.add(mlp_module(layer, mcfg, "act"), f"gate{li}")
            except AttributeError:
                pass  # architecture without a hookable act module (see models.yaml notes)
        if "mlp_data" in features:
            try:
                catcher.add(mlp_module(layer, mcfg, "up"), f"data{li}")
            except AttributeError:
                pass

    rows = []
    try:
        for i in tqdm(range(0, len(token_ids), batch_size), desc=desc):
            chunk = token_ids[i : i + batch_size]
            input_ids, attention_mask = spliced_batch(tok, chunk, device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=use_attn,
                use_cache=False,
            )
            parts = []
            if use_attn:
                for li in key_layers:
                    # [batch, heads, q, k] -> last query row, all heads flattened
                    a = out.attentions[li][:, :, -1, :]
                    parts.append(a.reshape(a.shape[0], -1).to(torch.float16).cpu())
            for li in key_layers:
                for prefix in ("gate", "data"):
                    key = f"{prefix}{li}"
                    if key in catcher.store:
                        parts.append(catcher.store[key])
            rows.append(torch.cat(parts, dim=1).numpy())
            catcher.store.clear()
    finally:
        catcher.remove()
    return np.concatenate(rows, axis=0)
