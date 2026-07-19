"""GlitchProber repair (paper Algorithm 2, Sec. 4.2).

Activation = the gated MLP product act(gate(x)) * up(x), captured at the input of
down_proj. From a sample of NORMAL tokens per key layer:
  Neun_up   : neurons with activation > m in >99% of normal tokens
  Neun_down : neurons with activation <= m in ALL normal tokens
During generation on a glitch token, a pre-hook on down_proj applies:
  act[Neun_up]   += beta     (promote under-activated should-fire neurons)
  act[Neun_down] /= alpha    (suppress abnormally-firing should-be-silent neurons)

alpha/beta come from either the 'adaptive' linear formulas (Eq. 9-12; constants
undisclosed in the paper -> ours are config values) or fixed 'rule_based' values.
"""
import torch
from tqdm import tqdm

from ..common.model_utils import decoder_layers, mlp_module
from ..common.prompts import spliced_batch
from ..ground_truth.sweep import repetition_sweep


@torch.no_grad()
def _collect_mlp_acts(model, tok, token_ids, mcfg, batch_size, desc, task="repetition"):
    """Last-position down_proj INPUT per key layer -> {layer: tensor [n, d_mlp]} (cpu fp32)."""
    device = next(model.parameters()).device
    layers = decoder_layers(model, mcfg)
    key_layers = mcfg["key_layers"]
    store = {li: [] for li in key_layers}
    handles = []

    def make_hook(li):
        def hook(_m, args):
            store[li].append(args[0][:, -1, :].detach().float().cpu())
        return hook

    for li in key_layers:
        handles.append(
            mlp_module(layers[li], mcfg, "down").register_forward_pre_hook(make_hook(li))
        )
    try:
        for i in tqdm(range(0, len(token_ids), batch_size), desc=desc):
            chunk = token_ids[i : i + batch_size]
            input_ids, attention_mask = spliced_batch(tok, chunk, device, task)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {li: torch.cat(v, dim=0) for li, v in store.items()}


def compute_neuron_stats(model, tok, normal_sample, mcfg, m: float, up_quantile: float,
                         batch_size: int, task="repetition") -> dict:
    """Per key layer: Neun_up / Neun_down index tensors + mean normal activations."""
    acts = _collect_mlp_acts(model, tok, normal_sample, mcfg, batch_size, "normal stats", task)
    stats = {}
    for li, A in acts.items():  # A: [n_normal, d_mlp]
        active_frac = (A > m).float().mean(dim=0)
        neun_up = torch.nonzero(active_frac > up_quantile).squeeze(-1)
        neun_down = torch.nonzero((A <= m).all(dim=0)).squeeze(-1)
        stats[li] = {
            "neun_up": neun_up,
            "neun_down": neun_down,
            "mean_normal": A.mean(dim=0),
        }
    return stats


def compute_adjustments(model, tok, glitch_sample, mcfg, stats, adaptive: dict,
                        batch_size: int, task="repetition") -> dict:
    """Paper Eq. 9-12: per key layer, beta = k1*dAct_up + b1, alpha = k2*dAct_down + b2."""
    acts = _collect_mlp_acts(model, tok, glitch_sample, mcfg, batch_size, "glitch stats", task)
    adj = {}
    eps = 1e-6
    for li, A in acts.items():
        s = stats[li]
        mean_glitch = A.mean(dim=0)
        up, down = s["neun_up"], s["neun_down"]
        d_up = (s["mean_normal"][up] - mean_glitch[up]).mean().item() if len(up) else 0.0
        d_down = (
            (mean_glitch[down] / (s["mean_normal"][down].abs() + eps)).mean().item()
            if len(down) else 1.0
        )
        beta = adaptive["k1"] * d_up + adaptive["b1"]
        alpha = adaptive["k2"] * d_down + adaptive["b2"]
        adj[li] = {"beta": beta, "alpha": max(alpha, 1.0)}  # never amplify via division
    return adj


class RepairHooks:
    """Context manager installing the activation-adjustment pre-hooks on down_proj."""

    def __init__(self, model, mcfg, stats, alpha_beta: dict):
        """alpha_beta: {layer: {'alpha': float, 'beta': float}} or {'alpha':..,'beta':..} for all."""
        self.model, self.mcfg, self.stats = model, mcfg, stats
        self.ab = alpha_beta
        self.handles = []

    def _get(self, li, key):
        return (self.ab.get(li, self.ab) if isinstance(next(iter(self.ab.values()), None), dict)
                else self.ab)[key]

    def __enter__(self):
        layers = decoder_layers(self.model, self.mcfg)
        device = next(self.model.parameters()).device
        for li in self.mcfg["key_layers"]:
            s = self.stats[li]
            up = s["neun_up"].to(device)
            down = s["neun_down"].to(device)
            beta = float(self._get(li, "beta"))
            alpha = float(self._get(li, "alpha"))

            def hook(_m, args, _up=up, _down=down, _b=beta, _a=alpha):
                x = args[0].clone()
                if len(_up):
                    x[..., _up] += _b
                if len(_down) and _a != 0:
                    x[..., _down] /= _a
                return (x,)

            self.handles.append(
                mlp_module(layers[li], self.mcfg, "down").register_forward_pre_hook(hook)
            )
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def evaluate_repair(model, tok, glitch_ids, normal_ids, mcfg, stats, alpha_beta,
                    batch_size, max_new_tokens, task="repetition", correct_fn=None) -> dict:
    """Repair rate on glitch tokens + collateral breakage on normal tokens."""
    with RepairHooks(model, mcfg, stats, alpha_beta):
        g = repetition_sweep(model, tok, glitch_ids, batch_size, max_new_tokens,
                             task=task, desc="repair glitch", correct_fn=correct_fn)
        n = repetition_sweep(model, tok, normal_ids, batch_size, max_new_tokens,
                             task=task, desc="repair normal", correct_fn=correct_fn)
    repaired = sum(1 for ok, _ in g.values() if ok)
    broken = sum(1 for ok, _ in n.values() if not ok)
    return {
        "n_glitch": len(glitch_ids),
        "repaired": repaired,
        "repair_rate": repaired / len(glitch_ids) if glitch_ids else 0.0,
        "n_normal_checked": len(normal_ids),
        "normal_broken": broken,
        "normal_break_rate": broken / len(normal_ids) if normal_ids else 0.0,
    }
