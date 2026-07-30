"""GlitchProber repair (paper Algorithm 2, Sec. 4.2).

WHAT THE PAPER SPECIFIES
------------------------
The paper's "MLP status" is two separate signals (Sec. 2.1, Eq. 3-5):
    MLP gate  = sigma(Z1)      -- output of the activation function
    MLP data  = Z2             -- output of the second (up) projection
and the gated output is their elementwise product Zt = sigma(Z1) * Z2.
Sec. 4.2 defines Neun_up / Neun_down over "the activation value of the i-th
neuron in the MLP module" and Fig. 5 shows the repair producing a "Modified MLP
gate" and a "Modified MLP data" - i.e. the two streams are adjusted SEPARATELY,
before they are multiplied.

CORRECTION HISTORY (documented for the thesis)
----------------------------------------------
v1 of this module profiled and modified the ALREADY-MULTIPLIED product at the
input of down_proj (one hook instead of two). That is a different intervention:
adding beta to a product is not the same as adding beta to its factors. All v1
repair numbers therefore measured a variant, not Algorithm 2. This version
implements the two streams separately; `stream_mode="product"` reproduces the v1
behaviour so the two can be compared directly rather than silently replaced.

POSITION SEMANTICS
------------------
The paper does not state which sequence position the statistics are taken from,
nor at which positions the correction is applied. We default to the position of
the token under test (`position="token"`), which is the reading that matches
"the activation states of glitch tokens", and expose `position="last"` (v1
behaviour: final prompt position) so the choice can be ablated rather than
assumed. During generation the correction is applied only in the prefill pass,
where the token under test is actually present.
"""
import torch
from tqdm import tqdm

from ..common.model_utils import decoder_layers, mlp_module
from ..common.prompts import spliced_batch, token_position
from ..ground_truth.sweep import repetition_sweep

STREAMS = ("gate", "data")
# module_map key that produces each stream's tensor as its OUTPUT
_STREAM_MODULE = {"gate": "act", "data": "up"}


@torch.no_grad()
def _collect_mlp_acts(model, tok, token_ids, mcfg, batch_size, desc,
                      task="repetition", position="token", stream_mode="separate"):
    """Per key layer and stream -> {(layer, stream): tensor [n_tokens, d_mlp]} on cpu fp32.

    stream_mode="separate": capture MLP gate (act_fn output) and MLP data (up_proj
        output) independently -- the paper's formulation.
    stream_mode="product":  capture the gated product at down_proj's input -- the
        v1 behaviour, retained for the ablation.
    """
    device = next(model.parameters()).device
    layers = decoder_layers(model, mcfg)
    key_layers = mcfg["key_layers"]
    pos = token_position(tok, task) if position == "token" else -1
    store, handles = {}, []

    def make_out_hook(key):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store.setdefault(key, []).append(t[:, pos, :].detach().float().cpu())
        return hook

    def make_pre_hook(key):
        def hook(_m, args):
            store.setdefault(key, []).append(args[0][:, pos, :].detach().float().cpu())
        return hook

    for li in key_layers:
        if stream_mode == "separate":
            for s in STREAMS:
                try:
                    mod = mlp_module(layers[li], mcfg, _STREAM_MODULE[s])
                except AttributeError:
                    continue  # architecture lacks a hookable module for this stream
                handles.append(mod.register_forward_hook(make_out_hook((li, s))))
        else:
            handles.append(
                mlp_module(layers[li], mcfg, "down").register_forward_pre_hook(
                    make_pre_hook((li, "product"))
                )
            )
    try:
        for i in tqdm(range(0, len(token_ids), batch_size), desc=desc):
            chunk = token_ids[i : i + batch_size]
            input_ids, attention_mask = spliced_batch(tok, chunk, device, task)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, dim=0) for k, v in store.items()}


def compute_neuron_stats(model, tok, normal_sample, mcfg, m: float, up_quantile: float,
                         batch_size: int, task="repetition", position="token",
                         stream_mode="separate") -> dict:
    """Paper Eq. 7-8, per key layer and stream.

      Neun_up   = {i : act[i] > m for more than `up_quantile` of normal tokens}
      Neun_down = {i : act[i] <= m for ALL normal tokens}
    """
    acts = _collect_mlp_acts(model, tok, normal_sample, mcfg, batch_size,
                             "normal stats", task, position, stream_mode)
    stats = {}
    for key, A in acts.items():  # A: [n_normal, d_mlp]
        active_frac = (A > m).float().mean(dim=0)
        stats[key] = {
            "neun_up": torch.nonzero(active_frac > up_quantile).squeeze(-1),
            "neun_down": torch.nonzero((A <= m).all(dim=0)).squeeze(-1),
            "mean_normal": A.mean(dim=0),
        }
    return stats


def neuron_stat_summary(stats: dict) -> dict:
    """Auditable record of how many neurons each criterion selected."""
    return {f"L{li}.{s}": {"n_neun_up": int(len(v["neun_up"])),
                           "n_neun_down": int(len(v["neun_down"])),
                           "d_mlp": int(v["mean_normal"].numel())}
            for (li, s), v in stats.items()}


def compute_adjustments(model, tok, glitch_sample, mcfg, stats, adaptive: dict,
                        batch_size: int, task="repetition", position="token",
                        stream_mode="separate") -> dict:
    """Paper Eq. 9-12.

        Eq. 9  dAct_up   = mean_{i in Neun_up}  ( Act_normal[i] - Act_glitch[i] )
        Eq. 10 dAct_down = mean_{i in Neun_down}( Act_glitch[i] / Act_normal[i] )
        Eq. 11 beta  = k1 * dAct_up   + b1
        Eq. 12 alpha = k2 * dAct_down + b2

    Eq. 10 is a SIGNED ratio in the paper; v1 of this module inserted an absolute
    value in the denominator, which is a deviation. We now follow the published
    form, guarding only against division by zero.

    k1, b1, k2, b2 are never given numerically in the paper ("derived through an
    adaptive process tailored to the specific dynamics of each model"), and the
    "range restriction" it mentions is likewise unspecified. Both therefore come
    from configs/glitchprober.yaml and are reported as OUR choice, not the
    paper's. `alpha_min` makes the clamp explicit instead of hard-coded.
    """
    acts = _collect_mlp_acts(model, tok, glitch_sample, mcfg, batch_size,
                             "glitch stats", task, position, stream_mode)
    eps = 1e-6
    alpha_min = float(adaptive.get("alpha_min", 1.0))
    adj = {}
    for key, A in acts.items():
        s = stats[key]
        mean_glitch = A.mean(dim=0)
        up, down = s["neun_up"], s["neun_down"]
        d_up = ((s["mean_normal"][up] - mean_glitch[up]).mean().item()
                if len(up) else 0.0)
        if len(down):
            denom = s["mean_normal"][down]
            denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
            d_down = (mean_glitch[down] / denom).mean().item()
        else:
            d_down = 1.0
        adj[key] = {
            "beta": adaptive["k1"] * d_up + adaptive["b1"],
            "alpha": max(adaptive["k2"] * d_down + adaptive["b2"], alpha_min),
            "d_act_up": d_up,
            "d_act_down": d_down,
        }
    return adj


def _resolve(ab: dict, key):
    """alpha_beta may be per-(layer,stream) or a single pair applied everywhere."""
    if key in ab:
        return ab[key]
    if "alpha" in ab and "beta" in ab:
        return ab
    raise KeyError(f"no alpha/beta for {key}")


class RepairHooks:
    """Install the activation adjustment of Algorithm 2 lines 8-13.

    For each key layer and stream:  act[Neun_up] += beta ; act[Neun_down] /= alpha
    """

    def __init__(self, model, mcfg, stats, alpha_beta: dict, task="repetition",
                 position="token", stream_mode="separate"):
        self.model, self.mcfg, self.stats, self.ab = model, mcfg, stats, alpha_beta
        self.task, self.position, self.stream_mode = task, position, stream_mode
        self.handles = []
        self._tok_pos = None  # set by bind_position(tok) before __enter__

    def bind_position(self, tok):
        self._tok_pos = token_position(tok, self.task) if self.position == "token" else -1
        return self

    def __enter__(self):
        layers = decoder_layers(self.model, self.mcfg)
        device = next(self.model.parameters()).device
        pos = self._tok_pos if self._tok_pos is not None else -1

        for key, s in self.stats.items():
            li, stream = key
            ab = _resolve(self.ab, key)
            up = s["neun_up"].to(device)
            down = s["neun_down"].to(device)
            beta, alpha = float(ab["beta"]), float(ab["alpha"])

            def edit(x, _up=up, _down=down, _b=beta, _a=alpha, _pos=pos):
                # Only the prefill pass contains the token under test; during
                # decoding (seq_len == 1) the token is no longer present, so the
                # correction must not fire.
                if _pos >= 0 and x.shape[1] <= _pos:
                    return x
                x = x.clone()
                idx = (slice(None), _pos) if _pos >= 0 else (slice(None), -1)
                if len(_up):
                    x[idx[0], idx[1], _up] += _b
                if len(_down) and _a != 0:
                    x[idx[0], idx[1], _down] /= _a
                return x

            if stream == "product":
                def pre_hook(_m, args, _edit=edit):
                    return (_edit(args[0]),)
                mod = mlp_module(layers[li], self.mcfg, "down")
                self.handles.append(mod.register_forward_pre_hook(pre_hook))
            else:
                def out_hook(_m, _inp, out, _edit=edit):
                    if isinstance(out, tuple):
                        return (_edit(out[0]),) + out[1:]
                    return _edit(out)
                mod = mlp_module(layers[li], self.mcfg, _STREAM_MODULE[stream])
                self.handles.append(mod.register_forward_hook(out_hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def evaluate_repair(model, tok, glitch_ids, normal_ids, mcfg, stats, alpha_beta,
                    batch_size, max_new_tokens, task="repetition", correct_fn=None,
                    position="token", stream_mode="separate", record_path=None) -> dict:
    """Repair rate on glitch tokens + collateral breakage on normal tokens.

    record_path: if given, per-token outcomes (id, was_glitch, repaired_ok, the
    generated text) are written there so every headline number can be
    reconstructed from raw evidence later.
    """
    hooks = RepairHooks(model, mcfg, stats, alpha_beta, task, position, stream_mode)
    hooks.bind_position(tok)
    with hooks:
        g = repetition_sweep(model, tok, glitch_ids, batch_size, max_new_tokens,
                             task=task, desc="repair glitch", correct_fn=correct_fn)
        n = repetition_sweep(model, tok, normal_ids, batch_size, max_new_tokens,
                             task=task, desc="repair normal", correct_fn=correct_fn)
    repaired = sum(1 for ok, _ in g.values() if ok)
    broken = sum(1 for ok, _ in n.values() if not ok)

    if record_path is not None:
        import csv
        from pathlib import Path
        p = Path(record_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_ALL)
            w.writerow(["token_id", "set", "ok_after_repair", "generated"])
            for tid, (ok, text) in g.items():
                w.writerow([tid, "glitch", int(ok), text])
            for tid, (ok, text) in n.items():
                w.writerow([tid, "normal", int(ok), text])

    return {
        "n_glitch": len(glitch_ids),
        "repaired": repaired,
        "repair_rate": repaired / len(glitch_ids) if glitch_ids else 0.0,
        "n_normal_checked": len(normal_ids),
        "normal_broken": broken,
        "normal_break_rate": broken / len(normal_ids) if normal_ids else 0.0,
    }
