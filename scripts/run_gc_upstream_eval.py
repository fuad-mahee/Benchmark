"""RQ4 anchor: evaluate GlitchCleaner's OWN released adapter on its OWN glitch set.

Everywhere else, RQ4 compares our retrained adapter against the paper's published
rate, which leaves "maybe your training differs from ours" as a permanent
alternative explanation. This script removes it: the authors' weights, the
authors' glitch-token list, the authors' prompt and decoding budget.

  python scripts/run_gc_upstream_eval.py --model mistral-7b-instruct-v01

Three populations are scored, and the differences between them are the point:
  * published   -- their CSV list (their denominator; the direct test of 94.80%)
  * ours        -- our census under the same protocol (2 tokens differ)
  * held-out    -- the subset of their list our OWN adapter never trained on,
                   so their adapter and ours are compared on identical tokens

plus clean-token controls with the gate on and forced off, and the gate statistics
that show whether lambda ever actually closed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.config import get_model_cfg, load_yaml, results_dir
from src.common.io_utils import run_metadata, save_json

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party" / "GlitchCleaner"
# Their checkpoints and token lists are keyed by the HF model name.
CKPT_NAME = {"mistral-7b-instruct-v01": "Mistral-7B-Instruct-v0.1",
             "llama2-7b-chat": "Llama-2-7b-chat",
             "gemma-2b-it": "gemma-2b-it",
             "yi-6b-chat": "Yi-6B-Chat",
             "qwen-7b-chat": "Qwen-7B-Chat"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=["gccode", "paper"], default="gccode",
                    help="gccode reproduces their released evaluation exactly; "
                         "paper uses the template their paper prints")
    ap.add_argument("--normal-sample", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N glitch tokens (smoke test)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    mcfg = get_model_cfg(args.model)
    batch = args.batch_size or mcfg["batch_size"]
    name = CKPT_NAME.get(args.model)
    if name is None:
        sys.exit(f"no upstream checkpoint mapping for '{args.model}'")
    ckpt = UPSTREAM / "LoRA-Parameter" / f"{name}.pt"
    tokens_csv = UPSTREAM / "Glitchtokens" / f"{name}-glitch-tokens.csv"
    for p in (ckpt, tokens_csv):
        if not p.exists():
            sys.exit(f"missing upstream artefact: {p}\n"
                     "re-clone third_party/GlitchCleaner (see HANDOVER.md section 2)")

    out = results_dir("gc_upstream", args.model)
    if args.protocol == "gccode":
        out = out / "gccode"
    if args.tag:
        out = out / args.tag

    # --- populations -----------------------------------------------------
    published = [int(x) for x in pd.read_csv(tokens_csv)[["index"]].iloc[:, 0]]
    gt = results_dir("ground_truth", args.model)
    gt_path = gt / ("gccode/tokens.csv" if args.protocol == "gccode" else "tokens.csv")
    df = pd.read_csv(gt_path)
    ours = df[df.category == "glitch"].token_id.astype(int).tolist()
    normal_pool = df[df.category == "normal"].token_id.tolist()

    # The tokens OUR adapter held out, intersected with THEIR list: the only
    # population on which the two adapters can be compared without a train/test
    # asymmetry favouring one of them.
    split_p = results_dir("gc", args.model) / ("gccode" if args.protocol == "gccode" else "") / "split.json"
    heldout = []
    if split_p.exists():
        with open(split_p, encoding="utf-8") as f:
            heldout = sorted(set(json.load(f)["heldout_ids"]) & set(published))

    # Tokens WE find glitchy that are absent from their published list. Their
    # adapter was trained on their list, so these are the only tokens in this
    # experiment it provably never saw -- the sole generalisation test available
    # for their weights. Small, but it is the only unseen population there is.
    unseen = sorted(set(ours) - set(published))

    if args.limit:
        published, ours = published[:args.limit], ours[:args.limit]
        heldout, unseen = heldout[:args.limit], unseen[:args.limit]

    rng = np.random.default_rng(load_yaml("glitchcleaner.yaml")["seed"])
    normal_sample = [int(x) for x in rng.choice(
        normal_pool, size=min(args.normal_sample, len(normal_pool)), replace=False)]

    # --- protocol --------------------------------------------------------
    from src.common.model_utils import load_model, load_tokenizer
    from src.glitchcleaner.upstream_adapter import load_upstream_adapter
    from src.ground_truth.sweep import repetition_sweep

    if args.protocol == "gccode":
        task, max_new = "repetition_gccode", 10

        def correct_fn(tok, tid, text):
            return tok.decode([tid]).lstrip() in text
    else:
        task, max_new, correct_fn = "repetition", 24, None

    tok = load_tokenizer(mcfg)
    model = load_model(mcfg, attn_impl=None)
    # The gate consults THEIR published list, as their code does.
    gc_model, meta = load_upstream_adapter(model, mcfg, ckpt, published)
    gc_model.eval()
    print(f"attached {meta['n_modules_attached']} LoRA branches, "
          f"layers {meta['layers_attached']}, config {meta['declared_config']}")

    def sweep(ids, desc):
        r = repetition_sweep(gc_model, tok, ids, batch, max_new, task=task,
                             desc=desc, correct_fn=correct_fn)
        ok = sum(1 for k, _ in r.values() if k)
        return ok, (ok / len(r) if r else 0.0)

    res = {}
    gc_model.set_gated(True)
    for key, ids, desc in [("published", published, "their list, their adapter"),
                           ("ours", ours, "our census, their adapter"),
                           ("heldout_shared", heldout, "our held-out, in their list"),
                           ("unseen_by_upstream", unseen, "NOT in their list (unseen)")]:
        if not ids:
            continue
        n, rate = sweep(ids, desc)
        res[f"{key}_n"], res[f"{key}_repaired"], res[f"{key}_repair_rate"] = len(ids), n, rate
        print(f"{desc:34s} {n}/{len(ids)} = {rate * 100:.2f}%")

    n_ok, r_ok = sweep(normal_sample, "clean tokens, gate active")
    res["normal_ok_gated"], res["normal_ok_rate_gated"] = n_ok, r_ok
    res["gate_stats"] = gc_model.gate_stats()

    gc_model.set_gated(False)
    n_f, r_f = sweep(normal_sample, "clean tokens, adapter forced on")
    res["normal_ok_adapter_forced_on"], res["normal_ok_rate_adapter_forced_on"] = n_f, r_f
    res["n_normal"] = len(normal_sample)

    # Control: with the LoRA branch removed the repair must collapse, otherwise
    # the "repair" is not coming from their weights.
    for m in model.modules():
        if hasattr(m, "scaling") and hasattr(m, "A"):
            m.scaling = 0.0
    n_off, r_off = sweep(heldout or published, "control: branch zeroed")
    res["control_zeroed_repaired"], res["control_zeroed_repair_rate"] = n_off, r_off

    res["paper_claim_repair_rate_mistral"] = 0.9480
    res["paper_claim_repaired_tokens"] = 2407
    res["upstream_meta"] = meta

    print(f"\nclean, gate active:   {r_ok * 100:.2f}%   "
          f"gate_stats={res['gate_stats']}")
    print(f"clean, forced on:     {r_f * 100:.2f}%")
    print(f"control (zeroed):     {r_off * 100:.2f}%")
    save_json(run_metadata(model=args.model, protocol=args.protocol, results=res),
              out / "eval.json")
    print(f"\nwrote {out / 'eval.json'}")


if __name__ == "__main__":
    main()
