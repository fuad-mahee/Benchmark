"""Recompute every derived number this study reports, from the artefacts under results/.

Several figures in the write-up are not the direct output of any run: they are sums,
set operations or re-scorings over stored files. Those are exactly the numbers that
drifted between drafts, so they are computed here rather than transcribed.

  python scripts/verify_claims.py --model mistral-7b-instruct-v01

Read-only: writes nothing under results/. Anything it prints can be pasted into the
study document and re-derived by a reader with the same repository.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.config import get_model_cfg, results_dir

REPO = Path(__file__).resolve().parents[1]


def _hdr(s):
    print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)


def _read_sweep(p):
    """Generations may be empty or whitespace; keep them as literal strings."""
    return pd.read_csv(p, keep_default_na=False, dtype={"text": str})


def census(model):
    _hdr("RQ1 - census, factorial and mechanism")
    gt = results_dir("ground_truth", model)
    paper = pd.read_csv(gt / "tokens.csv", keep_default_na=False, dtype={"token": str})
    code = pd.read_csv(gt / "gccode" / "tokens.csv", keep_default_na=False, dtype={"token": str})
    G_p = set(paper[paper.category == "glitch"].token_id)
    G_c = set(code[code.category == "glitch"].token_id)
    cand_p = set(paper[paper.category.isin(["normal", "glitch"])].token_id)
    cand_c = set(code[code.category.isin(["normal", "glitch"])].token_id)

    print(f"  glitch: P_paper={len(G_p)}  P_code={len(G_c)}  gap={len(G_c) - len(G_p):+d}")
    print(f"  candidate sets identical: {cand_p == cand_c}  (n={len(cand_p)})")
    print(f"  bidirectional: paper-only={len(G_p - G_c)}  code-only={len(G_c - G_p)}")

    b24 = gt / "gccode" / "budget24" / "tokens.csv"
    if b24.exists():
        G_24 = set(pd.read_csv(b24).query("category == 'glitch'").token_id)
        print(f"  generation budget 24->10 moves the count by {len(G_c) - len(G_24):+d} "
              f"(gccode@24 = {len(G_24)})")

    # Mechanism: the two sides must be scored with the SAME predicate.
    print("\n  mechanism - like-for-like predicates over all candidates:")
    for nm, p in [("P_paper", gt / "sweep_checkpoint.csv"),
                  ("P_code ", gt / "gccode" / "sweep_checkpoint.csv")]:
        t = _read_sweep(p)["text"].astype(str)
        nl = int(((t.str.len() > 0) & (t.str.replace("\n", "", regex=False) == "")).sum())
        ws = int((t.str.strip() == "").sum())
        print(f"    {nm}  newline-only={nl:5d}   whitespace-or-empty={ws:5d}")
    sw = _read_sweep(gt / "gccode" / "sweep_checkpoint.csv")
    g = sw[sw.token_id.isin(G_c)]["text"].astype(str)
    n_ws = int((g.str.strip() == "").sum())
    print(f"    of P_code's {len(g)} glitch tokens, {n_ws} emit pure whitespace "
          f"({n_ws / len(g) * 100:.1f}%)")

    # Construct validity: how much of the census rests on a near-empty match target.
    cand = paper[paper.category.isin(["normal", "glitch"])]
    tk = cand["token"].astype(str).str.strip()
    print(f"\n  oracle weakness: {int((tk == '').sum())} whitespace-only candidates; "
          f"{int((tk.str.len() <= 1).sum())} with stripped length <= 1")
    nrm = paper[paper.category == "normal"]
    n1 = int((nrm["token"].astype(str).str.strip().str.len() <= 1).sum())
    print(f"    {n1} of {len(nrm)} 'normal' verdicts ({n1 / len(nrm) * 100:.1f}%) rest on "
          f"a match of <= 1 character")
    return G_p, G_c


def factorial(model):
    """Isolate the three factors separating the two protocols.

    Each is measured with the other two held fixed, at BOTH settings of the
    remaining factors -- they interact, so a single additive decomposition would
    be misleading. Greedy decoding makes the stored generations re-scorable
    offline, which is what lets this be computed without re-running the model.
    """
    _hdr("RQ1 - factorial: which factor separates the two protocols?")
    from src.common.model_utils import load_tokenizer, token_str
    from src.common.prompts import is_repetition_correct
    tok = load_tokenizer(get_model_cfg(model))
    gt = results_dir("ground_truth", model)
    cand = set(pd.read_csv(gt / "tokens.csv")
               .query("category in ['normal','glitch']").token_id)

    def load(p):
        d = _read_sweep(p)
        # id 3 is NUL on this tokenizer and does not survive the CSV round trip
        return d[d.token_id.isin(cand) & (d.token_id != 3)]

    # Each protocol's oracle, exactly as its run applied it.
    oracles = {
        "paper": lambda tid, txt: is_repetition_correct(token_str(tok, int(tid)), txt),
        "gccode": lambda tid, txt: tok.decode([int(tid)]).lstrip() in txt,
    }
    gens = {"P_paper@24": gt / "sweep_checkpoint.csv",
            "P_code@10": gt / "gccode" / "sweep_checkpoint.csv",
            "P_code@24": gt / "gccode" / "budget24" / "sweep_checkpoint.csv"}

    n = {}
    print(f"  {'generations':14s} {'paper oracle':>14s} {'gccode oracle':>14s}")
    for gname, path in gens.items():
        if not path.exists():
            continue
        df = load(path)
        row = []
        for oname, fn in oracles.items():
            n[(gname, oname)] = sum(1 for t, x in zip(df.token_id, df.text.astype(str))
                                    if not fn(t, x))
            row.append(n[(gname, oname)])
        print(f"  {gname:14s} {row[0]:14d} {row[1]:14d}")

    gap = 1564
    print(f"\n  each factor isolated, at both settings of the others "
          f"(total published gap +{gap}):")
    if ("P_code@24", "gccode") in n:
        for o in ("paper", "gccode"):
            d = n[("P_code@10", o)] - n[("P_code@24", o)]
            print(f"    budget 24->10   ({o:6s} oracle)      {d:+6d}  {d / gap * 100:6.2f}%")
    for g in ("P_paper@24", "P_code@10"):
        d = n[(g, "gccode")] - n[(g, "paper")]
        print(f"    oracle paper->gccode ({g:10s}) {d:+6d}  {d / gap * 100:6.2f}%")
    for o in ("paper", "gccode"):
        d = n[("P_code@10", o)] - n[("P_paper@24", o)]
        print(f"    prompt wording  ({o:6s} oracle)      {d:+6d}  {d / gap * 100:6.2f}%")
    print("  => the factors INTERACT (the oracle effect changes sign), so report the"
          "\n     range rather than a single additive decomposition.")


def anchoring(model, G_c):
    _hdr("RQ1 - anchoring against GlitchCleaner's published list")
    p = REPO / "third_party" / "GlitchCleaner" / "Glitchtokens"
    f = p / "Mistral-7B-Instruct-v0.1-glitch-tokens.csv"
    if not f.exists():
        print("  third_party/GlitchCleaner missing - skipped")
        return None
    pub = pd.read_csv(f)
    G_pub = set(int(x) for x in pub[pub.columns[0]])
    inter = G_c & G_pub
    print(f"  |published| = {len(G_pub)}   recovered = {len(inter)}   "
          f"missed = {len(G_pub - G_c)}   ours-only = {len(G_c - G_pub)}")
    print(f"  Jaccard = {len(inter) / len(G_c | G_pub):.4f}")
    return G_pub


def gate(model, G_pub):
    """The set G is not only a denominator: GlitchCleaner consults it at inference."""
    _hdr("RQ4 - the lambda gate: how often does it open?")
    if G_pub is None:
        return
    from src.common.model_utils import load_tokenizer
    from src.common.prompts import template_ids
    tok = load_tokenizer(get_model_cfg(model))

    for task in ("repetition", "repetition_gccode"):
        pre, suf = template_ids(tok, task)
        hits = sorted(set(pre + suf) & G_pub)
        print(f"  template {task:18s} contains published-G ids {hits} "
              f"-> {[tok.decode([i]) for i in hits]}")
    print("  => the prompt boilerplate alone opens the gate, for every token under test")

    common = sorted(i for i in G_pub if i < 1000)
    print(f"\n  published G contains {len(common)} ids below 1000 (i.e. high-frequency): "
          f"{[tok.decode([i]) for i in common[:14]]} ...")

    # Ordinary prose: the papers' own text is a fair, offline sample.
    n_tot = n_open = 0
    who = Counter()
    for f in sorted((REPO / "papers" / "extracted").glob("*.txt")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw)
                 if 40 <= len(s.strip()) <= 400]
        k = 0
        for s in sents:
            hit = set(tok(s, add_special_tokens=False).input_ids) & G_pub
            if hit:
                k += 1
                who.update(hit)
        print(f"  {f.name:32s} {k:5d} / {len(sents):5d} sentences open the gate "
              f"({k / len(sents) * 100:.1f}%)")
        n_tot += len(sents)
        n_open += k
    print(f"  {'TOTAL':32s} {n_open:5d} / {n_tot:5d} ({n_open / n_tot * 100:.1f}%)")
    print("  top openers: " + ", ".join(
        f"{tok.decode([i])!r} ({n})" for i, n in who.most_common(6)))

    ev = results_dir("gc", model) / "gccode" / "eval.json"
    if ev.exists():
        r = json.loads(ev.read_text())["results"]
        gs = r["gate_stats"]
        print(f"\n  stored gccode eval: gate_on={gs['examples_gate_on']} "
              f"gate_off={gs['examples_gate_off']} "
              f"(expected off = n_normal = {r['n_normal']})")
        print(f"    clean gated={r['normal_ok_rate_gated']:.4f}  "
              f"forced-on={r['normal_ok_rate_adapter_forced_on']:.4f}  "
              f"-> identical because the gate never closed")


def gp_repair(model):
    _hdr("RQ3 - neuron selection, the 2x2, and the corrected grid")
    d = results_dir("gp_repair", model)
    rows = [("separate/token (paper reading)", ""), ("product/token", "ablation_product/"),
            ("separate/all-positions", "abl_separate_last/"),
            ("product/all-positions (v1)", "abl_product_last/"),
            ("separate/token, gccode", "gccode/")]
    print(f"  {'configuration':32s} {'Neun_up':>8s} {'Neun_down':>10s} "
          f"{'repair':>18s} {'collateral':>11s}")
    for label, sub in rows:
        p = d / sub / "summary.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        ns = j["neuron_selection"]
        rb = j["results"]["rule_based"]
        print(f"  {label:32s} {sum(v['n_neun_up'] for v in ns.values()):8d} "
              f"{sum(v['n_neun_down'] for v in ns.values()):10d} "
              f"{rb['repair_rate'] * 100:9.2f}% ({rb['repaired']:4d}/{rb['n_glitch']:4d}) "
              f"{rb['normal_break_rate'] * 100:10.2f}%")
    print(f"  candidate neurons = 10 layers x 2 streams x 14336 = {10 * 2 * 14336}")

    for nm, f in [("corrected", "alpha_beta_grid.csv"),
                  ("v1 (product/all-positions)", "alpha_beta_grid.v1_product_last.csv")]:
        p = d / f
        if not p.exists():
            continue
        g = pd.read_csv(p)
        # The v1 grid stored rates only; the corrected one also stores counts.
        n_cell = int(g.n_glitch.iloc[0]) if "n_glitch" in g else None
        am, bm = g.groupby("alpha").repair_rate.mean(), g.groupby("beta").repair_rate.mean()
        best = g[g.repair_rate == g.repair_rate.max()]
        pv = g[(g.alpha == 4.0) & (g.beta == 1.5)]
        print(f"\n  {nm} grid ({len(g)} cells"
              f"{f', n={n_cell} per cell' if n_cell else ', counts not stored'}):")
        print(f"    marginal spread: alpha {(am.max() - am.min()) * 100:.2f} pp   "
              f"beta {(bm.max() - bm.min()) * 100:.2f} pp")
        print(f"    max {g.repair_rate.max() * 100:.2f}% at "
              f"{[(float(r.alpha), float(r.beta)) for _, r in best.iterrows()]}")
        print(f"    at the paper's (4, 1.5): {pv.repair_rate.iloc[0] * 100:.2f}%")
        a1 = g[g.alpha == 1.0].sort_values("beta")
        counts = ([int(x) for x in a1.repaired] if "repaired" in g
                  else [round(r * n_cell) if n_cell else r for r in a1.repair_rate])
        print(f"    alpha=1 row across beta: {counts}"
              f"{f' of {n_cell}' if n_cell else ' (rates)'}")


def gc(model):
    _hdr("RQ4 - split sizes and denominators")
    for sub, nm in [("", "paper"), ("gccode/", "gccode")]:
        tm = results_dir("gc", model) / sub / "train_meta.json"
        ev = results_dir("gc", model) / sub / "eval.json"
        if not tm.exists():
            continue
        t = json.loads(tm.read_text())
        e = json.loads(ev.read_text())["results"]
        print(f"  {nm:7s} |G|={t['n_glitch_total']:5d} = train {t['n_train']} + heldout "
              f"{t['n_heldout']} + leaked-dropped {t['n_heldout_dropped_leaked']}"
              f"   (train {t.get('train_seconds', float('nan')):.1f}s)")
        print(f"          train {e['train_repair_rate'] * 100:.2f}%   "
              f"held-out {e['heldout_repair_rate'] * 100:.2f}%   "
              f"adapter-off {e['heldout_repair_rate_adapter_off'] * 100:.2f}%")


def rescaling():
    """GlitchCleaner's reported GlitchProber counts vs rate x GlitchCleaner's own |G|."""
    _hdr("Motivation - are GlitchCleaner's competitor counts derived?")
    # (model, GP's own count, GP's rate, GC's reported count, GC's rate, GC's |G|)
    tbl = [("Llama-2-7b-chat", 4021, 0.6258, 2968, 0.6258, 4743),
           ("Gemma-2b-it", 13638, 0.4877, 14548, 0.4877, 29831),
           ("Mistral-7B-Instruct-v0.1", 1045, 0.3760, 956, 0.3765, 2539),
           ("Qwen-7B-Chat", 14765, 0.4811, 13320, 0.4811, 27686),
           ("Yi-6B-Chat", 4317, 0.5326, 3188, 0.5327, 5985)]
    print(f"  {'model':26s} {'GP count':>9s} {'GC reports':>11s} {'rate x |G_GC|':>14s} "
          f"{'diff':>6s}  {'GP rate':>8s} {'GC rate':>8s}")
    for m, gp_n, gp_r, gc_n, gc_r, gN in tbl:
        pred = gp_r * gN
        print(f"  {m:26s} {gp_n:9d} {gc_n:11d} {pred:14.1f} {gc_n - pred:+6.1f}  "
              f"{gp_r * 100:7.2f}% {gc_r * 100:7.2f}%")
    print(f"\n  GlitchCleaner's average of its GlitchProber column: "
          f"{sum(r[4] for r in tbl) / 5 * 100:.2f}%   "
          f"GlitchProber's own published average: 50.06%")
    print("  The rate is recomputed from the rounded rescaled count (Mistral and Yi differ"
          "\n  from GlitchProber's printed rates), so the counts were produced first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-7b-instruct-v01")
    ap.add_argument("--skip-tokenizer", action="store_true",
                    help="skip the gate analysis (the only part that loads a tokenizer)")
    args = ap.parse_args()

    _G_p, G_c = census(args.model)
    G_pub = anchoring(args.model, G_c)
    if not args.skip_tokenizer:
        factorial(args.model)
    gp_repair(args.model)
    gc(args.model)
    rescaling()
    if not args.skip_tokenizer:
        gate(args.model, G_pub)


if __name__ == "__main__":
    main()
