"""Orchestrate the full benchmark pipeline across models, with logging and resume.

    python scripts/orchestrate.py --models qwen25-7b-instruct gemma-2b-it --phase census
    python scripts/orchestrate.py --models gemma-2b-it --phase full
    python scripts/orchestrate.py --models ... --phase full --dry-run

Design notes
------------
* Every step writes a complete log to logs/<model>/<step>.log (stdout+stderr,
  unfiltered) so a run can be audited after the fact rather than trusted.
* Progress is journalled to logs/progress.jsonl after EVERY step, so an
  interrupted run can be inspected and resumed without guessing.
* A step whose expected output already exists is SKIPPED unless --force. The
  underlying scripts also checkpoint internally, so a step interrupted midway
  resumes from its own checkpoint when re-run.
* Preflight is run first for every model and a FAILURE ABORTS that model. A wrong
  module_map does not crash the pipeline, it silently produces meaningless
  numbers, so this gate is not optional.
* Steps are ordered by value-per-GPU-hour, so that an interrupted run still
  leaves the most useful artefacts on disk.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

# (step id, script, extra args, expected output relative to results/)
# {m} is substituted with the model name.
STEPS = {
    "census": [
        ("census_paper", "run_ground_truth.py", [],
         "ground_truth/{m}/tokens.csv"),
        ("census_gccode", "run_ground_truth.py", ["--protocol", "gccode"],
         "ground_truth/{m}/gccode/tokens.csv"),
    ],
    "anchor": [
        # only meaningful where GlitchCleaner published an adapter + token list
        ("gc_upstream", "run_gc_upstream_eval.py", [],
         "gc_upstream/{m}/gccode/eval.json"),
    ],
    "repair": [
        ("gp_repair_paper", "run_gp_repair.py", [],
         "gp_repair/{m}/summary.json"),
        ("gp_repair_gccode", "run_gp_repair.py", ["--protocol", "gccode"],
         "gp_repair/{m}/gccode/summary.json"),
        ("gp_abl_product_token", "run_gp_repair.py",
         ["--stream-mode", "product", "--tag", "ablation_product"],
         "gp_repair/{m}/ablation_product/summary.json"),
        ("gp_abl_product_last", "run_gp_repair.py",
         ["--stream-mode", "product", "--position", "last", "--tag", "abl_product_last"],
         "gp_repair/{m}/abl_product_last/summary.json"),
        ("gp_abl_separate_last", "run_gp_repair.py",
         ["--stream-mode", "separate", "--position", "last", "--tag", "abl_separate_last"],
         "gp_repair/{m}/abl_separate_last/summary.json"),
        ("gp_m_sweep", "run_gp_alpha_beta_sweep.py", ["--sweep", "m"],
         "gp_repair/{m}/m_sweep.csv"),
        ("gp_grid", "run_gp_alpha_beta_sweep.py", [],
         "gp_repair/{m}/alpha_beta_grid.csv"),
    ],
    "gc": [
        ("gc_train_paper", "run_gc_train.py", [], "gc/{m}/train_meta.json"),
        ("gc_eval_paper", "run_gc_eval.py", [], "gc/{m}/eval.json"),
        ("gc_train_gccode", "run_gc_train.py", ["--protocol", "gccode"],
         "gc/{m}/gccode/train_meta.json"),
        ("gc_eval_gccode", "run_gc_eval.py", ["--protocol", "gccode"],
         "gc/{m}/gccode/eval.json"),
    ],
    "detect": [
        ("gp_detect_paper", "run_gp_detect.py", [], "gp_detect/{m}/summary.json"),
        ("gp_detect_gccode", "run_gp_detect.py", ["--protocol", "gccode"],
         "gp_detect/{m}/gccode/summary.json"),
    ],
    "speed": [
        ("speed", "run_speed.py", [], "side_effects/{m}/speed.json"),
    ],
}

# Value-per-GPU-hour ordering. Census first (RQ1 is the flagship result and every
# later step depends on it); detection last (3 seeds x full-vocabulary feature
# extraction is by far the most expensive step).
PHASE_ORDER = ["census", "anchor", "repair", "gc", "speed", "detect"]


def journal(entry: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_step(model: str, step_id: str, script: str, extra: list[str],
             expected: str, force: bool, dry: bool) -> dict:
    out_path = RESULTS / expected.format(m=model)
    log_path = LOGS / model / f"{step_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # A step is only "done" if the journal says it completed. Existence of the
    # output file is NOT sufficient: several steps write their output
    # incrementally (the m-sweep appends a row per threshold, the alpha/beta grid
    # a row per cell), so a step killed midway leaves a PARTIAL file that an
    # existence check would happily skip. Cross-checking the journal prevents a
    # truncated artefact from being mistaken for a finished one.
    if out_path.exists() and not force:
        completed = False
        jl = LOGS / "progress.jsonl"
        if jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if (e.get("model") == model and e.get("step") == step_id
                        and e.get("status") == "ok"):
                    completed = True
                    break
        if completed:
            print(f"  SKIP  {step_id:<22} (completed: {out_path.relative_to(ROOT)})")
            return {"model": model, "step": step_id, "status": "skipped"}
        print(f"  REDO  {step_id:<22} (output exists but no 'ok' in journal -> "
              f"treating as PARTIAL)")

    cmd = [PY, str(ROOT / "scripts" / script), "--model", model] + extra
    print(f"  RUN   {step_id:<22} {' '.join(cmd[2:])}")
    if dry:
        return {"model": model, "step": step_id, "status": "dry-run", "cmd": cmd}

    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {' '.join(cmd)}\n# started {started}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(ROOT), text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0 and out_path.exists()
    status = "ok" if ok else ("failed" if proc.returncode else "no-output")
    print(f"        -> {status} in {dt / 60:.1f} min   log: {log_path.relative_to(ROOT)}")

    entry = {"model": model, "step": step_id, "status": status,
             "returncode": proc.returncode, "seconds": round(dt, 1),
             "started": started, "log": str(log_path.relative_to(ROOT)),
             "output": str(out_path.relative_to(ROOT)) if out_path.exists() else None}
    journal(entry)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--phase", nargs="+", default=["full"],
                    help="one or more of: " + ", ".join(PHASE_ORDER) + ", or 'full'")
    ap.add_argument("--force", action="store_true", help="rerun steps whose output exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="NOT RECOMMENDED - preflight is what stops a bad config "
                         "producing plausible nonsense")
    args = ap.parse_args()

    phases = PHASE_ORDER if "full" in args.phase else [p for p in PHASE_ORDER if p in args.phase]
    if not phases:
        ap.error(f"no valid phase in {args.phase}")

    print(f"models : {', '.join(args.models)}")
    print(f"phases : {' -> '.join(phases)}")
    print(f"logs   : {LOGS.relative_to(ROOT)}/<model>/<step>.log")
    print(f"journal: {(LOGS / 'progress.jsonl').relative_to(ROOT)}\n")

    # ---- gate: preflight every model before spending any GPU time ------------
    if not args.skip_preflight and not args.dry_run:
        from scripts.preflight import check as preflight_check
        bad = []
        for m in args.models:
            ok, problems = preflight_check(m, verbose=False)
            print(f"preflight {m:<24} {'PASS' if ok else 'FAIL'}")
            for p in problems:
                print(f"    - {p}")
            if not ok:
                bad.append(m)
        if bad:
            print(f"\nABORT: preflight failed for {', '.join(bad)}. "
                  f"Fix configs/models.yaml before running.")
            sys.exit(1)
        print()

    journal({"event": "run_start", "models": args.models, "phases": phases,
             "timestamp": datetime.now().isoformat(timespec="seconds")})

    summary = []
    for phase in phases:
        for model in args.models:
            steps = STEPS[phase]
            if phase == "anchor":
                # GlitchCleaner released adapters only for the models it evaluated.
                # Match on the EXACT HuggingFace repo basename: a loose match would
                # happily load gemma-2b-it's adapter for gemma-7b-it and report the
                # result as an anchor.
                from src.common.config import get_model_cfg
                basename = get_model_cfg(model)["hf_id"].split("/")[-1].lower()
                pt = ROOT / "third_party" / "GlitchCleaner" / "LoRA-Parameter"
                names = {p.stem.lower() for p in pt.glob("*.pt")} if pt.exists() else set()
                if basename not in names:
                    print(f"[{phase}] {model}: no upstream adapter published "
                          f"(looked for '{basename}.pt') - skipping anchor")
                    continue
            print(f"\n=== phase '{phase}' | model '{model}' ===")
            for step_id, script, extra, expected in steps:
                if not (ROOT / "scripts" / script).exists():
                    print(f"  MISS  {step_id:<22} (script {script} not present)")
                    continue
                summary.append(run_step(model, step_id, script, extra, expected,
                                        args.force, args.dry_run))

    journal({"event": "run_end", "timestamp": datetime.now().isoformat(timespec="seconds")})
    ok = sum(1 for s in summary if s.get("status") == "ok")
    sk = sum(1 for s in summary if s.get("status") == "skipped")
    bad = [s for s in summary if s.get("status") in ("failed", "no-output")]
    print(f"\n{'=' * 70}\n{ok} ran, {sk} skipped, {len(bad)} failed")
    for b in bad:
        print(f"  FAILED {b['model']}/{b['step']} -> see {b.get('log')}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
