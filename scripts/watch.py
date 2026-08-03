r"""One-shot health check for a running orchestrate.py job.

    .\.venv\Scripts\python.exe scripts\watch.py

Answers, in order: is the process alive, is the GPU busy, what is it doing right
now, what has finished, and has anything failed. Exit code 0 = healthy,
1 = something failed, 2 = process is not running.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def c(s, code):          # tiny colour helper; harmless if the terminal ignores it
    return f"\033[{code}m{s}\033[0m"


GREEN, RED, YELLOW, DIM = "32", "31", "33", "90"


def proc_alive(pid: int) -> bool:
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], text=True, stderr=subprocess.DEVNULL)
        return str(pid) in out
    except Exception:
        return False


def main():
    print(f"\n{'=' * 74}\n BENCHMARK RUN STATUS   {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 74}")
    problems = 0

    # ---- 1. process ------------------------------------------------------
    pid_file = LOGS / "orchestrator.pid"
    alive = False
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        alive = proc_alive(pid)
        print(f" orchestrator : {c('RUNNING', GREEN) if alive else c('NOT RUNNING', RED)}  (pid {pid})")
    else:
        print(f" orchestrator : {c('no pid file - was it launched?', RED)}")

    # ---- 2. gpu ----------------------------------------------------------
    try:
        g = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        util, used, total = int(g[0]), int(g[1]), int(g[2])
        state = c(f"{util}% util, {used / 1024:.1f}/{total / 1024:.0f} GB", GREEN if util > 5 else YELLOW)
        print(f" gpu          : {state}")
    except Exception as e:
        print(f" gpu          : {c('nvidia-smi unavailable', YELLOW)} ({type(e).__name__})")

    # ---- 3. what is it doing right now -----------------------------------
    logs = sorted(LOGS.glob("*/*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        cur = logs[-1]
        age = time.time() - cur.stat().st_mtime
        tail = ""
        try:
            lines = [l.rstrip() for l in cur.read_text(encoding="utf-8", errors="replace")
                     .replace("\r", "\n").split("\n") if l.strip()]
            tail = lines[-1][:96] if lines else ""
        except Exception:
            pass
        fresh = age < 300
        print(f" current step : {cur.parent.name} / {cur.stem}")
        print(f"   last write : {c(f'{age:.0f}s ago', GREEN if fresh else RED)}"
              + ("" if fresh else c("  <-- STALLED? no output for >5 min", RED)))
        if tail:
            print(f"   last line  : {c(tail, DIM)}")
        if not fresh and alive:
            problems += 1

    # ---- 4. completed / failed steps -------------------------------------
    jl = LOGS / "progress.jsonl"
    done, failed = [], []
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("status") == "ok":
                done.append(e)
            elif e.get("status") in ("failed", "no-output"):
                failed.append(e)

    print(f"\n completed    : {len(done)} step(s)")
    for e in done[-8:]:
        print(f"   {c('OK', GREEN)}  {e['model']:<22} {e['step']:<22} {e['seconds'] / 60:6.1f} min")
    if len(done) > 8:
        print(f"   {c(f'... and {len(done) - 8} earlier', DIM)}")

    if failed:
        problems += len(failed)
        print(f"\n {c(f'FAILED: {len(failed)} step(s)', RED)}")
        for e in failed:
            print(f"   {c('FAIL', RED)}  {e['model']}/{e['step']}  rc={e.get('returncode')}")
            print(f"         log: {e.get('log')}")
            lp = ROOT / str(e.get("log", ""))
            if lp.exists():
                errs = [l.rstrip() for l in lp.read_text(encoding="utf-8", errors="replace").splitlines()
                        if any(k in l for k in ("Error", "error", "Traceback", "CUDA", "Killed"))]
                for l in errs[-3:]:
                    print(f"         {c(l[:92], DIM)}")
    else:
        print(f"\n failures     : {c('none', GREEN)}")

    # ---- 5. results actually on disk -------------------------------------
    models = ["qwen25-7b-instruct", "qwen25-14b-instruct", "gemma-2b-it", "gemma-7b-it"]
    print(f"\n {'model':<24}{'census':>9}{'gccode':>9}{'repair':>9}{'gc':>7}{'detect':>9}")
    R = ROOT / "results"
    for m in models:
        row = [
            "yes" if (R / "ground_truth" / m / "tokens.csv").exists() else "-",
            "yes" if (R / "ground_truth" / m / "gccode" / "tokens.csv").exists() else "-",
            "yes" if (R / "gp_repair" / m / "summary.json").exists() else "-",
            "yes" if (R / "gc" / m / "eval.json").exists() else "-",
            "yes" if (R / "gp_detect" / m / "summary.json").exists() else "-",
        ]
        cells = "".join(c(f"{v:>9}", GREEN) if v == "yes" else f"{v:>9}" for v in row[:3]) \
                + (c(f"{row[3]:>7}", GREEN) if row[3] == "yes" else f"{row[3]:>7}") \
                + (c(f"{row[4]:>9}", GREEN) if row[4] == "yes" else f"{row[4]:>9}")
        print(f" {m:<24}{cells}")

    print(f"\n{'=' * 74}")
    if not alive and not failed:
        print(c(" Process not running. If all steps show above, the run finished.", YELLOW))
        print(f"{'=' * 74}\n")
        sys.exit(2)
    if problems:
        print(c(f" ATTENTION: {problems} issue(s) above.", RED))
        print(f"{'=' * 74}\n")
        sys.exit(1)
    print(c(" Healthy - run is progressing normally.", GREEN))
    print(f"{'=' * 74}\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
