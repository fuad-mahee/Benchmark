"""Result writing with run metadata, so every thesis number is traceable."""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True if tracked files differ from HEAD.

    Recording this matters: a run launched from a modified working tree is NOT
    reproducible from the recorded commit alone. An earlier version of this file
    logged only the commit hash, which produced artifacts attributed to commits
    that could not have generated them.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=ROOT,
        )
        return bool(out.strip())
    except Exception:
        return True  # unknown state is not clean


def env_fingerprint() -> dict:
    fp = {}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
        fp["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    for mod in ("transformers", "peft", "sklearn"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            pass
    return fp


def run_metadata(**extra) -> dict:
    md = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "env": env_fingerprint(),
    }
    md.update(extra)
    return md


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
