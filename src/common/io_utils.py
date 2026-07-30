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


def checkpoint_guard(path, fingerprint: dict) -> bool:
    """Refuse to resume a checkpoint that was produced by a different configuration.

    Why this exists: an earlier run of the alpha/beta grid silently resumed a
    checkpoint written by a PREVIOUS, since-corrected implementation, so the
    "recomputed" grid was in fact the old grid with fresh metadata. A checkpoint
    is only safe to resume if the code path and settings that produced it match
    the ones about to extend it.

    Returns True if an existing checkpoint may be resumed, False if there is no
    checkpoint yet. Raises if a checkpoint exists but was produced by a different
    configuration - the caller must then move it aside deliberately.
    """
    import hashlib
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    fp_path = p.with_suffix(p.suffix + ".fingerprint.json")
    digest = hashlib.sha256(
        _json.dumps(fingerprint, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    if not p.exists():
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(_json.dumps({"digest": digest, **fingerprint}, indent=2,
                                       default=str), encoding="utf-8")
        return False

    if not fp_path.exists():
        raise RuntimeError(
            f"{p} exists but has no fingerprint: it predates checkpoint guarding and "
            f"cannot be shown to match the current configuration. Move it aside "
            f"(e.g. rename to *.v1.csv) and rerun."
        )
    prev = _json.loads(fp_path.read_text(encoding="utf-8"))
    if prev.get("digest") != digest:
        raise RuntimeError(
            f"{p} was produced by a different configuration and must not be extended.\n"
            f"  stored:  { {k: v for k, v in prev.items() if k != 'digest'} }\n"
            f"  current: {fingerprint}\n"
            f"Move the old checkpoint aside and rerun."
        )
    return True


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t0
