"""Config loading + environment bootstrap.

Import this module (and call setup_env) BEFORE importing transformers anywhere,
so HF_HOME points at the big drive instead of C:.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_env() -> dict:
    cfg = load_yaml("models.yaml")
    cache = cfg.get("hf_cache_dir")
    if cache:
        os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cfg


def get_model_cfg(name: str) -> dict:
    cfg = setup_env()
    models = cfg["models"]
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Known: {', '.join(models)}")
    merged = dict(cfg.get("defaults", {}))
    merged.update(models[name])
    merged["name"] = name
    return merged


def enabled_models() -> list[str]:
    cfg = setup_env()
    return [n for n, m in cfg["models"].items() if m.get("enabled")]


def results_dir(*parts: str) -> Path:
    p = RESULTS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
