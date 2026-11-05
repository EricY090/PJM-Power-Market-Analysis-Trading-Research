"""Load project configuration from config/config.yaml (falls back to the example file)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config dict, with the PJM_API_KEY env var overriding the file value.

    Relative paths under `paths:` are resolved against the project root.
    """
    if path is None:
        path = CONFIG_DIR / "config.yaml"
        if not path.exists():
            path = CONFIG_DIR / "config.example.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_key = os.environ.get("PJM_API_KEY")
    if env_key:
        cfg["pjm"]["api_key"] = env_key

    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def get_api_key(cfg: dict[str, Any]) -> str:
    """Return the PJM API key, raising a clear error if it has not been set."""
    key = cfg["pjm"].get("api_key", "")
    if not key or key.startswith("YOUR_"):
        raise RuntimeError(
            "PJM API key not set. Copy config/config.example.yaml to config/config.yaml "
            "and fill in pjm.api_key, or set the PJM_API_KEY environment variable."
        )
    return key
