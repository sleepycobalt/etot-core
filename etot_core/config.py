import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from etot_core import llm


def load_config(path: str | Path) -> dict:
    """Load a YAML config and the .env file. Returns a plain dict."""
    load_dotenv()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not os.getenv("ANTHROPIC_API_KEY") and not llm.has_context_key():
        raise RuntimeError("ANTHROPIC_API_KEY not set — check your .env file")
    cfg.setdefault("models", {})
    cfg.setdefault("loop", {})
    cfg["loop"].setdefault("max_iterations", 3)
    return cfg
