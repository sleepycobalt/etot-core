import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from etot_core import llm


def load_config(path: str | Path, *, require_key: bool = True) -> dict:
    """Load a YAML config and the .env file. Returns a plain dict.

    require_key=True (the default) raises when no Anthropic key is available, from the environment or a
    context key. require_key=False skips only that check, for tool paths that never call a model; the .env
    file is still loaded and the same defaults are applied."""
    load_dotenv()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if require_key and not os.getenv("ANTHROPIC_API_KEY") and not llm.has_context_key():
        raise RuntimeError("ANTHROPIC_API_KEY not set — check your .env file")
    cfg.setdefault("models", {})
    cfg.setdefault("loop", {})
    cfg["loop"].setdefault("max_iterations", 3)
    return cfg
