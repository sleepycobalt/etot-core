"""etot_core.config.load_config. Offline: no API calls, no .env read from the machine."""

import pytest

from etot_core import config, llm


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No key in the environment, no context key, and load_dotenv recorded instead of searching the disk."""
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: calls.append(1) or False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "c.yaml"
    p.write_text("models:\n  critic: claude-sonnet-5\nrules: []\n")
    return p, calls


# ── 0.1.0 behaviour, pinned: Motif depends on every line of it ─────────────────────────────────────

def test_default_raises_without_a_key(isolated):
    p, calls = isolated
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set — check your .env file"):
        config.load_config(p)
    assert calls == [1]


def test_default_loads_with_an_environment_key(isolated, monkeypatch):
    p, _ = isolated
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = config.load_config(p)
    assert cfg == {"models": {"critic": "claude-sonnet-5"}, "rules": [], "loop": {"max_iterations": 3}}


def test_default_loads_with_a_context_key(isolated):
    p, _ = isolated
    with llm.using_key("sk-test"):
        assert config.load_config(p)["loop"]["max_iterations"] == 3


def test_empty_file_gets_defaults(isolated, monkeypatch, tmp_path):
    _, _ = isolated
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert config.load_config(empty) == {"models": {}, "loop": {"max_iterations": 3}}


def test_existing_max_iterations_is_kept(isolated, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = tmp_path / "loop.yaml"
    p.write_text("loop:\n  max_iterations: 5\n")
    assert config.load_config(p)["loop"] == {"max_iterations": 5}


# ── 0.2.0: require_key ─────────────────────────────────────────────────────────────────────────────

def test_require_key_false_loads_without_a_key(isolated):
    p, calls = isolated
    cfg = config.load_config(p, require_key=False)
    assert cfg == {"models": {"critic": "claude-sonnet-5"}, "rules": [], "loop": {"max_iterations": 3}}
    assert calls == [1]                                   # .env is still loaded


def test_require_key_false_is_identical_when_a_key_exists(isolated, monkeypatch):
    p, _ = isolated
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert config.load_config(p, require_key=False) == config.load_config(p)


def test_require_key_is_keyword_only_and_defaults_true():
    import inspect
    param = inspect.signature(config.load_config).parameters["require_key"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is True
