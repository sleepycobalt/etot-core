# etot-core

Shared engine for [ETOT](https://etot.design) agentic loops: a `produce → check → revise` controller, a structured Claude client, a run logger, and config loading. It knows nothing about any one tool's prompts, rules, or corpus — those live beside each tool that uses it.

[Motif](https://github.com/sleepycobalt/motif) is the first tool built on it.

## What's in it

```
etot_core/
    loop.py      generic plan -> act -> check -> revise -> stop controller
    llm.py       thin Claude client: one call, structured JSON out, usage tracked
    logger.py    per-run directory with every prompt, response, and metric saved
    config.py    load YAML config with model roles and critic rules
```

## Install

```bash
pip install etot-core
```

Python 3.10+. Requires an Anthropic API key at call time (`ANTHROPIC_API_KEY` or a context-scoped key passed to `core.llm`).

## Use it in a new tool

```python
from etot_core.loop import run_loop
from etot_core.logger import RunLogger
from etot_core import llm

result = run_loop(
    state,
    produce=produce,   # state -> state: first draft into state
    check=check,       # state -> verdict: {"pass": bool, "failures": [...]}
    revise=revise,      # state, verdict -> state: address failures
    max_iterations=3,
)
```

Call models through `llm.call`, log through `RunLogger`, and keep tool-specific prompts, rules, and corpus handling in your own package — `etot_core` never talks to a tool's domain, and a tool never re-implements the loop.

## License

MIT. See [LICENSE](LICENSE).
