"""
Thin Claude client. Every agent goes through `call` so that usage,
latency, and raw prompts/responses are captured uniformly.
"""

import contextvars
import json
import re
import time
from contextlib import contextmanager

from anthropic import Anthropic

# Approximate $/MTok, used only for the running cost estimate in logs.
# Update from https://docs.claude.com/en/docs/about-claude/models when needed.
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5-1": (10.0, 50.0),
}

_client = None
# A client bound to the current context (thread or task). Set by `using_key` so a
# hosted job runs on its caller's key; concurrent jobs with different keys never mix.
_ctx_client: contextvars.ContextVar = contextvars.ContextVar("motif_anthropic_client", default=None)


def client() -> Anthropic:
    c = _ctx_client.get()
    if c is not None:
        return c
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def has_context_key() -> bool:
    return _ctx_client.get() is not None


@contextmanager
def using_key(api_key: str):
    """Route every `call` inside the block through a client built from api_key.
    The key lives only in that client object for the block's duration; it is never logged."""
    if not api_key or not api_key.strip():
        raise ValueError("empty API key")
    token = _ctx_client.set(Anthropic(api_key=api_key.strip()))
    try:
        yield
    finally:
        _ctx_client.reset(token)


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (in_tok * pin + out_tok * pout) / 1_000_000


def extract_json(text: str):
    """Pull the first JSON object/array out of a response, tolerating fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found in response")
    return json.loads(text[start:])


def call(*, model: str, system: str, user: str, max_tokens: int = 8000,
         json_out: bool = True, logger=None, label: str = "call") -> dict:
    """
    One request to Claude. Returns a dict:
        text, data (parsed JSON or None), model, in_tok, out_tok, cost, seconds
    If a logger is supplied, prompt and response are written to the run dir.
    """
    t0 = time.time()
    # Streaming is required by the SDK for large max_tokens (long requests);
    # we stream and collect the final message so callers see a normal response.
    with client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        resp = stream.get_final_message()
    seconds = time.time() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    block_types = [getattr(b, "type", "?") for b in resp.content]
    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
    result = {
        "label": label,
        "model": model,
        "text": text,
        "data": None,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "cost": estimate_cost(model, in_tok, out_tok),
        "seconds": round(seconds, 2),
        "stop_reason": resp.stop_reason,
        "block_types": block_types,
    }
    if json_out:
        try:
            result["data"] = extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            result["json_error"] = str(e)
    if logger:
        logger.record_call(label, system, user, result)
    return result
