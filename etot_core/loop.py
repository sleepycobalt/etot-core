"""
Generic agentic loop controller.

A tool supplies three callables:

    produce(state)          -> state      first draft (or full pipeline) into state
    check(state)            -> verdict    dict with at least {"pass": bool, "failures": [...]}
    revise(state, verdict)  -> state      address failures, return new state

and the controller runs:

    state = produce(state)
    for i in 1..max_iterations:
        verdict = check(state)
        if verdict.pass: stop("critic_pass")
        if unchanged_since_last(verdict): stop("no_progress")
        state = revise(state, verdict)
    stop("max_iterations")

The stop condition is explicit and logged, because *which* condition fires
is one of the research questions.

"Unchanged" compares the set of failure keys between consecutive checks.
failure_key(failure) names a failure; the default, (insight_id, rule), is
Motif's vocabulary. A tool whose failures are identified otherwise passes its
own key, so distinct objections never collapse into one and stop the loop
while progress is being made.
"""

from dataclasses import dataclass, field
from typing import Callable, Hashable


@dataclass
class LoopResult:
    state: dict
    iterations: int
    stop_reason: str
    verdicts: list = field(default_factory=list)


def _motif_key(failure: dict) -> Hashable:
    return (failure.get("insight_id"), failure.get("rule"))


def run_loop(*, state: dict, produce: Callable, check: Callable, revise: Callable,
             max_iterations: int = 3, logger=None, critic_enabled: bool = True,
             failure_key: Callable[[dict], Hashable] | None = None) -> LoopResult:
    key = failure_key or _motif_key
    state = produce(state)
    if logger:
        logger.record_iteration(0, {"stage": "produce", "state": state})

    if not critic_enabled:
        return LoopResult(state=state, iterations=0, stop_reason="critic_disabled")

    verdicts = []
    prev_failure_keys = None
    for i in range(1, max_iterations + 1):
        verdict = check(state)
        verdicts.append(verdict)
        failures = verdict.get("failures", [])
        if logger:
            logger.record_iteration(i, {"stage": "check", "verdict": verdict, "state": state})
            logger.note(f"iteration {i}: {len(failures)} failure(s), pass={verdict.get('pass')}")

        if verdict.get("pass"):
            return LoopResult(state, i, "critic_pass", verdicts)

        failure_keys = frozenset(key(f) for f in failures)
        if failure_keys and failure_keys == prev_failure_keys:
            return LoopResult(state, i, "no_progress", verdicts)
        prev_failure_keys = failure_keys

        if i == max_iterations:
            break
        state = revise(state, verdict)
        if logger:
            logger.record_iteration(i, {"stage": "revise", "state": state})

    return LoopResult(state, max_iterations, "max_iterations", verdicts)
