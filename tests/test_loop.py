"""etot_core.loop.run_loop. Offline, no model: produce / check / revise are plain functions."""

import pytest

from etot_core.loop import run_loop


class Script:
    """check() returns the next verdict from a list; revise() and produce() count their calls."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.checks = self.revisions = self.produced = 0

    def produce(self, state):
        self.produced += 1
        return {**state, "draft": 0}

    def check(self, state):
        self.checks += 1
        return self.verdicts.pop(0)

    def revise(self, state, verdict):
        self.revisions += 1
        return {**state, "draft": state["draft"] + 1}


def fail(*pairs):
    return {"pass": False, "failures": [{"insight_id": i, "rule": r} for i, r in pairs]}


PASS = {"pass": True, "failures": []}


# ── 0.2.0 behaviour, pinned: Motif depends on it ───────────────────────────────────────────────────

def test_critic_pass_stops_on_the_iteration_it_passes():
    s = Script([fail(("I-01", "unsupported")), PASS])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert (res.stop_reason, res.iterations, s.revisions, res.state["draft"]) == ("critic_pass", 2, 1, 1)
    assert len(res.verdicts) == 2


def test_same_insight_and_rule_twice_is_no_progress():
    s = Script([fail(("I-01", "unsupported")), fail(("I-01", "unsupported")), PASS])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert (res.stop_reason, res.iterations, s.revisions) == ("no_progress", 2, 1)


def test_default_key_is_insight_id_and_rule_only():
    # Details differ, keys do not: Motif's detector compares (insight_id, rule) and nothing else.
    a = {"pass": False, "failures": [{"insight_id": "I-01", "rule": "unsupported", "detail": "first"}]}
    b = {"pass": False, "failures": [{"insight_id": "I-01", "rule": "unsupported", "detail": "second"}]}
    s = Script([a, b])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert res.stop_reason == "no_progress"


def test_changing_failures_run_to_max_iterations_without_a_final_revise():
    s = Script([fail(("I-01", "unsupported")), fail(("I-02", "unsupported")), fail(("I-03", "unsupported"))])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert (res.stop_reason, res.iterations, s.checks, s.revisions) == ("max_iterations", 3, 3, 2)


def test_a_failing_verdict_with_no_failures_never_counts_as_no_progress():
    empty = {"pass": False, "failures": []}
    s = Script([empty, empty, empty])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert res.stop_reason == "max_iterations"


def test_critic_disabled_returns_after_produce():
    s = Script([])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, critic_enabled=False)
    assert (res.stop_reason, res.iterations, s.produced, s.checks) == ("critic_disabled", 0, 1, 0)


def test_logger_records_every_stage():
    class Log:
        def __init__(self):
            self.iterations, self.notes = [], []

        def record_iteration(self, n, state):
            self.iterations.append((n, state["stage"]))

        def note(self, msg):
            self.notes.append(msg)

    log = Log()
    s = Script([fail(("I-01", "unsupported")), PASS])
    run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3, logger=log)
    assert log.iterations == [(0, "produce"), (1, "check"), (1, "revise"), (2, "check")]
    assert log.notes == ["iteration 1: 1 failure(s), pass=False", "iteration 2: 0 failure(s), pass=True"]


# ── 0.3.0: failure_key ─────────────────────────────────────────────────────────────────────────────

def moire_fail(*pairs):
    return {"pass": False, "failures": [{"finding_id": i, "rule": r} for i, r in pairs]}


def test_default_key_collapses_failures_that_do_not_use_insight_id():
    # The defect the second consumer found: failures named by another field all key as (None, rule), so objections
    # to two different findings compare equal and the loop stops no_progress while progress is being made.
    s = Script([moire_fail(("F-001", "wrong_binding")), moire_fail(("F-007", "wrong_binding")), PASS])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3)
    assert res.stop_reason == "no_progress" and res.iterations == 2


def test_failure_key_keeps_distinct_failures_distinct():
    s = Script([moire_fail(("F-001", "wrong_binding")), moire_fail(("F-007", "wrong_binding")), PASS])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3,
                   failure_key=lambda f: (f["finding_id"], f["rule"]))
    assert (res.stop_reason, res.iterations, s.revisions) == ("critic_pass", 3, 2)


def test_failure_key_still_detects_no_progress():
    s = Script([moire_fail(("F-001", "wrong_binding")), moire_fail(("F-001", "wrong_binding")), PASS])
    res = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3,
                   failure_key=lambda f: (f["finding_id"], f["rule"]))
    assert (res.stop_reason, res.iterations) == ("no_progress", 2)


def test_failure_key_is_keyword_only_and_defaults_to_motif():
    import inspect
    p = inspect.signature(run_loop).parameters["failure_key"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    s = Script([fail(("I-01", "unsupported")), fail(("I-01", "unsupported"))])
    explicit = run_loop(state={}, produce=s.produce, check=s.check, revise=s.revise, max_iterations=3,
                        failure_key=lambda f: (f.get("insight_id"), f.get("rule")))
    assert explicit.stop_reason == "no_progress"
