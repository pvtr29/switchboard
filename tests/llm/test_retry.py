import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from switchboard.llm.retry import (
    Outcome,
    RetryPolicy,
    UnavailabilityTracker,
    Verdict,
    run_with_fallback,
)
from switchboard.llm.types import AttemptRecord, BadRequest, Unavailable

pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


class Down(Exception):
    pass


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, s: float) -> None:
        self.delays.append(s)


def classify(exc: Exception) -> tuple[Verdict, str]:
    if isinstance(exc, Down):
        return Verdict.RETRY, f"down: {exc}"
    if isinstance(exc, TimeoutError):
        return Verdict.RETRY, f"timeout: {exc}"
    if isinstance(exc, KeyError):
        return Verdict.NEXT_MODEL, "gone"
    return Verdict.RAISE, "bug"


def scripted(script: dict[str, list[object]]) -> tuple[Callable[[str], Awaitable[str]], list[str]]:
    calls: list[str] = []

    async def attempt(model: str) -> str:
        calls.append(model)
        step = script[model].pop(0)
        if isinstance(step, Exception):
            raise step
        if step == "hang":
            await asyncio.Event().wait()
        return str(step)

    return attempt, calls


FLAT = RetryPolicy(attempts_per_model=3, base_delay_s=1.0, max_delay_s=4.0, jitter_floor=1.0)


async def run(
    models: list[str],
    attempt: Callable[[str], Awaitable[str]],
    policy: RetryPolicy = FLAT,
    **kw: Any,
) -> tuple[Outcome[str], list[float]]:
    sleeps = Sleeps()
    out = await run_with_fallback(
        models, attempt, classify, policy, deadline_s=10, sleep=sleeps, **kw
    )
    return out, sleeps.delays


async def test_success_first_try_leaves_no_record() -> None:
    attempt, calls = scripted({"a": ["ok"]})
    out, delays = await run(["a"], attempt)
    assert (out.value, out.model, out.failed_attempts, delays, calls) == ("ok", "a", (), [], ["a"])


async def test_retry_then_success() -> None:
    attempt, calls = scripted({"a": [Down("503"), "ok"]})
    out, delays = await run(["a", "b"], attempt)
    assert out.value == "ok"
    assert out.model == "a"
    assert delays == [1.0]
    assert [r.action for r in out.failed_attempts] == ["retry"]
    assert out.failed_attempts[0].attempt == 1
    assert calls == ["a", "a"]


async def test_backoff_doubles_and_is_capped() -> None:
    policy = RetryPolicy(attempts_per_model=5, base_delay_s=1.0, max_delay_s=4.0, jitter_floor=1.0)
    attempt, _ = scripted({"a": [Down("x")] * 4 + ["ok"]})
    _, delays = await run(["a"], attempt, policy)
    assert delays == [1.0, 2.0, 4.0, 4.0]


async def test_jitter_stays_in_range() -> None:
    policy = RetryPolicy(base_delay_s=8.0, max_delay_s=8.0, jitter_floor=0.5)
    rng = random.Random(3)
    assert all(4.0 <= policy.backoff(1, rng) <= 8.0 for _ in range(100))


async def test_sustained_failure_falls_back_to_the_next_model() -> None:
    attempt, calls = scripted({"a": [Down("503")] * 3, "b": ["ok"]})
    out, delays = await run(["a", "b"], attempt)
    assert out.model == "b"
    assert calls == ["a", "a", "a", "b"]
    assert [(r.model, r.action) for r in out.failed_attempts] == [
        ("a", "retry"),
        ("a", "retry"),
        ("a", "next_model"),
    ]
    assert delays == [1.0, 2.0]  # no sleep before switching models


async def test_all_models_down_is_unavailable_with_the_full_record() -> None:
    attempt, calls = scripted({"a": [Down("503")] * 3, "b": [Down("504")] * 3})
    sleeps = Sleeps()
    with pytest.raises(Unavailable) as ei:
        await run_with_fallback(
            ["a", "b"], attempt, classify, FLAT, deadline_s=10, sleep=sleeps, rng=random.Random(0)
        )
    assert len(calls) == 6
    assert len(ei.value.attempts) == 6
    assert ei.value.attempts[-1].action == "give_up"
    assert "504" in str(ei.value)
    assert "a, b" in str(ei.value)


async def test_non_retryable_propagates_immediately() -> None:
    attempt, calls = scripted({"a": [ValueError("bug")], "b": ["ok"]})
    with pytest.raises(ValueError, match="bug"):
        await run(["a", "b"], attempt)
    assert calls == ["a"]


async def test_llm_errors_from_the_attempt_propagate_unclassified() -> None:
    attempt, calls = scripted({"a": [BadRequest("400")], "b": ["ok"]})
    with pytest.raises(BadRequest):
        await run(["a", "b"], attempt)
    assert calls == ["a"]


async def test_next_model_verdict_skips_remaining_attempts() -> None:
    attempt, calls = scripted({"a": [KeyError("404")], "b": ["ok"]})
    out, delays = await run(["a", "b"], attempt)
    assert out.model == "b"
    assert calls == ["a", "b"]
    assert delays == []


async def test_per_attempt_deadline_turns_a_hang_into_a_retry() -> None:
    attempt, calls = scripted({"a": ["hang", "ok"]})
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    sleeps = Sleeps()
    out = await run_with_fallback(
        ["a"], attempt, classify, FLAT, deadline_s=7.0, sleep=sleeps, rng=random.Random(0)
    )
    assert out.value == "ok"
    assert loop.time() - t0 == pytest.approx(7.0)  # the hang was cut at the deadline
    assert "within 7s" in out.failed_attempts[0].error
    assert calls == ["a", "a"]


async def test_max_total_stops_retrying_early() -> None:
    now = [0.0]

    async def sleep(s: float) -> None:
        now[0] += s

    attempt, calls = scripted({"a": [Down("x")] * 3, "b": ["ok"]})
    policy = RetryPolicy(attempts_per_model=3, base_delay_s=10, jitter_floor=1.0, max_total_s=5.0)

    async def timed(model: str) -> str:
        now[0] += 6.0  # every attempt "takes" 6 s: over the total budget after the first
        return await attempt(model)

    out = await run_with_fallback(
        ["a", "b"], timed, classify, policy, deadline_s=10, sleep=sleep, clock=lambda: now[0]
    )
    assert out.model == "b"
    assert calls == ["a", "b"]


async def test_observer_sees_every_failed_attempt() -> None:
    seen: list[AttemptRecord] = []
    attempt, _ = scripted({"a": [Down("x"), "ok"]})
    await run(["a"], attempt, observer=seen.append)
    assert [r.action for r in seen] == ["retry"]


async def test_tracker_counts_recent_failures_per_model() -> None:
    now = [0.0]
    tracker = UnavailabilityTracker(window_s=60, clock=lambda: now[0])
    rec = AttemptRecord("m", 1, "e", True, 0.0, "retry")
    for _ in range(3):
        tracker.record(rec)
    assert tracker.recent_failures("m") == 3
    assert tracker.degraded("m")
    assert not tracker.degraded("other")
    now[0] = 61.0
    assert tracker.recent_failures("m") == 0
    assert not tracker.degraded("m")


async def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="attempts_per_model"):
        RetryPolicy(attempts_per_model=0)
