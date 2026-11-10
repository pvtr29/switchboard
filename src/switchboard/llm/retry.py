"""One retry/backoff/fallback policy, independent of any provider SDK.

`run_with_fallback` tries the models in order. Each model gets up to
`policy.attempts_per_model` attempts, each under its own hard deadline (`asyncio.timeout`),
separated by capped exponential backoff with jitter. Only what the classifier calls RETRY is
retried; NEXT_MODEL skips straight to the next model (e.g. 404 model not found); RAISE
propagates at once (400s, programming errors). When every model is exhausted the result is
`Unavailable`, carrying the record of every failed attempt.

The SDK's own retries must be off: this is the only policy (docs/spike-notes.md).
"""

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from switchboard.llm.types import AttemptRecord, LLMError, Unavailable


class Verdict(StrEnum):
    RETRY = "retry"
    NEXT_MODEL = "next_model"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts_per_model: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    jitter_floor: float = 0.5  # delay is scaled by a factor drawn from [floor, 1]
    max_total_s: float | None = None  # stop starting new attempts after this long overall

    def __post_init__(self) -> None:
        if self.attempts_per_model < 1:
            raise ValueError("attempts_per_model must be >= 1")
        if not 0.0 <= self.jitter_floor <= 1.0:
            raise ValueError("jitter_floor must be in [0, 1]")

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Delay after the `attempt`-th failed attempt (1-based)."""
        raw = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return float(raw * rng.uniform(self.jitter_floor, 1.0))


type Classifier = Callable[[Exception], tuple[Verdict, str]]
"""Maps an exception to (verdict, short reason)."""
type Observer = Callable[[AttemptRecord], None]


@dataclass(frozen=True, slots=True)
class Outcome[T]:
    value: T
    model: str
    failed_attempts: tuple[AttemptRecord, ...]


async def run_with_fallback[T](
    models: Sequence[str],
    attempt: Callable[[str], Awaitable[T]],
    classify: Classifier,
    policy: RetryPolicy,
    *,
    deadline_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
    observer: Observer | None = None,
) -> Outcome[T]:
    if not models:
        raise ValueError("no models")
    rng = rng or random.Random()
    started = clock()
    failed: list[AttemptRecord] = []

    def note(rec: AttemptRecord) -> None:
        failed.append(rec)
        if observer is not None:
            observer(rec)

    def out_of_time() -> bool:
        return policy.max_total_s is not None and clock() - started >= policy.max_total_s

    for mi, model in enumerate(models):
        last_model = mi == len(models) - 1
        for n in range(1, policy.attempts_per_model + 1):
            try:
                async with asyncio.timeout(deadline_s):
                    value = await attempt(model)
            except LLMError:
                raise
            except Exception as exc:
                if isinstance(exc, TimeoutError) and not str(exc):
                    exc = TimeoutError(f"no complete answer within {deadline_s:g}s")
                verdict, reason = classify(exc)
                if verdict is Verdict.RAISE:
                    raise
                last_attempt = n == policy.attempts_per_model or verdict is Verdict.NEXT_MODEL
                exhausted = last_attempt or out_of_time()
                if exhausted and last_model:
                    action = "give_up"
                elif exhausted:
                    action = "next_model"
                else:
                    action = "retry"
                delay = policy.backoff(n, rng) if action == "retry" else 0.0
                note(AttemptRecord(model, n, reason, True, delay, action))
                if action == "give_up":
                    raise Unavailable(
                        f"all models unavailable ({', '.join(models)}); last error: {reason}",
                        attempts=tuple(failed),
                    ) from exc
                if action == "next_model":
                    break
                await sleep(delay)
            else:
                return Outcome(value, model, tuple(failed))
    raise AssertionError("unreachable")  # pragma: no cover


class UnavailabilityTracker:
    """Counts recent failed attempts per model, so a scheduler can back off globally instead of
    every agent hammering a sick API. Pass `tracker.record` as the client's observer; share one
    tracker between all clients."""

    def __init__(
        self, *, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[float]] = {}

    def record(self, rec: AttemptRecord) -> None:
        self._events.setdefault(rec.model, deque()).append(self._clock())

    def recent_failures(self, model: str) -> int:
        q = self._events.get(model)
        if not q:
            return 0
        cutoff = self._clock() - self.window_s
        while q and q[0] < cutoff:
            q.popleft()
        return len(q)

    def degraded(self, model: str, *, threshold: int = 3) -> bool:
        return self.recent_failures(model) >= threshold
