"""Immutable domain models. Pure data: no I/O, no clocks (callers pass timestamps in)."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from switchboard.core.states import JobState, transition


class JobKind(StrEnum):
    FAKE = "fake"
    ROLLOUT = "rollout"


class Verdict(StrEnum):
    # `pass` is a keyword, hence the trailing underscore on the member name.
    PASS_ = "pass"
    FAIL = "fail"
    SUSPECTED_HACK = "suspected_hack"


class EventType(StrEnum):
    JOB_CREATED = "job_created"
    STATE_CHANGED = "state_changed"
    STEP_COMPLETED = "step_completed"
    LLM_RESPONDED = "llm_responded"
    STEP_STARTED = "step_started"
    RESTORED = "restored"
    USAGE_RECORDED = "usage_recorded"
    BUDGET_RESERVED = "budget_reserved"
    BUDGET_SETTLED = "budget_settled"
    BUDGET_WARNING = "budget_warning"
    EFFECT_INTENT = "effect_intent"
    EFFECT_OUTCOME = "effect_outcome"
    LEASE_ACQUIRED = "lease_acquired"
    LEASE_LOST = "lease_lost"
    VERDICT_RECORDED = "verdict_recorded"


STEP_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.STEP_COMPLETED, EventType.LLM_RESPONDED, EventType.STEP_STARTED}
)
"""Event types that carry a step_id (and only these)."""


class Dimension(StrEnum):
    ACTIVE_SECONDS = "active_seconds"
    TOKENS = "tokens"
    USD_MICROS = "usd_micros"


def _check_non_negative(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Resources consumed. Money is integer micro-dollars: floats drift when summed."""

    active_seconds: float = 0.0
    tokens: int = 0
    usd_micros: int = 0

    def __post_init__(self) -> None:
        _check_non_negative(
            active_seconds=self.active_seconds, tokens=self.tokens, usd_micros=self.usd_micros
        )

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            active_seconds=self.active_seconds + other.active_seconds,
            tokens=self.tokens + other.tokens,
            usd_micros=self.usd_micros + other.usd_micros,
        )

    def get(self, dim: Dimension) -> float:
        match dim:
            case Dimension.ACTIVE_SECONDS:
                return self.active_seconds
            case Dimension.TOKENS:
                return self.tokens
            case Dimension.USD_MICROS:
                return self.usd_micros


@dataclass(frozen=True, slots=True, kw_only=True)
class Remaining:
    """Budget left per dimension; None means unlimited."""

    active_seconds: float | None = None
    tokens: int | None = None
    usd_micros: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Budget:
    """Limits per dimension; None means unlimited."""

    active_seconds: float | None = None
    tokens: int | None = None
    usd_micros: int | None = None

    def __post_init__(self) -> None:
        _check_non_negative(
            active_seconds=self.active_seconds, tokens=self.tokens, usd_micros=self.usd_micros
        )

    def limit(self, dim: Dimension) -> float | None:
        match dim:
            case Dimension.ACTIVE_SECONDS:
                return self.active_seconds
            case Dimension.TOKENS:
                return self.tokens
            case Dimension.USD_MICROS:
                return self.usd_micros

    def remaining(self, usage: Usage) -> Remaining:
        # Written per field so int dimensions stay int and float stays float.
        secs, tok, usd = self.active_seconds, self.tokens, self.usd_micros
        return Remaining(
            active_seconds=None if secs is None else max(secs - usage.active_seconds, 0.0),
            tokens=None if tok is None else max(tok - usage.tokens, 0),
            usd_micros=None if usd is None else max(usd - usage.usd_micros, 0),
        )

    def exhausted(self, usage: Usage) -> tuple[Dimension, ...]:
        """Dimensions where usage has reached the limit (a limit of 0 is always exhausted)."""
        return tuple(
            d for d in Dimension if (lim := self.limit(d)) is not None and usage.get(d) >= lim
        )

    def warnings(self, usage: Usage, threshold: float = 0.8) -> tuple[Dimension, ...]:
        """Dimensions at or above threshold * limit but not yet exhausted."""
        return tuple(
            d
            for d in Dimension
            if (lim := self.limit(d)) is not None
            and usage.get(d) < lim
            and usage.get(d) >= threshold * lim
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """One immutable fact in a job's log. `seq` is None until the store assigns it."""

    seq: int | None = None
    job_id: str
    epoch: int
    step_id: int | None = None
    ts: float
    type: EventType
    v: int = 1
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be non-empty")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if self.v < 1:
            raise ValueError("v must be >= 1")
        if (self.type in STEP_EVENT_TYPES) != (self.step_id is not None):
            raise ValueError(
                "step_id must be set if and only if type is step_completed, llm_responded "
                "or step_started"
            )
        if self.step_id is not None and self.step_id < 0:
            raise ValueError("step_id must be >= 0")
        # Copy so the caller's dict cannot mutate a frozen event; frozen needs object.__setattr__.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    job_id: str
    kind: JobKind
    task_ref: str
    state: JobState = JobState.QUEUED
    verdict: Verdict | None = None
    budget: Budget
    usage: Usage = Usage()
    owner: str | None = None
    lease_until: float | None = None
    epoch: int = 0
    last_step: int | None = None
    last_seq: int
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        if self.verdict is not None and self.state is not JobState.DONE:
            raise ValueError("verdict may only be set when state is done")

    def with_state(self, dst: JobState, *, at: float) -> "Job":
        """Return a copy moved to `dst`; raises IllegalTransition if the edge is not allowed."""
        return dataclasses.replace(self, state=transition(self.state, dst), updated_at=at)
