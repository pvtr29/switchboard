"""Budget enforcement helpers: reservations, the global cap, the rate limiter, deadlines.

Reservation (tokens and dollars). Before every sample call the runner asks `BudgetManager`
for a `Reservation`: `est_input` tokens plus an output allowance, the allowance being
`min(agent max_output_tokens, job remaining - est_input, global remaining - est_input)` where
"global remaining" already subtracts every reservation currently in flight. The same is done in
usd via the agent's `Pricing`. The granted allowance travels to the agent as
`StepContext.max_output_tokens` and the agent must not exceed it, so a settled call can never
take more than it reserved: with a well-behaved agent the overshoot is exactly 0, even when
many jobs race for the same global pool. `settle` then releases the unused part and charges
the actual usage; `release` drops a reservation without charging (sample cancelled or
preempted; the tokens a real provider may have billed for an abandoned call are not known, so
they are not counted: documented gap).

When is an `asyncio.Lock` needed? `reserve`, `settle` and `release` are synchronous: they
contain no `await`, and on one event loop nothing else runs between their first and last line,
so the read-modify-write on the counters is atomic WITHOUT a lock. Keep them synchronous. A
lock becomes necessary the moment a critical section spans an `await`, e.g. a reserve that
had to read the store for the job's usage: two jobs could both read "plenty left" and both
reserve. Two places here do span awaits and therefore use a lock: lazy initialisation of the
global usage from the store (`Supervisor`, see `_ensure_budget`) and `TokenBucket.acquire`
(waiting for refill must be serialised to be FIFO-fair).

In-flight reservations live in memory only; a crash drops them. Global usage is rebuilt from
the sum of all jobs' projected usage (`BudgetManager.from_store`), so it survives restarts.
The authoritative record stays `usage_recorded`; `budget_reserved` / `budget_settled` events
are NOT emitted (a reservation lives for one sample call; logging it would double the write
rate for no recovery value).

Global cap reached: every running job is denied at its next reservation and moves to
over_budget. Queued jobs are still admitted (slots are cheap) and go over_budget at their
first reservation; choosing this over refusing admission keeps admission policy-free and the
reason visible in each job's own log.

Active time: `DeadlineWatch` lets the supervisor move the deadline of a step that is already
running when a job is extended.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from switchboard.core import Budget, Dimension, Job, Usage
from switchboard.runtime.agent import Pricing
from switchboard.store import EventStore

DEFAULT_GRACE_S = 5.0
DEFAULT_MIN_OUTPUT_TOKENS = 16


class ReservationDenied(Exception):
    """The caps leave too little for a useful sample call."""

    def __init__(self, dimension: Dimension, scope: Literal["job", "global"]) -> None:
        super().__init__(f"{dimension.value} ({scope})")
        self.dimension = dimension
        self.scope = scope


@dataclass(slots=True)
class Reservation:
    job_id: str
    tokens: int  # reserved total: est_input + max_output_tokens
    usd_micros: int
    max_output_tokens: int  # the output cap granted to the agent
    active: bool = True  # False once settled or released


@dataclass(slots=True)
class BudgetStats:
    reserved_tokens: int = 0
    reserved_usd_micros: int = 0
    used_tokens: int = 0
    used_usd_micros: int = 0
    overshoots: int = 0  # settled calls whose actual usage exceeded the reservation
    overshoot_tokens: int = 0  # total tokens above reservation across those calls
    denials: int = 0


class BudgetManager:
    def __init__(
        self,
        global_budget: Budget | None = None,
        *,
        initial_usage: Usage | None = None,
        min_output_tokens: int = DEFAULT_MIN_OUTPUT_TOKENS,
    ) -> None:
        self._global = global_budget or Budget()
        self.min_output_tokens = min_output_tokens
        self._used_tokens = initial_usage.tokens if initial_usage else 0
        self._used_usd = initial_usage.usd_micros if initial_usage else 0
        self._reserved_tokens = 0
        self._reserved_usd = 0
        self._job_reserved_tokens: dict[str, int] = {}
        self._job_reserved_usd: dict[str, int] = {}
        self.overshoots = 0
        self.overshoot_tokens = 0
        self.denials = 0

    @classmethod
    async def from_store(
        cls,
        store: EventStore,
        global_budget: Budget | None = None,
        *,
        min_output_tokens: int = DEFAULT_MIN_OUTPUT_TOKENS,
    ) -> "BudgetManager":
        """Initialise global usage from the sum of every job's projected usage."""
        total = Usage()
        if global_budget is not None and (
            global_budget.tokens is not None or global_budget.usd_micros is not None
        ):
            for job in await store.list_jobs():
                total = total + job.usage
        return cls(global_budget, initial_usage=total, min_output_tokens=min_output_tokens)

    @property
    def stats(self) -> BudgetStats:
        return BudgetStats(
            self._reserved_tokens,
            self._reserved_usd,
            self._used_tokens,
            self._used_usd,
            self.overshoots,
            self.overshoot_tokens,
            self.denials,
        )

    def extend_global(self, *, tokens: int = 0, usd_micros: int = 0) -> None:
        self._global = self._global.extended(tokens=tokens, usd_micros=usd_micros)

    def global_remaining_tokens(self) -> int | None:
        if self._global.tokens is None:
            return None
        return self._global.tokens - self._used_tokens - self._reserved_tokens

    def global_remaining_usd(self) -> int | None:
        if self._global.usd_micros is None:
            return None
        return self._global.usd_micros - self._used_usd - self._reserved_usd

    def reserve(
        self, job: Job, est_input_tokens: int, max_output_tokens: int, pricing: Pricing
    ) -> Reservation:
        """Reserve for one sample call, capped to what remains. Synchronous: atomic on a loop.

        Raises ReservationDenied (naming dimension and job/global scope) if the output allowance
        would fall under `min_output_tokens` (or the agent's own max, if that is smaller).
        """
        floor = min(self.min_output_tokens, max_output_tokens)
        jid = job.job_id
        rem = job.budget.remaining(job.usage)
        grant = max_output_tokens
        denied: ReservationDenied | None = None

        def cap(
            left: int | None,
            fixed: int,
            per_token: int,
            dim: Dimension,
            scope: Literal["job", "global"],
        ) -> None:
            nonlocal grant, denied
            if left is None:
                return
            room = left - fixed
            # Free output (price 0): only the input has to fit.
            allowed = room // per_token if per_token > 0 else (grant if room >= 0 else -1)
            if allowed < grant:
                grant = allowed
            if allowed < floor and denied is None:
                denied = ReservationDenied(dim, scope)

        in_usd = est_input_tokens * pricing.input_usd_micros
        # Tokens: output allowance = remaining - est_input.
        cap(
            _sub(rem.tokens, self._job_reserved_tokens.get(jid, 0)),
            est_input_tokens,
            1,
            Dimension.TOKENS,
            "job",
        )
        cap(self.global_remaining_tokens(), est_input_tokens, 1, Dimension.TOKENS, "global")
        # Dollars: output allowance = (remaining - est_input cost) / output price.
        cap(
            _sub(rem.usd_micros, self._job_reserved_usd.get(jid, 0)),
            in_usd,
            pricing.output_usd_micros,
            Dimension.USD_MICROS,
            "job",
        )
        cap(
            self.global_remaining_usd(),
            in_usd,
            pricing.output_usd_micros,
            Dimension.USD_MICROS,
            "global",
        )
        if denied is not None:
            self.denials += 1
            raise denied
        usd = in_usd + grant * pricing.output_usd_micros
        tokens = est_input_tokens + grant
        self._reserved_tokens += tokens
        self._reserved_usd += usd
        self._job_reserved_tokens[jid] = self._job_reserved_tokens.get(jid, 0) + tokens
        self._job_reserved_usd[jid] = self._job_reserved_usd.get(jid, 0) + usd
        return Reservation(jid, tokens, usd, grant)

    def _drop(self, r: Reservation) -> bool:
        if not r.active:
            return False
        r.active = False
        self._reserved_tokens -= r.tokens
        self._reserved_usd -= r.usd_micros
        self._job_reserved_tokens[r.job_id] -= r.tokens
        self._job_reserved_usd[r.job_id] -= r.usd_micros
        if self._job_reserved_tokens[r.job_id] == 0:
            del self._job_reserved_tokens[r.job_id]
            del self._job_reserved_usd[r.job_id]
        return True

    def settle(self, r: Reservation, actual: Usage) -> None:
        """Release the reservation and charge `actual` to the global pool. Synchronous."""
        if not self._drop(r):
            return
        self._used_tokens += actual.tokens
        self._used_usd += actual.usd_micros
        if actual.tokens > r.tokens:
            self.overshoots += 1
            self.overshoot_tokens += actual.tokens - r.tokens

    def release(self, r: Reservation) -> None:
        """Drop a reservation without charging (call abandoned). Idempotent."""
        self._drop(r)


def _sub(left: int | None, reserved: int) -> int | None:
    return None if left is None else left - reserved


class TokenBucket:
    """Rate limiter shared by all agents: `rate` tokens/second refill, `capacity` burst.

    `acquire(n)` waits on the loop clock (so looptime tests see virtual time). Fairness comes
    from an `asyncio.Lock`, whose waiters are served FIFO: only the head of the line sleeps for
    the refill, everyone else queues behind the lock, so a big request is never starved by
    small ones. The lock section spans awaits (that is why it is a lock). Cancellation safe:
    tokens are taken in one synchronous step after the wait, so a cancelled waiter consumed
    nothing.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._stamp: float | None = None
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = asyncio.get_running_loop().time()
        if self._stamp is not None:
            self._tokens = min(self.capacity, self._tokens + (now - self._stamp) * self.rate)
        self._stamp = now

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    async def acquire(self, n: float) -> None:
        if n < 0 or n > self.capacity:
            raise ValueError(f"cannot acquire {n} from a bucket of capacity {self.capacity}")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= n - 1e-9:
                    self._tokens = max(self._tokens - n, 0.0)
                    return
                await asyncio.sleep((n - self._tokens) / self.rate)


@dataclass(slots=True)
class DeadlineWatch:
    """Lets the supervisor move the deadline of the step a runner is executing.

    The runner calls `arm` with the timeout context manager of the current scope and the
    "origin" (loop time at which active usage was zero: `loop.time() - usage.active_seconds`;
    constant while the job runs because recorded time equals elapsed time). The deadline is
    `origin + active limit + grace`. `limit_changed` stores the new limit and, if a scope is
    armed, reschedules it.
    """

    grace_s: float = DEFAULT_GRACE_S
    limit: float | None = None
    origin: float = 0.0
    _cm: asyncio.Timeout | None = field(default=None, repr=False)

    def deadline(self) -> float | None:
        return None if self.limit is None else self.origin + self.limit + self.grace_s

    def arm(self, cm: asyncio.Timeout) -> None:
        self._cm = cm
        cm.reschedule(self.deadline())

    def disarm(self) -> None:
        self._cm = None

    def limit_changed(self, limit: float | None) -> None:
        self.limit = limit
        if self._cm is not None:
            self._cm.reschedule(self.deadline())
