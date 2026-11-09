import asyncio
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Budget, Event, EventType, Job, JobKind, JobState, Usage
from switchboard.runtime.agent import ActResult, ModelResponse, Pricing, StepContext
from switchboard.runtime.budget import (
    BudgetManager,
    ReservationDenied,
    TokenBucket,
)
from switchboard.runtime.supervisor import NotExtendable, Supervisor
from switchboard.store import EventStore

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


class TimedAgent(FakeAgent):
    """Every act sleeps `act_s` virtual seconds (`slow_attempt0_s` on attempt 0, if given)."""

    def __init__(
        self, *, act_s: float = 1.0, slow_attempt0_s: float | None = None, steps: int = 20
    ) -> None:
        super().__init__(steps=steps, token_interval=0.0)
        self.act_s = act_s
        self.slow_attempt0_s = slow_attempt0_s

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        slow = self.slow_attempt0_s is not None and ctx.step == 0 and ctx.attempt == 0
        await asyncio.sleep(self.slow_attempt0_s if slow and self.slow_attempt0_s else self.act_s)
        return await super().act(ctx, response)


async def job_of(store: EventStore, job_id: str = "j1") -> Job:
    job = await store.get_job(job_id)
    assert job is not None
    return job


def of_type(events: list[Event], t: EventType) -> list[Event]:
    return [e for e in events if e.type is t]


def changes(events: list[Event]) -> list[tuple[str, str, str]]:
    return [
        (e.payload["from"], e.payload["to"], e.payload["reason"])
        for e in of_type(events, T.STATE_CHANGED)
    ]


# ---- Part A: active time ---------------------------------------------------------------------


async def test_queued_and_paused_time_is_not_charged(store: EventStore) -> None:
    await create_job(store, "a")
    await create_job(store, "b")
    sup = Supervisor(
        store, lambda job: TimedAgent(steps=3), clock=asyncio.get_running_loop().time, max_running=1
    )
    sup.start("a")
    sup.start("b")  # queued behind a for ~3 virtual seconds
    await sup.wait("a")
    await sup.wait("b")
    for jid in ("a", "b"):
        job = await job_of(store, jid)
        assert job.state is JobState.VERIFYING
        assert job.usage.active_seconds == pytest.approx(3.0)  # b's queue time is not in it

    await create_job(store, "p")
    sup.start("p")
    await asyncio.sleep(1.5)
    await sup.pause("p")
    await sup.wait("p")
    before = (await job_of(store, "p")).usage.active_seconds
    assert (await job_of(store, "p")).state is JobState.PAUSED
    await asyncio.sleep(10_000)
    assert (await job_of(store, "p")).usage.active_seconds == before
    await sup.resume("p")
    await sup.wait("p")
    job = await job_of(store, "p")
    assert job.state is JobState.VERIFYING
    assert job.usage.active_seconds == pytest.approx(3.0)
    await sup.aclose()


async def test_boundary_check_warning_and_rearm_after_extend(store: EventStore) -> None:
    await create_job(store, budget={"active_seconds": 3.5})
    sup = Supervisor(store, lambda job: TimedAgent(steps=8), clock=asyncio.get_running_loop().time)
    sup.start("j1")
    await sup.wait("j1")
    job = await job_of(store, "j1")
    assert job.state is JobState.OVER_BUDGET
    assert job.last_step == 3  # 4 steps of 1 s: the 4th ended past 3.5, the boundary stopped it
    events = await store.read_job("j1")
    assert changes(events)[-1] == ("running", "over_budget", "budget exhausted: active_seconds")
    warnings = of_type(events, T.BUDGET_WARNING)
    assert [(w.payload["dimension"], w.payload["limit"]) for w in warnings] == [
        ("active_seconds", 3.5)
    ]  # once, at 3.0 >= 2.8, not repeated at every step

    await sup.extend("j1", active_seconds=4.5)  # limit 8.0: re-queued, re-admitted
    await sup.wait("j1")
    job = await job_of(store, "j1")
    assert job.state is JobState.VERIFYING
    assert job.budget.active_seconds == 8.0
    warnings = of_type(await store.read_job("j1"), T.BUDGET_WARNING)
    assert [w.payload["limit"] for w in warnings] == [3.5, 8.0]  # re-armed for the new limit
    await sup.aclose()


async def test_hard_preemption_within_grace_commits_nothing(store: EventStore) -> None:
    await create_job(store, budget={"active_seconds": 3.0})
    loop = asyncio.get_running_loop()
    sup = Supervisor(store, lambda job: TimedAgent(act_s=1000.0), clock=loop.time)
    t0 = loop.time()
    sup.start("j1")
    await sup.wait("j1")
    job = await job_of(store, "j1")
    events = await store.read_job("j1")
    assert job.state is JobState.OVER_BUDGET
    over = [e for e in of_type(events, T.STATE_CHANGED) if e.payload["to"] == "over_budget"]
    assert over[0].payload["reason"] == "active_seconds (preempted)"
    assert over[0].ts <= t0 + 3.0 + 5.0 + 1e-6  # deadline + grace
    assert over[0].ts >= t0 + 3.0
    assert of_type(events, T.STEP_COMPLETED) == []  # nothing of the step committed
    assert job.last_step is None
    # The wasted active seconds were recorded, exactly once.
    assert job.usage.active_seconds == pytest.approx(over[0].ts - t0)
    await sup.aclose()


async def test_custom_grace(store: EventStore) -> None:
    await create_job(store, budget={"active_seconds": 3.0})
    loop = asyncio.get_running_loop()
    sup = Supervisor(store, lambda job: TimedAgent(act_s=1000.0), clock=loop.time, grace_s=0.5)
    t0 = loop.time()
    sup.start("j1")
    await sup.wait("j1")
    assert (await job_of(store, "j1")).usage.active_seconds == pytest.approx(3.5)
    assert loop.time() == pytest.approx(t0 + 3.5)
    await sup.aclose()


async def test_extension_mid_step_reschedules_the_deadline(store: EventStore) -> None:
    await create_job(store, budget={"active_seconds": 3.0})
    sup = Supervisor(
        store, lambda job: TimedAgent(act_s=20.0, steps=1), clock=asyncio.get_running_loop().time
    )
    sup.start("j1")  # would be preempted at 8 s
    await asyncio.sleep(2.0)
    await sup.extend("j1", active_seconds=30.0)
    await sup.wait("j1")
    job = await job_of(store, "j1")
    assert job.state is JobState.VERIFYING
    assert job.usage.active_seconds == pytest.approx(20.0)
    assert not [
        e
        for e in await store.read_job("j1")
        if e.type is T.STATE_CHANGED and e.payload["to"] == "over_budget"
    ]
    await sup.aclose()


async def test_extending_an_over_budget_job_requeues_and_it_completes(store: EventStore) -> None:
    await create_job(store, budget={"active_seconds": 3.0})
    sup = Supervisor(
        store,
        lambda job: TimedAgent(slow_attempt0_s=1000.0, act_s=1.0, steps=2),
        clock=asyncio.get_running_loop().time,
    )
    sup.start("j1")
    await sup.wait("j1")
    assert (await job_of(store, "j1")).state is JobState.OVER_BUDGET
    await sup.extend("j1", active_seconds=100.0)
    await sup.wait("j1")
    job = await job_of(store, "j1")
    assert job.state is JobState.VERIFYING
    events = await store.read_job("j1")
    assert [(f, t) for f, t, _ in changes(events) if t in ("over_budget", "queued")] == [
        ("running", "over_budget"),
        ("over_budget", "queued"),
    ]
    assert len(of_type(events, T.LLM_RESPONDED)) == 2  # step 0 was not sampled again
    assert [e.payload["attempt"] for e in of_type(events, T.STEP_STARTED)][:2] == [0, 1]
    assert job.usage.active_seconds == pytest.approx(8.0 + 2.0)  # wasted 8 s + two 1 s steps


async def test_extend_terminal_job_is_refused(store: EventStore) -> None:
    await create_job(store)
    sup = Supervisor(store, lambda job: TimedAgent(), clock=asyncio.get_running_loop().time)
    await sup.cancel("j1")
    with pytest.raises(NotExtendable):
        await sup.extend("j1", tokens=1)
    await sup.aclose()


async def test_unlimited_stays_unlimited(store: EventStore) -> None:
    await create_job(store, budget={"tokens": 100})
    sup = Supervisor(store, lambda job: TimedAgent(), clock=asyncio.get_running_loop().time)
    await sup.extend("j1", active_seconds=5.0, tokens=50, usd_micros=7)
    assert (await job_of(store, "j1")).budget == Budget(tokens=150)
    await sup.aclose()


async def test_restart_reconstructs_remaining_time_from_the_log(tmp_path: Path) -> None:
    path = tmp_path / "sb.db"
    loop = asyncio.get_running_loop()

    async def supervisor(s: EventStore) -> Supervisor:
        return Supervisor(s, lambda job: TimedAgent(steps=10), clock=loop.time)

    async with await EventStore.open(path) as s1:
        await create_job(s1, budget={"active_seconds": 2.5})
        sup1 = await supervisor(s1)
        sup1.start("j1")
        await asyncio.sleep(1.5)  # step 0 committed (1 s), step 1 in flight
        await sup1.aclose()
        assert (await job_of(s1, "j1")).usage.active_seconds == pytest.approx(1.0)
    await asyncio.sleep(10_000)  # daemon down: not charged
    async with await EventStore.open(path) as s2:
        sup2 = await supervisor(s2)
        sup2.start("j1")
        await sup2.wait("j1")
        job = await job_of(s2, "j1")
        # 1 s (before) + steps 1 and 2 (2 s): the budget carried over, a fresh 2.5 s would
        # have allowed 2 more steps.
        assert job.state is JobState.OVER_BUDGET
        assert job.last_step == 2
        assert job.usage.active_seconds == pytest.approx(3.0)
        await sup2.aclose()


# ---- Part B: reservations ---------------------------------------------------------------------


def make_job(budget: Budget, usage: Usage | None = None, job_id: str = "j") -> Job:
    return Job(
        job_id=job_id,
        kind=JobKind.FAKE,
        task_ref="t",
        budget=budget,
        usage=usage or Usage(),
        last_seq=1,
        created_at=0.0,
        updated_at=0.0,
    )


async def test_reserve_is_capped_by_job_and_global_and_settle_releases() -> None:
    bm = BudgetManager(Budget(tokens=300), min_output_tokens=16)
    free = Pricing()
    job = make_job(Budget(tokens=200), Usage(tokens=100))
    r = bm.reserve(job, 50, 1000, free)
    assert r.max_output_tokens == 50  # job: 100 left - 50 input
    assert r.tokens == 100
    r2 = bm.reserve(make_job(Budget(), job_id="k"), 50, 1000, free)
    assert r2.max_output_tokens == 150  # global: 300 - 100 reserved - 50 input
    with pytest.raises(ReservationDenied) as denied:
        bm.reserve(make_job(Budget(), job_id="m"), 50, 1000, free)  # 50 left, input alone
    assert (denied.value.dimension.value, denied.value.scope) == ("tokens", "global")
    bm.settle(r, Usage(tokens=60))
    bm.settle(r, Usage(tokens=60))  # idempotent
    bm.release(r2)
    s = bm.stats
    assert (s.reserved_tokens, s.used_tokens, s.overshoots) == (0, 60, 0)
    bm.settle(bm.reserve(job, 50, 10, free), Usage(tokens=70))  # an agent over its cap
    assert bm.stats.overshoot_tokens == 10


async def test_usd_reservation_uses_the_agent_price() -> None:
    bm = BudgetManager(Budget(usd_micros=1000))
    r = bm.reserve(make_job(Budget()), 100, 10_000, Pricing(2, 5))
    assert r.max_output_tokens == (1000 - 200) // 5
    assert r.usd_micros == 200 + 160 * 5


async def test_global_usage_survives_restart(store: EventStore) -> None:
    await create_job(store, "a")
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=2),
        clock=asyncio.get_running_loop().time,
        global_budget=Budget(tokens=10_000),
    )
    sup.start("a")
    await sup.wait("a")
    await sup.aclose()
    used = (await job_of(store, "a")).usage.tokens
    assert used > 0
    bm = await BudgetManager.from_store(store, Budget(tokens=10_000))
    assert bm.stats.used_tokens == used
    assert bm.global_remaining_tokens() == 10_000 - used


@pytest.mark.parametrize("cap", [2_500, 12_000])
async def test_global_token_cap_is_never_overshot(store: EventStore, cap: int) -> None:
    n = 100
    ids = [f"j{i}" for i in range(n)]
    for jid in ids:
        await create_job(store, jid, seed=hash(jid) % 1000)
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=3, token_interval=0.001),
        clock=asyncio.get_running_loop().time,
        max_running=n,
        global_budget=Budget(tokens=cap),
    )
    for jid in ids:
        sup.start(jid)
    await asyncio.gather(*(sup.wait(j) for j in ids))
    jobs = [await job_of(store, j) for j in ids]
    total = sum(j.usage.tokens for j in jobs)
    assert total <= cap  # the global cap, jointly
    bm = sup.budget_manager
    assert bm is not None
    assert bm.stats.overshoots == 0  # exactly 0 for reserved calls
    assert bm.stats.reserved_tokens == 0  # nothing left in flight
    assert bm.stats.used_tokens == total
    assert {j.state for j in jobs} <= {JobState.OVER_BUDGET, JobState.VERIFYING}
    # Whoever the cap did not stop finished; with the small cap at most a handful did.
    assert JobState.OVER_BUDGET in {j.state for j in jobs}
    assert sum(j.state is JobState.VERIFYING for j in jobs) <= cap // (3 * 53)
    reasons = {c[2] for j in ids for c in changes(await store.read_job(j)) if c[1] == "over_budget"}
    assert reasons == {"budget exhausted: tokens (global)"}
    # Observable slack: how close to the cap the pool got (less than one reservation).
    assert cap - total < 58 + 8


async def test_per_job_token_cap_is_respected(store: EventStore) -> None:
    ids = [f"j{i}" for i in range(10)]
    for jid in ids:
        await create_job(store, jid, seed=1, budget={"tokens": 200})
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=10, token_interval=0.001),
        clock=asyncio.get_running_loop().time,
        max_running=10,
        global_budget=Budget(tokens=1_000_000),
    )
    for jid in ids:
        sup.start(jid)
    await asyncio.gather(*(sup.wait(j) for j in ids))
    for jid in ids:
        job = await job_of(store, jid)
        assert job.state is JobState.OVER_BUDGET
        assert job.usage.tokens <= 200
        assert changes(await store.read_job(jid))[-1][2].startswith("budget exhausted: tokens")
    bm = sup.budget_manager
    assert bm is not None
    assert bm.stats.overshoots == 0


async def test_usd_cap_is_never_overshot(store: EventStore) -> None:
    n, cap = 30, 5_000
    ids = [f"j{i}" for i in range(n)]
    for jid in ids:
        await create_job(store, jid, seed=3)
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=4, token_interval=0.001),
        clock=asyncio.get_running_loop().time,
        max_running=n,
        global_budget=Budget(usd_micros=cap),
    )
    for jid in ids:
        sup.start(jid)
    await asyncio.gather(*(sup.wait(j) for j in ids))
    jobs = [await job_of(store, j) for j in ids]
    total = sum(j.usage.usd_micros for j in jobs)
    assert 0 < total <= cap
    assert {j.state for j in jobs} == {JobState.OVER_BUDGET}
    bm = sup.budget_manager
    assert bm is not None
    assert bm.stats.reserved_usd_micros == 0


async def test_rate_limiter_is_applied_before_each_sample(store: EventStore) -> None:
    await create_job(store, "j1")
    loop = asyncio.get_running_loop()
    bucket = TokenBucket(rate=100.0, capacity=100.0)
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=3, token_interval=0.0),
        clock=loop.time,
        rate_limiter=bucket,
    )
    t0 = loop.time()
    sup.start("j1")
    await sup.wait("j1")
    # Three samples of 58 reserved tokens against a burst of 100 refilling at 100/s.
    assert loop.time() - t0 == pytest.approx((3 * 58 - 100) / 100.0)
    await sup.aclose()


# ---- TokenBucket ------------------------------------------------------------------------------


async def test_token_bucket_rate_over_virtual_time() -> None:
    loop = asyncio.get_running_loop()
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    t0 = loop.time()
    await bucket.acquire(10)  # burst: immediate
    assert loop.time() == pytest.approx(t0)
    for _ in range(5):
        await bucket.acquire(10)
    assert loop.time() - t0 == pytest.approx(5.0)  # 50 tokens at 10/s
    with pytest.raises(ValueError, match="capacity"):
        await bucket.acquire(11)


async def test_token_bucket_is_fifo() -> None:
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    await bucket.acquire(10)
    order: list[str] = []

    async def take(name: str, n: float) -> None:
        await bucket.acquire(n)
        order.append(name)

    tasks = [
        asyncio.create_task(take("big", 10)),
        asyncio.create_task(take("small1", 1)),
        asyncio.create_task(take("small2", 1)),
    ]
    await asyncio.gather(*tasks)
    assert order == ["big", "small1", "small2"]  # small ones never overtake the head


async def test_token_bucket_cancel_while_waiting_consumes_nothing() -> None:
    loop = asyncio.get_running_loop()
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    await bucket.acquire(10)
    t0 = loop.time()
    waiter = asyncio.create_task(bucket.acquire(10))
    await asyncio.sleep(0.5)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await bucket.acquire(10)
    assert loop.time() - t0 == pytest.approx(1.0)  # not delayed by the cancelled request
