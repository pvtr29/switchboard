import asyncio
import logging
from collections.abc import Sequence

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Job, JobState
from switchboard.runtime.agent import ActResult, Message, ModelResponse, StepContext
from switchboard.runtime.slots import SlotPool
from switchboard.runtime.supervisor import Supervisor
from switchboard.store import EventStore

pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


class CountingAgent(FakeAgent):
    """Counts how many agents are between the start of sample and the end of act."""

    active = 0
    peak = 0

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        cls = CountingAgent
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        try:
            return await super().sample(ctx, history)
        except BaseException:
            cls.active -= 1
            raise

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        try:
            return await super().act(ctx, response)
        finally:
            CountingAgent.active -= 1


# ---- the pool itself ----------------------------------------------------------------------


async def test_pool_basic_fifo_and_idempotent_release() -> None:
    pool = SlotPool(2)
    a, b = await pool.acquire(), await pool.acquire()
    order: list[int] = []

    async def waiter(i: int) -> None:
        (await pool.acquire()).release()
        order.append(i)

    tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
    await asyncio.sleep(0)
    assert pool.waiting == 3
    a.release()
    a.release()  # idempotent: must not free a second slot
    await asyncio.sleep(0)
    assert pool.held == 2  # b, plus the first waiter's slot (a's double release freed only one)
    b.release()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2]
    assert (pool.held, pool.waiting, pool.max_seen) == (0, 0, 2)
    with pytest.raises(ValueError, match="capacity"):
        SlotPool(0)


async def test_pool_cancel_while_waiting_leaks_nothing() -> None:
    pool = SlotPool(1)
    held = await pool.acquire()
    t = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    assert pool.waiting == 1
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert pool.waiting == 0
    held.release()
    assert pool.held == 0
    (await pool.acquire()).release()


async def test_pool_cancel_racing_the_grant_leaks_nothing() -> None:
    pool = SlotPool(1)
    held = await pool.acquire()
    t = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    held.release()  # grants the slot to t's future ...
    t.cancel()  # ... and the cancel lands before t resumes
    with pytest.raises(asyncio.CancelledError):
        await t
    assert pool.held == 0
    assert pool.waiting == 0
    slot = await pool.acquire()  # capacity is intact
    slot.release()


# ---- the supervisor -----------------------------------------------------------------------


async def test_limit_five_with_fifty_jobs(
    store: EventStore, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="switchboard.supervisor")
    CountingAgent.active = CountingAgent.peak = 0
    n, limit = 50, 5
    ids = [f"j{i:02d}" for i in range(n)]
    for i, j in enumerate(ids):
        await create_job(store, j, seed=i)
    sup = Supervisor(store, lambda job: CountingAgent(steps=3), max_running=limit)
    samples: list[tuple[int, int]] = []
    stop = asyncio.Event()

    async def probe() -> None:
        while not stop.is_set():
            samples.append((sup.stats.running, sup.stats.waiting))
            await asyncio.sleep(0.001)

    probe_task = asyncio.create_task(probe())
    for j in ids:
        sup.start(j)
    await asyncio.sleep(0.0005)
    assert sup.stats.running == limit
    assert sup.stats.waiting == n - limit
    for j in ids:
        await sup.wait(j)
    stop.set()
    await probe_task
    await sup.aclose()

    assert max(r for r, _ in samples) == limit
    assert all(r <= limit for r, _ in samples)
    assert any(w > 0 for _, w in samples)
    assert CountingAgent.peak <= limit
    assert sup.stats.max_running_seen == limit
    assert (sup.stats.running, sup.stats.waiting) == (0, 0)
    for j in ids:
        job = await store.get_job(j)
        assert job is not None
        assert job.state is JobState.VERIFYING
    admitted = [
        r.getMessage().split("job_id=")[1]
        for r in caplog.records
        if r.getMessage().startswith("job admitted")
    ]
    assert admitted == ids  # FIFO


async def test_waiting_jobs_stay_queued_until_admitted(store: EventStore) -> None:
    for i in range(3):
        await create_job(store, f"j{i}", seed=i)
    sup = Supervisor(store, lambda job: FakeAgent(steps=50), max_running=1)
    for i in range(3):
        sup.start(f"j{i}")
    await asyncio.sleep(0.5)
    states = {}
    for i in range(3):
        job = await store.get_job(f"j{i}")
        assert job is not None
        states[f"j{i}"] = job.state
    assert states["j0"] is JobState.RUNNING
    assert states["j1"] is states["j2"] is JobState.QUEUED
    assert (sup.stats.running, sup.stats.waiting) == (1, 2)
    await sup.aclose()
    assert (sup.stats.running, sup.stats.waiting) == (0, 0)


async def test_cancel_waiting_job_does_not_leak_a_slot(store: EventStore) -> None:
    limit = 5
    ids = [f"j{i}" for i in range(8)]
    for i, j in enumerate(ids):
        await create_job(store, j, seed=i)
    sup = Supervisor(store, lambda job: FakeAgent(steps=4), max_running=limit)
    for j in ids:
        sup.start(j)
    await asyncio.sleep(0.01)
    assert sup.stats.waiting == 3
    await sup.cancel("j6")  # waiting
    await sup.cancel("j7")  # waiting
    assert sup.stats.waiting == 1
    cancelled = await store.get_job("j6")
    assert cancelled is not None
    assert cancelled.state is JobState.CANCELLED
    for j in ids:
        await sup.wait(j)
    assert sup.stats.running == 0
    # capacity is still 5: five fresh long jobs all run at once, the sixth waits
    for i in range(6):
        await create_job(store, f"k{i}", seed=i)
        sup.start(f"k{i}")
    await asyncio.sleep(0.001)
    assert (sup.stats.running, sup.stats.waiting) == (limit, 1)
    await sup.aclose()
    assert sup.stats.running == 0
    j0: Job | None = await store.get_job("j0")
    assert j0 is not None
    assert j0.state is JobState.VERIFYING
