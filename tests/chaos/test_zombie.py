"""Zombie owners: a runner that does not know its job was taken away from it.

A zombie is an owner that stopped (a frozen process, a GC pause, a partition) while another one
took its job over, and then carries on as if nothing happened. Epoch fencing must refuse every
write it makes after the takeover: no step commit, no logged response, no effect intent (so no
external effect), no failing the job on the new owner's behalf.

Two `EventStore`s on one database stand in for two daemon processes: separate writer threads and
connections, separate job caches. That is the case fencing exists for, and the one a stale cache
would get wrong (see SQLiteEventLog: the cache is dropped when another writer appended).
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, Job, JobState, StaleEpoch
from switchboard.runtime.agent import ActResult, ModelResponse, StepContext
from switchboard.runtime.recovery import recover
from switchboard.runtime.runner import run_job
from switchboard.runtime.sinks import SQLiteSink
from switchboard.runtime.supervisor import Supervisor, revoke
from switchboard.store import EventStore

T = EventType
pytestmark = pytest.mark.asyncio
STEPS = 6


class Frozen(FakeAgent):
    """Freezes inside the act phase of `freeze_at` (before its effect) until released."""

    def __init__(self, frozen: asyncio.Event, release: asyncio.Event, freeze_at: int = 1) -> None:
        super().__init__(steps=STEPS, token_interval=0, effect_every=1)
        self.frozen, self.release, self.freeze_at = frozen, release, freeze_at

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        if ctx.step == self.freeze_at and not self.frozen.is_set():
            self.frozen.set()
            await self.release.wait()
        return await super().act(ctx, response)


def fresh() -> FakeAgent:
    return FakeAgent(steps=STEPS, token_interval=0, effect_every=1)


@pytest_asyncio.fixture
async def two(tmp_path: Path) -> AsyncIterator[tuple[EventStore, EventStore, SQLiteSink]]:
    """Two writers on one log (two 'processes') and a shared external sink."""
    db = tmp_path / "log.db"
    async with (
        await EventStore.open(db) as old,
        await EventStore.open(db) as new,
        await SQLiteSink.open(tmp_path / "sink.db") as sink,
    ):
        await create_job(old, "j1", seed=4)
        yield old, new, sink


async def current(store: EventStore) -> Job:
    job = await store.get_job("j1")
    assert job is not None
    return job


def assert_no_writes_after_takeover(events: list[Event], zombie_epoch: int) -> None:
    takeover = next(e.seq or 0 for e in events if e.type is T.LEASE_LOST)
    late = [e.type.value for e in events if e.epoch <= zombie_epoch and (e.seq or 0) > takeover]
    assert late == []


def assert_each_step_once(events: list[Event]) -> None:
    for kind in (T.LLM_RESPONDED, T.STEP_COMPLETED):
        assert [e.step_id for e in events if e.type is kind] == list(range(STEPS))


async def test_a_stale_owner_cannot_write_after_a_takeover(
    two: tuple[EventStore, EventStore, SQLiteSink],
) -> None:
    old, new, sink = two
    frozen, release = asyncio.Event(), asyncio.Event()
    zombie = asyncio.create_task(
        run_job(old, "j1", Frozen(frozen, release), sink=sink, worker="old")
    )
    await frozen.wait()  # step 0 committed, step 1 sampled, act frozen before its effect
    job = await current(new)
    assert (job.state, job.owner, job.epoch) == (JobState.RUNNING, "old", 1)

    # The new process decides the old owner is dead: fence it off, re-queue, run to the end.
    await new.append_group(revoke(job, JobState.QUEUED, "owner presumed dead", 1.0))
    done = await run_job(new, "j1", fresh(), sink=sink, worker="new")
    assert (done.state, done.owner, done.epoch) == (JobState.VERIFYING, "new", 3)

    release.set()  # the zombie wakes up and carries on with step 1
    with pytest.raises(StaleEpoch):
        await zombie
    events = await new.read_job("j1")
    assert_no_writes_after_takeover(events, zombie_epoch=1)
    assert_each_step_once(events)
    counts = sink.counts()
    assert (counts.applications, counts.duplicates) == (STEPS, 0)  # its effect never ran
    assert (await current(new)).state is JobState.VERIFYING


async def test_two_owners_racing_for_one_job_exactly_one_runs_it(
    two: tuple[EventStore, EventStore, SQLiteSink],
) -> None:
    old, new, sink = two
    outcomes = await asyncio.gather(
        run_job(old, "j1", fresh(), sink=sink, worker="a"),
        run_job(new, "j1", fresh(), sink=sink, worker="b"),
        return_exceptions=True,
    )
    finished = [o for o in outcomes if isinstance(o, Job)]
    fenced = [o for o in outcomes if isinstance(o, StaleEpoch)]
    assert (len(finished), len(fenced)) == (1, 1), outcomes
    events = await new.read_job("j1")
    assert [e.payload["worker"] for e in events if e.type is T.LEASE_ACQUIRED] == [
        finished[0].owner
    ]
    assert_each_step_once(events)
    assert sink.counts().duplicates == 0


async def test_a_cancel_from_another_process_fences_the_owner(
    two: tuple[EventStore, EventStore, SQLiteSink],
) -> None:
    old, new, sink = two
    frozen, release = asyncio.Event(), asyncio.Event()
    zombie = asyncio.create_task(run_job(old, "j1", Frozen(frozen, release), sink=sink))
    await frozen.wait()
    canceller = Supervisor(new, lambda job: fresh())
    await canceller.cancel("j1", reason="stop it")
    await canceller.aclose()
    release.set()
    with pytest.raises(StaleEpoch):
        await zombie
    events = await new.read_job("j1")
    assert_no_writes_after_takeover(events, zombie_epoch=1)
    assert events[-1].payload == {"from": "running", "to": "cancelled", "reason": "stop it"}
    assert sink.counts().applications == 1  # step 0 only


async def test_the_supervisor_records_nothing_for_a_fenced_job(
    two: tuple[EventStore, EventStore, SQLiteSink],
) -> None:
    old, new, sink = two
    frozen, release = asyncio.Event(), asyncio.Event()
    sup = Supervisor(
        old, lambda job: Frozen(frozen, release), sink_factory=lambda job: sink, worker_id="old"
    )
    sup.start("j1")
    await frozen.wait()

    # A second daemon starts on the same database and recovers it: the job looks orphaned.
    other = Supervisor(new, lambda job: fresh(), sink_factory=lambda job: sink, worker_id="new")
    report = await recover(new, other)
    assert (report.requeued, report.admitted) == (1, 1)
    await other.wait("j1")
    await other.aclose()

    release.set()
    await sup.wait("j1")
    outcome = sup.outcome("j1")
    assert outcome is not None
    assert isinstance(outcome.error, StaleEpoch)
    assert (sup.stats.fenced, sup.errors) == (1, [])
    await sup.aclose()
    events = await new.read_job("j1")
    assert_no_writes_after_takeover(events, zombie_epoch=1)
    assert not [e for e in events if e.payload.get("to") == "failed"]  # not failed for them
    assert (await current(new)).state is JobState.VERIFYING
    assert sink.counts().duplicates == 0
