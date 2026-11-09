import asyncio
from collections.abc import Sequence

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import EventType, JobState
from switchboard.runtime.agent import Message, ModelResponse, StepContext
from switchboard.runtime.runner import run_job
from switchboard.runtime.supervisor import Supervisor, SupervisorClosed
from switchboard.store import EventStore

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


async def test_run_job_should_stop_returns_at_a_boundary_and_writes_nothing(
    store: EventStore,
) -> None:
    await create_job(store, "j1")
    agent = FakeAgent(steps=50)
    stop_after = 2

    def should_stop() -> bool:
        return len(agent.act_calls) >= stop_after

    job = await run_job(store, "j1", agent, should_stop=should_stop)
    assert job.state is JobState.RUNNING  # not paused, not failed: recovery can resume it
    events = await store.read_job("j1")
    completed = [e.step_id for e in events if e.type is T.STEP_COMPLETED]
    assert completed == [0, 1]
    assert {e.step_id for e in events if e.type is T.LLM_RESPONDED} == {0, 1}
    assert not [e for e in events if e.type is T.STATE_CHANGED and e.payload["to"] == "paused"]
    # and the job really is resumable
    done = await run_job(store, "j1", FakeAgent(steps=4))
    assert done.state is JobState.VERIFYING


async def test_supervisor_drain_stops_everything_without_writing(store: EventStore) -> None:
    for i in range(4):
        await create_job(store, f"j{i}", seed=i)
    sup = Supervisor(store, lambda job: FakeAgent(steps=1000), max_running=3)
    for i in range(4):
        sup.start(f"j{i}")
    await asyncio.sleep(1.0)
    assert sup.stats.running == 3
    sup.drain()
    assert await sup.wait_idle(5.0)
    marker = await store.last_seq()
    await asyncio.sleep(5.0)
    assert await store.last_seq() == marker  # nothing is written after the boundary stop
    states = {j: (await store.get_job(j)) for j in ("j0", "j1", "j2", "j3")}
    # the three admitted jobs stay running; the one waiting for a slot never started
    assert sorted(j.state.value for j in states.values() if j) == [
        "queued",
        "running",
        "running",
        "running",
    ]
    for j in ("j0", "j1", "j2"):
        events = await store.read_job(j)
        responded = {e.step_id for e in events if e.type is T.LLM_RESPONDED}
        completed = {e.step_id for e in events if e.type is T.STEP_COMPLETED}
        assert completed <= responded
        assert completed == set(range(len(completed)))
    with pytest.raises(SupervisorClosed):
        sup.start("j3")
    await sup.aclose()


async def test_wait_idle_times_out_when_a_job_does_not_reach_a_boundary(
    store: EventStore,
) -> None:
    class Stuck(FakeAgent):
        async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
            await asyncio.sleep(3600)
            return await super().sample(ctx, history)

    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: Stuck(steps=3))
    sup.start("j1")
    await asyncio.sleep(1)
    sup.drain()
    assert not await sup.wait_idle(2.0)
    await sup.aclose()  # cancels the stuck job, writing nothing
    job = await store.get_job("j1")
    assert job is not None
    assert job.state is JobState.RUNNING
