"""Zombie owners: two runners for the same job at the same time (a stale owner that does not
know it was replaced).

Without leases (phase 5) nothing stops the second runner from running. What holds today is the
database: unique indexes make a second llm_responded / step_completed / effect_intent for the
same step or key a DUPLICATE the runner survives, and the idempotent sink applies each key at
most once however many runners call it. What does NOT hold yet is epoch fencing (a write from a
stale epoch being rejected); that test is a strict xfail until phase 5.
"""

import asyncio
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState
from switchboard.runtime.agent import ActResult, ModelResponse, StepContext
from switchboard.runtime.runner import run_job
from switchboard.runtime.sinks import SQLiteSink
from switchboard.store import EventStore

T = EventType
pytestmark = pytest.mark.asyncio
STEPS = 6


async def test_two_runners_apply_each_effect_at_most_once(
    store: EventStore, tmp_path: Path
) -> None:
    await create_job(store, "j1", seed=4)
    seen = 0

    def stop_after_two_steps() -> bool:
        nonlocal seen
        seen += 1
        return seen > 2

    async with await SQLiteSink.open(tmp_path / "sink.db") as sink:
        first = await run_job(
            store,
            "j1",
            FakeAgent(steps=STEPS, token_interval=0, effect_every=1),
            sink=sink,
            should_stop=stop_after_two_steps,
        )
        assert first.state is JobState.RUNNING  # the "stale owner" stopped mid-job
        # now two owners resume the same running job concurrently
        jobs = await asyncio.gather(
            *(
                run_job(
                    store,
                    "j1",
                    FakeAgent(steps=STEPS, token_interval=0.001, effect_every=1),
                    sink=sink,
                )
                for _ in range(2)
            )
        )
        assert {j.state for j in jobs} == {JobState.VERIFYING}
        per_key = sink.counts()
        assert per_key.duplicates == 0
        assert per_key.applications == STEPS
    events = await store.read_job("j1")
    for kind in (T.LLM_RESPONDED, T.STEP_COMPLETED):
        assert [e.step_id for e in events if e.type is kind] == list(range(STEPS))
    intents = [e.payload["effect_id"] for e in events if e.type is T.EFFECT_INTENT]
    assert len(intents) == len(set(intents)) == STEPS


@pytest.mark.xfail(
    strict=True,
    reason="epoch fencing needs leases (phase 5): today a stale epoch's writes are accepted",
)
async def test_stale_epoch_writes_are_rejected(store: EventStore) -> None:
    await create_job(store, "j1", seed=4)
    in_sample = asyncio.Event()
    release = asyncio.Event()

    class Blocked(FakeAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            in_sample.set()
            await release.wait()  # the zombie is paused (GC, partition) inside its step
            return await super().act(ctx, response)

    zombie = asyncio.create_task(run_job(store, "j1", Blocked(steps=3, token_interval=0)))
    await in_sample.wait()
    # a new owner takes the job over: epoch 1
    await store.append(
        Event(
            job_id="j1",
            epoch=1,
            ts=1.0,
            type=T.LEASE_ACQUIRED,
            payload={"worker": "w2", "lease_until": 99.0},
        )
    )
    release.set()
    await asyncio.wait({zombie})
    stale = [e for e in await store.read_job("j1") if e.epoch == 0 and e.type is T.STEP_COMPLETED]
    assert stale == []  # the zombie's commit must have been refused
