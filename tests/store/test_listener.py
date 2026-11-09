import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.core import Event, EventType
from switchboard.store import EventStore

pytestmark = pytest.mark.asyncio


def usage(job_id: str, step: int = 0) -> Event:
    return Event(
        job_id=job_id, epoch=0, ts=0.0, type=EventType.USAGE_RECORDED, payload={"tokens_in": step}
    )


async def test_listener_gets_inserted_events_with_seqs(store: EventStore) -> None:
    seen: list[list[Event]] = []
    store.add_listener(lambda events: seen.append(list(events)))
    await create_job(store, "a")
    await store.append_group([usage("a", 1), usage("a", 2)])
    flat = [e for batch in seen for e in batch]
    assert [e.type for e in flat] == [
        EventType.JOB_CREATED,
        EventType.USAGE_RECORDED,
        EventType.USAGE_RECORDED,
    ]
    assert [e.seq for e in flat] == [1, 2, 3]


async def test_duplicates_and_rejected_groups_are_not_reported(store: EventStore) -> None:
    await create_job(store, "a")
    seen: list[Event] = []
    store.add_listener(seen.extend)
    with pytest.raises(ValueError, match="does not exist"):
        await store.append_group([usage("ghost")])  # no job_created: rejected
    assert seen == []
    llm = Event(job_id="a", epoch=0, step_id=0, ts=0.0, type=EventType.LLM_RESPONDED, payload={})
    await store.append(llm)
    await store.append(llm)  # duplicate (job, type, step): stored once, reported once
    assert [e.type for e in seen] == [EventType.LLM_RESPONDED]


async def test_failing_listener_never_breaks_the_writer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good: list[Sequence[Event]] = []

    def bad(_: Sequence[Event]) -> None:
        raise RuntimeError("listener bug")

    async with await EventStore.open(tmp_path / "sb.db") as s:
        s.add_listener(bad)
        s.add_listener(good.append)
        with caplog.at_level(logging.ERROR, logger="switchboard.store"):
            await create_job(s, "a")
            await s.append(usage("a"))
        assert len(good) == 2  # later listeners still ran, and the appends were acked
        assert (await s.get_job("a")) is not None
    assert "listener bug" in caplog.text
