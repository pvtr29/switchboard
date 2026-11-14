"""NDJSON encoding of events for client feeds."""

import json
from pathlib import Path

import pytest

from switchboard.core import Event, EventType
from switchboard.daemon.protocol import dumps_line, event_line, event_message
from switchboard.store import EventStore


@pytest.mark.asyncio
async def test_event_line_splices_the_stored_payload_unchanged(tmp_path: Path) -> None:
    payload = {"z": [1, {"é": None}], "a": 'quote " and \\ backslash', "f": 0.1}
    async with await EventStore.open(tmp_path / "p.db") as store:
        seen: list[Event] = []
        store.add_listener(seen.extend)
        await store.append(
            Event(
                job_id='j"1',
                epoch=0,
                ts=1.25,
                type=EventType.JOB_CREATED,
                payload={"kind": "fake", "task_ref": "t", "seed": 0, **payload},
            )
        )
        (read,) = await store.read_since(0)
    (live,) = seen
    for e in (live, read):
        assert e.payload_json is not None
        line = event_line("sub\n1", e)
        assert line.endswith(b"\n")
        assert line.count(b"\n") == 1
        assert json.loads(line) == json.loads(dumps_line(event_message("sub\n1", e)))
    assert live == read  # payload_json is not part of equality


def test_event_line_without_stored_payload_falls_back_to_a_full_encode() -> None:
    e = Event(seq=3, job_id="j", epoch=0, ts=1.0, type=EventType.USAGE_RECORDED, payload={"x": 1})
    assert event_line("s", e) == dumps_line(event_message("s", e))
