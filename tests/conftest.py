import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from switchboard.core import Event, EventType
from switchboard.store import EventStore


@pytest.fixture(scope="session", autouse=True)
def _main_thread_event_loop() -> Iterator[None]:
    # looptime's pytest_pyfunc_call hook calls policy.get_event_loop() for every test. In a sync
    # test with no loop set, Python 3.12 emits a DeprecationWarning (fatal under
    # filterwarnings=error) or implicitly creates a loop that is never closed. Owning one
    # explicit loop for the session avoids both, whatever order the tests run in.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    asyncio.set_event_loop(None)
    loop.close()


async def create_job(
    store: EventStore,
    job_id: str = "j1",
    *,
    seed: int = 7,
    instructions: Sequence[Mapping[str, Any]] = (),
    config_hash: str = "",
    budget: Mapping[str, Any] | None = None,
) -> None:
    """Append job_created for `job_id` (state queued)."""
    await store.append(
        Event(
            job_id=job_id,
            epoch=0,
            ts=0.0,
            type=EventType.JOB_CREATED,
            payload={
                "kind": "fake",
                "task_ref": "t",
                "budget": dict(budget or {}),
                "seed": seed,
                "instructions": [dict(m) for m in instructions],
                "config_hash": config_hash,
            },
        )
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[EventStore]:
    async with await EventStore.open(tmp_path / "sb.db") as s:
        yield s
