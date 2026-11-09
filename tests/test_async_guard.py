"""The guard fixture's detector must catch the two failures filterwarnings cannot see."""

import asyncio
import gc

from conftest import AsyncioErrorCapture


def _destroy_pending_task() -> None:
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(1000))
        loop.run_until_complete(asyncio.sleep(0))  # let it start and suspend
        del task
    finally:
        loop.close()


def _drop_unretrieved_exception() -> None:
    async def boom() -> None:
        raise RuntimeError("nobody reads this")

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(boom())
        loop.run_until_complete(asyncio.sleep(0))
        assert task.done()
        del task
    finally:
        loop.close()


def test_detects_destroyed_pending_task() -> None:
    with AsyncioErrorCapture(consume=True) as capture:
        _destroy_pending_task()
        gc.collect()
    assert any("Task was destroyed but it is pending" in m for m in capture.messages())


def test_detects_exception_never_retrieved() -> None:
    with AsyncioErrorCapture(consume=True) as capture:
        _drop_unretrieved_exception()
        gc.collect()
    assert any("exception was never retrieved" in m for m in capture.messages())
