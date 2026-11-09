"""Smoke tests that the async test tooling behaves as later commits will rely on."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_event_loop_runs() -> None:
    assert await asyncio.sleep(0, result=42) == 42


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_looptime_fast_forwards_sleep() -> None:
    # With looptime, loop.time() is virtual: an hour of sleeping costs no wall time,
    # and asyncio.timeout deadlines follow the same virtual clock.
    loop = asyncio.get_running_loop()
    start = loop.time()
    async with asyncio.timeout(3601):
        await asyncio.sleep(3600)
    assert loop.time() - start == pytest.approx(3600)
