import asyncio
import time

import pytest

from switchboard.obs import LoopLagMonitor


@pytest.mark.asyncio
async def test_idle_loop_has_near_zero_lag() -> None:
    async with LoopLagMonitor(interval=0.01, window=100) as mon:
        await asyncio.sleep(0.5)
        stats = mon.stats()
    assert stats.samples >= 20
    assert stats.p50_s < 0.02
    assert stats.p99_s < 0.1  # generous: shared CI machines hiccup, a blocked loop is >= 0.3


@pytest.mark.asyncio
async def test_a_blocking_call_shows_up_as_lag() -> None:
    async with LoopLagMonitor(interval=0.02, window=100) as mon:
        await asyncio.sleep(0.1)
        time.sleep(0.3)  # noqa: ASYNC251 - the point of the test
        await asyncio.sleep(0.1)
        stats = mon.stats()
    assert stats.max_s >= 0.25
    assert stats.max_ever_s >= 0.25
    assert stats.p50_s < 0.1  # one stall among many samples


@pytest.mark.asyncio
async def test_window_forgets_but_max_ever_does_not() -> None:
    mon = LoopLagMonitor(interval=0.01, window=5)
    mon.start()
    await asyncio.sleep(0.05)
    time.sleep(0.15)  # noqa: ASYNC251
    await asyncio.sleep(0.3)  # many quiet samples push the stall out of the 5-sample window
    stats = mon.stats()
    await mon.stop()
    assert stats.window == 5
    assert stats.max_s < 0.1
    assert stats.max_ever_s >= 0.1
    mon.reset()
    assert mon.stats().samples == 0


@pytest.mark.asyncio
async def test_start_stop_are_idempotent_and_leave_no_task() -> None:
    before = asyncio.all_tasks()
    mon = LoopLagMonitor(interval=0.01)
    mon.start()
    mon.start()
    assert mon.running
    await asyncio.sleep(0.03)
    await mon.stop()
    await mon.stop()
    assert not mon.running
    assert asyncio.all_tasks() == before


def test_bad_interval() -> None:
    with pytest.raises(ValueError, match="interval"):
        LoopLagMonitor(interval=0)
