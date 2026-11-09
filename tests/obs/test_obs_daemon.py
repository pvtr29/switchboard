"""Commit 20's proof: a deliberate blocking call is visible in the daemon's stats AND in the
asyncio debug log."""

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from conftest import wait_state

from switchboard.agents.fake import TokenHook
from switchboard.client.cli import format_stats
from switchboard.client.conn import Client
from switchboard.daemon.server import Daemon
from switchboard.runtime.agent import StepContext

pytestmark = pytest.mark.asyncio

BLOCK_S = 0.3


async def started(
    sock_dir: Path,
    tmp_path: Path,
    *,
    debug_asyncio: bool = False,
    on_token: TokenHook | None = None,
) -> tuple[Daemon, Client]:
    d = Daemon(
        socket_path=sock_dir / "s",
        db_path=tmp_path / "sb.db",
        envs_dir=tmp_path / "envs",
        shutdown_timeout=5.0,
        lag_interval=0.02,
        debug_asyncio=debug_asyncio,
        on_token=on_token,
    )
    await d.start()
    return d, await Client.connect(str(d.socket_path))


@pytest_asyncio.fixture
async def blocked_daemon(sock_dir: Path, tmp_path: Path) -> AsyncIterator[tuple[Daemon, Client]]:
    fired = False

    def block_once(ctx: StepContext, i: int) -> None:
        nonlocal fired
        if not fired:
            fired = True
            time.sleep(BLOCK_S)

    hook: TokenHook = block_once
    d, c = await started(sock_dir, tmp_path, debug_asyncio=True, on_token=hook)
    try:
        yield d, c
    finally:
        await c.aclose()
        await d.shutdown()


async def test_blocking_sleep_shows_in_stats_and_debug_log(
    blocked_daemon: tuple[Daemon, Client], caplog: pytest.LogCaptureFixture
) -> None:
    daemon, client = blocked_daemon
    with caplog.at_level(logging.WARNING, logger="asyncio"):
        job = (await client.request("submit", steps=2, token_interval=0.001))["job_id"]
        await wait_state(client, job, {"verifying"})
        stats = await client.request("stats")
    assert daemon.debug_asyncio
    assert stats["debug_asyncio"] is True
    assert stats["loop_lag"]["max_s"] >= 0.25
    assert stats["loop_lag"]["max_ever_s"] >= 0.25
    slow = [m for m in caplog.messages if re.search(r"Executing .* took 0\.[3-9]\d* seconds", m)]
    assert slow, caplog.messages
    assert "job:" in slow[0]  # the job's task is the culprit
    assert "max=" in format_stats(stats)


async def test_idle_daemon_reports_near_zero_lag_and_the_new_stats(
    daemon: Daemon, client: Client
) -> None:
    await asyncio.sleep(0.4)
    stats = await client.request("stats")
    lag = stats["loop_lag"]
    assert lag["samples"] >= 3
    assert lag["max_s"] < 0.2  # nothing blocks: a real stall is >= 0.3
    assert stats["debug_asyncio"] is True  # pytest runs every loop in debug mode
    assert set(stats["store_latency"]) == {"commit_s", "ack_s", "batch_events", "batch_groups"}
    assert stats["store"]["queue_depth"] == 0
    text = format_stats(stats)
    for needle in ("loop_lag:", "store_commit:", "store_ack:", "slots:", "bus:", "recovery:"):
        assert needle in text


async def test_monitor_task_is_stopped_on_shutdown(sock_dir: Path, tmp_path: Path) -> None:
    d, c = await started(sock_dir, tmp_path)
    assert d.lag.running
    await c.aclose()
    await d.shutdown()
    assert not d.lag.running
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "sb-loop-lag"]


async def test_store_latency_in_stats_after_a_job(client: Client) -> None:
    job = (await client.request("submit", steps=3, token_interval=0.001))["job_id"]
    await wait_state(client, job, {"verifying"})
    lat = (await client.request("stats"))["store_latency"]
    assert lat["commit_s"]["count"] >= 3
    assert lat["ack_s"]["count"] >= 3
    assert 0 < lat["commit_s"]["p50"] <= lat["commit_s"]["p99"]
    assert lat["batch_events"]["mean"] >= 1
