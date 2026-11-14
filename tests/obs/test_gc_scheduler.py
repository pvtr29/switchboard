"""GcScheduler: the loop runs young collections and freezes survivors; full sweeps find cycles
among frozen objects; stop() restores the collector."""

import asyncio
import gc
import weakref
from pathlib import Path

import pytest

from switchboard.client.cli import format_stats
from switchboard.client.conn import Client
from switchboard.daemon.server import Daemon
from switchboard.obs import GcScheduler

pytestmark = pytest.mark.asyncio


class Node:
    def __init__(self) -> None:
        self.other: Node | None = None


def make_cycle() -> weakref.ref[Node]:
    a, b = Node(), Node()
    a.other, b.other = b, a
    return weakref.ref(a)


async def test_takes_over_collection_and_restores_it() -> None:
    assert gc.isenabled()
    sched = GcScheduler(young_s=0.01)
    sched.start()
    try:
        assert not gc.isenabled()
        await asyncio.sleep(0.1)
        st = sched.stats()
        assert st.young.count >= 3
        assert st.frozen > 0  # survivors moved to the permanent generation
    finally:
        await sched.stop()
    assert gc.isenabled()
    assert gc.get_freeze_count() == 0
    await sched.stop()  # idempotent


async def test_young_collection_frees_a_fresh_cycle() -> None:
    sched = GcScheduler(young_s=3600)  # driven by hand below
    sched.start()
    try:
        ref = make_cycle()
        sched.collect_young()
        assert ref() is None
        assert sched.stats().garbage_found >= 2
    finally:
        await sched.stop()


async def test_a_cycle_formed_among_frozen_objects_waits_for_the_full_sweep() -> None:
    sched = GcScheduler(young_s=3600)
    sched.start()
    try:
        a, b = Node(), Node()
        sched.collect_young()  # a and b survive and are frozen
        a.other, b.other = b, a
        ref = weakref.ref(a)
        del a, b
        sched.collect_young()
        assert ref() is not None  # frozen: young collections never look at it
        sched.sweep()
        assert ref() is None
        assert sched.stats().full_sweeps == 1
    finally:
        await sched.stop()


async def test_full_sweeps_run_on_their_interval() -> None:
    sched = GcScheduler(young_s=0.01, full_every_s=0.03)
    sched.start()
    try:
        await asyncio.sleep(0.15)
        assert sched.stats().full_sweeps >= 2
    finally:
        await sched.stop()


async def test_rejects_non_positive_intervals() -> None:
    with pytest.raises(ValueError, match="> 0"):
        GcScheduler(young_s=0)


async def test_the_daemon_runs_it_and_reports_it(sock_dir: Path, tmp_path: Path) -> None:
    d = Daemon(
        socket_path=sock_dir / "s",
        db_path=tmp_path / "sb.db",
        envs_dir=tmp_path / "envs",
        shutdown_timeout=5.0,
        gc_scheduler=GcScheduler(young_s=0.01),
    )
    await d.start()
    try:
        assert not gc.isenabled()
        async with await Client.connect(str(sock_dir / "s")) as c:
            await asyncio.sleep(0.05)
            st = await c.request("stats")
        assert st["gc"]["young"]["count"] > 0
        assert "gc: young p50=" in format_stats(st)
    finally:
        await d.shutdown()
    assert gc.isenabled()
    assert gc.get_freeze_count() == 0
