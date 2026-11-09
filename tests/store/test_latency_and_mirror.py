"""Write-latency stats, the in-memory job mirror behind `get_job`, and group-commit linger."""

import asyncio
from pathlib import Path

import pytest

from switchboard.core import Event, EventType, JobState, Usage
from switchboard.store import EventStore, writer

pytestmark = pytest.mark.asyncio


def created(job: str) -> Event:
    return Event(
        job_id=job,
        epoch=0,
        ts=0.5,
        type=EventType.JOB_CREATED,
        payload={"kind": "fake", "task_ref": "t", "budget": {}, "seed": 0},
    )


def usage(job: str, tokens: int = 1) -> Event:
    return Event(
        job_id=job,
        epoch=0,
        ts=1.0,
        type=EventType.USAGE_RECORDED,
        payload={"tokens_in": tokens, "tokens_out": 0},
    )


async def test_latency_counts_match_the_work_done(store: EventStore) -> None:
    await store.append_group([created("a"), created("b")])
    await asyncio.gather(*(store.append(usage("a")) for _ in range(30)))
    st, lat = store.stats, store.latency
    assert lat.commit_s.count == st.transactions
    assert lat.batch_groups.count == st.transactions
    assert lat.ack_s.count == 31  # one per append call that was acked
    assert lat.batch_events.mean * lat.batch_events.count == pytest.approx(st.events_written)
    assert lat.batch_events.max_ever == st.max_batch_seen
    assert 0 < lat.commit_s.p50 <= lat.commit_s.p99 <= lat.commit_s.max
    assert lat.ack_s.p50 > 0
    assert lat.ack_s.max >= lat.commit_s.p50  # an ack includes at least a commit


async def test_latency_memory_is_bounded_and_resettable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(writer, "LATENCY_WINDOW", 8)
    async with await EventStore.open(tmp_path / "sb.db") as s:
        await s.append(created("a"))
        for _ in range(40):
            await s.append(usage("a"))
        lat = s.latency
        assert lat.commit_s.count == 41
        assert lat.commit_s.window == 8
        assert lat.ack_s.window == 8
        s.reset_latency()
        assert s.latency.commit_s.count == 0


async def test_get_job_sees_own_writes_without_touching_the_reader(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object) -> None:
        raise AssertionError("reader used for a job that was just written")

    await store.append(created("a"))
    assert store._rlog is not None
    monkeypatch.setattr(store._rlog, "get_job", boom)
    job = await store.get_job("a")
    assert job is not None
    assert job.state is JobState.QUEUED
    await store.append(usage("a", 5))
    job = await store.get_job("a")
    assert job is not None
    assert job.usage.tokens == 5  # mirrored before the ack, so never stale for the writer


async def test_mirror_matches_the_table_and_ignores_rejected_groups(store: EventStore) -> None:
    await store.append_group([created("a"), created("b")])
    await store.append(usage("a", 3))
    bad = Event(
        job_id="b",
        epoch=0,
        ts=2.0,
        type=EventType.STATE_CHANGED,
        payload={"from": "running", "to": "done", "reason": "x"},  # b is queued: illegal
    )
    with pytest.raises(ValueError, match="from=running"):
        await store.append_group([usage("b", 9), bad])
    table = {j.job_id: j for j in await store.list_jobs()}
    assert await store.get_job("a") == table["a"]
    assert await store.get_job("b") == table["b"]
    assert table["b"].usage == Usage()  # the rejected group left no trace in either


async def test_evicted_jobs_are_read_from_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(writer, "JOB_MIRROR_SIZE", 2)
    async with await EventStore.open(tmp_path / "sb.db") as s:
        for j in ("a", "b", "c"):
            await s.append(created(j))
        assert len(s._jobs) == 2
        assert "a" not in s._jobs
        job = await s.get_job("a")
        assert job is not None
        assert job.job_id == "a"
        assert await s.get_job("zzz") is None


async def test_get_job_after_close_raises(tmp_path: Path) -> None:
    s = await EventStore.open(tmp_path / "sb.db")
    await s.append(created("a"))
    await s.aclose()
    with pytest.raises(writer.StoreClosed):
        await s.get_job("a")


async def test_linger_groups_callers_only_under_load(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db", linger=0.05) as s:
        await s.append_group([created(f"j{i}") for i in range(20)])
        assert s.stats.transactions == 1
        t0 = asyncio.get_running_loop().time()
        await s.append(usage("j0"))  # idle store: the previous batch had one group, no linger
        assert asyncio.get_running_loop().time() - t0 < 0.04
        before = s.stats.transactions
        results = await asyncio.gather(*(s.append(usage(f"j{i}")) for i in range(20)))
        assert all(r.seq is not None for r in results)
        assert s.stats.transactions - before <= 2  # not 20
        assert s.stats.max_batch_seen >= 19


async def test_negative_linger_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="linger"):
        EventStore(tmp_path / "x.db", linger=-1.0)
