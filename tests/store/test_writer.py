import asyncio
import itertools
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from switchboard.core import Event, EventType
from switchboard.store import (
    AppendResult,
    AppendStatus,
    EventStore,
    GroupResult,
    StoreClosed,
    WriterCrashed,
)


def ev(job: str = "j", step: int | None = None, **payload: object) -> Event:
    t = EventType.STEP_COMPLETED if step is not None else EventType.USAGE_RECORDED
    return Event(job_id=job, epoch=0, step_id=step, ts=1.0, type=t, payload=payload)


async def make_jobs(store: EventStore, *jobs: str) -> None:
    """job_created for each id: every other event type requires its job to exist."""
    created = [
        Event(
            job_id=j,
            epoch=0,
            ts=0.5,
            type=EventType.JOB_CREATED,
            payload={"kind": "fake", "task_ref": "t", "budget": {}, "seed": 0},
        )
        for j in jobs
    ]
    await store.append_group(created)


async def wait_for(flag: threading.Event) -> None:
    assert await asyncio.to_thread(flag.wait, 5)


def patch_append_groups(
    store: EventStore,
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[
        [Callable[[Sequence[Sequence[Event]]], list[GroupResult]], Sequence[Sequence[Event]]],
        list[GroupResult],
    ],
) -> None:
    real = store._log.append_groups
    monkeypatch.setattr(store._log, "append_groups", lambda groups: wrapper(real, groups))


@pytest.mark.asyncio
async def test_200_producers_10k_events(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db", max_batch=512) as store:
        await make_jobs(store, *(f"j{i}" for i in range(200)))

        async def producer(i: int) -> list[int]:
            seqs: list[int] = []
            for n in range(50):
                r = await store.append(ev(job=f"j{i}", n=n))
                assert r.status is AppendStatus.INSERTED
                assert r.seq is not None
                seqs.append(r.seq)
            return seqs

        per_caller = await asyncio.gather(*(producer(i) for i in range(200)))
        all_seqs = [s for seqs in per_caller for s in seqs]
        assert len(set(all_seqs)) == 10_000
        for seqs in per_caller:
            assert seqs == sorted(seqs)
        events = await store.read_since(200, limit=20_000)  # after the 200 job_created
        assert len(events) == 10_000
        got = [e.seq for e in events if e.seq is not None]
        assert len(got) == 10_000
        assert all(a < b for a, b in itertools.pairwise(got))
        st = store.stats
        assert st.events_written == 10_200
        assert st.transactions < 10_000
        assert st.max_batch_seen <= 512
        assert await store.last_seq() == got[-1]
        assert len(await store.read_job("j7")) == 51


@pytest.mark.asyncio
async def test_group_is_atomic_and_consecutive_under_load(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db", max_batch=8) as store:
        await make_jobs(store, "g", "big", *(f"n{i}" for i in range(10)))

        async def noise(i: int) -> None:
            for n in range(30):
                await store.append(ev(job=f"n{i}", n=n))

        async def groups() -> list[list[AppendResult]]:
            return [
                await store.append_group([ev(job="g", step=3 * k + i) for i in range(3)])
                for k in range(20)
            ]

        _, res = await asyncio.gather(asyncio.gather(*(noise(i) for i in range(10))), groups())
        for g in res:
            seqs = [r.seq for r in g if r.seq is not None]
            assert len(seqs) == 3
            assert seqs == list(range(seqs[0], seqs[0] + 3))
        # oversized group (> max_batch) is still committed whole
        big = await store.append_group([ev(job="big", step=i) for i in range(20)])
        first = big[0].seq
        assert first is not None
        assert [r.seq for r in big] == list(range(first, first + 20))


@pytest.mark.asyncio
async def test_duplicate_only_affects_its_caller(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db") as store:
        await make_jobs(store, "j", "other")
        first = await store.append(ev(step=1))
        assert first.status is AppendStatus.INSERTED
        results = await asyncio.gather(
            store.append(ev(step=1)),
            store.append(ev(step=2)),
            store.append(ev(job="other", step=1)),
            store.append(ev(n=1)),
        )
        assert [r.status for r in results] == [
            AppendStatus.DUPLICATE,
            AppendStatus.INSERTED,
            AppendStatus.INSERTED,
            AppendStatus.INSERTED,
        ]
        assert results[0].seq is None
        assert store.stats.duplicates == 1


@pytest.mark.asyncio
async def test_invalid_event_only_fails_its_caller(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db") as store:
        await make_jobs(store, "ok", "bad", "j")
        bad_payload = ev(x=object())
        good = [store.append(ev(job="ok", n=i)) for i in range(20)]
        bad = store.append_group([ev(job="bad", n=1), bad_payload])
        bad_seq = store.append(
            Event(
                seq=5,
                job_id="bad",
                epoch=0,
                step_id=None,
                ts=1.0,
                type=EventType.USAGE_RECORDED,
                payload={},
            )
        )
        res = await asyncio.gather(*good, bad, bad_seq, return_exceptions=True)
        assert all(isinstance(r, AppendResult) for r in res[:20])
        assert isinstance(res[20], ValueError)
        assert isinstance(res[21], ValueError)
        assert len(await store.read_job("bad")) == 1  # only its job_created
        assert len(await store.read_job("ok")) == 21


@pytest.mark.asyncio
async def test_backpressure_bounds_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with await EventStore.open(tmp_path / "sb.db", max_batch=1, queue_size=2) as store:
        await make_jobs(store, "j")
        entered = threading.Event()
        gate = threading.Event()

        def slow(
            real: Callable[[Sequence[Sequence[Event]]], list[GroupResult]],
            evs: Sequence[Sequence[Event]],
        ) -> list[GroupResult]:
            entered.set()
            assert gate.wait(timeout=10)
            return real(evs)

        patch_append_groups(store, monkeypatch, slow)
        tasks = [asyncio.create_task(store.append(ev(n=i))) for i in range(10)]
        await wait_for(entered)
        for _ in range(20):
            await asyncio.sleep(0.001)
            assert store.stats.queue_depth <= 2
        # 1 in flight + 2 queued; the other 7 producers are blocked in put()
        assert store.stats.queue_depth == 2
        assert not any(t.done() for t in tasks)
        gate.set()
        results = await asyncio.gather(*tasks)
        assert len({r.seq for r in results}) == 10
        assert store.stats.queue_depth == 0


@pytest.mark.asyncio
async def test_caller_cancellation_keeps_writer_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with await EventStore.open(tmp_path / "sb.db") as store:
        await make_jobs(store, "victim", "after")
        entered = threading.Event()
        gate = threading.Event()

        def slow(
            real: Callable[[Sequence[Sequence[Event]]], list[GroupResult]],
            evs: Sequence[Sequence[Event]],
        ) -> list[GroupResult]:
            entered.set()
            assert gate.wait(timeout=10)
            return real(evs)

        patch_append_groups(store, monkeypatch, slow)
        victim = asyncio.create_task(store.append(ev(job="victim")))
        await wait_for(entered)
        victim.cancel()
        with pytest.raises(asyncio.CancelledError):
            await victim
        gate.set()
        r = await store.append(ev(job="after"))
        assert r.status is AppendStatus.INSERTED
        # already queued, so the cancelled caller's event is still written (documented)
        assert len(await store.read_job("victim")) == 2


@pytest.mark.asyncio
async def test_batch_failure_fails_batch_then_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with await EventStore.open(tmp_path / "sb.db", max_batch=1000) as store:
        await make_jobs(store, "j")
        entered = threading.Event()
        gate = threading.Event()
        calls = 0

        def flaky(
            real: Callable[[Sequence[Sequence[Event]]], list[GroupResult]],
            evs: Sequence[Sequence[Event]],
        ) -> list[GroupResult]:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                assert gate.wait(timeout=10)
                raise OSError("disk full")
            return real(evs)

        patch_append_groups(store, monkeypatch, flaky)
        t0 = asyncio.create_task(store.append(ev(n=0)))
        await wait_for(entered)
        # these queue up behind the in-flight batch and form the *second* batch
        rest = [asyncio.create_task(store.append(ev(n=i))) for i in range(1, 6)]
        await asyncio.sleep(0.01)
        gate.set()
        with pytest.raises(OSError, match="disk full"):
            await t0
        assert all(isinstance(r, AppendResult) for r in await asyncio.gather(*rest))

        calls = 0  # next call fails again, with several callers in the same batch
        entered.clear()
        gate.clear()
        calls = 0
        batch = [asyncio.create_task(store.append(ev(n=i))) for i in range(5)]
        await wait_for(entered)
        gate.set()
        outcomes = await asyncio.gather(*batch, return_exceptions=True)
        assert any(isinstance(o, OSError) for o in outcomes)
        assert isinstance((await store.append(ev(n=99))).seq, int)


@pytest.mark.asyncio
async def test_writer_crash_fails_pending_and_new_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = await EventStore.open(tmp_path / "sb.db", max_batch=1)
    await make_jobs(store, "j")
    entered = threading.Event()
    gate = threading.Event()

    def broken(
        real: Callable[[Sequence[Sequence[Event]]], list[GroupResult]],
        evs: Sequence[Sequence[Event]],
    ) -> list[GroupResult]:
        entered.set()
        assert gate.wait(timeout=10)
        return []  # violates the contract: the writer's invariant check kills the task

    patch_append_groups(store, monkeypatch, broken)
    pending = [asyncio.create_task(store.append(ev(n=i))) for i in range(5)]
    await wait_for(entered)
    gate.set()
    async with asyncio.timeout(5):
        outcomes = await asyncio.gather(*pending, return_exceptions=True)
        assert all(isinstance(o, WriterCrashed) for o in outcomes)
        with pytest.raises(WriterCrashed) as info:
            await store.append(ev())
        assert isinstance(info.value.__cause__, RuntimeError)
        await store.aclose()


@pytest.mark.asyncio
async def test_aclose_drains_and_rejects(tmp_path: Path) -> None:
    path = tmp_path / "sb.db"
    store = await EventStore.open(path, max_batch=16)
    await make_jobs(store, "j")
    tasks = [asyncio.create_task(store.append(ev(n=i))) for i in range(300)]
    await asyncio.sleep(0)  # let producers enqueue
    await store.aclose()
    results = await asyncio.gather(*tasks)
    assert len({r.seq for r in results}) == 300
    with pytest.raises(StoreClosed):
        await store.append(ev())
    with pytest.raises(StoreClosed):
        await store.read_since(0)
    await store.aclose()  # idempotent
    async with await EventStore.open(path) as again:
        assert len(await again.read_since(0)) == 301


@pytest.mark.asyncio
async def test_db_calls_run_on_dedicated_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with await EventStore.open(tmp_path / "sb.db") as store:
        await make_jobs(store, "j")
        names: list[str] = []

        def spy(
            real: Callable[[Sequence[Sequence[Event]]], list[GroupResult]],
            evs: Sequence[Sequence[Event]],
        ) -> list[GroupResult]:
            names.append(threading.current_thread().name)
            return real(evs)

        patch_append_groups(store, monkeypatch, spy)
        await asyncio.gather(*(store.append(ev(n=i)) for i in range(50)))
        assert names
        assert {n.rsplit("_", 1)[0] for n in names} == {"sb-store-writer"}
        assert len(set(names)) == 1  # always the same single thread, never a to_thread pool
        assert not any(n.startswith("asyncio") for n in names)
        # reads work from their own connection on their own thread
        assert len(await store.read_since(0)) == 51
