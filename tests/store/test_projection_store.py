"""The jobs projection inside the store: same-transaction updates, isolation, rebuild."""

import asyncio
import dataclasses
import random
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from switchboard.core import (
    TRANSITIONS,
    Event,
    EventType,
    Job,
    JobState,
    ProjectionError,
    Usage,
    fold,
)
from switchboard.store import AppendStatus, EventStore, SQLiteEventLog, open_db

T = EventType


def mk(
    job: str,
    type_: EventType,
    payload: dict[str, object] | None = None,
    *,
    epoch: int = 0,
    ts: float = 1.0,
    step_id: int | None = None,
) -> Event:
    return Event(job_id=job, type=type_, payload=payload or {}, epoch=epoch, ts=ts, step_id=step_id)


def created(job: str) -> Event:
    return mk(
        job, T.JOB_CREATED, {"kind": "fake", "task_ref": "t", "budget": {"tokens": 9}, "seed": 1}
    )


def move(job: str, src: str, dst: str) -> Event:
    return mk(job, T.STATE_CHANGED, {"from": src, "to": dst, "reason": "r"})


def usage(job: str, tokens: int = 1) -> Event:
    return mk(job, T.USAGE_RECORDED, {"tokens_in": tokens})


def step(job: str, n: int) -> Event:
    return mk(job, T.STEP_COMPLETED, step_id=n)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(tmp_path / "sb.db")
    yield conn
    conn.close()


@pytest.fixture
def log(db: sqlite3.Connection) -> SQLiteEventLog:
    return SQLiteEventLog(db)


def live(log: SQLiteEventLog) -> dict[str, Job]:
    return {j.job_id: j for j in log.list_jobs()}


def all_events(log: SQLiteEventLog) -> list[Event]:
    return log.read_since(0, limit=1_000_000)


def test_projection_updated_with_events(log: SQLiteEventLog) -> None:
    assert log.get_job("a") is None
    log.append_many([created("a"), usage("a", 3), step("a", 0)])
    job = log.get_job("a")
    assert job is not None
    assert (job.state, job.usage, job.last_step, job.last_seq) == (
        JobState.QUEUED,
        Usage(tokens=3),
        0,
        3,
    )
    assert job.budget.tokens == 9


def test_list_jobs_filters_by_state(log: SQLiteEventLog) -> None:
    log.append_many([created("a"), created("b"), move("b", "queued", "provisioning")])
    assert [j.job_id for j in log.list_jobs()] == ["a", "b"]
    assert [j.job_id for j in log.list_jobs(JobState.PROVISIONING)] == ["b"]
    assert log.list_jobs(JobState.DONE) == []


def test_projection_error_rolls_back_group(log: SQLiteEventLog) -> None:
    log.append_many([created("a")])
    with pytest.raises(ProjectionError):
        log.append_many([usage("a"), move("a", "queued", "done")])
    assert log.last_seq() == 1
    job = log.get_job("a")
    assert job is not None
    assert job.usage == Usage()


def test_group_events_see_each_other(log: SQLiteEventLog) -> None:
    # created and moved in ONE group: the fold must carry state between events of the group.
    log.append_many([created("a"), move("a", "queued", "provisioning")])
    job = log.get_job("a")
    assert job is not None
    assert job.state is JobState.PROVISIONING


def test_group_isolation_in_one_batch(log: SQLiteEventLog) -> None:
    log.append_many([created("a"), created("b")])
    bad = [usage("a", 5), move("a", "queued", "running")]  # illegal edge, after a valid event
    out = log.append_groups(
        [
            [usage("a", 1), usage("b", 1)],
            bad,
            [usage("b", 2), move("b", "queued", "provisioning")],
            [usage("ghost")],  # unknown job
            [usage("a", 10)],
        ]
    )
    assert isinstance(out[1], ProjectionError)
    assert isinstance(out[3], ProjectionError)
    for i in (0, 2, 4):
        assert not isinstance(out[i], Exception)
    assert [len(o) for o in out if not isinstance(o, Exception)] == [2, 2, 1]
    # rejected groups left no events and no projection effects (and consumed no state)
    assert [e.payload.get("tokens_in") for e in all_events(log) if e.type is T.USAGE_RECORDED] == [
        1,
        1,
        2,
        10,
    ]
    a, b = log.get_job("a"), log.get_job("b")
    assert a is not None
    assert b is not None
    assert (a.usage.tokens, a.state) == (11, JobState.QUEUED)
    assert (b.usage.tokens, b.state) == (3, JobState.PROVISIONING)
    assert log.get_job("ghost") is None
    assert live(log) == fold(all_events(log))


def test_rejected_group_does_not_leak_into_later_groups_of_same_job(log: SQLiteEventLog) -> None:
    out = log.append_groups(
        [
            [created("a")],
            [move("a", "queued", "provisioning"), move("a", "queued", "provisioning")],
            [move("a", "queued", "provisioning")],  # valid only if the rejected group left no trace
        ]
    )
    assert not isinstance(out[0], Exception)
    assert isinstance(out[1], ProjectionError)
    assert not isinstance(out[2], Exception)
    job = log.get_job("a")
    assert job is not None
    assert job.state is JobState.PROVISIONING


def test_duplicate_step_is_not_folded_twice(log: SQLiteEventLog) -> None:
    log.append_many([created("a")])
    dup = Event(
        job_id="a",
        epoch=0,
        step_id=0,
        ts=1.0,
        type=T.STEP_COMPLETED,
        payload={},
    )
    log.append_many([dup, usage("a", 4)])
    before = log.get_job("a")
    out = log.append_many([dup, dup])
    assert [r.status for r in out] == [AppendStatus.DUPLICATE] * 2
    assert log.get_job("a") == before  # last_seq / updated_at untouched too
    assert live(log) == fold(all_events(log))


def test_rebuild_restores_emptied_and_corrupted_projection(
    db: sqlite3.Connection, log: SQLiteEventLog
) -> None:
    log.append_many(
        [created("a"), created("b"), usage("a", 2), move("b", "queued", "provisioning")]
    )
    log.append_many([step("a", 0), usage("b", 7)])
    expected = live(log)
    db.execute("DELETE FROM jobs")
    assert log.list_jobs() == []
    assert log.rebuild() == 2
    assert live(log) == expected
    db.execute("UPDATE jobs SET state = 'done', usage = '{}', last_seq = 0")
    db.execute(
        "INSERT INTO jobs SELECT 'zombie', kind, task_ref, state, NULL, budget, usage, "
        "NULL, NULL, 0, NULL, 1, 0, 0 FROM jobs LIMIT 1"
    )
    assert log.rebuild() == 2
    assert live(log) == expected
    assert log.rebuild() == 2  # idempotent
    assert live(log) == expected


def test_cache_backed_log_matches_fresh_log_and_rebuild(tmp_path: Path) -> None:
    path = tmp_path / "c.db"
    conn = open_db(path)
    try:
        log = SQLiteEventLog(conn)
        log.append_many([created(f"j{i}") for i in range(5)])
        for n in range(30):  # many batches: later ones are served from the cache
            log.append_groups(
                [[usage(f"j{n % 5}", n), step(f"j{n % 5}", n)], [usage(f"j{(n + 1) % 5}")]]
            )
        log.append_groups([[move("j0", "queued", "running")]])  # rejected: cache must stay valid
        log.append_many([move("j0", "queued", "provisioning")])
        cached = live(log)
        with closing(open_db(path)) as other:
            assert live(SQLiteEventLog(other)) == cached
        assert log.rebuild() == 5
        assert live(log) == cached
        assert cached["j0"].state is JobState.PROVISIONING
    finally:
        conn.close()


def test_cache_cleared_when_transaction_fails(db: sqlite3.Connection, log: SQLiteEventLog) -> None:
    log.append_many([created("a")])
    db.execute(
        "CREATE TRIGGER boom BEFORE INSERT ON events WHEN NEW.payload LIKE '%boom%' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.DatabaseError, match="boom"):
        log.append_groups([[usage("a", 5)], [mk("a", T.USAGE_RECORDED, {"x": "boom"})]])
    db.execute("DROP TRIGGER boom")
    assert log.get_job("a") == fold(all_events(log))["a"]
    log.append_many([usage("a", 2)])  # uses the (re-read) projection, not stale cache state
    assert live(log) == fold(all_events(log))
    assert log.get_job("a") is not None


def test_rebuild_on_empty_log(log: SQLiteEventLog) -> None:
    assert log.rebuild() == 0


def test_rebuild_failure_keeps_old_projection(db: sqlite3.Connection, log: SQLiteEventLog) -> None:
    log.append_many([created("a")])
    db.execute(
        "INSERT INTO events (job_id, epoch, ts, type, v, payload) "
        "VALUES ('ghost', 0, 0, 'usage_recorded', 1, '{}')"
    )
    before = live(log)
    with pytest.raises(ProjectionError):
        log.rebuild()
    assert not db.in_transaction
    assert live(log) == before


@pytest.mark.asyncio
async def test_writer_rejects_only_the_bad_group(tmp_path: Path) -> None:
    async with await EventStore.open(tmp_path / "sb.db", max_batch=1000) as store:
        await store.append_group([created("a"), created("b")])
        good = [store.append_group([usage("a"), usage("b")]) for _ in range(20)]
        bad = store.append_group([usage("a"), move("a", "queued", "done")])
        ghost = store.append(usage("ghost"))
        res = await asyncio.gather(*good, bad, ghost, return_exceptions=True)
        assert all(isinstance(r, list) for r in res[:20])
        assert isinstance(res[20], ProjectionError)
        assert isinstance(res[21], ProjectionError)
        st_ = store.stats
        assert st_.rejected_groups == 2
        assert st_.events_written == 2 + 40
        a = await store.get_job("a")
        assert a is not None
        assert a.usage.tokens == 20
        assert [j.job_id for j in await store.list_jobs(JobState.QUEUED)] == ["a", "b"]
        assert await store.get_job("ghost") is None
        assert len(await store.read_job("a")) == 21
        # the writer keeps working afterwards
        assert (await store.append(usage("a"))).status is AppendStatus.INSERTED


# ---- property test -----------------------------------------------------------------------


def generate(ops: Sequence[tuple[int, int]]) -> list[Event]:
    """Turn random (job, choice) pairs into a VALID event sequence via per-job simulation."""
    state: dict[str, JobState] = {}
    next_step: dict[str, int] = {}
    out: list[Event] = []
    for job_n, choice in ops:
        job = f"job{job_n}"
        if job not in state:
            out.append(created(job))
            state[job] = JobState.QUEUED
            next_step[job] = 0
            continue
        rng = random.Random(choice)
        moves = sorted(TRANSITIONS[state[job]], key=lambda s: s.value)
        kind = rng.choice(["move", "move", "usage", "step", "lease", "lease_lost", "noise"])
        if kind == "move" and moves:
            dst = rng.choice(moves)
            out.append(move(job, state[job].value, dst.value))
            if dst is JobState.DONE:
                v = rng.choice(["pass", "fail", "suspected_hack"])
                out.append(mk(job, T.VERDICT_RECORDED, {"verdict": v}))
            state[job] = dst
        elif kind == "step":
            out.append(step(job, next_step[job]))
            next_step[job] += 1
        elif kind == "lease":
            out.append(
                mk(
                    job,
                    T.LEASE_ACQUIRED,
                    {"worker": f"w{choice % 3}", "lease_until": choice * 0.5},
                )
            )
        elif kind == "lease_lost":
            out.append(mk(job, T.LEASE_LOST, {"worker": "w", "reason": "takeover"}))
        elif kind == "noise":
            payload = {
                "effect_id": str(choice),
                "kind": "k",
                "args_hash": "h",
                "policy": "keyed",
                "step": 0,
            }
            out.append(mk(job, T.EFFECT_INTENT, payload))
        else:
            out.append(
                mk(
                    job,
                    T.USAGE_RECORDED,
                    {
                        "tokens_in": choice % 7,
                        "tokens_out": choice % 3,
                        "usd_micros": choice,
                        "active_seconds": choice / 4,
                    },
                    ts=float(choice),
                )
            )
    return with_epochs(out)


def with_epochs(events: list[Event]) -> list[Event]:
    """Give every event its job's current epoch, and ownership events current + 1 (fencing)."""
    epoch: dict[str, int] = {}
    out: list[Event] = []
    for e in events:
        n = epoch.get(e.job_id, 0)
        if e.type in (T.LEASE_ACQUIRED, T.LEASE_LOST):
            n += 1
        epoch[e.job_id] = n
        out.append(dataclasses.replace(e, epoch=n))
    return out


ops_strategy = st.lists(st.tuples(st.integers(0, 3), st.integers(0, 10_000)), max_size=60)
sizes_strategy = st.lists(st.integers(1, 6), min_size=1, max_size=10)


def chunk[X](items: Sequence[X], sizes: Sequence[int]) -> list[list[X]]:
    out: list[list[X]] = []
    pos, i = 0, 0
    while pos < len(items):
        n = sizes[i % len(sizes)]
        out.append(list(items[pos : pos + n]))
        pos += n
        i += 1
    return out


@settings(deadline=None, max_examples=40)
@given(ops=ops_strategy, sizes=sizes_strategy, batch=st.integers(1, 4))
def test_property_live_equals_rebuild_equals_fold(
    ops: list[tuple[int, int]], sizes: list[int], batch: int
) -> None:
    events = generate(ops)
    groups = chunk(events, sizes)
    with tempfile.TemporaryDirectory() as tmp:
        conn = open_db(Path(tmp) / "p.db")
        try:
            log = SQLiteEventLog(conn)
            for i in range(0, len(groups), batch):  # several groups per transaction
                for outcome in log.append_groups(groups[i : i + batch]):
                    assert not isinstance(outcome, Exception)
            expected = fold(all_events(log))
            assert live(log) == expected
            assert log.rebuild() == len(expected)
            assert live(log) == expected
        finally:
            conn.close()


@settings(deadline=None, max_examples=15)
@given(ops=ops_strategy, sizes=sizes_strategy)
def test_property_concurrent_store(ops: list[tuple[int, int]], sizes: list[int]) -> None:
    events = generate(ops)
    per_job: dict[str, list[Event]] = {}
    for e in events:
        per_job.setdefault(e.job_id, []).append(e)

    async def scenario(path: Path) -> None:
        async with await EventStore.open(path, max_batch=8) as store:

            async def producer(evs: list[Event]) -> None:
                # one producer per job keeps that job's events ordered; jobs interleave freely
                for g in chunk(evs, sizes):
                    await store.append_group(g)

            await asyncio.gather(*(producer(evs) for evs in per_job.values()))
            stored = await store.read_since(0, limit=1_000_000)
            expected = fold(stored)
            assert {j.job_id: j for j in await store.list_jobs()} == expected
            assert store.stats.rejected_groups == 0
        conn = open_db(path)
        try:
            log = SQLiteEventLog(conn)
            assert log.rebuild() == len(expected)
            assert live(log) == expected
        finally:
            conn.close()

    with tempfile.TemporaryDirectory() as tmp:
        loop = asyncio.new_event_loop()  # not set as current: leaves conftest's loop alone
        try:
            loop.run_until_complete(scenario(Path(tmp) / "p.db"))
        finally:
            loop.close()
