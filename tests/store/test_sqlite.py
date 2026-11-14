import math
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from switchboard.core import Event, EventType, StaleEpoch
from switchboard.store import (
    PAYLOAD_VERSIONS,
    SCHEMA_VERSION,
    AppendStatus,
    EventLog,
    SchemaVersionError,
    SQLiteEventLog,
    UnknownEventVersion,
    migrate,
    open_db,
)


def ev(
    job: str = "j1",
    step: int | None = None,
    type_: EventType | None = None,
    **payload: object,
) -> Event:
    t = type_ or (EventType.STEP_COMPLETED if step is not None else EventType.USAGE_RECORDED)
    return Event(job_id=job, epoch=0, step_id=step, ts=1.5, type=t, payload=payload)


def created(*jobs: str) -> list[Event]:
    """job_created events: every other event type requires its job to exist."""
    return [
        Event(
            job_id=j,
            epoch=0,
            ts=0.5,
            type=EventType.JOB_CREATED,
            payload={"kind": "fake", "task_ref": "t", "budget": {}, "seed": 0},
        )
        for j in jobs
    ]


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    # Python 3.13+ warns (ResourceWarning, fatal here) about sqlite3 connections never closed.
    conn = open_db(tmp_path / "sb.db")
    yield conn
    conn.close()


@pytest.fixture
def log(db: sqlite3.Connection) -> SQLiteEventLog:
    return SQLiteEventLog(db)


@pytest.fixture
def jlog(log: SQLiteEventLog) -> SQLiteEventLog:
    """A log in which jobs j, j1, a and b already exist (seqs 1-4)."""
    log.append_many(created("j", "j1", "a", "b"))
    return log


def test_migrate_fresh_creates_everything(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "raw.db", isolation_level=None)) as conn:
        assert migrate(conn) == 3 == SCHEMA_VERSION
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
        }
        assert {"schema_version", "events", "snapshots", "effects", "jobs"} <= names
        assert {"events_job", "events_step"} <= names
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(3,)]


def test_migrate_twice_is_noop(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "raw.db", isolation_level=None)) as conn:
        migrate(conn)
        assert migrate(conn) == 3
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone() == (1,)


def test_future_version_refused(tmp_path: Path) -> None:
    path = tmp_path / "raw.db"
    conn = sqlite3.connect(path, isolation_level=None)
    migrate(conn)
    conn.execute("UPDATE schema_version SET version = 99")
    conn.close()
    with pytest.raises(SchemaVersionError, match="99"):
        open_db(path)


def test_pragmas(db: sqlite3.Connection) -> None:
    assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert db.execute("PRAGMA synchronous").fetchone() == (1,)


def test_roundtrip_all_fields(jlog: SQLiteEventLog) -> None:
    payload = {"a": [1, {"b": None, "c": "é"}], "f": 0.5, "t": True}
    src = Event(
        job_id="j1",
        epoch=0,
        step_id=7,
        ts=123.25,
        type=EventType.STEP_COMPLETED,
        v=1,
        payload=payload,
    )
    (res,) = jlog.append_many([src])
    assert res.status is AppendStatus.INSERTED
    assert res.seq == 5
    (got,) = jlog.read_since(4)
    assert got.seq == 5
    assert (got.job_id, got.epoch, got.step_id, got.ts, got.type, got.v) == (
        "j1",
        0,
        7,
        123.25,
        EventType.STEP_COMPLETED,
        1,
    )
    assert dict(got.payload) == payload


def test_seq_increasing_in_input_order(jlog: SQLiteEventLog) -> None:
    res = jlog.append_many([ev(n=i) for i in range(5)])
    seqs = [r.seq for r in res]
    assert seqs == [5, 6, 7, 8, 9]
    assert len(set(seqs)) == 5
    more = jlog.append_many([ev(n=9)])
    assert more[0].seq is not None
    assert seqs[-1] is not None
    assert more[0].seq > seqs[-1]
    assert [e.payload["n"] for e in jlog.read_since(4)] == [0, 1, 2, 3, 4, 9]


def test_duplicate_in_batch_only_affects_that_row(jlog: SQLiteEventLog) -> None:
    res = jlog.append_many([ev(n=1), ev(step=1), ev(step=1), ev(n=2)])
    assert [r.status for r in res] == [
        AppendStatus.INSERTED,
        AppendStatus.INSERTED,
        AppendStatus.DUPLICATE,
        AppendStatus.INSERTED,
    ]
    assert res[2].seq is None
    assert len(jlog.read_since(4)) == 3


def test_duplicate_across_batches(jlog: SQLiteEventLog) -> None:
    jlog.append_many([ev(step=1)])
    (res,) = jlog.append_many([ev(step=1)])
    assert res.status is AppendStatus.DUPLICATE
    assert len(jlog.read_since(4)) == 1


def test_same_step_other_job_and_null_step_never_conflict(jlog: SQLiteEventLog) -> None:
    res = jlog.append_many([ev("a", step=1), ev("b", step=1), ev("a", n=1), ev("a", n=1)])
    assert all(r.status is AppendStatus.INSERTED for r in res)


def test_serialization_failure_writes_nothing(jlog: SQLiteEventLog) -> None:
    with pytest.raises(ValueError, match="JSON"):
        jlog.append_many([ev(n=1), ev(bad=object()), ev(n=3)])
    with pytest.raises(ValueError, match="JSON"):
        jlog.append_many([ev(n=1), ev(bad=math.nan)])
    assert jlog.read_since(4) == []
    assert jlog.last_seq() == 4


def test_mid_transaction_failure_rolls_back(db: sqlite3.Connection, jlog: SQLiteEventLog) -> None:
    db.execute(
        "CREATE TRIGGER boom BEFORE INSERT ON events WHEN NEW.payload LIKE '%boom%' "
        "BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.DatabaseError, match="boom"):
        jlog.append_many([ev(n=1), ev(n="boom")])
    assert not db.in_transaction
    assert jlog.read_since(4) == []


def test_seq_set_rejected(log: SQLiteEventLog) -> None:
    e = Event(seq=5, job_id="j", epoch=0, ts=0.0, type=EventType.USAGE_RECORDED, payload={})
    with pytest.raises(ValueError, match="seq"):
        log.append_many([e])


def test_every_event_type_has_version() -> None:
    assert set(PAYLOAD_VERSIONS) == set(EventType)


def test_future_version_rejected_on_append(log: SQLiteEventLog) -> None:
    e = Event(job_id="j", epoch=0, ts=0.0, type=EventType.USAGE_RECORDED, v=2, payload={})
    with pytest.raises(ValueError, match="version"):
        log.append_many([e])
    assert log.last_seq() == 0


def _raw_insert(db: sqlite3.Connection, type_: str, v: int) -> None:
    db.execute(
        "INSERT INTO events (job_id, epoch, ts, type, v, payload) VALUES ('j', 0, 0, ?, ?, '{}')",
        (type_, v),
    )


def test_unknown_version_row_errors_on_read(db: sqlite3.Connection, log: SQLiteEventLog) -> None:
    _raw_insert(db, "usage_recorded", 2)
    with pytest.raises(UnknownEventVersion, match="version 2"):
        log.read_since(0)
    with pytest.raises(UnknownEventVersion):
        log.read_job("j")


def test_unknown_type_row_errors_on_read(db: sqlite3.Connection, log: SQLiteEventLog) -> None:
    _raw_insert(db, "mystery", 1)
    with pytest.raises(UnknownEventVersion, match="mystery"):
        log.read_since(0)


def test_read_since_pagination(log: SQLiteEventLog) -> None:
    log.append_many(created("j1"))
    log.append_many([ev(n=i) for i in range(9)])
    first = log.read_since(0, limit=4)
    assert [e.seq for e in first] == [1, 2, 3, 4]
    assert first[-1].seq is not None
    nxt = log.read_since(first[-1].seq, limit=4)
    assert [e.seq for e in nxt] == [5, 6, 7, 8]
    assert [e.seq for e in log.read_since(8)] == [9, 10]
    assert log.read_since(10) == []


def test_read_job_filters(jlog: SQLiteEventLog) -> None:
    jlog.append_many([ev("a", n=1), ev("b", n=2), ev("a", n=3), ev("a", step=1)])
    assert [e.seq for e in jlog.read_job("a")] == [3, 5, 7, 8]
    assert [e.seq for e in jlog.read_job("a", after_seq=5)] == [7, 8]
    assert jlog.read_job("zzz") == []


def test_last_seq(log: SQLiteEventLog) -> None:
    assert log.last_seq() == 0
    log.append_many([*created("j1"), ev()])
    assert log.last_seq() == 2


def test_persistence_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "p.db"
    conn = open_db(path)
    SQLiteEventLog(conn).append_many([*created("j1"), ev(step=1, k="v")])
    conn.close()
    log = SQLiteEventLog(open_db(path))
    e = log.read_since(0)[1]
    assert e.step_id == 1
    assert e.payload["k"] == "v"
    assert log.append_many([ev(step=1)])[0].status is AppendStatus.DUPLICATE
    log.close()


def test_satisfies_protocol(log: SQLiteEventLog) -> None:
    store: EventLog = log
    assert store.last_seq() == 0


def test_migration_2_upgrades_a_v1_database(tmp_path: Path) -> None:
    from switchboard.store.schema import MIGRATIONS

    path = tmp_path / "v1.db"
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        conn.executescript(
            f"BEGIN;\n{MIGRATIONS[0]}\nINSERT INTO schema_version (version) VALUES (1);\nCOMMIT;"
        )
        conn.execute(
            "INSERT INTO events (job_id, epoch, step_id, ts, type, v, payload) "
            "VALUES ('j', 0, 1, 1.0, 'step_completed', 1, '{}')"
        )
        # v1 index: step_id alone is unique per job, so llm_responded(1) would collide.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO events (job_id, epoch, step_id, ts, type, v, payload) "
                "VALUES ('j', 0, 1, 1.0, 'llm_responded', 1, '{}')"
            )
        assert migrate(conn) == 3
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone() == (1,)
        conn.execute(
            "INSERT INTO events (job_id, epoch, step_id, ts, type, v, payload) "
            "VALUES ('j', 0, 1, 1.0, 'llm_responded', 1, '{}')"
        )


def _step_ev(type_: EventType, step: int) -> Event:
    return Event(job_id="j", epoch=0, step_id=step, ts=1.0, type=type_, payload={})


def test_step_uniqueness_rules(jlog: SQLiteEventLog) -> None:
    T = EventType
    res = jlog.append_many(
        [
            _step_ev(T.STEP_STARTED, 0),
            _step_ev(T.STEP_STARTED, 0),
            _step_ev(T.STEP_STARTED, 0),
            _step_ev(T.LLM_RESPONDED, 0),
            _step_ev(T.LLM_RESPONDED, 0),
            _step_ev(T.STEP_COMPLETED, 0),
            _step_ev(T.STEP_COMPLETED, 0),
            _step_ev(T.LLM_RESPONDED, 1),
        ]
    )
    assert [r.status is AppendStatus.INSERTED for r in res] == [
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        True,
    ]


def test_a_second_writer_on_the_same_database_invalidates_the_job_cache(tmp_path: Path) -> None:
    """Fencing must hold across processes: a log whose cached projection is stale (another
    writer advanced the epoch) must fold against the table, not its cache."""

    def at(epoch: int, type_: EventType, **payload: object) -> Event:
        return Event(job_id="j1", epoch=epoch, ts=1.0, type=type_, payload=payload)

    with closing(open_db(tmp_path / "x.db")) as c1, closing(open_db(tmp_path / "x.db")) as c2:
        old, new = SQLiteEventLog(c1), SQLiteEventLog(c2)
        old.append_many([*created("j1"), at(1, EventType.LEASE_ACQUIRED, worker="old")])
        old.append_many([at(1, EventType.USAGE_RECORDED)])  # j1 is in old's cache at epoch 1
        new.append_many([at(2, EventType.LEASE_LOST, reason="takeover")])
        with pytest.raises(StaleEpoch):
            old.append_many([at(1, EventType.USAGE_RECORDED)])
        assert old.append_many([at(2, EventType.USAGE_RECORDED)])[0].status is AppendStatus.INSERTED
        assert [e.epoch for e in new.read_job("j1")] == [0, 1, 1, 2, 2]
