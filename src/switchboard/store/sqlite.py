"""SQLite event log. Synchronous; one connection owned by one thread."""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from switchboard.core import (
    Event,
    EventType,
    Job,
    JobKind,
    JobState,
    Verdict,
    apply,
    budget_from_dict,
    budget_to_dict,
    usage_from_dict,
    usage_to_dict,
)
from switchboard.store.interface import (
    AppendResult,
    AppendStatus,
    GroupResult,
    UnknownEventVersion,
)
from switchboard.store.schema import migrate

# Current payload schema version per event type. Bump when a payload shape changes.
PAYLOAD_VERSIONS: Mapping[EventType, int] = MappingProxyType({t: 1 for t in EventType})

_COLUMNS = "seq, job_id, epoch, step_id, ts, type, v, payload"


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a database configured for the event log.

    The connection keeps the default check_same_thread=True: the single-writer thread that
    commit 05 adds creates and owns it, and sqlite3 then enforces that nobody else touches it.

    isolation_level=None turns off sqlite3's implicit transactions; we issue BEGIN IMMEDIATE
    ourselves. synchronous=NORMAL in WAL mode skips the fsync on each commit but still writes
    to the WAL before COMMIT returns, so commits survive a process crash or kill -9 (our
    failure model). Only an OS crash or power loss can lose the last few commits.
    """
    conn = sqlite3.connect(path, isolation_level=None)
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        conn.close()
        raise RuntimeError(f"could not enable WAL mode (got {mode!r}); is {path} on a local disk?")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def _json_default(obj: object) -> object:
    # Event wraps its payload in a read-only mapping, which json does not know.
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def _dump_payload(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not JSON-serializable: {exc}") from exc


def _row_to_event(row: tuple[Any, ...]) -> Event:
    seq, job_id, epoch, step_id, ts, type_, v, payload = row
    try:
        etype = EventType(type_)
    except ValueError:
        raise UnknownEventVersion(f"event seq={seq} has unknown type {type_!r}") from None
    if v > PAYLOAD_VERSIONS[etype]:
        raise UnknownEventVersion(
            f"event seq={seq} type={type_} has payload version {v}, "
            f"newer than supported {PAYLOAD_VERSIONS[etype]}"
        )
    return Event(
        seq=seq,
        job_id=job_id,
        epoch=epoch,
        step_id=step_id,
        ts=ts,
        type=etype,
        v=v,
        payload=json.loads(payload),
    )


_Row = tuple[str, int, int | None, float, str, int, str]


def encode_event(e: Event) -> _Row:
    """Validate an event for appending and return its row tuple (without seq).

    Raises ValueError if seq is set, the payload version is from the future, or the payload is
    not JSON. The async writer calls this per group before enqueueing, so one bad event fails
    only its own caller instead of a whole group-committed batch.
    """
    if e.seq is not None:
        raise ValueError("events passed to append_many must have seq=None")
    if e.v > PAYLOAD_VERSIONS[e.type]:
        raise ValueError(
            f"unknown future payload version {e.v} for {e.type} "
            f"(supported: {PAYLOAD_VERSIONS[e.type]})"
        )
    return (e.job_id, e.epoch, e.step_id, e.ts, e.type.value, e.v, _dump_payload(e.payload))


_JOB_COLUMNS = (
    "job_id, kind, task_ref, state, verdict, budget, usage, owner, lease_until, epoch, "
    "last_step, last_seq, created_at, updated_at"
)
_UPSERT_JOB = (
    f"INSERT INTO jobs ({_JOB_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (job_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _JOB_COLUMNS.split(", ")[1:])
)


_dumps = json.JSONEncoder(separators=(",", ":"), allow_nan=False).encode


def _job_to_row(j: Job) -> tuple[Any, ...]:
    return (
        j.job_id,
        j.kind.value,
        j.task_ref,
        j.state.value,
        None if j.verdict is None else j.verdict.value,
        _dumps(budget_to_dict(j.budget)),
        _dumps(usage_to_dict(j.usage)),
        j.owner,
        j.lease_until,
        j.epoch,
        j.last_step,
        j.last_seq,
        j.created_at,
        j.updated_at,
    )


def _row_to_job(row: tuple[Any, ...]) -> Job:
    (job_id, kind, task_ref, state, verdict, budget, usage, owner, lease_until) = row[:9]
    epoch, last_step, last_seq, created_at, updated_at = row[9:]
    return Job(
        job_id=job_id,
        kind=JobKind(kind),
        task_ref=task_ref,
        state=JobState(state),
        verdict=None if verdict is None else Verdict(verdict),
        budget=budget_from_dict(json.loads(budget)),
        usage=usage_from_dict(json.loads(usage)),
        owner=owner,
        lease_until=lease_until,
        epoch=epoch,
        last_step=last_step,
        last_seq=last_seq,
        created_at=created_at,
        updated_at=updated_at,
    )


def _with_seq(e: Event, seq: int) -> Event:
    # Much cheaper than dataclasses.replace on this hot path.
    return Event(
        seq=seq,
        job_id=e.job_id,
        epoch=e.epoch,
        step_id=e.step_id,
        ts=e.ts,
        type=e.type,
        v=e.v,
        payload=e.payload,
    )


class SQLiteEventLog:
    """Event log plus jobs projection over one connection owned by one thread.

    Write-through job cache: `append_groups` keeps the projection of every job it has touched
    in `self._cache`, so a batch does not re-read and re-decode jobs from SQLite. This is only
    correct under the single-writer invariant: the jobs table is written ONLY by this log
    object, on the writer thread (a log that has the cache assumes it is the only writer of the
    database). The reader connection uses a separate SQLiteEventLog whose reads (`get_job`,
    `list_jobs`) always go to the table and never to the cache. Code or tests that modify the
    jobs table behind the log's back must call `rebuild()` (which clears the cache) or use a
    fresh log. Any exception inside `append_groups` clears the cache entirely.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, Job] = {}

    def append_many(self, events: Sequence[Event]) -> list[AppendResult]:
        """Append one group atomically (see `append_groups`); raises if it is rejected."""
        if not events:
            return []
        (outcome,) = self.append_groups([events])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def append_groups(self, groups: Sequence[Sequence[Event]]) -> list[GroupResult]:
        """Append groups in ONE transaction, projecting each into `jobs`; one outcome per group.

        Each group runs in its own SAVEPOINT: its events are inserted (per-row ON CONFLICT DO
        NOTHING, so a duplicate only affects that row), then the events actually inserted are
        folded into the affected jobs (a duplicate was folded when first written). If the fold
        raises ProjectionError the savepoint is rolled back and that group's exception is
        returned in its slot; other groups are unaffected. A sqlite error aborts the whole
        transaction and is raised.

        Everything that can be rejected statelessly (encoding) is checked before BEGIN. Jobs
        are loaded at most once per batch and written once at the end: BEGIN IMMEDIATE holds
        the write lock, so the in-memory view cannot go stale, and a rejected group's changes
        are only merged into the view after it succeeded.
        """
        encoded = [[encode_event(e) for e in g] for g in groups]
        if not any(encoded):
            return [[] for _ in groups]

        conn = self._conn
        outcomes: list[GroupResult] = []
        pending: dict[str, Job] = {}  # committed-by-savepoint, not yet by the transaction
        conn.execute("BEGIN IMMEDIATE")
        try:
            for group, rows in zip(groups, encoded, strict=True):
                if not rows:
                    outcomes.append([])
                    continue
                conn.execute("SAVEPOINT grp")
                results: list[AppendResult] = []
                staged: dict[str, Job | None] = {}
                try:
                    for event, row in zip(group, rows, strict=True):
                        got = conn.execute(
                            "INSERT INTO events (job_id, epoch, step_id, ts, type, v, payload) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING seq",
                            row,
                        ).fetchone()
                        if got is None:
                            results.append(AppendResult(AppendStatus.DUPLICATE, None))
                            continue
                        seq = int(got[0])
                        results.append(AppendResult(AppendStatus.INSERTED, seq))
                        jid = event.job_id
                        if jid not in staged:
                            staged[jid] = self._lookup(pending, jid)
                        staged[jid] = apply(staged[jid], _with_seq(event, seq))
                # ValueError, not just ProjectionError: any validation failure in the fold (e.g. a
                # model __post_init__ check) must stay this group's problem, never abort the batch.
                # sqlite3.Error is not a ValueError, so real database errors still abort.
                except ValueError as exc:
                    conn.execute("ROLLBACK TO grp")
                    conn.execute("RELEASE grp")
                    outcomes.append(exc)
                    continue
                conn.execute("RELEASE grp")
                for jid, job in staged.items():
                    if job is not None:
                        pending[jid] = job
                outcomes.append(results)
            conn.executemany(_UPSERT_JOB, [_job_to_row(j) for j in pending.values()])
            conn.execute("COMMIT")
        except BaseException:
            self._cache.clear()
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        self._cache.update(pending)
        return outcomes

    def _lookup(self, pending: dict[str, Job], job_id: str) -> Job | None:
        job = pending.get(job_id) or self._cache.get(job_id)
        if job is None:
            job = self.get_job(job_id)
            if job is not None:
                self._cache[job_id] = job  # unchanged table row, so valid even if we roll back
        return job

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return None if row is None else _row_to_job(row)

    def list_jobs(self, state: JobState | None = None) -> list[Job]:
        if state is None:
            cur = self._conn.execute(f"SELECT {_JOB_COLUMNS} FROM jobs ORDER BY created_at, job_id")
        else:
            cur = self._conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM jobs WHERE state = ? ORDER BY created_at, job_id",
                (state.value,),
            )
        return [_row_to_job(r) for r in cur.fetchall()]

    def rebuild(self) -> int:
        """Drop the jobs projection and re-fold every event in seq order; returns job count.

        One transaction: readers see either the old or the new projection. Raises
        ProjectionError (leaving the old projection) if the log itself cannot be folded.
        """
        conn = self._conn
        self._cache.clear()  # repopulated lazily from the new table
        conn.execute("BEGIN IMMEDIATE")
        try:
            jobs: dict[str, Job] = {}
            cur = conn.execute(f"SELECT {_COLUMNS} FROM events ORDER BY seq")
            while chunk := cur.fetchmany(1000):
                for row in chunk:
                    e = _row_to_event(row)
                    jobs[e.job_id] = apply(jobs.get(e.job_id), e)
            conn.execute("DELETE FROM jobs")
            conn.executemany(_UPSERT_JOB, [_job_to_row(j) for j in jobs.values()])
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return len(jobs)

    def read_since(self, after_seq: int, limit: int = 1000) -> list[Event]:
        cur = self._conn.execute(
            f"SELECT {_COLUMNS} FROM events WHERE seq > ? ORDER BY seq LIMIT ?",
            (after_seq, limit),
        )
        return [_row_to_event(r) for r in cur.fetchall()]

    def read_job(self, job_id: str, after_seq: int = 0) -> list[Event]:
        cur = self._conn.execute(
            f"SELECT {_COLUMNS} FROM events WHERE job_id = ? AND seq > ? ORDER BY seq",
            (job_id, after_seq),
        )
        return [_row_to_event(r) for r in cur.fetchall()]

    def last_seq(self) -> int:
        row = self._conn.execute("SELECT MAX(seq) FROM events").fetchone()
        return int(row[0]) if row[0] is not None else 0

    def close(self) -> None:
        self._conn.close()
