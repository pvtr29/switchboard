"""SQLite event log. Synchronous; one connection owned by one thread."""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from switchboard.core import Event, EventType
from switchboard.store.interface import AppendResult, AppendStatus, UnknownEventVersion
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


class SQLiteEventLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append_many(self, events: Sequence[Event]) -> list[AppendResult]:
        """Append a batch in one transaction; results are in input order.

        Everything that can be rejected is checked before BEGIN, so a bad event never leaves
        a partially written batch. Each row uses INSERT ... ON CONFLICT DO NOTHING so a
        duplicate (job_id, step_id) only affects that row: a plain INSERT would abort the
        whole transaction and fail unrelated callers sharing the batch. BEGIN IMMEDIATE takes
        the write lock up front, so a busy database fails at BEGIN (honouring busy_timeout)
        rather than midway when a deferred read lock cannot upgrade.
        """
        rows = [encode_event(e) for e in events]
        if not rows:
            return []

        results: list[AppendResult] = []
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                got = conn.execute(
                    "INSERT INTO events (job_id, epoch, step_id, ts, type, v, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING seq",
                    row,
                ).fetchone()
                if got is None:
                    results.append(AppendResult(AppendStatus.DUPLICATE, None))
                else:
                    results.append(AppendResult(AppendStatus.INSERTED, int(got[0])))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return results

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
