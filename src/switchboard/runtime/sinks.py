"""Fake outside worlds that count applications: `FakeExternalSink` (in memory) and
`SQLiteSink` (its ledger lives in its own SQLite file, so it survives kill -9 of the daemon).

Duplicates are counted by action identity, not by bytes: two applications of the same key are a
duplicate even if their payloads differ, and two keys with identical bytes are not. A sink
sees the idempotency key, so `counts()` groups by key. The chaos harness's naive baseline has no
idempotency keys at all (a restarted job gets a new job id, hence new keys), so there the action
identity is the logical action itself, `(kind, canonical args)`; see `counts_of(by_action=True)`.
Both measure the journal against a fake sink, not against a real external system.
"""

import asyncio
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self

from switchboard.runtime.effects import canonical_json


@dataclass(frozen=True, slots=True)
class EffectCounts:
    applications: int  # total external actions performed
    keys: int  # distinct keys applied
    duplicates: int  # keys applied more than once
    extra_applications: int  # applications beyond the first, summed over keys


class FakeExternalSink:
    """`idempotent=True`: a known key returns its stored outcome without applying again.
    `idempotent=False`: every call applies (the naive world). `supports_lookup=False` makes
    `lookup` unusable, so unsafe effects cannot be reconciled. `before_apply` / `after_apply`
    run (with the key) right before / after a REAL application is recorded; tests raise from
    them to crash on either side of the external action."""

    def __init__(
        self,
        *,
        idempotent: bool = True,
        supports_lookup: bool = True,
        before_apply: Callable[[str], None] | None = None,
        after_apply: Callable[[str], None] | None = None,
    ) -> None:
        self.idempotent = idempotent
        self.supports_lookup = supports_lookup
        self.before_apply = before_apply
        self.after_apply = after_apply
        self.ledger: list[tuple[str, str, Mapping[str, Any]]] = []
        self._outcomes: dict[str, Mapping[str, Any]] = {}

    async def apply(self, key: str, kind: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.idempotent and key in self._outcomes:
            return self._outcomes[key]
        if self.before_apply is not None:
            self.before_apply(key)
        self.ledger.append((key, kind, dict(args)))
        outcome = {"receipt": f"{key[:12]}-{self.applications(key)}", "kind": kind}
        # idempotent sinks keep the first outcome; a naive one just overwrites
        self._outcomes.setdefault(key, outcome)
        if self.after_apply is not None:
            self.after_apply(key)
        return self._outcomes[key] if self.idempotent else outcome

    async def lookup(self, key: str) -> Mapping[str, Any] | None:
        if not self.supports_lookup:
            raise RuntimeError("this sink does not support lookup")
        return self._outcomes.get(key)

    def applications(self, key: str) -> int:
        return sum(1 for k, _, _ in self.ledger if k == key)

    def duplicates(self) -> dict[str, int]:
        """Keys applied more than once, with their application counts."""
        return {k: n for k, n in Counter(k for k, _, _ in self.ledger).items() if n > 1}

    def counts(self) -> EffectCounts:
        per_key = Counter(k for k, _, _ in self.ledger)
        return EffectCounts(
            applications=len(self.ledger),
            keys=len(per_key),
            duplicates=sum(1 for n in per_key.values() if n > 1),
            extra_applications=sum(n - 1 for n in per_key.values()),
        )


# ---- SQLiteSink ------------------------------------------------------------------------------

_SINK_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
  n    INTEGER PRIMARY KEY,
  key  TEXT NOT NULL,
  kind TEXT NOT NULL,
  args TEXT NOT NULL,
  ts   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS applications_key ON applications(key);
CREATE TABLE IF NOT EXISTS outcomes (
  key     TEXT PRIMARY KEY,
  outcome TEXT NOT NULL
);
"""


class Application(NamedTuple):
    """One row of the sink's ledger: a real external action."""

    n: int
    key: str
    kind: str
    args: str  # canonical JSON
    ts: float

    @property
    def action(self) -> str:
        """The action's identity independent of any idempotency key."""
        return f"{self.kind}|{self.args}"


def read_applications(path: str | Path) -> list[Application]:
    """The ledger of the sink database at `path`, in application order (blocking; opens its own
    read-only connection, so it is safe while a daemon is writing and after it was killed)."""
    uri = f"file:{path}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10.0)) as conn:
        rows = conn.execute("SELECT n, key, kind, args, ts FROM applications ORDER BY n")
        return [Application(*r) for r in rows.fetchall()]


def counts_of(applications: Sequence[Application], *, by_action: bool = False) -> EffectCounts:
    """Counts over a ledger, grouped by idempotency key or (`by_action`) by logical action."""
    per = Counter(a.action if by_action else a.key for a in applications)
    return EffectCounts(
        applications=len(applications),
        keys=len(per),
        duplicates=sum(1 for n in per.values() if n > 1),
        extra_applications=sum(n - 1 for n in per.values()),
    )


def _open_sink_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=10.0)  # explicit transactions
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")  # safe against process kill; not power loss
    conn.executescript(_SINK_SCHEMA)
    return conn


def _stored_outcome(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT outcome FROM outcomes WHERE key = ?", (key,)).fetchone()
    return None if row is None else dict(json.loads(row[0]))


def _insert_application(
    conn: sqlite3.Connection, key: str, kind: str, args: str, ts: float, idempotent: bool
) -> tuple[dict[str, Any], bool]:
    """Record an application and its outcome in ONE transaction. Returns (outcome, inserted).
    Idempotent: a known key returns its stored outcome and inserts nothing."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if idempotent:
            known = _stored_outcome(conn, key)
            if known is not None:
                conn.execute("COMMIT")
                return known, False
        conn.execute(
            "INSERT INTO applications (key, kind, args, ts) VALUES (?, ?, ?, ?)",
            (key, kind, args, ts),
        )
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE key = ?", (key,)
        ).fetchone()
        outcome = {"receipt": f"{key[:12]}-{count}", "kind": kind}
        conn.execute(
            "INSERT OR IGNORE INTO outcomes (key, outcome) VALUES (?, ?)",
            (key, json.dumps(outcome, sort_keys=True)),
        )
        stored = _stored_outcome(conn, key) if idempotent else outcome
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    assert stored is not None
    return stored, True


def _close_sink_db(conn: sqlite3.Connection) -> None:
    conn.close()


class SQLiteSink:
    """`FakeExternalSink` with a persistent ledger in its OWN SQLite file (separate from the
    event store), so applications survive kill -9 and can be counted afterwards.

    Tables: `applications(n, key, kind, args, ts)` (one row per real external action) and
    `outcomes(key PRIMARY KEY, outcome)`. `idempotent=True`: a known key returns its stored
    outcome, else application and outcome are inserted in ONE transaction. `idempotent=False`:
    every call inserts an application (the naive world). Hooks as in `FakeExternalSink`:
    `before_apply` runs before a REAL application is committed, `after_apply` right after.

    Blocking SQLite runs on a dedicated single-thread executor that owns the connection (the
    same pattern as EventStore). `counts()` and friends are synchronous reads through a fresh
    read-only connection: for tests and harnesses, not for the daemon's loop.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        idempotent: bool = True,
        supports_lookup: bool = True,
        before_apply: Callable[[str], None] | None = None,
        after_apply: Callable[[str], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.idempotent = idempotent
        self.supports_lookup = supports_lookup
        self.before_apply = before_apply
        self.after_apply = after_apply
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sb-sink")
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        idempotent: bool = True,
        supports_lookup: bool = True,
        before_apply: Callable[[str], None] | None = None,
        after_apply: Callable[[str], None] | None = None,
    ) -> Self:
        sink = cls(
            path,
            idempotent=idempotent,
            supports_lookup=supports_lookup,
            before_apply=before_apply,
            after_apply=after_apply,
        )
        loop = asyncio.get_running_loop()
        try:
            sink.path.parent.mkdir(parents=True, exist_ok=True)
            sink._conn = await loop.run_in_executor(sink._exec, _open_sink_db, str(sink.path))
        except BaseException:
            sink._exec.shutdown(wait=True)
            raise
        return sink

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the connection on its own thread. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._conn is not None:
            await asyncio.get_running_loop().run_in_executor(self._exec, _close_sink_db, self._conn)
        self._exec.shutdown(wait=True)

    def _connection(self) -> sqlite3.Connection:
        assert not self._closed, "sink is closed"
        assert self._conn is not None, "sink is not open"
        return self._conn

    async def apply(self, key: str, kind: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        loop = asyncio.get_running_loop()
        conn = self._connection()
        if self.idempotent:
            known = await loop.run_in_executor(self._exec, _stored_outcome, conn, key)
            if known is not None:
                return known
        if self.before_apply is not None:
            self.before_apply(key)
        outcome, inserted = await loop.run_in_executor(
            self._exec,
            _insert_application,
            conn,
            key,
            kind,
            canonical_json(args),
            time.time(),
            self.idempotent,
        )
        if inserted and self.after_apply is not None:
            self.after_apply(key)
        return outcome

    async def lookup(self, key: str) -> Mapping[str, Any] | None:
        if not self.supports_lookup:
            raise RuntimeError("this sink does not support lookup")
        return await asyncio.get_running_loop().run_in_executor(
            self._exec, _stored_outcome, self._connection(), key
        )

    # ---- reads (blocking, fresh read-only connection) --------------------------------------
    def ledger(self) -> list[Application]:
        return read_applications(self.path)

    def applications(self, key: str) -> int:
        return sum(1 for a in self.ledger() if a.key == key)

    def duplicates(self) -> dict[str, int]:
        """Keys applied more than once, with their application counts."""
        return {k: n for k, n in Counter(a.key for a in self.ledger()).items() if n > 1}

    def counts(self) -> EffectCounts:
        return counts_of(self.ledger())
