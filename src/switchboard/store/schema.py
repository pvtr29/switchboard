"""SQLite schema and forward-only migrations."""

import sqlite3

MIGRATIONS: tuple[str, ...] = (
    # 1: tables exactly as in docs/plan.md "Data model". Only `events` is used so far; the
    # others exist so later commits do not need a schema change for them.
    """
    CREATE TABLE schema_version (
      version INTEGER NOT NULL CHECK (version >= 1)
    ) STRICT;

    CREATE TABLE events (
      seq     INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id  TEXT NOT NULL,
      epoch   INTEGER NOT NULL CHECK (epoch >= 0),
      step_id INTEGER CHECK (step_id IS NULL OR step_id >= 0),
      ts      REAL NOT NULL,
      type    TEXT NOT NULL,
      v       INTEGER NOT NULL CHECK (v >= 1),
      payload TEXT NOT NULL
    ) STRICT;
    CREATE INDEX events_job ON events(job_id, seq);
    CREATE UNIQUE INDEX events_step ON events(job_id, step_id) WHERE step_id IS NOT NULL;

    CREATE TABLE snapshots (
      job_id      TEXT NOT NULL,
      step_id     INTEGER NOT NULL CHECK (step_id >= 0),
      agent_state BLOB NOT NULL,
      env_ref     TEXT NOT NULL,
      PRIMARY KEY (job_id, step_id)
    ) STRICT;

    CREATE TABLE effects (
      effect_id TEXT PRIMARY KEY,
      job_id    TEXT NOT NULL,
      step_id   INTEGER NOT NULL CHECK (step_id >= 0),
      kind      TEXT NOT NULL,
      args_hash TEXT NOT NULL,
      status    TEXT NOT NULL,
      outcome   TEXT
    ) STRICT;

    CREATE TABLE jobs (
      job_id      TEXT PRIMARY KEY,
      kind        TEXT NOT NULL,
      task_ref    TEXT NOT NULL,
      state       TEXT NOT NULL,
      verdict     TEXT,
      budget      TEXT NOT NULL,
      usage       TEXT NOT NULL,
      owner       TEXT,
      lease_until REAL,
      epoch       INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0),
      last_step   INTEGER,
      last_seq    INTEGER NOT NULL,
      created_at  REAL NOT NULL,
      updated_at  REAL NOT NULL
    ) STRICT;
    """,
    # 2: llm_responded and step_started now carry a step_id too, so uniqueness widens to
    # (job_id, type, step_id): one llm_responded and one step_completed per step. step_started
    # is exempt: it is written once per attempt and its count is the restart counter.
    """
    DROP INDEX events_step;
    CREATE UNIQUE INDEX events_step ON events(job_id, type, step_id)
      WHERE step_id IS NOT NULL AND type <> 'step_started';
    """,
    # 3: effect journal. Effect events carry no step_id, so events_step cannot dedupe them;
    # instead two partial unique indexes over an expression on the payload's effect_id make
    # "one intent and one outcome per effect" a database fact (an index on an expression is
    # simpler than a generated column and needs no table rewrite). The effects projection
    # also remembers the policy stored with the intent: it is never upgraded afterwards.
    # DEFAULT 'unsafe' is the conservative value for any row without a recorded policy.
    """
    CREATE UNIQUE INDEX events_effect_intent ON events(json_extract(payload, '$.effect_id'))
      WHERE type = 'effect_intent';
    CREATE UNIQUE INDEX events_effect_outcome ON events(json_extract(payload, '$.effect_id'))
      WHERE type = 'effect_outcome';
    ALTER TABLE effects ADD COLUMN policy TEXT NOT NULL DEFAULT 'unsafe';
    """,
    # 4: density. The two indexes keyed by job (events_job on (job_id, seq), events_step on
    # (job_id, type, step_id)) made every append a random B-tree insert, and at 10k concurrent
    # jobs they were most of the writer's time (benchmarks/density.py; docs/benchmarks.md).
    # They are replaced by things the writer already has in hand:
    # - prev_seq: each event points at its job's previous event, and jobs.last_seq at the
    #   newest, so read_job walks the chain by primary key (a recursive CTE) instead of an index;
    # - step uniqueness is checked against the projection (jobs.last_step / last_responded)
    #   under the write lock: steps are written in order, so n is a duplicate iff n <= the
    #   highest recorded. Backfilled here from the log, then the indexes go.
    """
    ALTER TABLE events ADD COLUMN prev_seq INTEGER CHECK (prev_seq IS NULL OR prev_seq < seq);
    UPDATE events SET prev_seq = (
      SELECT MAX(p.seq) FROM events p WHERE p.job_id = events.job_id AND p.seq < events.seq
    );
    ALTER TABLE jobs ADD COLUMN last_responded INTEGER;
    UPDATE jobs SET last_responded = (
      SELECT MAX(e.step_id) FROM events e
      WHERE e.job_id = jobs.job_id AND e.type = 'llm_responded'
    );
    DROP INDEX events_job;
    DROP INDEX events_step;
    """,
)

SCHEMA_VERSION = len(MIGRATIONS)


class SchemaVersionError(RuntimeError):
    """The database was written by a newer version of the code."""


def current_version(conn: sqlite3.Connection) -> int:
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if has_table is None:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations and return the resulting version. Idempotent.

    Each migration and its version bump share one transaction, so a crash leaves either the
    old version or the new one, never a half-applied schema. `executescript` does not join an
    open transaction, so BEGIN/COMMIT are part of the script itself.
    """
    version = current_version(conn)
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"database schema version {version} is newer than supported version {SCHEMA_VERSION}"
        )
    for n, script in enumerate(MIGRATIONS[version:], start=version + 1):
        try:
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nDELETE FROM schema_version;\n"
                f"INSERT INTO schema_version (version) VALUES ({int(n)});\nCOMMIT;"
            )
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return SCHEMA_VERSION
