"""Event store: schema, single-writer task, group commit."""

from switchboard.store.interface import (
    AppendResult,
    AppendStatus,
    EventLog,
    UnknownEventVersion,
)
from switchboard.store.schema import SCHEMA_VERSION, SchemaVersionError, migrate
from switchboard.store.sqlite import PAYLOAD_VERSIONS, SQLiteEventLog, encode_event, open_db
from switchboard.store.writer import EventStore, StoreClosed, StoreStats, WriterCrashed

__all__ = [
    "PAYLOAD_VERSIONS",
    "SCHEMA_VERSION",
    "AppendResult",
    "AppendStatus",
    "EventLog",
    "EventStore",
    "SQLiteEventLog",
    "SchemaVersionError",
    "StoreClosed",
    "StoreStats",
    "UnknownEventVersion",
    "WriterCrashed",
    "encode_event",
    "migrate",
    "open_db",
]
