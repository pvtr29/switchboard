"""Store interface the rest of the system depends on (SQLite now, Postgres in phase 5)."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from switchboard.core import Event


class AppendStatus(StrEnum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Outcome for one event of a batch. `seq` is None for a duplicate."""

    status: AppendStatus
    seq: int | None


class UnknownEventVersion(Exception):
    """An event has a type or payload version this code does not know."""


class EventLog(Protocol):
    def append_many(self, events: Sequence[Event]) -> list[AppendResult]: ...

    def read_since(self, after_seq: int, limit: int = 1000) -> list[Event]: ...

    def read_job(self, job_id: str, after_seq: int = 0) -> list[Event]: ...

    def last_seq(self) -> int: ...
