"""Store interface the rest of the system depends on (SQLite now, Postgres in phase 5)."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from switchboard.core import Event, Job, JobState


class AppendStatus(StrEnum):
    INSERTED = "inserted"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Outcome for one event of a batch. `seq` is None for a duplicate."""

    status: AppendStatus
    seq: int | None


type GroupResult = list[AppendResult] | Exception
"""Per-group outcome of `append_groups`: results in input order, or the reason it was rejected."""


class UnknownEventVersion(Exception):
    """An event has a type or payload version this code does not know."""


class EventLog(Protocol):
    def append_many(self, events: Sequence[Event]) -> list[AppendResult]: ...

    def append_groups(self, groups: Sequence[Sequence[Event]]) -> list[GroupResult]: ...

    def read_since(self, after_seq: int, limit: int = 1000) -> list[Event]: ...

    def read_job(self, job_id: str, after_seq: int = 0) -> list[Event]: ...

    def last_seq(self) -> int: ...

    def get_job(self, job_id: str) -> Job | None: ...

    def list_jobs(self, state: JobState | None = None) -> list[Job]: ...

    def rebuild(self) -> int: ...
