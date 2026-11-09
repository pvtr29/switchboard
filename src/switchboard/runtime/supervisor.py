"""Supervisor: one isolated `asyncio.Task` per job, with crashes recorded, never lost.

Why not `asyncio.TaskGroup`: a TaskGroup is fail-fast. When one child raises, the group cancels
every sibling and re-raises as an ExceptionGroup. That is right for sub-tasks of one unit of work
and exactly wrong here, where a crash in one agent must leave the other jobs untouched. Jobs are
therefore plain tasks created with `asyncio.create_task`, each with its own done callback.

Why a strong reference: the event loop keeps only weak references to tasks, so a task nobody
holds can be garbage-collected while it is still running ("Task was destroyed but it is
pending!"). `_tasks` (and `_followups` for the supervisor's own bookkeeping tasks) hold the
references until the done callback removes them.

Why a done callback always reads the outcome: calling `task.result()` / `task.exception()`
marks the exception as retrieved, so asyncio never logs "Task exception was never retrieved".

Outcome recording (`_on_done`, synchronous):

- `run_job` returned a Job: nothing to record; it already wrote verifying or failed.
- It raised a StoreClosed (incl. WriterCrashed): the store cannot take writes, so appending is
  pointless. The error goes to `errors` for recovery to deal with later.
- It raised anything else (including a BaseException that is not CancelledError): the job is
  moved to failed with reason "supervisor: <type>: <msg>" by a tracked follow-up task, but only
  if its projected state still allows it (provisioning or running); otherwise it is only logged.
- Cancelled: nothing appended (cancel semantics arrive with commit 11); just cleanup.

`wait(job_id)` returns None once the job's task has finished and never re-raises what the task
raised; use `outcome(job_id)` for the Job, the exception, or the cancelled flag.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from switchboard.core import Event, EventType, Job, JobState
from switchboard.env.base import Environment
from switchboard.runtime.agent import Agent
from switchboard.runtime.runner import JobNotRunnable, run_job
from switchboard.store import EventStore, StoreClosed

logger = logging.getLogger("switchboard.supervisor")


class AlreadyRunning(RuntimeError):
    """`start` was called for a job that already has a live task."""


class SupervisorClosed(RuntimeError):
    """`start` was called after `aclose`."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """How a job's task ended: exactly one of job, error, cancelled is set."""

    job: Job | None = None
    error: BaseException | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class SupervisorStats:
    started: int
    finished: int  # run_job returned
    crashed: int  # run_job raised (not cancelled)
    cancelled: int


class Supervisor:
    def __init__(
        self,
        store: EventStore,
        agent_factory: Callable[[Job], Agent],
        env_factory: Callable[[Job], Environment | None] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._agent_factory = agent_factory
        self._env_factory = env_factory
        self._clock = clock
        self._tasks: dict[str, asyncio.Task[Job]] = {}
        self._followups: set[asyncio.Task[None]] = set()
        self._outcomes: dict[str, Outcome] = {}
        self._closed = False
        self._started = 0
        self._finished = 0
        self._crashed = 0
        self._cancelled = 0
        self.errors: list[tuple[str, BaseException]] = []
        """Infrastructure errors (job_id, exception) that were deliberately not written to the
        log, for recovery to look at."""

    @property
    def running(self) -> Mapping[str, asyncio.Task[Job]]:
        return MappingProxyType(self._tasks)

    @property
    def stats(self) -> SupervisorStats:
        return SupervisorStats(self._started, self._finished, self._crashed, self._cancelled)

    def outcome(self, job_id: str) -> Outcome | None:
        """The outcome of the job's most recent finished task, or None if none finished yet."""
        return self._outcomes.get(job_id)

    def start(self, job_id: str) -> None:
        if self._closed:
            raise SupervisorClosed("supervisor is closed")
        if job_id in self._tasks:
            raise AlreadyRunning(f"job {job_id} already has a live task")
        task = asyncio.create_task(self._run(job_id), name=f"job:{job_id}")
        self._tasks[job_id] = task
        self._outcomes.pop(job_id, None)
        self._started += 1
        task.add_done_callback(lambda t: self._on_done(job_id, t))

    async def wait(self, job_id: str) -> None:
        """Return when the job's live task has finished (immediately if there is none)."""
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait({task})

    async def aclose(self) -> None:
        """Cancel every live job, await them and the follow-ups. Idempotent."""
        self._closed = True
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Done callbacks of the job tasks ran before gather resumed, so any follow-up they
        # scheduled is already registered; follow-ups never schedule more.
        while self._followups:
            await asyncio.gather(*list(self._followups), return_exceptions=True)
            await asyncio.sleep(0)  # let their done callbacks discard them

    async def _run(self, job_id: str) -> Job:
        job = await self._store.get_job(job_id)
        if job is None:
            raise JobNotRunnable(f"unknown job {job_id}")
        agent = self._agent_factory(job)
        env = self._env_factory(job) if self._env_factory is not None else None
        return await run_job(self._store, job_id, agent, clock=self._clock, env=env)

    def _on_done(self, job_id: str, task: asyncio.Task[Job]) -> None:
        if self._tasks.get(job_id) is task:
            del self._tasks[job_id]
        if task.cancelled():
            self._cancelled += 1
            self._outcomes[job_id] = Outcome(cancelled=True)
            logger.info(
                "job cancelled job_id=%s outcome=cancelled", job_id, extra={"job_id": job_id}
            )
            return
        exc = task.exception()  # marks the exception retrieved
        if exc is None:
            job = task.result()
            self._finished += 1
            self._outcomes[job_id] = Outcome(job=job)
            logger.info(
                "job finished job_id=%s outcome=%s",
                job_id,
                job.state.value,
                extra={"job_id": job_id},
            )
            return
        self._crashed += 1
        self._outcomes[job_id] = Outcome(error=exc)
        if isinstance(exc, StoreClosed):
            self.errors.append((job_id, exc))
            logger.error(
                "job hit store error job_id=%s outcome=store_error error=%r",
                job_id,
                exc,
                extra={"job_id": job_id},
            )
            return
        logger.error(
            "job crashed job_id=%s outcome=crashed error=%r",
            job_id,
            exc,
            extra={"job_id": job_id},
        )
        followup = asyncio.create_task(self._record_crash(job_id, exc), name=f"crash:{job_id}")
        self._followups.add(followup)
        followup.add_done_callback(lambda t: self._on_followup_done(job_id, t))

    async def _record_crash(self, job_id: str, exc: BaseException) -> None:
        job = await self._store.get_job(job_id)
        if job is None or job.state not in (JobState.PROVISIONING, JobState.RUNNING):
            logger.warning(
                "crash not recorded job_id=%s state=%s",
                job_id,
                job.state.value if job else None,
                extra={"job_id": job_id},
            )
            return
        reason = f"supervisor: {type(exc).__name__}: {exc}"
        await self._store.append_group(
            [
                Event(
                    job_id=job_id,
                    epoch=job.epoch,
                    ts=self._clock(),
                    type=EventType.STATE_CHANGED,
                    payload={
                        "from": job.state.value,
                        "to": JobState.FAILED.value,
                        "reason": reason,
                    },
                )
            ]
        )

    def _on_followup_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        self._followups.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.errors.append((job_id, exc))
            logger.error(
                "could not record crash job_id=%s error=%r", job_id, exc, extra={"job_id": job_id}
            )


__all__ = [
    "AlreadyRunning",
    "Outcome",
    "Supervisor",
    "SupervisorClosed",
    "SupervisorStats",
]
