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
- Cancelled: nothing appended here; `cancel()` appends `-> cancelled` after the task is gone.
  `aclose` (shutdown) cancels tasks WITHOUT recording anything: those jobs stay resumable.

Slots (`SlotPool`, see slots.py): a job task first waits for a slot (the job stays `queued` in
the projection meanwhile), runs `run_job` while holding it and releases it in `finally`.
Admission is FIFO. `pause` makes `run_job` return at a step boundary, which ends the task and
frees the slot; `resume` queues a new task that waits for a slot like any other.

Cancel: `cancel()` cancels the task, waits for it to finish, then appends `<state> ->
cancelled` with the reason. The task itself writes nothing on cancel. A queued job (waiting for
a slot) is cancelled the same way. Cancelling a cancelled job is a no-op; done/failed raise.

Pause: cooperative. `pause()` sets a flag that `run_job` polls at the step boundary (before
`step_started` of the next step), so an in-flight step always commits first. Decision for a
job that is still waiting for a slot (or provisioning): the pause is remembered and takes
effect at its first boundary, i.e. it is admitted, provisions, and goes running -> paused
without running a step (there is no legal queued -> paused edge). `resume()` before that
cancels the pending pause. Pausing a job without a live task raises NotPausable. `pause()`
returns as soon as the request is recorded; `wait(job_id)` returns once the task ended.

`wait(job_id)` returns None once the job's task has finished and never re-raises what the task
raised; use `outcome(job_id)` for the Job, the exception, or the cancelled flag.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from switchboard.core import TERMINAL_STATES, Event, EventType, Job, JobState, ProjectionError
from switchboard.env.base import Environment
from switchboard.runtime.agent import Agent
from switchboard.runtime.runner import JobNotRunnable, run_job
from switchboard.runtime.slots import SlotPool
from switchboard.store import EventStore, StoreClosed

logger = logging.getLogger("switchboard.supervisor")


class AlreadyRunning(RuntimeError):
    """`start` was called for a job that already has a live task."""


class SupervisorClosed(RuntimeError):
    """`start` was called after `aclose`."""


class UnknownJob(LookupError):
    """The job id has no job_created event."""


class NotCancellable(RuntimeError):
    """`cancel` was called for a job that already finished (done or failed)."""


class NotPausable(RuntimeError):
    """`pause` was called for a job that is not queued or running under this supervisor."""


class NotResumable(RuntimeError):
    """`resume` was called for a job that is not paused."""


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
    running: int  # slots held right now (provisioning or running jobs)
    waiting: int  # job tasks waiting for a slot
    max_running_seen: int  # high-water mark of `running`


class Supervisor:
    def __init__(
        self,
        store: EventStore,
        agent_factory: Callable[[Job], Agent],
        env_factory: Callable[[Job], Environment | None] | None = None,
        *,
        clock: Callable[[], float] = time.time,
        max_running: int = 8,
    ) -> None:
        self._pool = SlotPool(max_running)
        self._pause_requested: set[str] = set()
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
        return SupervisorStats(
            self._started,
            self._finished,
            self._crashed,
            self._cancelled,
            self._pool.held,
            self._pool.waiting,
            self._pool.max_seen,
        )

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

    async def cancel(self, job_id: str, reason: str = "cancelled by user") -> None:
        """Cancel a job wherever it is: waiting for a slot, running, or paused.

        The task (if any) gets CancelledError and writes nothing; once it is gone this appends
        `<current state> -> cancelled`. Idempotent for an already cancelled job.
        """
        job = await self._store.get_job(job_id)
        if job is None:
            raise UnknownJob(f"unknown job {job_id}")
        if job.state in TERMINAL_STATES:
            self._require_cancelled(job)
            return
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
            await asyncio.wait({task})
        # Re-read: the task may have finished (or a concurrent cancel won) while we waited.
        job = await self._store.get_job(job_id)
        assert job is not None
        if job.state in TERMINAL_STATES:
            self._require_cancelled(job)
            return
        try:
            await self._append_state(job, JobState.CANCELLED, reason)
        except ProjectionError:
            job = await self._store.get_job(job_id)
            assert job is not None
            self._require_cancelled(job)  # lost a race: fine only if it ended cancelled

    @staticmethod
    def _require_cancelled(job: Job) -> None:
        if job.state is not JobState.CANCELLED:
            raise NotCancellable(f"job {job.job_id} is already {job.state.value}")

    async def pause(self, job_id: str) -> None:
        """Ask a live job to pause at its next step boundary (see module docstring)."""
        job = await self._store.get_job(job_id)
        if job is None:
            raise UnknownJob(f"unknown job {job_id}")
        if job.state is JobState.PAUSED:
            return
        task = self._tasks.get(job_id)
        if (
            task is None
            or task.done()
            or job.state not in (JobState.QUEUED, JobState.PROVISIONING, JobState.RUNNING)
        ):
            raise NotPausable(f"job {job_id} is {job.state.value} and cannot be paused")
        self._pause_requested.add(job_id)

    async def resume(self, job_id: str) -> None:
        """paused -> queued, then start the job again through normal slot admission."""
        task = self._tasks.get(job_id)
        if task is not None and not task.done() and job_id in self._pause_requested:
            self._pause_requested.discard(job_id)  # not paused yet: just withdraw the request
            return
        job = await self._store.get_job(job_id)
        if job is None:
            raise UnknownJob(f"unknown job {job_id}")
        if job.state is not JobState.PAUSED:
            raise NotResumable(f"job {job_id} is {job.state.value}, not paused")
        await self.wait(job_id)  # the paused task may still be releasing its slot
        try:
            await self._append_state(job, JobState.QUEUED, "resumed by user")
        except ProjectionError as exc:
            raise NotResumable(f"job {job_id} was resumed concurrently") from exc
        self.start(job_id)

    async def _append_state(self, job: Job, to: JobState, reason: str) -> None:
        await self._store.append_group(
            [
                Event(
                    job_id=job.job_id,
                    epoch=job.epoch,
                    ts=self._clock(),
                    type=EventType.STATE_CHANGED,
                    payload={"from": job.state.value, "to": to.value, "reason": reason},
                )
            ]
        )

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
        # Acquire first, with no await before it, so admission order is start() order.
        slot = await self._pool.acquire()
        try:
            logger.info("job admitted job_id=%s", job_id, extra={"job_id": job_id})
            job = await self._store.get_job(job_id)
            if job is None:
                raise JobNotRunnable(f"unknown job {job_id}")
            agent = self._agent_factory(job)
            env = self._env_factory(job) if self._env_factory is not None else None
            return await run_job(
                self._store,
                job_id,
                agent,
                clock=self._clock,
                env=env,
                should_pause=lambda: job_id in self._pause_requested,
            )
        finally:
            slot.release()  # cleanup only: never write to the store here

    def _on_done(self, job_id: str, task: asyncio.Task[Job]) -> None:
        if self._tasks.get(job_id) is task:
            del self._tasks[job_id]
            self._pause_requested.discard(job_id)
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
    "NotCancellable",
    "NotPausable",
    "NotResumable",
    "Outcome",
    "Supervisor",
    "SupervisorClosed",
    "SupervisorStats",
    "UnknownJob",
]
