"""Recovery on startup: turn the log a dead process left behind into running jobs again.

`recover(store, supervisor)` runs once, before the daemon accepts connections (docs/plan.md
"Execution model", item 5 of a step's recovery). It never needs the dead process to have done
anything: `kill -9` leaves the log at some commit boundary, and the log alone is enough.

1. Optional consistency check: the live `jobs` projection is compared with a fold of the job's
   events. A mismatch means the projection table is corrupt; it is reported (and logged as an
   error) but nothing is "repaired" here: `rebuild()` re-derives it from the log. Default
   (`verify_projection=None`): check every job when there are at most `AUTO_VERIFY_MAX_JOBS`,
   else a seeded sample of `VERIFY_SAMPLE_SIZE`; `True` forces all jobs, `False` skips it.
2. Every job in `provisioning` or `running` is moved to `queued` (reason "recovery: daemon
   restart"). Nothing of such a job is alive in this process, so the previous owner is gone.
   The move is behind a `lease_lost` that advances the epoch (`supervisor.revoke`), so if the
   previous owner is NOT dead after all (a frozen daemon that wakes up), its writes are refused.
   `provisioning -> queued` is a legal edge for exactly this case (a crash during provisioning
   must be re-queueable); the runner provisions again, which is idempotent.
3. `queued` jobs (including the ones just re-queued) are admitted through the supervisor in
   creation order, so slot admission is FIFO by age. Each resumes from its last checkpoint:
   the runner restores S(n) (logging `restored`), a logged `llm_responded` is never re-sampled,
   and an `effect_intent` without an outcome is reconciled inline when the step re-runs and
   reaches the same `ctx.effect` call, by the policy stored with the intent. There is no
   separate reconciliation pass: only the re-run of the step knows the call's arguments, and
   the step is guaranteed to re-run because it has no `step_completed`.
4. `paused` and `over_budget` jobs stay as they are (a human or an extension moves them);
   terminal jobs and `verifying` jobs are untouched.
5. `prepare(job_id)` (the daemon's `_ensure_spec`) is awaited for every job before it is
   admitted, so the agent and environment factories can build it from the log.

The restart counter (`step_started` events per step) is what bounds a crash loop: a job whose
step kills the process every time reaches `max_attempts_per_step` and then fails with "restart
limit" instead of being re-admitted forever.

Jobs that already have a live task in `supervisor` are skipped (counted as `live`), so calling
`recover` on a supervisor that is already running jobs does no harm.
"""

import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from switchboard.core import Event, Job, JobState, ProjectionError, fold
from switchboard.runtime.supervisor import AlreadyRunning, Supervisor, revoke
from switchboard.store import EventStore

logger = logging.getLogger("switchboard.recovery")

REASON_RESTART = "recovery: daemon restart"
AUTO_VERIFY_MAX_JOBS = 1000
VERIFY_SAMPLE_SIZE = 100


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    requeued: int  # provisioning/running jobs moved to queued
    admitted: int  # jobs handed to the supervisor (requeued + already queued)
    left: dict[str, int] = field(default_factory=dict)  # state -> jobs not touched
    verified: int = 0  # jobs whose projection was compared with a fold of their log
    mismatches: tuple[str, ...] = ()  # job ids whose projection differs from the fold
    errors: tuple[str, ...] = ()  # "job_id: what went wrong"; those jobs were not admitted
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.errors


async def verify_projection(
    store: EventStore, jobs: list[Job], *, sample: int | None = None, seed: int = 0
) -> tuple[int, tuple[str, ...]]:
    """Compare the live projection of `jobs` with a fold of their events; returns (jobs checked,
    ids that differ). `sample` limits the check to that many jobs, picked with `seed`."""
    chosen = jobs
    if sample is not None and len(jobs) > sample:
        chosen = random.Random(seed).sample(jobs, sample)
    bad: list[str] = []
    for job in chosen:
        events: list[Event] = await store.read_job(job.job_id)
        try:
            folded = fold(events).get(job.job_id)
        except ProjectionError as exc:
            logger.error("log of %s cannot be folded: %s", job.job_id, exc)
            bad.append(job.job_id)
            continue
        if folded != job:
            logger.error("projection of %s differs from its log: %r != %r", job.job_id, job, folded)
            bad.append(job.job_id)
    return len(chosen), tuple(sorted(bad))


async def recover(
    store: EventStore,
    supervisor: Supervisor,
    *,
    prepare: Callable[[str], Awaitable[None]] | None = None,
    verify: bool | None = None,
    clock: Callable[[], float] = time.time,
) -> RecoveryReport:
    started = time.monotonic()
    jobs = await store.list_jobs()  # creation order
    checked = 0
    mismatches: tuple[str, ...] = ()
    if verify is not False:
        sample = None if verify or len(jobs) <= AUTO_VERIFY_MAX_JOBS else VERIFY_SAMPLE_SIZE
        checked, mismatches = await verify_projection(
            store, jobs, sample=sample, seed=await store.last_seq()
        )

    requeued = admitted = 0
    left: Counter[str] = Counter()
    errors: list[str] = []
    for job in jobs:
        if job.job_id in supervisor.running:
            left["live"] += 1
            continue
        if job.state in (JobState.PROVISIONING, JobState.RUNNING):
            try:
                await store.append_group(revoke(job, JobState.QUEUED, REASON_RESTART, clock()))
            except ProjectionError as exc:  # someone moved the job meanwhile: leave it
                errors.append(f"{job.job_id}: could not re-queue: {exc}")
                continue
            requeued += 1
        elif job.state is not JobState.QUEUED:
            left[job.state.value] += 1  # paused, over_budget, verifying, terminal
            continue
        try:
            if prepare is not None:
                await prepare(job.job_id)
            supervisor.start(job.job_id)
        except AlreadyRunning:
            left["live"] += 1
        except Exception as exc:
            logger.exception("job %s could not be admitted", job.job_id)
            errors.append(f"{job.job_id}: {type(exc).__name__}: {exc}")
        else:
            admitted += 1
    report = RecoveryReport(
        requeued=requeued,
        admitted=admitted,
        left=dict(left),
        verified=checked,
        mismatches=mismatches,
        errors=tuple(errors),
        duration_s=round(time.monotonic() - started, 4),
    )
    logger.info(
        "recovery: requeued=%d admitted=%d left=%s verified=%d mismatches=%d errors=%d in %.3fs",
        report.requeued,
        report.admitted,
        report.left,
        report.verified,
        len(report.mismatches),
        len(report.errors),
        report.duration_s,
    )
    return report


__all__ = [
    "AUTO_VERIFY_MAX_JOBS",
    "REASON_RESTART",
    "VERIFY_SAMPLE_SIZE",
    "RecoveryReport",
    "recover",
    "verify_projection",
]
