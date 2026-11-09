"""The step loop: `run_job` drives one job through two-phase steps against the event log.

Per step n (see docs/plan.md "Execution model"):

1. `step_started(n, attempt)` is appended; the number of these is the restart counter.
2. Sample phase: unless `llm_responded(n)` is already logged, call `agent.sample` and append
   one group [llm_responded, usage_recorded]. A logged response is never sampled again.
3. Act phase: `agent.act` runs the tool calls.
4. Commit: one group [step_completed(n)] (+ running -> verifying when the agent is done).
   `step_completed` is the checkpoint: its delta is the conversation added by the step.

The runner never writes in a `finally`, and nothing about a half-finished step is committed,
so cancellation and crashes (including BaseException) simply leave the log at the last
boundary. Agent state at step n is job_created.instructions plus the deltas of steps 0..n-1.

Environment (optional, `env=`): a fresh job is provisioned and S(0) is snapshotted as
"initial" (its ref rides in the provisioning -> running state_changed payload as `env_ref`).
After the act phase and BEFORE the commit group the runner snapshots S(n+1) as
`s{n+1}-a{attempt}` and puts the ref in step_completed. The attempt suffix means a snapshot
orphaned by a crash between snapshot and commit can never be mistaken for a committed one; it
is harmless and the next gc deletes it. After each commit gc keeps the initial ref and the
last `keep_last` committed refs. On resume (job already running) the runner ALWAYS restores
S(n) first, because an interrupted act phase may have left uncommitted changes, and logs a
`restored` event. A missing snapshot fails the job with reason "snapshot missing".

Failure policy: an `Exception` from the agent fails the job (state_changed running -> failed,
reason "agent error: ...") and `run_job` RETURNS the failed job instead of raising, so a
supervisor needs no exception handling to learn the outcome. CancelledError and other
BaseExceptions propagate untouched. Store errors propagate.

Budgets and active time (docs/plan.md "Budgets"):

- Active time is measured with the event loop clock (`loop.time()`, so looptime tests work)
  from just after the boundary checks to the commit: restore, step_started, sample, act,
  snapshot. It lands in the log exactly once: sample time in the sample group's
  usage_recorded, act + snapshot time as `active_seconds` in the `step_completed` payload (the
  projection adds it; atomic with the commit, no extra event). A step that fails or is
  preempted records the unrecorded remainder in a usage_recorded in the same group as the
  state change. A step cancelled by the user (or a shutdown) loses its unrecorded time.
  Queued, paused and daemon-down time is never measured. The agent's own
  `Usage.active_seconds` is ignored: the runtime's measurement is the record.
- Boundary check (before step_started): an exhausted dimension moves running -> over_budget
  (reason "budget exhausted: <dimension>"); at >= 80% one `budget_warning` per (dimension,
  limit) is appended, deduplicated against the log, so it survives restarts and re-arms when
  an extension changes the limit.
- Hard preemption: sample and act (+ snapshot) each run under `asyncio.timeout` whose deadline
  is `origin + active limit + grace`, origin being the loop time at which recorded active
  usage was zero (recomputed from the projection every step: never from in-memory state).
  The store appends between the scopes are deliberately outside them so a preemption never
  lands in the middle of a group. On expiry nothing more of the step is committed (the logged
  llm_responded, if any, stays; the re-run act reuses it) and running -> over_budget with
  reason "active_seconds (preempted)" is recorded together with the wasted active seconds.
  Every attempt counts towards `max_attempts_per_step`.
- Tokens/usd: with a `BudgetManager`, each sample is preceded by a reservation (see
  budget.py); denial moves the job to over_budget before sampling.

Usage payload choice: `Usage` has one `tokens` total, so usage_recorded gets
tokens_in=usage.tokens and tokens_out=0 (the projection sums both). The full usage is also
stored in llm_responded so a logged response decodes losslessly. A duplicate llm_responded
still records its usage_recorded: wasted sampling is counted, not hidden.
"""

import asyncio
import dataclasses
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from switchboard.core import (
    Event,
    EventType,
    Job,
    JobState,
    ProjectionError,
    usage_from_dict,
    usage_to_dict,
)
from switchboard.env.base import Environment, SnapshotMissing
from switchboard.runtime.agent import (
    ActResult,
    Agent,
    Message,
    ModelResponse,
    ReplayPolicy,
    StepContext,
    ToolCall,
)
from switchboard.runtime.budget import (
    DEFAULT_GRACE_S,
    BudgetManager,
    DeadlineWatch,
    Reservation,
    ReservationDenied,
    TokenBucket,
)
from switchboard.runtime.effects import EffectJournal, ExternalSink
from switchboard.store import AppendStatus, EventStore

REASON_CONFIG_CHANGED = "config changed"
REASON_RESTART_LIMIT = "restart limit"
REASON_SNAPSHOT_MISSING = "snapshot missing"
INITIAL_SNAPSHOT = "initial"
PREEMPTED = "active_seconds (preempted)"


class JobNotRunnable(RuntimeError):
    """The job does not exist or is in a state `run_job` does not handle."""


class _LogView:
    """The parts of a job's event log the runner needs, decoded once per read."""

    def __init__(self, events: Sequence[Event]) -> None:
        self.created: Event | None = None
        self.responded: dict[int, Event] = {}
        self.completed: dict[int, Event] = {}
        self.started: Counter[int] = Counter()
        self.initial_ref: str | None = None
        self.ran = False  # the job has been in `running` at least once (see run_job)
        self.warned: set[tuple[str, float]] = set()  # (dimension, limit) already warned about
        for e in events:
            match e.type:
                case EventType.JOB_CREATED:
                    self.created = e
                case EventType.LLM_RESPONDED:
                    assert e.step_id is not None
                    self.responded[e.step_id] = e
                case EventType.STEP_COMPLETED:
                    assert e.step_id is not None
                    self.completed[e.step_id] = e
                case EventType.STATE_CHANGED:
                    if e.payload.get("to") == JobState.RUNNING.value:
                        self.ran = True
                        if e.payload.get("env_ref"):
                            self.initial_ref = str(e.payload["env_ref"])
                case EventType.STEP_STARTED:
                    assert e.step_id is not None
                    self.started[e.step_id] += 1
                case EventType.BUDGET_WARNING:
                    self.warned.add((str(e.payload["dimension"]), float(e.payload["limit"])))
                case _:
                    pass
        if self.created is None:
            raise JobNotRunnable("job has no job_created event")
        if sorted(self.completed) != list(range(len(self.completed))):
            raise RuntimeError(
                f"step_completed events are not contiguous: {sorted(self.completed)}"
            )

    @property
    def seed(self) -> int:
        assert self.created is not None
        return int(self.created.payload["seed"])

    @property
    def next_step(self) -> int:
        return len(self.completed)

    def history(self) -> list[Message]:
        assert self.created is not None
        out: list[Message] = list(self.created.payload.get("instructions", []))
        for n in range(len(self.completed)):
            out.extend(self.completed[n].payload["delta"])
        return out

    def ref_before(self, n: int) -> str | None:
        """S(n): the env_ref of step_completed(n-1), or the initial ref for n == 0."""
        if n == 0:
            return self.initial_ref
        ref = self.completed[n - 1].payload.get("env_ref")
        return None if ref is None else str(ref)

    def keep_refs(self, last: int) -> list[str]:
        refs = [
            r
            for n in range(max(0, len(self.completed) - last), len(self.completed))
            if (r := self.completed[n].payload.get("env_ref"))
        ]
        return [*([self.initial_ref] if self.initial_ref else []), *map(str, refs)]

    def recorded_hashes(self) -> set[str]:
        assert self.created is not None
        hashes = {
            str(e.payload["config_hash"])
            for e in (*self.responded.values(), *self.completed.values())
            if "config_hash" in e.payload
        }
        initial = self.created.payload.get("config_hash", "")
        if initial:
            hashes.add(str(initial))
        return hashes


def _decode_response(e: Event) -> ModelResponse:
    """Rebuild a ModelResponse from llm_responded. The logged `replay` is not decoded: it is
    the policy in force when the call was made and commit 12 reads it from the log."""
    p: Mapping[str, Any] = e.payload
    return ModelResponse(
        message=p["message"],
        tool_calls=tuple(
            ToolCall(id=c["id"], name=c["name"], args=c["args"]) for c in p["tool_calls"]
        ),
        usage=usage_from_dict(p["usage"]),
        model=p["model"],
    )


def _step_rng(seed: int, step: int) -> random.Random:
    return random.Random(f"switchboard:{seed}:{step}")


async def run_job(
    store: EventStore,
    job_id: str,
    agent: Agent,
    *,
    max_attempts_per_step: int = 5,
    clock: Callable[[], float] = time.time,
    env: Environment | None = None,
    keep_last: int = 2,
    should_pause: Callable[[], bool] | None = None,
    should_stop: Callable[[], bool] | None = None,
    sink: ExternalSink | None = None,
    budget: BudgetManager | None = None,
    rate_limiter: TokenBucket | None = None,
    grace_s: float = DEFAULT_GRACE_S,
    watch: DeadlineWatch | None = None,
) -> Job:
    """Run `job_id` until it is verifying, failed, or no longer ours; return its projection.

    A queued job is moved to running; a running job is resumed from its log. Any other state
    raises JobNotRunnable (recovery and the supervisor own those transitions later).

    A `queued` job is either fresh or a resumed one (paused or re-queued by recovery). The log
    tells them apart: a job whose log holds a `* -> running` state change has run before, so it
    is NOT provisioned again (that would wipe its environment); it goes queued -> provisioning
    -> running in one group and then takes the resume path (restore S(n)).

    `should_pause` is polled at every step boundary, before `step_started` of the next step
    (including the first). When it returns True the runner appends running -> paused and
    returns the paused job. It is cooperative: a step in flight always commits first.

    `should_stop` is polled at the same boundary, before `should_pause`. When it returns True the
    runner returns the job as it is and appends NOTHING: the job stays `running` and recovery
    resumes it (this is the daemon's graceful shutdown; contrast with pause).

    `sink` enables `ctx.effect`: each step attempt gets an `EffectJournal` over it (see
    runtime/effects.py). Effects are reconciled lazily, when the re-run act phase reaches the
    same `ctx.effect` call; restoring S(n) does not undo them.

    Budgets (see module docstring): `budget` enables token/usd reservation (and the granted
    output cap in `ctx.max_output_tokens`), `rate_limiter` is awaited before each sample,
    `grace_s` is the hard-preemption grace, `watch` lets the supervisor move a running step's
    deadline after an extension. Active time is always enforced.
    """
    job = await store.get_job(job_id)
    if job is None:
        raise JobNotRunnable(f"unknown job {job_id}")
    log = _LogView(await store.read_job(job_id))
    epoch = job.epoch

    def event(type_: EventType, payload: Mapping[str, Any], step: int | None = None) -> Event:
        return Event(
            job_id=job_id, epoch=epoch, step_id=step, ts=clock(), type=type_, payload=payload
        )

    def state_change(src: JobState, dst: JobState, reason: str) -> Event:
        return event(
            EventType.STATE_CHANGED, {"from": src.value, "to": dst.value, "reason": reason}
        )

    loop = asyncio.get_running_loop()
    watch = watch if watch is not None else DeadlineWatch(grace_s)
    watch.grace_s = grace_s
    mark: float | None = None  # loop time up to which active seconds are recorded in the log

    def take_active() -> float:
        """Active seconds since `mark`, advancing it: the caller MUST append them."""
        nonlocal mark
        now = loop.time()
        spent = 0.0 if mark is None else max(now - mark, 0.0)
        mark = now
        return spent

    def active_events() -> list[Event]:
        """A usage_recorded for the unrecorded active seconds of an unfinished step."""
        if mark is None:
            return []
        spent = take_active()
        return [
            event(EventType.USAGE_RECORDED, {"active_seconds": spent, "reason": "step aborted"})
        ]

    async def fail(reason: str) -> Job:
        await store.append_group(
            [*active_events(), state_change(JobState.RUNNING, JobState.FAILED, reason)]
        )
        return await _current()

    async def over_budget(reason: str, *, record_active: bool) -> Job:
        await store.append_group(
            [
                *(active_events() if record_active else []),
                state_change(JobState.RUNNING, JobState.OVER_BUDGET, reason),
            ]
        )
        return await _current()

    async def _current() -> Job:
        current = await store.get_job(job_id)
        assert current is not None
        return current

    resumed = job.state is JobState.RUNNING
    if job.state is JobState.QUEUED and log.ran:
        await store.append_group(
            [
                state_change(JobState.QUEUED, JobState.PROVISIONING, "resumed"),
                state_change(JobState.PROVISIONING, JobState.RUNNING, "resumed"),
            ]
        )
        resumed = True
        job = await _current()
    elif env is not None and job.state in (JobState.QUEUED, JobState.PROVISIONING):
        # A crash while provisioning leaves the job in `provisioning`; provision() is
        # idempotent, so redoing it is safe.
        if job.state is JobState.QUEUED:
            await store.append_group(
                [state_change(JobState.QUEUED, JobState.PROVISIONING, "runner started")]
            )
        await env.provision()
        initial = await env.snapshot(INITIAL_SNAPSHOT)
        running = state_change(JobState.PROVISIONING, JobState.RUNNING, "environment ready")
        await store.append_group(
            [event(EventType.STATE_CHANGED, {**running.payload, "env_ref": initial})]
        )
        log = _LogView(await store.read_job(job_id))
        job = await _current()
    elif job.state is JobState.QUEUED:
        await store.append_group(
            [
                state_change(JobState.QUEUED, JobState.PROVISIONING, "runner started"),
                state_change(JobState.PROVISIONING, JobState.RUNNING, "no environment"),
            ]
        )
        job = await _current()
    elif job.state is not JobState.RUNNING:
        raise JobNotRunnable(f"job {job_id} is {job.state.value}, expected queued or running")

    # Config check happens after the queued -> running move, so a mismatch can fail the job
    # through a legal edge; a pure resume appends nothing but the failure.
    if log.recorded_hashes() - {agent.config_hash}:
        return await fail(REASON_CONFIG_CHANGED)

    replay_by_tool = {t.name: t.replay for t in agent.tools}
    needs_restore = resumed and env is not None

    while True:
        job = await _current()
        # Set synchronously after the read: an extension landing later calls limit_changed.
        watch.limit = job.budget.active_seconds
        mark = None  # nothing of this step is unrecorded yet
        if job.state is not JobState.RUNNING:
            return job  # finished, failed or cancelled by someone else
        if should_stop is not None and should_stop():
            return job  # drain: a boundary, nothing written, the job stays running
        if should_pause is not None and should_pause():
            # A boundary: the previous step is committed and the next has not started, so the
            # log is consistent. Nothing but the state move is written.
            await store.append_group(
                [state_change(JobState.RUNNING, JobState.PAUSED, "paused by user")]
            )
            return await _current()
        # Budget boundary (cooperative level). warnings() excludes exhausted dimensions, so a
        # boundary either warns or stops, never both.
        exhausted = job.budget.exhausted(job.usage)
        if exhausted:
            return await over_budget(f"budget exhausted: {exhausted[0].value}", record_active=False)
        warnings = [
            event(
                EventType.BUDGET_WARNING,
                {
                    "dimension": d.value,
                    "used": job.usage.get(d),
                    "limit": lim,
                    "threshold": 0.8,
                },
            )
            for d in job.budget.warnings(job.usage)
            if (lim := job.budget.limit(d)) is not None and (d.value, lim) not in log.warned
        ]
        if warnings:
            await store.append_group(warnings)
            log.warned.update(
                (str(w.payload["dimension"]), float(w.payload["limit"])) for w in warnings
            )
        n = log.next_step
        attempt = log.started[n]
        if attempt >= max_attempts_per_step:
            return await fail(REASON_RESTART_LIMIT)
        mark = loop.time()
        watch.origin = mark - job.usage.active_seconds
        if env is not None and needs_restore:
            needs_restore = False
            ref = log.ref_before(n)
            try:
                if ref is None:
                    raise SnapshotMissing(f"no snapshot recorded for S({n})")
                await env.restore(ref)
            except SnapshotMissing:
                return await fail(REASON_SNAPSHOT_MISSING)
            await store.append_group(
                [event(EventType.RESTORED, {"step": n, "env_ref": ref, "reason": "resume"})]
            )
        await store.append_group([event(EventType.STEP_STARTED, {"attempt": attempt}, n)])

        ctx = StepContext(
            job_id=job_id,
            step=n,
            attempt=attempt,
            rng=_step_rng(log.seed, n),
            budget=job.budget,
            usage=job.usage,
            workdir=env.workdir if env is not None else None,
            effects=(
                EffectJournal(store, sink, job_id=job_id, step=n, epoch=epoch, clock=clock)
                if sink is not None
                else None
            ),
        )
        history = log.history()

        # Only the agent's own calls are wrapped: an exception there is the agent's failure and
        # fails the job. Store errors (StoreClosed, disk errors) are infrastructure faults and
        # must propagate to the supervisor/recovery instead of being recorded as agent errors.
        # ---- sample phase -----------------------------------------------------------------
        logged = log.responded.get(n)
        if logged is not None:
            response = _decode_response(logged)
        else:
            reservation: Reservation | None = None
            if budget is not None:
                try:
                    reservation = budget.reserve(
                        job,
                        agent.estimate_input_tokens(history),
                        agent.max_output_tokens,
                        agent.pricing,
                    )
                except ReservationDenied as denied:
                    return await over_budget(
                        f"budget exhausted: {denied.dimension.value} ({denied.scope})",
                        record_active=True,
                    )
                ctx = dataclasses.replace(ctx, max_output_tokens=reservation.max_output_tokens)
            sample_error: Exception | None = None
            try:
                async with asyncio.timeout(None) as cm:
                    watch.arm(cm)
                    try:
                        if rate_limiter is not None and reservation is not None:
                            await rate_limiter.acquire(
                                min(reservation.tokens, rate_limiter.capacity)
                            )
                        response = await agent.sample(ctx, tuple(history))
                    except Exception as exc:
                        sample_error = exc
            except TimeoutError:
                if budget is not None and reservation is not None:
                    budget.release(reservation)
                if cm.expired():
                    return await over_budget(PREEMPTED, record_active=True)
                raise
            except BaseException:
                if budget is not None and reservation is not None:
                    budget.release(reservation)
                raise
            finally:
                watch.disarm()
            if sample_error is not None:
                if budget is not None and reservation is not None:
                    budget.release(reservation)
                return await fail(f"agent error: {type(sample_error).__name__}: {sample_error}")
            if budget is not None and reservation is not None:
                budget.settle(reservation, response.usage)
            results = await store.append_group(
                [
                    event(
                        EventType.LLM_RESPONDED,
                        {
                            "message": response.message,
                            "tool_calls": [
                                {
                                    "id": c.id,
                                    "name": c.name,
                                    "args": c.args,
                                    "replay": replay_by_tool.get(c.name, ReplayPolicy.UNSAFE).value,
                                }
                                for c in response.tool_calls
                            ],
                            "usage": usage_to_dict(response.usage),
                            "model": response.model,
                            "config_hash": agent.config_hash,
                        },
                        n,
                    ),
                    event(
                        EventType.USAGE_RECORDED,
                        {
                            "tokens_in": response.usage.tokens,
                            "tokens_out": 0,
                            "usd_micros": response.usage.usd_micros,
                            # Measured, not the agent's claim: the sample phase's active time.
                            "active_seconds": take_active(),
                            "model": response.model,
                        },
                    ),
                ]
            )
            if results[0].status is AppendStatus.DUPLICATE:
                # Another runner logged n first: its response is the one of record.
                log = _LogView(await store.read_job(job_id))
                response = _decode_response(log.responded[n])

        # ---- act phase (+ snapshot): inside the preemption scope -----------------------------
        act_error: Exception | None = None
        env_ref: str | None = None
        try:
            async with asyncio.timeout(None) as cm:
                watch.arm(cm)
                try:
                    result: ActResult = await agent.act(ctx, response)
                except Exception as exc:
                    act_error = exc
                else:
                    if env is not None:
                        env_ref = await env.snapshot(f"s{n + 1}-a{attempt}")
        except TimeoutError:
            if cm.expired():
                return await over_budget(PREEMPTED, record_active=True)
            raise
        finally:
            watch.disarm()
        if act_error is not None:
            return await fail(f"agent error: {type(act_error).__name__}: {act_error}")

        # ---- commit -----------------------------------------------------------------------
        delta: list[Message] = [response.message, *result.messages]
        group = [
            event(
                EventType.STEP_COMPLETED,
                {
                    "delta": delta,
                    "done": result.done,
                    "summary": result.summary,
                    "config_hash": agent.config_hash,
                    "env_ref": env_ref,
                    # Act + commit time rides in the commit itself: atomic, no extra event.
                    "active_seconds": take_active(),
                },
                n,
            )
        ]
        if result.done:
            group.append(state_change(JobState.RUNNING, JobState.VERIFYING, "agent finished"))
        try:
            committed = await store.append_group(group)
        except ProjectionError:
            # The state move was refused: someone else already moved the job (and the group
            # rolled back). Re-read below and let the log decide.
            committed = []
        log = _LogView(await store.read_job(job_id))
        if env is not None and n in log.completed:
            await env.gc(log.keep_refs(keep_last))
        if committed and committed[0].status is AppendStatus.INSERTED and result.done:
            return await _current()
        # Otherwise (duplicate commit, refused group) the log is authoritative; the loop top
        # returns if the job is no longer running, else continues from the log's next step.
        if n not in log.completed:
            raise RuntimeError(f"step {n} of job {job_id} was neither committed nor completed")


__all__ = ["JobNotRunnable", "run_job"]
