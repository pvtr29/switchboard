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

Failure policy: an `Exception` from the agent fails the job (state_changed running -> failed,
reason "agent error: ...") and `run_job` RETURNS the failed job instead of raising, so a
supervisor needs no exception handling to learn the outcome. CancelledError and other
BaseExceptions propagate untouched. Store errors propagate.

Usage payload choice: `Usage` has one `tokens` total, so usage_recorded gets
tokens_in=usage.tokens and tokens_out=0 (the projection sums both). The full usage is also
stored in llm_responded so a logged response decodes losslessly. A duplicate llm_responded
still records its usage_recorded: wasted sampling is counted, not hidden.
"""

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
from switchboard.runtime.agent import (
    ActResult,
    Agent,
    Message,
    ModelResponse,
    ReplayPolicy,
    StepContext,
    ToolCall,
)
from switchboard.store import AppendStatus, EventStore

REASON_CONFIG_CHANGED = "config changed"
REASON_RESTART_LIMIT = "restart limit"


class JobNotRunnable(RuntimeError):
    """The job does not exist or is in a state `run_job` does not handle."""


class _LogView:
    """The parts of a job's event log the runner needs, decoded once per read."""

    def __init__(self, events: Sequence[Event]) -> None:
        self.created: Event | None = None
        self.responded: dict[int, Event] = {}
        self.completed: dict[int, Event] = {}
        self.started: Counter[int] = Counter()
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
                case EventType.STEP_STARTED:
                    assert e.step_id is not None
                    self.started[e.step_id] += 1
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
) -> Job:
    """Run `job_id` until it is verifying, failed, or no longer ours; return its projection.

    A queued job is moved to running; a running job is resumed from its log. Any other state
    raises JobNotRunnable (recovery and the supervisor own those transitions later).
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

    async def fail(reason: str) -> Job:
        await store.append_group([state_change(JobState.RUNNING, JobState.FAILED, reason)])
        return await _current()

    async def _current() -> Job:
        current = await store.get_job(job_id)
        assert current is not None
        return current

    if job.state is JobState.QUEUED:
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

    while True:
        job = await _current()
        if job.state is not JobState.RUNNING:
            return job  # finished, failed or cancelled by someone else
        n = log.next_step
        attempt = log.started[n]
        if attempt >= max_attempts_per_step:
            return await fail(REASON_RESTART_LIMIT)
        await store.append_group([event(EventType.STEP_STARTED, {"attempt": attempt}, n)])

        ctx = StepContext(
            job_id=job_id,
            step=n,
            attempt=attempt,
            rng=_step_rng(log.seed, n),
            budget=job.budget,
            usage=job.usage,
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
            try:
                response = await agent.sample(ctx, tuple(history))
            except Exception as exc:
                return await fail(f"agent error: {type(exc).__name__}: {exc}")
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
                            "active_seconds": response.usage.active_seconds,
                            "model": response.model,
                        },
                    ),
                ]
            )
            if results[0].status is AppendStatus.DUPLICATE:
                # Another runner logged n first: its response is the one of record.
                log = _LogView(await store.read_job(job_id))
                response = _decode_response(log.responded[n])

        # ---- act phase -------------------------------------------------------------------
        try:
            result: ActResult = await agent.act(ctx, response)
        except Exception as exc:
            return await fail(f"agent error: {type(exc).__name__}: {exc}")

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
                    "env_ref": None,
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
        if committed and committed[0].status is AppendStatus.INSERTED and result.done:
            return await _current()
        # Otherwise (duplicate commit, refused group) the log is authoritative; the loop top
        # returns if the job is no longer running, else continues from the log's next step.
        if n not in log.completed:
            raise RuntimeError(f"step {n} of job {job_id} was neither committed nor completed")


__all__ = ["JobNotRunnable", "run_job"]
