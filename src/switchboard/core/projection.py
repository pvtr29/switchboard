"""Pure fold of events into the `jobs` projection. No I/O.

`apply` folds one event into a job's current projection; `fold` folds a whole log. The store
runs the same `apply` inside the transaction that writes the events, and `rebuild` re-runs it
over the full log, so live projection and rebuilt projection agree by construction.

Strictness: payloads are decoded with explicit type checks and a wrong or missing field raises
ProjectionError naming the event. Illegal state moves are also reported as ProjectionError (the
original IllegalTransition is chained as `__cause__`), so callers need to catch only one type.
`step_completed` may carry `active_seconds` (act + commit time), added to usage.
`budget_extended` adds its deltas to the job's limits (unlimited stays unlimited) and is legal
in any non-terminal state.
Unknown extra payload keys are ignored (payloads may grow within a version).

JSON conventions for Budget/Usage: `budget_to_dict` omits unlimited (None) dimensions, and
`budget_from_dict` treats a missing key (or an explicit null) as unlimited. Usage always
writes all three keys.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from switchboard.core.models import Budget, Event, EventType, Job, JobKind, Usage, Verdict
from switchboard.core.states import TERMINAL_STATES, IllegalTransition, JobState


class ProjectionError(ValueError):
    """An event cannot be folded into the job projection (bad payload or invalid sequence)."""


# ---- JSON helpers ------------------------------------------------------------------------


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def budget_to_dict(b: Budget) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    if b.active_seconds is not None:
        out["active_seconds"] = b.active_seconds
    if b.tokens is not None:
        out["tokens"] = b.tokens
    if b.usd_micros is not None:
        out["usd_micros"] = b.usd_micros
    return out


def budget_from_dict(d: object) -> Budget:
    if not isinstance(d, Mapping):
        raise ProjectionError(f"budget must be an object, got {type(d).__name__}")
    unknown = set(d) - {"active_seconds", "tokens", "usd_micros"}
    if unknown:
        raise ProjectionError(f"budget has unknown keys {sorted(unknown)}")
    secs, tok, usd = d.get("active_seconds"), d.get("tokens"), d.get("usd_micros")
    if secs is not None and not _is_num(secs):
        raise ProjectionError(f"budget.active_seconds must be a number, got {secs!r}")
    if tok is not None and not _is_int(tok):
        raise ProjectionError(f"budget.tokens must be an integer, got {tok!r}")
    if usd is not None and not _is_int(usd):
        raise ProjectionError(f"budget.usd_micros must be an integer, got {usd!r}")
    try:
        return Budget(active_seconds=secs, tokens=tok, usd_micros=usd)
    except ValueError as exc:
        raise ProjectionError(f"invalid budget: {exc}") from exc


def usage_to_dict(u: Usage) -> dict[str, float | int]:
    return {"active_seconds": u.active_seconds, "tokens": u.tokens, "usd_micros": u.usd_micros}


def usage_from_dict(d: object) -> Usage:
    if not isinstance(d, Mapping):
        raise ProjectionError(f"usage must be an object, got {type(d).__name__}")
    secs, tok, usd = d.get("active_seconds", 0.0), d.get("tokens", 0), d.get("usd_micros", 0)
    if not _is_num(secs) or not _is_int(tok) or not _is_int(usd):
        raise ProjectionError(f"invalid usage object: {dict(d)!r}")
    try:
        return Usage(active_seconds=float(secs), tokens=tok, usd_micros=usd)
    except ValueError as exc:
        raise ProjectionError(f"invalid usage: {exc}") from exc


# ---- payload decoding ----------------------------------------------------------------------


def _where(e: Event) -> str:
    return f"{e.type.value} event seq={e.seq} job={e.job_id}"


def _str(e: Event, key: str) -> str:
    v = e.payload.get(key)
    if not isinstance(v, str):
        raise ProjectionError(f"{_where(e)}: payload.{key} must be a string, got {v!r}")
    return v


def _int_default(e: Event, key: str) -> int:
    v = e.payload.get(key, 0)
    if not _is_int(v) or v < 0:
        raise ProjectionError(f"{_where(e)}: payload.{key} must be an integer >= 0, got {v!r}")
    return int(v)


def _enum[E: (JobState, Verdict, JobKind)](e: Event, key: str, cls: type[E]) -> E:
    raw = _str(e, key)
    try:
        return cls(raw)
    except ValueError:
        raise ProjectionError(
            f"{_where(e)}: payload.{key} is not a valid {cls.__name__}: {raw!r}"
        ) from None


# ---- fold ------------------------------------------------------------------------------------


def apply(job: Job | None, event: Event) -> Job:
    """Fold one event into `job` (None if the job does not exist yet) and return the result."""
    if event.seq is None:
        raise ProjectionError(f"{_where(event)}: cannot fold an event without a seq")
    if event.type is EventType.JOB_CREATED:
        if job is not None:
            raise ProjectionError(f"{_where(event)}: job already exists")
        return _created(event)
    if job is None:
        raise ProjectionError(f"{_where(event)}: job does not exist (no job_created)")
    if job.job_id != event.job_id:
        raise ProjectionError(f"{_where(event)}: event applied to job {job.job_id}")

    changes: dict[str, Any] = {}
    match event.type:
        case EventType.STATE_CHANGED:
            src = _enum(event, "from", JobState)
            dst = _enum(event, "to", JobState)
            _str(event, "reason")
            if src is not job.state:
                raise ProjectionError(
                    f"{_where(event)}: from={src.value} but job is {job.state.value}"
                )
            try:
                job.with_state(dst, at=event.ts)
            except IllegalTransition as exc:
                raise ProjectionError(f"{_where(event)}: {exc}") from exc
            changes["state"] = dst
        case EventType.USAGE_RECORDED:
            model = event.payload.get("model")
            if model is not None and not isinstance(model, str):
                raise ProjectionError(f"{_where(event)}: payload.model must be a string")
            secs = event.payload.get("active_seconds", 0.0)
            if not _is_num(secs) or secs < 0:
                raise ProjectionError(f"{_where(event)}: payload.active_seconds invalid: {secs!r}")
            delta = Usage(
                active_seconds=float(secs),
                tokens=_int_default(event, "tokens_in") + _int_default(event, "tokens_out"),
                usd_micros=_int_default(event, "usd_micros"),
            )
            changes["usage"] = job.usage + delta
        case EventType.VERDICT_RECORDED:
            verdict = _enum(event, "verdict", Verdict)
            if job.state is not JobState.DONE:
                raise ProjectionError(
                    f"{_where(event)}: verdict requires state done, job is {job.state.value}"
                )
            changes["verdict"] = verdict
        case EventType.STEP_COMPLETED:
            changes["last_step"] = event.step_id
            # Active seconds of the act/commit portion of the step ride in the commit itself,
            # so they are recorded atomically with it. Absent in older logs: counts as 0.
            secs = event.payload.get("active_seconds", 0.0)
            if not _is_num(secs) or secs < 0:
                raise ProjectionError(f"{_where(event)}: payload.active_seconds invalid: {secs!r}")
            if secs:
                changes["usage"] = job.usage + Usage(active_seconds=float(secs))
        case EventType.BUDGET_EXTENDED:
            if job.state in TERMINAL_STATES:
                raise ProjectionError(f"{_where(event)}: cannot extend a {job.state.value} job")
            by = event.payload
            secs = by.get("active_seconds", 0.0)
            if not _is_num(secs) or secs < 0:
                raise ProjectionError(f"{_where(event)}: payload.active_seconds invalid: {secs!r}")
            changes["budget"] = job.budget.extended(
                active_seconds=float(secs),
                tokens=_int_default(event, "tokens"),
                usd_micros=_int_default(event, "usd_micros"),
            )
        case EventType.LEASE_ACQUIRED:
            lease_until = event.payload.get("lease_until")
            if not isinstance(lease_until, int | float) or isinstance(lease_until, bool):
                raise ProjectionError(f"{_where(event)}: payload.lease_until must be a number")
            changes.update(
                owner=_str(event, "worker"), lease_until=float(lease_until), epoch=event.epoch
            )
        case EventType.LEASE_LOST:
            changes.update(owner=None, lease_until=None)
        case _:
            pass  # budget_*, effect_*: journal-only for the projection
    return dataclasses.replace(job, last_seq=event.seq, updated_at=event.ts, **changes)


def _created(e: Event) -> Job:
    assert e.seq is not None
    seed = e.payload.get("seed")
    if not _is_int(seed):
        raise ProjectionError(f"{_where(e)}: payload.seed must be an integer, got {seed!r}")
    return Job(
        job_id=e.job_id,
        kind=_enum(e, "kind", JobKind),
        task_ref=_str(e, "task_ref"),
        state=JobState.QUEUED,
        budget=budget_from_dict(e.payload.get("budget", {})),
        usage=Usage(),
        epoch=e.epoch,
        last_seq=e.seq,
        created_at=e.ts,
        updated_at=e.ts,
    )


def fold(events: Iterable[Event]) -> dict[str, Job]:
    """Fold events (in seq order) into projections keyed by job id."""
    jobs: dict[str, Job] = {}
    for e in events:
        jobs[e.job_id] = apply(jobs.get(e.job_id), e)
    return jobs
