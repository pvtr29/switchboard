import dataclasses
from typing import Any

import pytest

from switchboard.core import (
    STEP_EVENT_TYPES,
    Budget,
    Event,
    EventType,
    Job,
    JobKind,
    JobState,
    ProjectionError,
    StaleEpoch,
    Usage,
    Verdict,
    apply,
    budget_from_dict,
    budget_to_dict,
    fold,
    usage_from_dict,
    usage_to_dict,
)

T = EventType
S = JobState


def ev(
    type_: EventType,
    payload: dict[str, Any] | None = None,
    *,
    seq: int | None = 1,
    ts: float = 10.0,
    epoch: int = 0,
    step: int | None = None,
    job: str = "j",
) -> Event:
    return Event(
        seq=seq, job_id=job, epoch=epoch, step_id=step, ts=ts, type=type_, payload=payload or {}
    )


CREATED = {"kind": "fake", "task_ref": "task-1", "budget": {"tokens": 100}, "seed": 7}


def created(epoch: int = 0, **over: Any) -> Job:
    return apply(None, ev(T.JOB_CREATED, {**CREATED, **over}, seq=1, ts=5.0, epoch=epoch))


def moved(job: Job, to: JobState, seq: int = 2, ts: float = 6.0) -> Job:
    payload = {"from": job.state.value, "to": to.value, "reason": "test"}
    return apply(job, ev(T.STATE_CHANGED, payload, seq=seq, ts=ts))


def test_job_created() -> None:
    job = created(epoch=2)
    assert job == Job(
        job_id="j",
        kind=JobKind.FAKE,
        task_ref="task-1",
        state=S.QUEUED,
        budget=Budget(tokens=100),
        usage=Usage(),
        epoch=2,
        last_seq=1,
        created_at=5.0,
        updated_at=5.0,
    )


def test_job_created_budget_defaults_to_unlimited() -> None:
    payload = {k: v for k, v in CREATED.items() if k != "budget"}
    assert apply(None, ev(T.JOB_CREATED, payload)).budget == Budget()


@pytest.mark.parametrize(
    "over",
    [
        {"kind": "nope"},
        {"kind": 3},
        {"task_ref": None},
        {"seed": "7"},
        {"seed": True},
        {"budget": [1]},
        {"budget": {"tokens": 1.5}},
        {"budget": {"tokens": -1}},
        {"budget": {"usd": 3}},
        {"budget": {"active_seconds": "5"}},
    ],
)
def test_job_created_bad_payload(over: dict[str, Any]) -> None:
    with pytest.raises(ProjectionError):
        apply(None, ev(T.JOB_CREATED, {**CREATED, **over}))


@pytest.mark.parametrize("missing", ["kind", "task_ref", "seed"])
def test_job_created_missing_field(missing: str) -> None:
    payload = {k: v for k, v in CREATED.items() if k != missing}
    with pytest.raises(ProjectionError, match=missing):
        apply(None, ev(T.JOB_CREATED, payload))


def test_second_job_created_rejected() -> None:
    with pytest.raises(ProjectionError, match="already exists"):
        apply(created(), ev(T.JOB_CREATED, CREATED, seq=2))


@pytest.mark.parametrize("type_", [t for t in EventType if t is not T.JOB_CREATED])
def test_event_for_unknown_job_rejected(type_: EventType) -> None:
    step = 0 if type_ in STEP_EVENT_TYPES else None
    with pytest.raises(ProjectionError, match="does not exist"):
        apply(None, ev(type_, step=step))


def test_seq_required() -> None:
    with pytest.raises(ProjectionError, match="seq"):
        apply(None, ev(T.JOB_CREATED, CREATED, seq=None))
    with pytest.raises(ProjectionError, match="seq"):
        apply(created(), ev(T.USAGE_RECORDED, seq=None))


def test_event_for_other_job_rejected() -> None:
    with pytest.raises(ProjectionError, match="applied to job"):
        apply(created(), ev(T.USAGE_RECORDED, seq=2, job="other"))


def test_state_changed() -> None:
    job = moved(created(), S.PROVISIONING, seq=9, ts=12.5)
    assert job.state is S.PROVISIONING
    assert (job.last_seq, job.updated_at, job.created_at) == (9, 12.5, 5.0)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"from": "running", "to": "paused", "reason": "x"}, "from=running but job is queued"),
        ({"from": "queued", "to": "running", "reason": "x"}, "illegal job transition"),
        ({"from": "queued", "to": "queued", "reason": "x"}, "illegal job transition"),
        ({"from": "queued", "to": "bogus", "reason": "x"}, "JobState"),
        ({"from": "bogus", "to": "provisioning", "reason": "x"}, "JobState"),
        ({"to": "provisioning", "reason": "x"}, "payload.from"),
        ({"from": "queued", "reason": "x"}, "payload.to"),
        ({"from": "queued", "to": "provisioning"}, "payload.reason"),
        ({"from": "queued", "to": "provisioning", "reason": 1}, "payload.reason"),
    ],
)
def test_state_changed_errors(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ProjectionError, match=match):
        apply(created(), ev(T.STATE_CHANGED, payload, seq=2))


def test_illegal_transition_is_chained() -> None:
    from switchboard.core import IllegalTransition

    payload = {"from": "queued", "to": "done", "reason": "x"}
    with pytest.raises(ProjectionError) as info:
        apply(created(), ev(T.STATE_CHANGED, payload, seq=2))
    assert isinstance(info.value.__cause__, IllegalTransition)


def test_usage_accumulates() -> None:
    job = created()
    job = apply(
        job,
        ev(
            T.USAGE_RECORDED,
            {
                "model": "m",
                "tokens_in": 10,
                "tokens_out": 5,
                "usd_micros": 300,
                "active_seconds": 1.5,
            },
            seq=2,
        ),
    )
    assert job.usage == Usage(active_seconds=1.5, tokens=15, usd_micros=300)
    job = apply(job, ev(T.USAGE_RECORDED, {"tokens_in": 1, "active_seconds": 2}, seq=3))
    assert job.usage == Usage(active_seconds=3.5, tokens=16, usd_micros=300)
    job = apply(job, ev(T.USAGE_RECORDED, seq=4))  # all fields default to 0
    assert job.usage == Usage(active_seconds=3.5, tokens=16, usd_micros=300)
    assert job.last_seq == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"tokens_in": "1"},
        {"tokens_out": 1.5},
        {"tokens_in": -1},
        {"usd_micros": True},
        {"usd_micros": 0.5},
        {"active_seconds": -1.0},
        {"active_seconds": "1"},
        {"model": 5},
    ],
)
def test_usage_bad_payload(payload: dict[str, Any]) -> None:
    with pytest.raises(ProjectionError):
        apply(created(), ev(T.USAGE_RECORDED, payload, seq=2))


def done_job() -> Job:
    job = created()
    for n, to in enumerate((S.PROVISIONING, S.RUNNING, S.VERIFYING, S.DONE), start=2):
        job = moved(job, to, seq=n)
    return job


@pytest.mark.parametrize("verdict", list(Verdict))
def test_verdict_recorded(verdict: Verdict) -> None:
    job = apply(
        done_job(), ev(T.VERDICT_RECORDED, {"verdict": verdict.value, "evidence": {}}, seq=9)
    )
    assert job.verdict is verdict
    assert job.state is S.DONE


@pytest.mark.parametrize(
    ("payload", "state"),
    [
        ({"verdict": "pass"}, S.QUEUED),
        ({"verdict": "pass"}, S.VERIFYING),
        ({"verdict": "maybe"}, S.DONE),
        ({}, S.DONE),
        ({"verdict": 1}, S.DONE),
    ],
)
def test_verdict_errors(payload: dict[str, Any], state: JobState) -> None:
    job = done_job() if state is S.DONE else created()
    if state is S.VERIFYING:
        for n, to in enumerate((S.PROVISIONING, S.RUNNING, S.VERIFYING), start=2):
            job = moved(job, to, seq=n)
    with pytest.raises(ProjectionError):
        apply(job, ev(T.VERDICT_RECORDED, payload, seq=20))


def test_step_completed() -> None:
    job = apply(created(), ev(T.STEP_COMPLETED, seq=2, step=4))
    assert job.last_step == 4
    job = apply(job, ev(T.STEP_COMPLETED, seq=3, step=5))
    assert job.last_step == 5


def test_lease_events() -> None:
    job = apply(
        created(),
        ev(T.LEASE_ACQUIRED, {"worker": "w1", "lease_until": 99.5}, seq=2, epoch=1),
    )
    assert (job.owner, job.lease_until, job.epoch) == ("w1", 99.5, 1)
    job = apply(job, ev(T.LEASE_LOST, {"reason": "recovery"}, seq=3, epoch=2))
    assert (job.owner, job.lease_until, job.epoch) == (None, None, 2)
    job = apply(job, ev(T.LEASE_ACQUIRED, {"worker": "w2"}, seq=4, epoch=3))
    assert (job.owner, job.lease_until, job.epoch) == ("w2", None, 3)  # no expiry


def test_events_must_carry_the_current_epoch() -> None:
    job = apply(created(), ev(T.LEASE_ACQUIRED, {"worker": "w1"}, seq=2, epoch=1))
    step = {"delta": [], "done": False}
    ok = apply(job, ev(T.STEP_COMPLETED, step, seq=3, epoch=1, step=0))
    assert ok.last_step == 0
    for epoch in (0, 2):  # the stale owner, and a writer from the future
        with pytest.raises(StaleEpoch, match="expected 1"):
            apply(job, ev(T.STEP_COMPLETED, step, seq=3, epoch=epoch, step=0))


@pytest.mark.parametrize("type_", [T.LEASE_ACQUIRED, T.LEASE_LOST])
@pytest.mark.parametrize("epoch", [0, 1, 3])
def test_ownership_events_must_advance_the_epoch_by_exactly_one(
    type_: EventType, epoch: int
) -> None:
    job = apply(created(), ev(T.LEASE_ACQUIRED, {"worker": "w1"}, seq=2, epoch=1))
    payload = {"worker": "w2", "reason": "takeover"}
    with pytest.raises(StaleEpoch, match="expected 2"):
        apply(job, ev(type_, payload, seq=3, epoch=epoch))
    assert apply(job, ev(type_, payload, seq=3, epoch=2)).epoch == 2


def test_two_claims_from_the_same_epoch_exactly_one_wins() -> None:
    job = created()
    first = apply(job, ev(T.LEASE_ACQUIRED, {"worker": "a"}, seq=2, epoch=1))
    with pytest.raises(StaleEpoch):
        apply(first, ev(T.LEASE_ACQUIRED, {"worker": "b"}, seq=3, epoch=1))


def test_stale_epoch_is_a_projection_error() -> None:
    assert issubclass(StaleEpoch, ProjectionError)  # one except clause still catches both


@pytest.mark.parametrize(
    "payload",
    [
        {"lease_until": 1.0},
        {"worker": 1, "lease_until": 1.0},
        {"worker": "w", "lease_until": "x"},
        {"worker": "w", "lease_until": True},
    ],
)
def test_lease_acquired_bad_payload(payload: dict[str, Any]) -> None:
    with pytest.raises(ProjectionError):
        apply(created(), ev(T.LEASE_ACQUIRED, payload, seq=2, epoch=1))


@pytest.mark.parametrize(
    "type_",
    [T.BUDGET_RESERVED, T.BUDGET_SETTLED, T.BUDGET_WARNING, T.EFFECT_INTENT, T.EFFECT_OUTCOME],
)
def test_journal_only_events_only_do_bookkeeping(type_: EventType) -> None:
    before = created()
    after = apply(before, ev(type_, {"anything": [1]}, seq=8, ts=77.0))
    assert after == dataclasses.replace(before, last_seq=8, updated_at=77.0)


def test_full_lifecycle() -> None:
    log = [
        ev(T.JOB_CREATED, CREATED, seq=1, ts=1.0),
        ev(
            T.STATE_CHANGED,
            {"from": "queued", "to": "provisioning", "reason": "slot"},
            seq=2,
            ts=2.0,
        ),
        ev(
            T.STATE_CHANGED,
            {"from": "provisioning", "to": "running", "reason": "ready"},
            seq=3,
            ts=3.0,
        ),
        ev(T.LEASE_ACQUIRED, {"worker": "w", "lease_until": 50.0}, seq=4, ts=3.5, epoch=1),
        ev(
            T.USAGE_RECORDED,
            {"tokens_in": 4, "tokens_out": 6, "active_seconds": 1.0},
            seq=5,
            ts=4.0,
            epoch=1,
        ),
        ev(T.STEP_COMPLETED, seq=6, ts=4.5, epoch=1, step=0),
        ev(
            T.STATE_CHANGED,
            {"from": "running", "to": "verifying", "reason": "finished"},
            seq=7,
            ts=5.0,
            epoch=1,
        ),
        ev(
            T.STATE_CHANGED,
            {"from": "verifying", "to": "done", "reason": "verified"},
            seq=8,
            ts=6.0,
            epoch=1,
        ),
        ev(T.VERDICT_RECORDED, {"verdict": "pass"}, seq=9, ts=6.0, epoch=1),
        ev(T.JOB_CREATED, CREATED, seq=10, ts=7.0, job="other"),
    ]
    jobs = fold(log)
    assert set(jobs) == {"j", "other"}
    j = jobs["j"]
    assert (j.state, j.verdict) == (S.DONE, Verdict.PASS_)
    assert j.usage == Usage(active_seconds=1.0, tokens=10)
    assert (j.last_step, j.owner, j.last_seq, j.epoch) == (0, "w", 9, 1)
    assert (j.created_at, j.updated_at) == (1.0, 6.0)
    assert jobs["other"].state is S.QUEUED
    assert fold([]) == {}


def test_fold_propagates_errors() -> None:
    with pytest.raises(ProjectionError):
        fold([ev(T.USAGE_RECORDED, seq=1)])


def test_json_helpers_roundtrip() -> None:
    b = Budget(active_seconds=2.5, tokens=10)
    assert budget_to_dict(b) == {"active_seconds": 2.5, "tokens": 10}  # None omitted
    assert budget_from_dict(budget_to_dict(b)) == b
    assert budget_from_dict({"tokens": None}) == Budget()
    assert budget_to_dict(Budget()) == {}
    u = Usage(active_seconds=1.0, tokens=2, usd_micros=3)
    assert usage_from_dict(usage_to_dict(u)) == u
    assert usage_from_dict({}) == Usage()
    for bad in ("x", {"tokens": "1"}, {"active_seconds": -1}):
        with pytest.raises(ProjectionError):
            usage_from_dict(bad)


@pytest.mark.parametrize("type_", [T.LLM_RESPONDED, T.STEP_STARTED, T.RESTORED])
def test_runtime_bookkeeping_events_change_nothing_but_seq(type_: EventType) -> None:
    job = created()
    step = 3 if type_ in STEP_EVENT_TYPES else None
    after = apply(job, ev(type_, {"anything": 1}, seq=2, step=step, ts=11.0))
    assert after == dataclasses.replace(job, last_seq=2, updated_at=11.0)
    assert after.last_step is None


def test_job_created_accepts_instructions_and_config_hash() -> None:
    payload = {**CREATED, "instructions": [{"role": "user", "content": "x"}], "config_hash": "h"}
    assert apply(None, ev(T.JOB_CREATED, payload)).task_ref == "task-1"
