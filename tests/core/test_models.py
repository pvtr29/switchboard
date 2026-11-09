import dataclasses
from typing import Any

import pytest

from switchboard.core import (
    Budget,
    Dimension,
    Event,
    EventType,
    IllegalTransition,
    Job,
    JobKind,
    JobState,
    Remaining,
    Usage,
    Verdict,
)

D = Dimension


def test_enum_values() -> None:
    assert Verdict.PASS_.value == "pass"
    assert {v.value for v in Verdict} == {"pass", "fail", "suspected_hack"}
    assert {k.value for k in JobKind} == {"fake", "rollout"}
    assert len(EventType) == 12
    assert EventType.STEP_COMPLETED.value == "step_completed"


# --- Usage ---------------------------------------------------------------------------------


def test_usage_defaults_and_add() -> None:
    assert Usage() == Usage(active_seconds=0.0, tokens=0, usd_micros=0)
    a = Usage(active_seconds=1.5, tokens=10, usd_micros=100)
    b = Usage(active_seconds=0.5, tokens=5, usd_micros=250)
    assert a + b == Usage(active_seconds=2.0, tokens=15, usd_micros=350)
    assert a == Usage(active_seconds=1.5, tokens=10, usd_micros=100)  # operands untouched


def test_usage_add_wrong_type() -> None:
    with pytest.raises(TypeError):
        Usage() + 1  # type: ignore[operator]


@pytest.mark.parametrize("field", ["active_seconds", "tokens", "usd_micros"])
def test_usage_rejects_negative(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Usage(**{field: -1})


# --- Budget --------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["active_seconds", "tokens", "usd_micros"])
def test_budget_rejects_negative(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Budget(**{field: -1})


def test_budget_unlimited() -> None:
    b = Budget()
    u = Usage(active_seconds=1e9, tokens=10**12, usd_micros=10**12)
    assert b.remaining(u) == Remaining()
    assert b.exhausted(u) == ()
    assert b.warnings(u) == ()


def test_budget_remaining() -> None:
    b = Budget(active_seconds=10.0, tokens=100, usd_micros=None)
    r = b.remaining(Usage(active_seconds=4.0, tokens=30, usd_micros=999))
    assert r == Remaining(active_seconds=6.0, tokens=70, usd_micros=None)


def test_budget_remaining_clamped_at_zero() -> None:
    b = Budget(active_seconds=10.0, tokens=100, usd_micros=5)
    r = b.remaining(Usage(active_seconds=11.0, tokens=500, usd_micros=5))
    assert r == Remaining(active_seconds=0, tokens=0, usd_micros=0)


def test_exhausted_exactly_at_limit_and_over() -> None:
    b = Budget(active_seconds=10.0, tokens=100, usd_micros=1000)
    assert b.exhausted(Usage(active_seconds=9.99, tokens=99, usd_micros=999)) == ()
    assert b.exhausted(Usage(active_seconds=10.0, tokens=99, usd_micros=999)) == (D.ACTIVE_SECONDS,)
    assert b.exhausted(Usage(active_seconds=11.0, tokens=100, usd_micros=1000)) == (
        D.ACTIVE_SECONDS,
        D.TOKENS,
        D.USD_MICROS,
    )


def test_zero_limit_is_exhausted_immediately() -> None:
    b = Budget(tokens=0)
    assert b.exhausted(Usage()) == (D.TOKENS,)
    assert b.warnings(Usage()) == ()


def test_warnings_threshold_boundaries() -> None:
    b = Budget(tokens=100, usd_micros=1000)
    assert b.warnings(Usage(tokens=79, usd_micros=799)) == ()
    # exactly at the threshold warns
    assert b.warnings(Usage(tokens=80, usd_micros=799)) == (D.TOKENS,)
    # exactly at the limit is exhausted, not a warning
    assert b.warnings(Usage(tokens=100, usd_micros=900)) == (D.USD_MICROS,)
    assert b.exhausted(Usage(tokens=100, usd_micros=900)) == (D.TOKENS,)


def test_warnings_custom_threshold() -> None:
    b = Budget(tokens=100)
    assert b.warnings(Usage(tokens=50), threshold=0.5) == (D.TOKENS,)
    assert b.warnings(Usage(tokens=49), threshold=0.5) == ()


def test_budget_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Budget().tokens = 1  # type: ignore[misc]


# --- Event ---------------------------------------------------------------------------------


def make_event(**overrides: Any) -> Event:
    fields: dict[str, Any] = {
        "job_id": "j1",
        "epoch": 0,
        "ts": 1.0,
        "type": EventType.JOB_CREATED,
        "payload": {},
    }
    fields.update(overrides)
    return Event(**fields)


def test_event_defaults() -> None:
    e = make_event()
    assert e.seq is None
    assert e.step_id is None
    assert e.v == 1


def test_event_step_completed_requires_step_id() -> None:
    with pytest.raises(ValueError, match="step_id"):
        make_event(type=EventType.STEP_COMPLETED)
    e = make_event(type=EventType.STEP_COMPLETED, step_id=0)
    assert e.step_id == 0


def test_event_step_id_forbidden_on_other_types() -> None:
    with pytest.raises(ValueError, match="step_id"):
        make_event(type=EventType.USAGE_RECORDED, step_id=3)


def test_event_negative_step_id() -> None:
    with pytest.raises(ValueError, match="step_id"):
        make_event(type=EventType.STEP_COMPLETED, step_id=-1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"epoch": -1}, "epoch"),
        ({"v": 0}, "v must"),
        ({"job_id": ""}, "job_id"),
    ],
)
def test_event_field_validation(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_event(**overrides)


def test_event_payload_is_read_only_copy() -> None:
    src: dict[str, Any] = {"a": 1}
    e = make_event(payload=src)
    src["a"] = 2
    src["b"] = 3
    assert dict(e.payload) == {"a": 1}
    with pytest.raises(TypeError):
        e.payload["a"] = 5  # type: ignore[index]


def test_event_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_event().seq = 1  # type: ignore[misc]


# --- Job -----------------------------------------------------------------------------------


def make_job(**overrides: Any) -> Job:
    fields: dict[str, Any] = {
        "job_id": "j1",
        "kind": JobKind.FAKE,
        "task_ref": "task/1",
        "budget": Budget(),
        "last_seq": 0,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    fields.update(overrides)
    return Job(**fields)


def test_job_defaults() -> None:
    j = make_job()
    assert j.state is JobState.QUEUED
    assert j.verdict is None
    assert j.usage == Usage()
    assert j.owner is None
    assert j.epoch == 0
    assert j.last_step is None


def test_with_state_legal() -> None:
    j = make_job()
    j2 = j.with_state(JobState.PROVISIONING, at=5.0)
    assert j2.state is JobState.PROVISIONING
    assert j2.updated_at == 5.0
    assert j2.created_at == 1.0
    assert j.state is JobState.QUEUED  # original unchanged


def test_with_state_illegal() -> None:
    j = make_job()
    with pytest.raises(IllegalTransition) as exc:
        j.with_state(JobState.DONE, at=2.0)
    assert exc.value.from_state is JobState.QUEUED
    assert exc.value.to_state is JobState.DONE


def test_with_state_from_terminal_is_illegal() -> None:
    j = make_job(state=JobState.CANCELLED)
    with pytest.raises(IllegalTransition):
        j.with_state(JobState.QUEUED, at=2.0)


def test_job_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_job().state = JobState.DONE  # type: ignore[misc]


@pytest.mark.parametrize("state", [s for s in JobState if s is not JobState.DONE])
def test_verdict_only_when_done(state: JobState) -> None:
    with pytest.raises(ValueError, match="verdict"):
        make_job(state=state, verdict=Verdict.PASS_)


def test_verdict_allowed_when_done() -> None:
    assert make_job(state=JobState.DONE, verdict=Verdict.SUSPECTED_HACK).verdict is (
        Verdict.SUSPECTED_HACK
    )


def test_verifying_to_done_then_verdict_via_replace() -> None:
    j = make_job(state=JobState.VERIFYING).with_state(JobState.DONE, at=3.0)
    assert dataclasses.replace(j, verdict=Verdict.PASS_).verdict is Verdict.PASS_
