import itertools

import pytest

from switchboard.core import (
    TERMINAL_STATES,
    TRANSITIONS,
    IllegalTransition,
    JobState,
    can_transition,
    transition,
)

S = JobState

# Written out literally on purpose: an independent spec, not derived from TRANSITIONS.
LEGAL: frozenset[tuple[JobState, JobState]] = frozenset(
    {
        (S.QUEUED, S.PROVISIONING),
        (S.PROVISIONING, S.RUNNING),
        (S.PROVISIONING, S.FAILED),
        (S.PROVISIONING, S.QUEUED),  # recovery: a crash during provisioning is re-queueable
        (S.RUNNING, S.PAUSED),
        (S.RUNNING, S.OVER_BUDGET),
        (S.RUNNING, S.QUEUED),
        (S.RUNNING, S.VERIFYING),
        (S.RUNNING, S.FAILED),
        (S.PAUSED, S.QUEUED),
        (S.OVER_BUDGET, S.QUEUED),
        (S.VERIFYING, S.DONE),
        (S.VERIFYING, S.FAILED),
        (S.QUEUED, S.CANCELLED),
        (S.PROVISIONING, S.CANCELLED),
        (S.RUNNING, S.CANCELLED),
        (S.PAUSED, S.CANCELLED),
        (S.OVER_BUDGET, S.CANCELLED),
        (S.VERIFYING, S.CANCELLED),
    }
)

ALL_PAIRS = list(itertools.product(JobState, JobState))
NON_TERMINAL = [s for s in JobState if s not in TERMINAL_STATES]


def test_pair_count() -> None:
    assert len(ALL_PAIRS) == 81
    assert len(LEGAL) == 19


@pytest.mark.parametrize(("src", "dst"), ALL_PAIRS)
def test_transition_table(src: JobState, dst: JobState) -> None:
    if (src, dst) in LEGAL:
        assert can_transition(src, dst)
        assert transition(src, dst) is dst
    else:
        assert not can_transition(src, dst)
        with pytest.raises(IllegalTransition) as exc:
            transition(src, dst)
        assert exc.value.from_state is src
        assert exc.value.to_state is dst
        assert src.value in str(exc.value)
        assert dst.value in str(exc.value)


def test_illegal_transition_is_value_error() -> None:
    assert issubclass(IllegalTransition, ValueError)


def test_terminal_states() -> None:
    assert frozenset({S.DONE, S.FAILED, S.CANCELLED}) == TERMINAL_STATES


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES))
def test_terminal_states_have_no_outgoing(state: JobState) -> None:
    assert TRANSITIONS[state] == frozenset()


@pytest.mark.parametrize("state", NON_TERMINAL)
def test_non_terminal_can_be_cancelled(state: JobState) -> None:
    assert can_transition(state, S.CANCELLED)


@pytest.mark.parametrize("state", list(JobState))
def test_no_self_transitions(state: JobState) -> None:
    assert not can_transition(state, state)


def test_every_state_has_an_entry() -> None:
    assert set(TRANSITIONS) == set(JobState)


def test_transitions_are_immutable() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[S.QUEUED] = frozenset()  # type: ignore[index]
    with pytest.raises(AttributeError):
        TRANSITIONS[S.QUEUED].add(S.DONE)  # type: ignore[attr-defined]


def test_state_values() -> None:
    assert S.OVER_BUDGET.value == "over_budget"
    assert {s.value for s in S} == {
        "queued",
        "provisioning",
        "running",
        "paused",
        "over_budget",
        "verifying",
        "done",
        "failed",
        "cancelled",
    }
