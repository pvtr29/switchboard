"""Job state machine: the states and the only legal moves between them."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class JobState(StrEnum):
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    PAUSED = "paused"
    OVER_BUDGET = "over_budget"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.DONE, JobState.FAILED, JobState.CANCELLED}
)

_S = JobState

# running -> queued exists for recovery: after a daemon restart or a lost lease the job is
# re-queued and resumes from its last checkpoint rather than being failed.
# provisioning -> queued exists for the same reason: a crash while the environment is being
# provisioned must leave the job re-queueable (provision() is idempotent, so it simply redoes it).
_EXPLICIT: dict[JobState, frozenset[JobState]] = {
    _S.QUEUED: frozenset({_S.PROVISIONING}),
    _S.PROVISIONING: frozenset({_S.RUNNING, _S.FAILED, _S.QUEUED}),
    _S.RUNNING: frozenset({_S.PAUSED, _S.OVER_BUDGET, _S.QUEUED, _S.VERIFYING, _S.FAILED}),
    _S.PAUSED: frozenset({_S.QUEUED}),
    _S.OVER_BUDGET: frozenset({_S.QUEUED}),
    _S.VERIFYING: frozenset({_S.DONE, _S.FAILED}),
}

# Cancellation is reachable from every non-terminal state; terminal states go nowhere.
TRANSITIONS: Mapping[JobState, frozenset[JobState]] = MappingProxyType(
    {
        state: (
            frozenset()
            if state in TERMINAL_STATES
            else _EXPLICIT.get(state, frozenset()) | {JobState.CANCELLED}
        )
        for state in JobState
    }
)


class IllegalTransition(ValueError):
    """Raised when a job is asked to move along an edge that is not in TRANSITIONS."""

    def __init__(self, from_state: JobState, to_state: JobState) -> None:
        super().__init__(f"illegal job transition: {from_state.value} -> {to_state.value}")
        self.from_state = from_state
        self.to_state = to_state


def can_transition(src: JobState, dst: JobState) -> bool:
    return dst in TRANSITIONS[src]


def transition(src: JobState, dst: JobState) -> JobState:
    """Return `dst` if src -> dst is legal, else raise IllegalTransition."""
    if not can_transition(src, dst):
        raise IllegalTransition(src, dst)
    return dst
