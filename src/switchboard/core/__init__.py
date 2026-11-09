"""Pure domain logic: models, job state machine, projection fold. No I/O."""

from switchboard.core.models import (
    Budget,
    Dimension,
    Event,
    EventType,
    Job,
    JobKind,
    Remaining,
    Usage,
    Verdict,
)
from switchboard.core.states import (
    TERMINAL_STATES,
    TRANSITIONS,
    IllegalTransition,
    JobState,
    can_transition,
    transition,
)

__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "Budget",
    "Dimension",
    "Event",
    "EventType",
    "IllegalTransition",
    "Job",
    "JobKind",
    "JobState",
    "Remaining",
    "Usage",
    "Verdict",
    "can_transition",
    "transition",
]
