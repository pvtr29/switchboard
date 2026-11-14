"""Pure domain logic: models, job state machine, projection fold. No I/O."""

from switchboard.core.models import (
    STEP_EVENT_TYPES,
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
from switchboard.core.projection import (
    ProjectionError,
    StaleEpoch,
    apply,
    budget_from_dict,
    budget_to_dict,
    fold,
    usage_from_dict,
    usage_to_dict,
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
    "STEP_EVENT_TYPES",
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
    "ProjectionError",
    "Remaining",
    "StaleEpoch",
    "Usage",
    "Verdict",
    "apply",
    "budget_from_dict",
    "budget_to_dict",
    "can_transition",
    "fold",
    "transition",
    "usage_from_dict",
    "usage_to_dict",
]
