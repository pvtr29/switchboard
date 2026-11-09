"""Agent protocol: the runtime owns the step loop, the agent supplies two callbacks.

`sample` asks the model what to do; `act` carries the chosen tool calls out. The runtime logs
the sample result (`llm_responded`) before `act` runs and commits the step afterwards, so an
agent never touches the store and a logged response is never sampled twice.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from switchboard.core import Budget, Remaining, Usage

type Message = Mapping[str, Any]
"""A JSON-compatible chat message, e.g. {"role": "assistant", "content": "..."}."""


class ReplayPolicy(StrEnum):
    """What recovery may do with a tool call whose outcome is unknown (enforced in commit 12)."""

    SAFE = "safe"  # only touches the environment, which recovery restores: re-run freely
    KEYED = "keyed"  # re-run with the same idempotency key
    UNSAFE = "unsafe"  # never re-run; reconcile with the sink


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    replay: ReplayPolicy


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    message: Message  # the assistant message to append to the history
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    model: str


@dataclass(frozen=True, slots=True)
class ActResult:
    messages: tuple[Message, ...]  # tool results to append to the history
    done: bool
    summary: str = ""


@dataclass(frozen=True, slots=True)
class StepContext:
    """What an agent may know about the step it is running. Read-only by convention.

    `rng` is seeded from (job seed, step) only, not the attempt, so a re-run of a step draws
    the same numbers. `budget` and `usage` are the projection's view when the step began;
    enforcement is the runtime's business (commit 13), agents merely see how much is left.
    """

    job_id: str
    step: int
    attempt: int
    rng: random.Random
    budget: Budget = field(default_factory=Budget)
    usage: Usage = field(default_factory=Usage)
    workdir: Path | None = None  # the job's environment directory, if it has one

    @property
    def remaining(self) -> Remaining:
        return self.budget.remaining(self.usage)


class Agent(Protocol):
    config_hash: str  # hash of model id, system prompt and tool schemas
    tools: Sequence[ToolSpec]

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse: ...

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult: ...
