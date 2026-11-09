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
    """What recovery may do with a tool call of unknown outcome (enforced by the effect journal)."""

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
class EffectResult:
    """What `ctx.effect` returns. `status` is applied, reconciled or interrupted; an
    interrupted effect was NOT re-applied (its outcome is unknown) and the agent decides."""

    effect_id: str
    status: str
    outcome: Mapping[str, Any]

    @property
    def interrupted(self) -> bool:
        return self.status == "interrupted"


class EffectCaller(Protocol):
    """The journal as an agent sees it (implemented by `runtime.effects.EffectJournal`)."""

    async def effect(
        self, kind: str, args: Mapping[str, Any], *, policy: ReplayPolicy
    ) -> EffectResult: ...


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
    effects: EffectCaller | None = None  # the effect journal, when the runner has a sink

    async def effect(
        self, kind: str, args: Mapping[str, Any], *, policy: ReplayPolicy
    ) -> EffectResult:
        """Perform an external effect through the journal (intent, apply, outcome)."""
        if self.effects is None:
            raise RuntimeError("no external sink configured: ctx.effect is unavailable")
        return await self.effects.effect(kind, args, policy=policy)

    @property
    def remaining(self) -> Remaining:
        return self.budget.remaining(self.usage)


class Agent(Protocol):
    config_hash: str  # hash of model id, system prompt and tool schemas
    tools: Sequence[ToolSpec]

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse: ...

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult: ...
