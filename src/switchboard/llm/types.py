"""Provider-neutral types of the LLM layer: what a model turn is, and how a call can fail."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StopKind(StrEnum):
    TOOL_USE = "tool_use"  # the model asked for tool calls
    END = "end"  # the model stopped on its own, without tool calls
    MAX_TOKENS = "max_tokens"  # cut off by the output cap
    REFUSED = "refused"  # blocked by the provider's safety policy
    MALFORMED = "malformed"  # the provider could not parse the model's own tool call
    OTHER = "other"  # any finish reason we do not map


@dataclass(frozen=True, slots=True)
class ToolDef:
    """A tool as the model sees it. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TurnUsage:
    """Token counts of one call. `input_tokens` includes `cached_tokens`; `thinking_tokens` are
    billed and capped as output but reported apart by the provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "thinking": self.thinking_tokens,
            "cached": self.cached_tokens,
        }


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One FAILED attempt (retried, or the end of a model's attempts). A successful call leaves
    no record; the turn's `model` says who answered."""

    model: str
    attempt: int  # 1-based, per model
    error: str
    retryable: bool
    delay_s: float  # backoff slept before the next attempt (0 when none follows)
    action: str  # "retry" | "next_model" | "give_up"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "attempt": self.attempt,
            "error": self.error,
            "retryable": self.retryable,
            "delay_s": round(self.delay_s, 3),
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One model answer.

    `message` is the provider-native assistant content as JSON; stored in the event log and
    handed back to the client later, it must round-trip EXACTLY (thought signatures included).
    Shape: {"role": "assistant", "content": text, "tool_calls": [...], "native": {...}}.
    """

    message: Mapping[str, Any]
    tool_calls: tuple[ToolCall, ...]
    text: str
    stop: StopKind
    stop_raw: str
    usage: TurnUsage
    model: str  # the model that actually answered (may be a fallback)
    ttfc_s: float | None = None  # time to first stream chunk
    call_s: float = 0.0  # duration of the successful attempt
    failed_attempts: tuple[AttemptRecord, ...] = field(default=())

    @property
    def fell_back(self) -> bool:
        """True if an earlier model of the list was given up on before this one answered."""
        return any(a.action == "next_model" for a in self.failed_attempts)


class LLMError(Exception):
    """Base of every error the LLM layer raises on purpose."""

    def __init__(self, message: str, *, attempts: tuple[AttemptRecord, ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts


class Unavailable(LLMError):
    """Every model stayed unavailable through all its retries (503/504/timeouts/...)."""


class Refused(LLMError):
    """The provider refused to answer (safety block). Retrying the same input will not help."""


class BadRequest(LLMError):
    """The provider rejected the request (4xx other than 429): a bug or bad history, not
    transient. Never retried, never falls back."""


class Truncated(LLMError):
    """The output cap cut the answer off in the middle of a tool call."""
