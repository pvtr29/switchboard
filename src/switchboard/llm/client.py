"""The provider-agnostic client protocol and the JSON message format agents keep history in.

History is a list of plain JSON messages (the runtime stores them in the event log, so this is
also what a resumed job hands back). Four shapes:

    {"role": "user", "content": "text"}
    {"role": "assistant", "content": "text", "tool_calls": [{"id", "name", "args"}],
     "native": {"provider": "...", "content": {...}}}          # a ModelTurn.message
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "text",
     "is_error": false}                                         # one per tool call
    any message may carry extra keys (e.g. "recovery": true); clients ignore what they do not know.

A client converts that to its provider's types on every call; `native` is what makes the
assistant turn round-trip byte-for-byte (thought signatures), the rest is a readable fallback.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from switchboard.llm.types import ModelTurn, ToolCall, ToolDef

type JsonMessage = Mapping[str, Any]

ROLES = ("user", "assistant", "tool")


class LLMClient(Protocol):
    def fingerprint(self) -> Mapping[str, Any]:
        """JSON describing everything about the client that changes behaviour (provider, model
        list, effort): it goes into the agent's config hash."""
        ...

    async def complete(
        self,
        history: Sequence[JsonMessage],
        tools: Sequence[ToolDef],
        *,
        system: str,
        max_output_tokens: int | None,
        deadline_s: float,
    ) -> ModelTurn:
        """One model answer for `history`. `max_output_tokens` is a hard cap (thinking included
        for providers that count it); `deadline_s` bounds each ATTEMPT. Raises `LLMError`s."""
        ...


def user_message(text: str, **extra: Any) -> dict[str, Any]:
    return {"role": "user", "content": text, **extra}


def assistant_message(turn: ModelTurn) -> dict[str, Any]:
    return dict(turn.message)


def tool_result_message(
    call: ToolCall | tuple[str, str], content: str, *, is_error: bool = False
) -> dict[str, Any]:
    """The result of one tool call. `call` is a ToolCall or an (id, name) pair."""
    call_id, name = (call.id, call.name) if isinstance(call, ToolCall) else call
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
        "is_error": is_error,
    }


def history_from_messages(messages: Sequence[JsonMessage]) -> list[dict[str, Any]]:
    """Validate JSON-form history (e.g. the replayed event log) and return plain dict copies.

    Checks the shape a provider would choke on: known roles, a tool result needs an id and a
    name, and a tool result must follow an assistant turn or another tool result. Raises
    ValueError with the offending index.
    """
    out: list[dict[str, Any]] = []
    prev: str | None = None
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ROLES:
            raise ValueError(f"message {i}: unknown role {role!r}")
        if role == "tool":
            if not m.get("tool_call_id") or not m.get("name"):
                raise ValueError(f"message {i}: tool result needs tool_call_id and name")
            if prev not in ("assistant", "tool"):
                raise ValueError(f"message {i}: tool result without a preceding assistant turn")
        out.append(dict(m))
        prev = str(role)
    return out


def append_tool_results(
    history: list[dict[str, Any]], results: Sequence[tuple[ToolCall, str, bool]]
) -> None:
    """Append (call, content, is_error) results as tool messages, in call order."""
    for call, content, is_error in results:
        history.append(tool_result_message(call, content, is_error=is_error))
