"""A scripted LLMClient: answers with a fixed sequence of tool calls, no network.

Call k (k = number of assistant turns already in the history) gets `script[k]`, so a replayed
step gets the same answer. Used to drive planted trajectories (verify.planted) through the real
runtime, sandbox and grader, and by the tests.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from switchboard.llm.types import ModelTurn, StopKind, ToolCall, ToolDef, TurnUsage

type Script = Sequence[tuple[str, Mapping[str, Any]]]


class ScriptedClient:
    def __init__(self, script: Script, *, name: str = "scripted") -> None:
        self.script = list(script)
        self.name = name

    def fingerprint(self) -> Mapping[str, Any]:
        return {"provider": "scripted", "models": [self.name]}

    async def complete(
        self,
        history: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolDef],
        *,
        system: str,
        max_output_tokens: int | None,
        deadline_s: float,
    ) -> ModelTurn:
        step = sum(1 for m in history if m["role"] == "assistant")
        name: str
        args: Mapping[str, Any]
        if step >= len(self.script):
            name, args = "finish", {"summary": "script ended"}
        else:
            name, args = self.script[step]
        call = ToolCall(f"call-{step}", name, dict(args))
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call.id, "name": name, "args": dict(args)}],
        }
        return ModelTurn(
            message=message,
            tool_calls=(call,),
            text="",
            stop=StopKind.TOOL_USE,
            stop_raw="STOP",
            usage=TurnUsage(input_tokens=100, output_tokens=20),
            model=self.name,
        )


__all__ = ["Script", "ScriptedClient"]
