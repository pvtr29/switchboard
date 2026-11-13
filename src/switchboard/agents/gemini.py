"""GeminiAgent: a tool-using coding agent on the runtime's Agent protocol, over any LLMClient.

(Named for the first provider; nothing here is Gemini-specific, the client does the
provider work.) History is plain JSON messages (see `llm.client`): the task as a user
message, then per step the logged assistant turn and one tool message per call.

`sample` makes exactly one model call (with retries and fallback inside the client) and maps
how it stopped:

- tool_use: the normal case.
- end (text, no tool call): the model forgot to call `finish`. `act` nudges it, up to
  `max_recovery_rounds` times in a row; after that the rollout simply ends (the verifier still
  judges the work).
- max_tokens without a tool call: the rollout ends with the text as summary. With a tool call
  the call may be cut off mid-argument, so it raises `Truncated` (fails the job, by design).
- refused: raises `Refused` (fails the job, by design).
- malformed / other: `act` sends an error message back, again bounded; beyond the bound
  `AgentError`.

The bound is a count of trailing recovery messages IN THE HISTORY, never in memory, so it
survives a crash and a replay.

`act` is a pure function of (logged response, environment state): tool calls run
sequentially, outputs have timings and host paths scrubbed, so re-running a replayed step
against the restored S(n) gives the same result messages.

Money: tokens are always counted. Dollars only when `pricing` is given (no price is invented);
otherwise `usd_micros` is 0, and the meta says `"usd_tracked": false`.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from switchboard.agents.tools import TOOL_DEFS, TOOL_SPECS, CommandRunner, ToolBox
from switchboard.core import Usage
from switchboard.llm.client import (
    LLMClient,
    history_from_messages,
    tool_result_message,
    user_message,
)
from switchboard.llm.types import ModelTurn, QuotaExhausted, Refused, StopKind, Truncated
from switchboard.llm.types import ToolCall as LLMToolCall
from switchboard.runtime.agent import (
    ActResult,
    Message,
    ModelResponse,
    Pricing,
    RetryLater,
    StepContext,
    ToolCall,
    ToolSpec,
)

SYSTEM_PROMPT = (
    "You are fixing a bug in the repository in your working directory. "
    "Run the tests with run_tests (`pytest -q`), find the cause, and make a minimal fix to the "
    "source. run_tests only runs pytest; use list_files, read_file and write_file to inspect "
    "and edit files. Do not modify the tests. When the tests pass, call the finish tool with a "
    "short summary."
)

CHARS_PER_TOKEN = 3
"""`estimate_input_tokens` assumes ONE token per 3 characters of what is sent. Typical English
and code run at about 4 chars/token, so this over-estimates on purpose: a reservation made from
it is an upper bound, which is what keeps token overshoot at 0 (the cost is a little budget
held back until settlement)."""

NUDGE = (
    "You stopped without calling a tool. Use the tools to continue; when the tests pass, call "
    "`finish` with a short summary."
)


class AgentError(RuntimeError):
    """The agent cannot make progress (bounded recovery exhausted)."""


def instructions(task: str) -> list[Message]:
    """The `instructions` for job_created: the task as the first user message."""
    return [user_message(task)]


def _recovery_rounds(history: Sequence[Mapping[str, Any]]) -> int:
    """Trailing recovery messages (consecutive, assistant turns in between ignored)."""
    n = 0
    for m in reversed(history):
        if m["role"] == "assistant":
            continue
        if m["role"] == "user" and m.get("recovery"):
            n += 1
            continue
        break
    return n


class GeminiAgent:
    def __init__(
        self,
        client: LLMClient,
        runner_factory: Callable[[StepContext], CommandRunner],
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_output_tokens: int = 16_384,
        pricing: Pricing | None = None,
        deadline_s: float = 120.0,
        max_recovery_rounds: int = 3,
    ) -> None:
        self.client = client
        self.runner_factory = runner_factory
        self.system_prompt = system_prompt
        self.deadline_s = deadline_s
        self.max_recovery_rounds = max_recovery_rounds
        self.max_output_tokens = max_output_tokens
        self.usd_tracked = pricing is not None
        self.pricing = pricing or Pricing()
        self.tools: Sequence[ToolSpec] = TOOL_SPECS
        self.config_hash = hashlib.sha256(
            json.dumps(
                {
                    "client": client.fingerprint(),
                    "system_prompt": system_prompt,
                    "tools": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in TOOL_DEFS
                    ],
                    "replay": [[s.name, s.replay.value] for s in TOOL_SPECS],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self._fixed_chars = len(system_prompt) + sum(
            len(json.dumps(t.parameters)) + len(t.description) + len(t.name) for t in TOOL_DEFS
        )

    def estimate_input_tokens(self, history: Sequence[Message]) -> int:
        """Upper-bound estimate: characters actually sent / CHARS_PER_TOKEN, rounded up.

        An assistant message with a `native` payload is sent as that payload only, so its
        readable duplicate (`content`, `tool_calls`) is not counted."""
        chars = self._fixed_chars
        for m in history:
            body = (
                {k: v for k, v in m.items() if k not in ("content", "tool_calls")}
                if m.get("native") is not None
                else m
            )
            chars += len(json.dumps(body, separators=(",", ":"), ensure_ascii=False))
        return math.ceil(chars / CHARS_PER_TOKEN)

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        messages = history_from_messages(history)
        if not messages:
            raise AgentError("empty history: job_created.instructions must hold the task")
        rounds = _recovery_rounds(messages)
        try:
            turn = await self.client.complete(
                messages,
                TOOL_DEFS,
                system=self.system_prompt,
                max_output_tokens=ctx.max_output_tokens
                if ctx.max_output_tokens is not None
                else self.max_output_tokens,
                deadline_s=self.deadline_s,
            )
        except QuotaExhausted as exc:
            raise RetryLater(str(exc), retry_after_s=exc.retry_after_s) from exc
        message: dict[str, Any] = dict(turn.message)
        message["stop"] = turn.stop.value
        calls: tuple[LLMToolCall, ...] = turn.tool_calls
        match turn.stop:
            case StopKind.REFUSED:
                raise Refused(f"the model refused to answer ({turn.stop_raw})")
            case StopKind.MAX_TOKENS if calls:
                raise Truncated("output limit reached in the middle of a tool call")
            case StopKind.END if rounds >= self.max_recovery_rounds:
                message["final"] = True
            case StopKind.MALFORMED | StopKind.OTHER:
                if rounds >= self.max_recovery_rounds:
                    raise AgentError(
                        f"model kept failing to produce a valid call ({turn.stop_raw}) "
                        f"after {rounds} recovery rounds"
                    )
                # Whatever partial call came with it is dropped: an unanswered function call
                # in the history would make the next request invalid.
                calls = ()
                message.pop("native", None)
                message["tool_calls"] = []
                message["error"] = turn.stop_raw
            case _:
                pass
        return ModelResponse(
            message=message,
            tool_calls=tuple(ToolCall(c.id, c.name, c.args) for c in calls),
            usage=self._usage(turn),
            model=turn.model,
            meta=self._meta(turn),
        )

    def _usage(self, turn: ModelTurn) -> Usage:
        u = turn.usage
        usd = 0
        if self.usd_tracked:
            # Cached tokens are billed as plain input (an upper bound), thinking as output.
            usd = (
                u.input_tokens * self.pricing.input_usd_micros
                + (u.output_tokens + u.thinking_tokens) * self.pricing.output_usd_micros
            )
        return Usage(tokens=u.total_tokens, usd_micros=usd)

    def _meta(self, turn: ModelTurn) -> dict[str, Any]:
        return {
            "stop_raw": turn.stop_raw,
            "tokens": turn.usage.to_dict(),
            "usd_tracked": self.usd_tracked,
            "failed_attempts": [a.to_dict() for a in turn.failed_attempts],
            "fell_back": turn.fell_back,
            "ttfc_s": None if turn.ttfc_s is None else round(turn.ttfc_s, 3),
            "call_s": round(turn.call_s, 3),
        }

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        stop = response.message.get("stop")
        text = str(response.message.get("content") or "")
        if not response.tool_calls:
            if stop == StopKind.MAX_TOKENS or response.message.get("final"):
                return ActResult((), done=True, summary=text or "stopped: output limit reached")
            if stop == StopKind.END:
                return ActResult((user_message(NUDGE, recovery=True),), done=False)
            err = response.message.get("error", "unknown")
            note = f"Your last response could not be used ({err}). Call one of the tools again."
            return ActResult((user_message(note, recovery=True),), done=False)
        if ctx.workdir is None:
            raise RuntimeError("GeminiAgent needs an environment: ctx.workdir is None")
        box = ToolBox(ctx.workdir, self.runner_factory(ctx))
        results: list[Message] = []
        done = False
        summary = ""
        for call in response.tool_calls:
            out = await box.execute(call.name, call.args)
            results.append(
                tool_result_message((call.id, call.name), out.content, is_error=out.is_error)
            )
            if out.done:
                done = True
                summary = str(call.args.get("summary", ""))
        return ActResult(messages=tuple(results), done=done, summary=summary)
