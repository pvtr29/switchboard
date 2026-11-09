"""Deterministic fake agent and model for tests, benchmarks and the demo CLI.

Everything random comes from `ctx.rng`, which the runtime seeds from (job seed, step), so a
re-sampled or re-run step reproduces the same response and tool results exactly.
"""

import asyncio
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from switchboard.core import Usage
from switchboard.runtime.agent import (
    ActResult,
    Message,
    ModelResponse,
    ReplayPolicy,
    StepContext,
    ToolCall,
    ToolSpec,
)

PROMPT_TOKENS = 50
USD_MICROS_PER_TOKEN = 3

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("add", ReplayPolicy.SAFE),
    ToolSpec("concat", ReplayPolicy.SAFE),
    ToolSpec("echo", ReplayPolicy.SAFE),
)

FILE_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("write_file", ReplayPolicy.SAFE),  # idempotent: overwrites files/f{k}.txt
    # NOT idempotent on its own: re-running it without restoring the environment first
    # appends a second line, which is why recovery must restore S(n).
    ToolSpec("append_file", ReplayPolicy.SAFE),
)

type FileWriteHook = Callable[[StepContext, str, Path], None]
"""Called after each file tool finishes with (ctx, tool name, path); tests raise to crash
mid-act with a partial write on disk."""

type TokenHook = Callable[[StepContext, int], None]
"""Called after each streamed token with (ctx, token index); tests raise from it to crash."""


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _append_file(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def run_tool(name: str, args: Mapping[str, Any]) -> Any:
    match name:
        case "add":
            return args["a"] + args["b"]
        case "concat":
            return "".join(args["xs"])
        case "echo":
            return args["text"]
    raise ValueError(f"unknown tool {name!r}")


class FakeModel:
    """Streams k fake tokens (one `asyncio.sleep` each) and returns 0-3 tool calls."""

    def __init__(
        self,
        name: str = "fake-model-1",
        *,
        token_interval: float = 0.01,
        on_token: TokenHook | None = None,
    ) -> None:
        self.name = name
        self.token_interval = token_interval
        self.on_token = on_token

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        rng = ctx.rng
        k = rng.randint(3, 8)
        tokens: list[str] = []
        for i in range(k):
            await asyncio.sleep(self.token_interval)
            tokens.append(f"tok{rng.randint(0, 999)}")
            if self.on_token is not None:
                self.on_token(ctx, i)
        calls: list[ToolCall] = []
        for i in range(rng.randint(0, 3)):
            tool = rng.choice(TOOLS).name
            args: dict[str, Any]
            match tool:
                case "add":
                    args = {"a": rng.randint(0, 100), "b": rng.randint(0, 100)}
                case "concat":
                    args = {"xs": [f"x{rng.randint(0, 9)}" for _ in range(rng.randint(1, 3))]}
                case _:
                    args = {"text": f"t{rng.randint(0, 999)}"}
            calls.append(ToolCall(id=f"call-{ctx.step}-{i}", name=tool, args=args))
        message: dict[str, Any] = {
            "role": "assistant",
            "content": " ".join(tokens),
            "tool_calls": [{"id": c.id, "name": c.name, "args": dict(c.args)} for c in calls],
        }
        total = PROMPT_TOKENS + k
        return ModelResponse(
            message=message,
            tool_calls=tuple(calls),
            usage=Usage(tokens=total, usd_micros=total * USD_MICROS_PER_TOKEN),
            model=self.name,
        )


class FakeAgent:
    """Runs for `steps` steps. `fail_at_step` raises RuntimeError inside act at that step.

    `sample_calls` / `act_calls` record (step, attempt) per invocation for test assertions.
    """

    def __init__(
        self,
        *,
        steps: int = 5,
        token_interval: float = 0.01,
        system_prompt: str = "You are a fake agent.",
        fail_at_step: int | None = None,
        on_token: TokenHook | None = None,
        on_file_write: FileWriteHook | None = None,
    ) -> None:
        self.on_file_write = on_file_write
        self.steps = steps
        self.fail_at_step = fail_at_step
        self.system_prompt = system_prompt
        self.model = FakeModel(token_interval=token_interval, on_token=on_token)
        self.tools: Sequence[ToolSpec] = (*TOOLS, *FILE_TOOLS)
        self.config_hash = hashlib.sha256(
            json.dumps(
                {
                    "model": self.model.name,
                    "system_prompt": system_prompt,
                    "tools": [[t.name, t.replay.value] for t in self.tools],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.sample_calls: list[tuple[int, int]] = []
        self.act_calls: list[tuple[int, int]] = []

    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        self.sample_calls.append((ctx.step, ctx.attempt))
        return await self.model.sample(ctx, history)

    async def _file_effects(
        self, ctx: StepContext, response: ModelResponse
    ) -> list[dict[str, Any]]:
        """Deterministic edits of ctx.workdir for this step.

        ctx.rng is already advanced by sampling in an uninterrupted run but fresh when a logged
        response is replayed, so it cannot drive the edits. They are seeded from the (logged)
        response instead, which is itself a pure function of the rng: same run, same edits.
        """
        assert ctx.workdir is not None
        seed = hashlib.sha256(json.dumps(response.message, sort_keys=True).encode()).hexdigest()
        r = random.Random(f"files:{seed}")
        out: list[dict[str, Any]] = []

        async def effect(name: str, rel: str, text: str) -> None:
            path = ctx.workdir / rel if ctx.workdir else Path(rel)
            fn = _write_file if name == "write_file" else _append_file
            await asyncio.to_thread(fn, path, text)
            out.append({"role": "tool", "name": name, "content": rel})
            if self.on_file_write is not None:
                self.on_file_write(ctx, name, path)

        await effect(
            "write_file", f"files/f{r.randint(0, 3)}.txt", f"step {ctx.step} v{r.randint(0, 9999)}"
        )
        if ctx.step % 2 == 0 or r.random() < 0.5:
            await effect("append_file", "log.txt", f"step {ctx.step} n{r.randint(0, 9999)}")
        return out

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        self.act_calls.append((ctx.step, ctx.attempt))
        if ctx.step == self.fail_at_step:
            raise RuntimeError(f"fake failure at step {ctx.step}")
        results = tuple(
            {
                "role": "tool",
                "tool_call_id": c.id,
                "name": c.name,
                "content": run_tool(c.name, c.args),
            }
            for c in response.tool_calls
        )
        if ctx.workdir is not None:
            results = (*results, *await self._file_effects(ctx, response))
        done = ctx.step + 1 >= self.steps
        return ActResult(messages=results, done=done, summary=f"step {ctx.step} done")
