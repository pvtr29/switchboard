"""Scripted stand-in for the google-genai client, at the `aio.models.generate_content_stream`
boundary, plus builders for stream chunks. No network, no sleeping (unless a step hangs)."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any

from switchboard.llm._sdk import errors, types

SIGNATURE = b"\x00\xff\xfe-sig\x80\x81bytes/+="  # not valid UTF-8, has base64-special bytes


class Hang:
    """A step whose stream never produces anything (a stalled connection)."""


type Step = Sequence[types.GenerateContentResponse] | Exception | Hang


def usage(
    prompt: int = 100, out: int = 10, thoughts: int = 5, cached: int = 0
) -> types.GenerateContentResponseUsageMetadata:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=out,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached or None,
    )


def chunk(
    parts: Sequence[types.Part] = (),
    *,
    finish: types.FinishReason | None = None,
    usage_md: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    cand = types.Candidate(
        content=types.Content(role="model", parts=list(parts)) if parts else None,
        finish_reason=finish,
    )
    return types.GenerateContentResponse(candidates=[cand], usage_metadata=usage_md)


def call_part(
    name: str, args: dict[str, Any], *, id: str | None = None, signature: bytes | None = None
) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(id=id, name=name, args=args), thought_signature=signature
    )


def tool_turn(
    name: str, args: dict[str, Any], *, text: str = "", signature: bytes | None = SIGNATURE
) -> list[types.GenerateContentResponse]:
    """A typical stream: a thought, then (maybe text and) one call carrying a signature."""
    parts = [types.Part(text="thinking...", thought=True)]
    if text:
        parts.append(types.Part(text=text))
    parts.append(call_part(name, args, signature=signature))
    return [
        chunk(parts[:1]),
        chunk(parts[1:], finish=types.FinishReason.STOP, usage_md=usage()),
    ]


def text_turn(text: str, finish: types.FinishReason = types.FinishReason.STOP) -> list[Any]:
    return [chunk([types.Part(text=text)], finish=finish, usage_md=usage())]


def server_error(code: int = 503) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


def client_error(code: int = 400) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}})


class Call:
    def __init__(
        self, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> None:
        self.model = model
        self.contents = contents
        self.config = config


class _Models:
    def __init__(self, sdk: "FakeSdk") -> None:
        self.sdk = sdk

    async def generate_content_stream(
        self, *, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> AsyncIterator[types.GenerateContentResponse]:
        self.sdk.calls.append(Call(model, contents, config))
        queue = self.sdk.by_model.get(model, self.sdk.steps)
        if not queue:
            raise AssertionError(f"FakeSdk: no scripted step left for {model}")
        step = queue.popleft()
        if isinstance(step, Exception):
            raise step
        return self._stream(step)

    @staticmethod
    async def _stream(step: Step) -> AsyncIterator[types.GenerateContentResponse]:
        if isinstance(step, Hang):
            await asyncio.Event().wait()
        else:
            assert not isinstance(step, Exception)
            for c in step:
                yield c
                await asyncio.sleep(0)


class _Aio:
    def __init__(self, sdk: "FakeSdk") -> None:
        self._models = _Models(sdk)
        self.closed = False

    @property
    def models(self) -> _Models:
        return self._models

    async def aclose(self) -> None:
        self.closed = True


class FakeSdk:
    """`FakeSdk(step, step, ...)` serves steps in call order; `by_model={"m": [steps]}` serves a
    model from its own queue instead. A step is a list of chunks, an exception raised when the
    call is made, or `Hang()`."""

    def __init__(self, *steps: Step, by_model: dict[str, list[Step]] | None = None) -> None:
        self.steps: deque[Step] = deque(steps)
        self.by_model: dict[str, deque[Step]] = {m: deque(s) for m, s in (by_model or {}).items()}
        self.calls: list[Call] = []
        self._aio = _Aio(self)

    @property
    def aio(self) -> _Aio:
        return self._aio
