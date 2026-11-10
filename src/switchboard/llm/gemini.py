"""GeminiClient: the LLMClient for Google's Gemini API (google-genai 3.x).

Decisions, mostly from docs/spike-notes.md:

- One retry policy, ours (`retry.run_with_fallback`). The SDK's retries are off
  (`HttpRetryOptions(attempts=1)`), because stacked policies turned a 503 storm into a silent
  10 minute hang. Each attempt runs under `asyncio.timeout(deadline_s)`, and the SDK's HTTP
  timeout is set to the same figure as a backstop.
- Only 408/429/5xx, timeouts, connection and DNS errors are retried. Any other 4xx is a
  `BadRequest` at once and never falls back. A 404 (model gone) skips to the next model.
- Fallback: when a model stays unavailable through all its attempts the next model in
  `models` is tried; the answering model is `ModelTurn.model` and every failed attempt is in
  `ModelTurn.failed_attempts` (and goes to `observer`, e.g. an `UnavailabilityTracker`).
- Streaming: `generate_content_stream` is an `async def` returning an async iterator, so it is
  awaited first, then iterated. Parts are collected VERBATIM, never merged: a thought
  signature can sit in its own part and must come back byte-for-byte. Stored as JSON via
  `Content.model_dump(mode="json")`, where bytes become urlsafe base64 and `model_validate`
  restores them exactly (tests/llm/test_gemini_client.py).
- A model that exhausted its attempts is tried last for `cooldown_s` (default 30 s): during
  an outage the retries are paid once, not at every step.
- `max_output_tokens` is passed straight through; for Gemini it also caps thinking tokens.
- No prices live here: money is the agent's business (the pricing comes from configuration).
"""

import asyncio
import dataclasses
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from switchboard.llm._sdk import errors, genai, types
from switchboard.llm.client import JsonMessage, history_from_messages
from switchboard.llm.retry import Observer, RetryPolicy, Verdict, run_with_fallback
from switchboard.llm.types import (
    AttemptRecord,
    BadRequest,
    ModelTurn,
    StopKind,
    ToolCall,
    ToolDef,
    TurnUsage,
    Unavailable,
)

DEFAULT_MODELS: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.5-flash")
PROVIDER = "gemini"

EFFORT_TO_THINKING: Mapping[str, str] = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}

_REFUSED = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}
_MALFORMED = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "MALFORMED_RESPONSE"}


class _Models(Protocol):
    async def generate_content_stream(
        self,
        *,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> AsyncIterator[types.GenerateContentResponse]: ...


class _Aio(Protocol):
    @property
    def models(self) -> _Models: ...

    async def aclose(self) -> None: ...


class SdkClient(Protocol):
    """The slice of `genai.Client` used here: the seam tests replace with a scripted fake."""

    @property
    def aio(self) -> _Aio: ...


def http_options(timeout_s: float) -> types.HttpOptions:
    """SDK transport settings: timeout in ms, and the SDK's own retries OFF (attempts=1)."""
    return types.HttpOptions(
        timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
    )


def _intify(v: Any) -> Any:
    """JSON numbers may come back as floats; make integral ones ints so validation is fair."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _intify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_intify(x) for x in v]
    return v


def _format_api_error(e: errors.APIError) -> str:
    return f"Gemini API error {e.code} {e.status or ''}: {str(e.message or e)[:300]}".strip()


_TRANSPORT_LIBS = {"httpx", "httpx2", "httpcore", "httpcore2", "aiohttp", "requests", "urllib3"}


def _is_transport_error(exc: Exception) -> bool:
    """A connection/timeout/protocol failure from the HTTP library under the SDK. Matched by
    module name because the SDK picks its transport (httpx2, httpx, aiohttp) at runtime."""
    return any(c.__module__.split(".")[0] in _TRANSPORT_LIBS for c in type(exc).__mro__)


_RETRY_IN = re.compile(r"retry in ((?:\d+(?:\.\d+)?[hms])+)", re.IGNORECASE)
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0}
LONG_QUOTA_WAIT_S = 120.0
"""A 429 that says to retry in longer than this (a daily quota) is not worth retrying: the next
model is tried instead. Shorter waits are ordinary rate limiting and are retried with backoff."""


def _retry_in_s(message: str) -> float | None:
    m = _RETRY_IN.search(message)
    if m is None:
        return None
    return sum(float(n) * _UNIT_S[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1)))


def classify_error(exc: Exception) -> tuple[Verdict, str]:
    """The retry classifier for Gemini: what to do about an exception from one attempt."""
    if isinstance(exc, errors.APIError):
        code = exc.code or 0
        reason = _format_api_error(exc)
        wait = _retry_in_s(str(exc.message or "")) if code == 429 else None
        if wait is not None and wait > LONG_QUOTA_WAIT_S:
            return Verdict.NEXT_MODEL, f"quota exhausted (retry in {wait:.0f}s): {reason[:160]}"
        if code in (408, 429) or 500 <= code < 600:
            return Verdict.RETRY, reason
        if code == 404:
            return Verdict.NEXT_MODEL, reason
        return Verdict.RAISE, reason
    if isinstance(exc, TimeoutError | OSError) or _is_transport_error(exc):
        return Verdict.RETRY, f"{type(exc).__name__}: {exc}"
    return Verdict.RAISE, f"{type(exc).__name__}: {exc}"


def contents_from_history(history: Sequence[JsonMessage]) -> list[types.Content]:
    """JSON-form history -> Gemini `Content`s. Consecutive tool results become ONE user Content
    (the API wants all responses to a turn's calls together)."""
    contents: list[types.Content] = []
    pending: list[types.Part] = []
    native_ids: set[str] = set()

    def flush() -> None:
        if pending:
            contents.append(types.Content(role="user", parts=list(pending)))
            pending.clear()

    for m in history_from_messages(history):
        role = m["role"]
        if role == "user":
            flush()
            contents.append(types.Content(role="user", parts=[types.Part(text=str(m["content"]))]))
        elif role == "assistant":
            flush()
            content = _model_content(m)
            native_ids = {
                p.function_call.id
                for p in content.parts or []
                if p.function_call is not None and p.function_call.id
            }
            if content.parts:  # an empty model Content would be rejected by the API
                contents.append(content)
        else:
            body = str(m["content"])
            part = types.Part.from_function_response(
                name=str(m["name"]),
                response={"error": body} if m.get("is_error") else {"output": body},
            )
            if part.function_response is not None and m["tool_call_id"] in native_ids:
                part.function_response.id = str(m["tool_call_id"])
            pending.append(part)
    flush()
    return contents


def _model_content(m: Mapping[str, Any]) -> types.Content:
    native = m.get("native")
    if isinstance(native, Mapping) and native.get("provider") == PROVIDER:
        return types.Content.model_validate(native["content"])
    parts: list[types.Part] = []
    if m.get("content"):
        parts.append(types.Part(text=str(m["content"])))
    for c in m.get("tool_calls") or []:
        parts.append(types.Part.from_function_call(name=c["name"], args=dict(c["args"])))
    return types.Content(role="model", parts=parts)


def _map_stop(
    raw: str,
    saw_candidate: bool,
    has_calls: bool,
    prompt_feedback: types.GenerateContentResponsePromptFeedback | None,
) -> tuple[StopKind, str]:
    if not saw_candidate:
        reason = prompt_feedback.block_reason if prompt_feedback is not None else None
        if reason is not None:
            return StopKind.REFUSED, f"prompt_blocked:{reason.name}"
        return StopKind.OTHER, "no_candidates"
    if raw in _REFUSED:
        return StopKind.REFUSED, raw
    if raw in _MALFORMED:
        return StopKind.MALFORMED, raw
    if raw == "MAX_TOKENS":
        return StopKind.MAX_TOKENS, raw
    if raw in ("STOP", "", "FINISH_REASON_UNSPECIFIED"):
        return (StopKind.TOOL_USE if has_calls else StopKind.END), raw
    return StopKind.OTHER, raw


def _usage(md: types.GenerateContentResponseUsageMetadata | None) -> TurnUsage:
    if md is None:
        return TurnUsage()
    cached = md.cached_content_token_count or 0
    return TurnUsage(
        input_tokens=md.prompt_token_count or 0,  # includes the cached tokens
        output_tokens=md.candidates_token_count or 0,
        thinking_tokens=md.thoughts_token_count or 0,
        cached_tokens=cached,
    )


class GeminiClient:
    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODELS,
        api_key: str | None = None,
        *,
        effort: str = "medium",
        policy: RetryPolicy | None = None,
        observer: Observer | None = None,
        sdk: SdkClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        rng: random.Random | None = None,
        default_http_timeout_s: float = 180.0,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not models:
            raise ValueError("models must not be empty")
        if effort.lower() not in EFFORT_TO_THINKING:
            raise ValueError(f"effort must be one of {sorted(EFFORT_TO_THINKING)}, got {effort!r}")
        self.models: tuple[str, ...] = tuple(models)
        self.effort = effort.lower()
        self.policy = policy or RetryPolicy()
        self.observer = observer
        self.http_options = http_options(default_http_timeout_s)
        # Without api_key the SDK reads GOOGLE_API_KEY, else GEMINI_API_KEY (GOOGLE wins).
        self._sdk: SdkClient = sdk or genai.Client(api_key=api_key, http_options=self.http_options)
        self._sleep = sleep or asyncio.sleep
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._cooling: dict[str, float] = {}  # model -> time it may be tried first again
        self._rng = rng

    def fingerprint(self) -> Mapping[str, Any]:
        return {"provider": PROVIDER, "models": list(self.models), "effort": self.effort}

    async def aclose(self) -> None:
        await self._sdk.aio.aclose()

    def _config(
        self,
        tools: Sequence[ToolDef],
        system: str,
        max_output_tokens: int | None,
        deadline_s: float,
    ) -> types.GenerateContentConfig:
        declarations = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters_json_schema=dict(t.parameters)
            )
            for t in tools
        ]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=types.ThinkingLevel(EFFORT_TO_THINKING[self.effort]),
            ),
            http_options=http_options(deadline_s),
        )

    async def complete(
        self,
        history: Sequence[JsonMessage],
        tools: Sequence[ToolDef],
        *,
        system: str,
        max_output_tokens: int | None,
        deadline_s: float,
    ) -> ModelTurn:
        contents = contents_from_history(history)
        config = self._config(tools, system, max_output_tokens, deadline_s)

        async def attempt(model: str) -> ModelTurn:
            try:
                return await self._stream_once(model, contents, config)
            except errors.APIError as exc:
                verdict, reason = classify_error(exc)
                if verdict is Verdict.RAISE:
                    raise BadRequest(reason) from exc
                raise

        try:
            outcome = await run_with_fallback(
                self._order(),
                attempt,
                classify_error,
                self.policy,
                deadline_s=deadline_s,
                rng=self._rng,
                observer=self.observer,
                sleep=self._sleep,
            )
        except Unavailable as exc:
            self._cool_down(exc.attempts)
            raise
        self._cool_down(outcome.failed_attempts)
        return dataclasses.replace(outcome.value, failed_attempts=outcome.failed_attempts)

    def _order(self) -> list[str]:
        """Models to try, in order: a model that just exhausted its attempts is tried last for
        `cooldown_s`, so a sustained outage costs the retries once, not once per call."""
        now = self._clock()
        ready = [m for m in self.models if self._cooling.get(m, 0.0) <= now]
        return ready + [m for m in self.models if m not in ready]

    def _cool_down(self, attempts: Sequence[AttemptRecord]) -> None:
        for a in attempts:
            if a.action in ("next_model", "give_up"):
                self._cooling[a.model] = self._clock() + self.cooldown_s

    async def _stream_once(
        self, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> ModelTurn:
        t0 = time.perf_counter()
        ttfc: float | None = None
        parts: list[types.Part] = []
        usage_md: types.GenerateContentResponseUsageMetadata | None = None
        finish: types.FinishReason | None = None
        feedback: types.GenerateContentResponsePromptFeedback | None = None
        saw_candidate = False
        stream = await self._sdk.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        async for chunk in stream:
            if ttfc is None:
                ttfc = time.perf_counter() - t0
            if chunk.usage_metadata is not None:
                usage_md = chunk.usage_metadata
            if chunk.prompt_feedback is not None:
                feedback = chunk.prompt_feedback
            cand = chunk.candidates[0] if chunk.candidates else None
            if cand is None:
                continue
            saw_candidate = True
            if cand.finish_reason is not None:
                finish = cand.finish_reason
            if cand.content is not None:
                parts.extend(cand.content.parts or [])  # verbatim, no merging
        call_s = time.perf_counter() - t0

        calls: list[ToolCall] = []
        texts: list[str] = []
        for p in parts:
            if p.function_call is not None:
                fc = p.function_call
                calls.append(
                    ToolCall(
                        fc.id or f"call-{len(calls)}", fc.name or "", _intify(dict(fc.args or {}))
                    )
                )
            elif p.text is not None and not p.thought:
                texts.append(p.text)
        text = "".join(texts)
        stop, raw = _map_stop(
            finish.name if finish is not None else "", saw_candidate, bool(calls), feedback
        )
        content = types.Content(role="model", parts=parts)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "tool_calls": [{"id": c.id, "name": c.name, "args": dict(c.args)} for c in calls],
            "native": {
                "provider": PROVIDER,
                "content": content.model_dump(mode="json", exclude_none=True),
            },
        }
        return ModelTurn(
            message=message,
            tool_calls=tuple(calls),
            text=text,
            stop=stop,
            stop_raw=raw,
            usage=_usage(usage_md),
            model=model,
            ttfc_s=ttfc,
            call_s=call_s,
        )
