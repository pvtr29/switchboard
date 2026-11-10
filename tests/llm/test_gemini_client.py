import asyncio
import json
import random
from typing import Any

import pytest
from gemini_fakes import (
    SIGNATURE,
    FakeSdk,
    Hang,
    Step,
    call_part,
    chunk,
    client_error,
    server_error,
    text_turn,
    tool_turn,
    usage,
)

from switchboard.llm._sdk import errors, types
from switchboard.llm.client import (
    append_tool_results,
    assistant_message,
    history_from_messages,
    tool_result_message,
    user_message,
)
from switchboard.llm.gemini import GeminiClient, contents_from_history
from switchboard.llm.retry import RetryPolicy
from switchboard.llm.types import (
    AttemptRecord,
    BadRequest,
    ModelTurn,
    StopKind,
    ToolCall,
    ToolDef,
    Unavailable,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]

TOOLS = [ToolDef("echo", "echo", {"type": "object", "properties": {}})]
FLAT = RetryPolicy(attempts_per_model=3, base_delay_s=1.0, max_delay_s=4.0, jitter_floor=1.0)


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, s: float) -> None:
        self.delays.append(s)


def make(
    sdk: FakeSdk,
    models: tuple[str, ...] = ("m1", "m2"),
    sleeps: Sleeps | None = None,
    **kw: Any,
) -> GeminiClient:
    return GeminiClient(
        models,
        sdk=sdk,
        policy=kw.pop("policy", FLAT),
        sleep=sleeps or Sleeps(),
        rng=random.Random(0),
        **kw,
    )


async def complete(
    client: GeminiClient,
    history: list[dict[str, Any]] | None = None,
    *,
    max_output_tokens: int | None = 1000,
    deadline_s: float = 30.0,
) -> ModelTurn:
    return await client.complete(
        history or [user_message("hi")],
        TOOLS,
        system="sys",
        max_output_tokens=max_output_tokens,
        deadline_s=deadline_s,
    )


# ---------------------------------------------------------------- history round-trip


async def test_thought_signature_survives_the_event_log_roundtrip() -> None:
    sdk = FakeSdk(tool_turn("echo", {"x": 1}, text="calling"))
    turn = await complete(make(sdk))
    assert turn.stop is StopKind.TOOL_USE
    # What the runtime does: JSON into the event log, JSON back out.
    stored = json.loads(json.dumps(assistant_message(turn)))
    assert stored == assistant_message(turn)
    history = [user_message("hi"), stored, tool_result_message(turn.tool_calls[0], "done")]
    contents = contents_from_history(history_from_messages(history))
    model_content = contents[1]
    original = [
        types.Part(text="thinking...", thought=True),
        types.Part(text="calling"),
        call_part("echo", {"x": 1}, signature=SIGNATURE),
    ]
    assert model_content.parts == original
    assert model_content.parts is not None
    assert model_content.parts[2].thought_signature == SIGNATURE  # the very same bytes


async def test_parts_are_kept_verbatim_not_merged() -> None:
    # Gemini may send a signature in a part of its own after the text; merging would lose it.
    parts = [
        types.Part(text="a"),
        types.Part(text="b"),
        types.Part(text="", thought_signature=b"s"),
    ]
    sdk = FakeSdk([chunk(parts[:2]), chunk(parts[2:], finish=types.FinishReason.STOP)])
    turn = await complete(make(sdk))
    stored = types.Content.model_validate(turn.message["native"]["content"])
    assert stored.parts == parts
    assert turn.text == "ab"


async def test_tool_results_become_one_user_content_with_ids_only_when_native() -> None:
    sdk = FakeSdk(
        [
            chunk(
                [call_part("a", {}, id="native-1"), call_part("b", {})],
                finish=types.FinishReason.STOP,
            )
        ]
    )
    turn = await complete(make(sdk))
    assert [c.id for c in turn.tool_calls] == ["native-1", "call-1"]  # second one synthesized
    history = [user_message("go"), assistant_message(turn)]
    append_tool_results(
        history,
        [(turn.tool_calls[0], "r1", False), (turn.tool_calls[1], "oops", True)],
    )
    contents = contents_from_history(history)
    assert [c.role for c in contents] == ["user", "model", "user"]
    responses = [p.function_response for p in contents[2].parts or []]
    assert [r.id if r else None for r in responses] == ["native-1", None]
    assert [r.response if r else None for r in responses] == [
        {"output": "r1"},
        {"error": "oops"},
    ]


async def test_assistant_without_native_payload_is_rebuilt_from_readable_fields() -> None:
    history = [
        user_message("go"),
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "x", "name": "f", "args": {"k": 1}}],
        },
        tool_result_message(ToolCall("x", "f", {}), "done"),
        user_message("again", recovery=True),
    ]
    contents = contents_from_history(history)
    model_parts = contents[1].parts or []
    assert model_parts[0].text == "ok"
    assert model_parts[1].function_call is not None
    assert model_parts[1].function_call.args == {"k": 1}
    assert [c.role for c in contents] == ["user", "model", "user", "user"]


async def test_empty_model_turn_is_not_sent() -> None:
    history = [
        user_message("go"),
        {"role": "assistant", "content": "", "tool_calls": []},
        user_message("nudge"),
    ]
    assert [c.role for c in contents_from_history(history)] == ["user", "user"]


@pytest.mark.parametrize(
    "bad",
    [
        [{"role": "system", "content": "x"}],
        [{"role": "tool", "tool_call_id": "a", "name": "f", "content": "x"}],  # no assistant first
        [user_message("a"), {"role": "tool", "name": "f", "content": "x"}],  # no id
    ],
)
async def test_history_validation(bad: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="message"):
        history_from_messages(bad)


# ---------------------------------------------------------------- mapping


async def test_usage_and_latency_mapping() -> None:
    sdk = FakeSdk(
        [
            chunk([types.Part(text="x")]),
            chunk(
                finish=types.FinishReason.STOP,
                usage_md=usage(prompt=900, out=20, thoughts=7, cached=300),
            ),
        ]
    )
    turn = await complete(make(sdk))
    u = turn.usage
    assert (u.input_tokens, u.output_tokens, u.thinking_tokens, u.cached_tokens) == (
        900,
        20,
        7,
        300,
    )
    assert u.total_tokens == 927  # cached tokens are part of the prompt count, not extra
    assert turn.ttfc_s is not None
    assert turn.model == "m1"
    assert turn.stop_raw == "STOP"


@pytest.mark.parametrize(
    ("finish", "with_call", "kind"),
    [
        (types.FinishReason.STOP, True, StopKind.TOOL_USE),
        (types.FinishReason.STOP, False, StopKind.END),
        (types.FinishReason.MAX_TOKENS, False, StopKind.MAX_TOKENS),
        (types.FinishReason.MAX_TOKENS, True, StopKind.MAX_TOKENS),
        (types.FinishReason.SAFETY, False, StopKind.REFUSED),
        (types.FinishReason.PROHIBITED_CONTENT, False, StopKind.REFUSED),
        (types.FinishReason.MALFORMED_FUNCTION_CALL, False, StopKind.MALFORMED),
        (types.FinishReason.OTHER, False, StopKind.OTHER),
    ],
)
async def test_finish_reason_mapping(
    finish: types.FinishReason, with_call: bool, kind: StopKind
) -> None:
    parts = [call_part("echo", {})] if with_call else [types.Part(text="t")]
    turn = await complete(make(FakeSdk([chunk(parts, finish=finish, usage_md=usage())])))
    assert turn.stop is kind
    assert bool(turn.tool_calls) is with_call


async def test_blocked_prompt_is_refused() -> None:
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    turn = await complete(make(FakeSdk([blocked])))
    assert turn.stop is StopKind.REFUSED
    assert turn.stop_raw == "prompt_blocked:SAFETY"


async def test_float_args_are_made_ints() -> None:
    sdk = FakeSdk(tool_turn("echo", {"n": 3.0, "xs": [1.0, 2.5], "d": {"k": 4.0}}))
    turn = await complete(make(sdk))
    args = turn.tool_calls[0].args
    assert args == {"n": 3, "xs": [1, 2.5], "d": {"k": 4}}
    assert type(args["n"]) is int


# ---------------------------------------------------------------- request config


async def test_max_output_tokens_and_thinking_are_passed_through() -> None:
    sdk = FakeSdk(text_turn("a"), text_turn("b"))
    client = make(sdk, effort="high")
    await complete(client, max_output_tokens=123, deadline_s=12.5)
    cfg = sdk.calls[0].config
    assert cfg.max_output_tokens == 123
    assert cfg.system_instruction == "sys"
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.include_thoughts is True
    assert cfg.thinking_config.thinking_level is types.ThinkingLevel.HIGH
    assert cfg.automatic_function_calling is not None
    assert cfg.automatic_function_calling.disable is True
    assert cfg.tools is not None
    tool = cfg.tools[0]
    assert isinstance(tool, types.Tool)
    assert tool.function_declarations is not None
    assert tool.function_declarations[0].name == "echo"
    await complete(client, max_output_tokens=None)
    assert sdk.calls[1].config.max_output_tokens is None


async def test_sdk_retries_are_disabled_everywhere() -> None:
    sdk = FakeSdk(text_turn("a"))
    client = make(sdk)
    await complete(client, deadline_s=12.5)
    for opts in (client.http_options, sdk.calls[0].config.http_options):
        assert opts is not None
        assert opts.retry_options is not None
        assert opts.retry_options.attempts == 1
    http = sdk.calls[0].config.http_options
    assert http is not None
    assert http.timeout == 12_500  # milliseconds, the same figure as the asyncio deadline


async def test_real_sdk_client_is_built_with_retries_off() -> None:
    client = GeminiClient(api_key="not-a-real-key")
    try:
        assert client.http_options.retry_options is not None
        assert client.http_options.retry_options.attempts == 1
        assert client.models == ("gemini-3.6-flash", "gemini-3.5-flash")
    finally:
        await client.aclose()


async def test_effort_is_validated_and_in_the_fingerprint() -> None:
    with pytest.raises(ValueError, match="effort"):
        GeminiClient(sdk=FakeSdk(), effort="turbo")
    fp = make(FakeSdk(), effort="LOW").fingerprint()
    assert fp == {"provider": "gemini", "models": ["m1", "m2"], "effort": "low"}


# ---------------------------------------------------------------- retries and fallback


async def test_503_then_success_is_retried_with_backoff() -> None:
    sdk = FakeSdk(server_error(503), text_turn("ok"))
    sleeps = Sleeps()
    turn = await complete(make(sdk, sleeps=sleeps))
    assert turn.text == "ok"
    assert turn.model == "m1"
    assert [c.model for c in sdk.calls] == ["m1", "m1"]
    assert sleeps.delays == [1.0]
    assert [a.action for a in turn.failed_attempts] == ["retry"]
    assert "503" in turn.failed_attempts[0].error
    assert not turn.fell_back


@pytest.mark.parametrize("code", [429, 500, 502, 504])
async def test_retryable_codes(code: int) -> None:
    turn = await complete(make(FakeSdk(server_error(code), text_turn("ok"))))
    assert turn.text == "ok"


async def test_sustained_503_falls_back_to_the_second_model() -> None:
    sdk = FakeSdk(by_model={"m1": [server_error(503)] * 3, "m2": [text_turn("from m2")]})
    seen: list[AttemptRecord] = []
    turn = await complete(make(sdk, observer=seen.append))
    assert turn.model == "m2"
    assert turn.text == "from m2"
    assert turn.fell_back
    assert [c.model for c in sdk.calls] == ["m1", "m1", "m1", "m2"]
    assert [(a.model, a.action) for a in turn.failed_attempts][-1] == ("m1", "next_model")
    assert len(seen) == 3  # the observer (e.g. an UnavailabilityTracker) saw them live


async def test_all_models_down_raises_unavailable() -> None:
    sdk = FakeSdk(by_model={"m1": [server_error(503)] * 3, "m2": [server_error(504)] * 3})
    with pytest.raises(Unavailable) as ei:
        await complete(make(sdk))
    assert len(sdk.calls) == 6
    assert len(ei.value.attempts) == 6


async def test_per_attempt_deadline_is_enforced_on_a_stalled_stream() -> None:
    sdk = FakeSdk(Hang(), text_turn("ok"))
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    turn = await complete(make(sdk), deadline_s=9.0)
    assert turn.text == "ok"
    assert loop.time() - t0 >= 9.0
    assert "within 9s" in turn.failed_attempts[0].error


async def test_timeouts_everywhere_exhaust_into_unavailable() -> None:
    sdk = FakeSdk(by_model={"m1": [Hang()] * 3, "m2": [Hang()] * 3})
    with pytest.raises(Unavailable):
        await complete(make(sdk), deadline_s=2.0)


async def test_connection_errors_are_retried() -> None:
    sdk = FakeSdk(ConnectionResetError("reset"), OSError("dns"), text_turn("ok"))
    turn = await complete(make(sdk))
    assert turn.text == "ok"
    assert len(turn.failed_attempts) == 2


async def test_400_is_not_retried_and_does_not_fall_back() -> None:
    sdk = FakeSdk(client_error(400), text_turn("never"))
    sleeps = Sleeps()
    with pytest.raises(BadRequest, match="400"):
        await complete(make(sdk, sleeps=sleeps))
    assert len(sdk.calls) == 1
    assert sleeps.delays == []


async def test_404_skips_to_the_next_model_without_retrying() -> None:
    sdk = FakeSdk(by_model={"m1": [client_error(404)], "m2": [text_turn("ok")]})
    turn = await complete(make(sdk))
    assert turn.model == "m2"
    assert [c.model for c in sdk.calls] == ["m1", "m2"]


async def test_programming_errors_are_not_swallowed() -> None:
    sdk = FakeSdk(KeyError("bug"), text_turn("never"))
    with pytest.raises(KeyError):
        await complete(make(sdk))


async def test_a_model_that_just_failed_is_tried_last_until_its_cooldown_ends() -> None:
    now = [0.0]
    m1: list[Step] = [*[server_error(503)] * 3, text_turn("m1 is back")]
    m2: list[Step] = [text_turn("one"), text_turn("two")]
    sdk = FakeSdk(by_model={"m1": m1, "m2": m2})
    client = make(sdk, cooldown_s=30.0, clock=lambda: now[0])
    first = await complete(client)
    assert first.model == "m2"
    second = await complete(client)  # goes straight to m2: no retries paid again
    assert second.model == "m2"
    assert second.failed_attempts == ()
    assert [c.model for c in sdk.calls] == ["m1", "m1", "m1", "m2", "m2"]
    now[0] = 31.0
    third = await complete(client)
    assert third.model == "m1"
    assert third.text == "m1 is back"


async def test_daily_quota_429_skips_to_the_next_model_but_short_429s_retry() -> None:
    quota = errors.ClientError(
        429,
        {
            "error": {
                "message": "You exceeded your current quota. Please retry in 13h41m23.9s.",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    sdk = FakeSdk(by_model={"m1": [quota], "m2": [text_turn("ok")]})
    turn = await complete(make(sdk))
    assert turn.model == "m2"
    assert [c.model for c in sdk.calls] == ["m1", "m2"]  # one try, no retries on m1
    assert "quota exhausted" in turn.failed_attempts[0].error
    short = errors.ClientError(
        429, {"error": {"message": "Please retry in 2.5s.", "status": "RESOURCE_EXHAUSTED"}}
    )
    sdk2 = FakeSdk(short, text_turn("ok"))
    turn2 = await complete(make(sdk2))
    assert turn2.model == "m1"
    assert [a.action for a in turn2.failed_attempts] == ["retry"]
