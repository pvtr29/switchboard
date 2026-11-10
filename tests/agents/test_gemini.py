import asyncio
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import create_job
from gemini_fakes import SIGNATURE, FakeSdk, Step, chunk, tool_turn, usage

from switchboard.agents.gemini import (
    NUDGE,
    SYSTEM_PROMPT,
    AgentError,
    GeminiAgent,
    instructions,
)
from switchboard.agents.tools import (
    TOOL_DEFS,
    CommandRunner,
    LocalSandboxRunner,
    RunResult,
    ToolBox,
)
from switchboard.core import Budget, Event, EventType, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.llm._sdk import types
from switchboard.llm.client import tool_result_message, user_message
from switchboard.llm.gemini import GeminiClient
from switchboard.llm.retry import RetryPolicy
from switchboard.llm.types import (
    ModelTurn,
    Refused,
    StopKind,
    ToolCall,
    ToolDef,
    Truncated,
    TurnUsage,
)
from switchboard.runtime.agent import (
    ActResult,
    ModelResponse,
    Pricing,
    ReplayPolicy,
    StepContext,
)
from switchboard.runtime.budget import BudgetManager
from switchboard.runtime.runner import run_job
from switchboard.store import EventStore

TOY = Path(__file__).resolve().parent.parent / "fixtures" / "toy_repo"
TASK = "Fix the bug in textstats.top_words without touching the tests."
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------- helpers


class PlainRunner:
    """Test-only CommandRunner: runs pytest in a plain subprocess (no sandbox)."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResult:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        assert argv[0] == "pytest"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            *argv[1:],
            cwd=self.workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async with asyncio.timeout(timeout_s):
            data, _ = await proc.communicate()
        return RunResult(proc.returncode or 0, data.decode())


def plain_runner(ctx: StepContext) -> CommandRunner:
    assert ctx.workdir is not None
    return PlainRunner(ctx.workdir)


def ctx_for(workdir: Path | None = None, *, max_output_tokens: int | None = None) -> StepContext:
    import random

    return StepContext(
        job_id="j",
        step=0,
        attempt=0,
        rng=random.Random(0),
        workdir=workdir,
        max_output_tokens=max_output_tokens,
    )


def turn(
    stop: StopKind = StopKind.END,
    text: str = "",
    calls: Sequence[ToolCall] = (),
    *,
    usage_: TurnUsage | None = None,
    native: bool = True,
) -> ModelTurn:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": c.id, "name": c.name, "args": dict(c.args)} for c in calls],
    }
    if native:
        message["native"] = {"provider": "scripted", "content": {"x": 1}}
    return ModelTurn(
        message=message,
        tool_calls=tuple(calls),
        text=text,
        stop=stop,
        stop_raw=stop.name,
        usage=usage_ or TurnUsage(input_tokens=100, output_tokens=10, thinking_tokens=5),
        model="scripted-1",
    )


class ScriptedClient:
    def __init__(self, *turns: ModelTurn | Exception, models: Sequence[str] = ("m",)) -> None:
        self.turns = list(turns)
        self.models = list(models)
        self.calls: list[dict[str, Any]] = []

    def fingerprint(self) -> Mapping[str, Any]:
        return {"provider": "scripted", "models": self.models, "effort": "medium"}

    async def complete(
        self,
        history: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolDef],
        *,
        system: str,
        max_output_tokens: int | None,
        deadline_s: float,
    ) -> ModelTurn:
        self.calls.append(
            {
                "history": [dict(m) for m in history],
                "tools": list(tools),
                "system": system,
                "max_output_tokens": max_output_tokens,
                "deadline_s": deadline_s,
            }
        )
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


def agent_for(client: ScriptedClient | GeminiClient, **kw: Any) -> GeminiAgent:
    return GeminiAgent(client, plain_runner, **kw)


# ---------------------------------------------------------------- the agent surface


async def test_tools_and_config_hash() -> None:
    a = agent_for(ScriptedClient())
    assert {t.name for t in a.tools} == {t.name for t in TOOL_DEFS}
    assert all(t.replay is ReplayPolicy.SAFE for t in a.tools)
    assert len(a.config_hash) == 64
    assert a.config_hash == agent_for(ScriptedClient()).config_hash  # stable
    assert a.max_output_tokens > 0


async def test_config_hash_covers_models_effort_prompt_and_tools() -> None:
    base = agent_for(ScriptedClient()).config_hash
    assert agent_for(ScriptedClient(models=("other",))).config_hash != base
    assert agent_for(ScriptedClient(), system_prompt=SYSTEM_PROMPT + " more").config_hash != base
    low = GeminiClient(sdk=FakeSdk(), effort="low")
    high = GeminiClient(sdk=FakeSdk(), effort="high")
    assert agent_for(low).config_hash != agent_for(high).config_hash
    same = GeminiClient(sdk=FakeSdk(), effort="low")
    assert agent_for(low).config_hash == agent_for(same).config_hash
    # unrelated knobs do not invalidate checkpoints
    assert agent_for(ScriptedClient(), max_output_tokens=999).config_hash == base
    assert agent_for(ScriptedClient(), deadline_s=1.0).config_hash == base


async def test_estimate_input_tokens_is_a_documented_upper_bound() -> None:
    a = agent_for(ScriptedClient())
    short = a.estimate_input_tokens(instructions("hi"))
    big = a.estimate_input_tokens([*instructions("hi"), user_message("x" * 3000)])
    assert big - short >= 1000  # one token per 3 chars of what is sent
    # An assistant turn with a native payload is counted by that payload only.
    msg = turn(text="y" * 3000).message
    with_native = a.estimate_input_tokens([*instructions("hi"), msg])
    assert with_native - short < 100


# ---------------------------------------------------------------- sample


async def test_sample_returns_the_turn_with_usage_and_meta() -> None:
    call = ToolCall("c1", "run_tests", {})
    client = ScriptedClient(turn(StopKind.TOOL_USE, "go", [call]))
    agent = agent_for(client)
    resp = await agent.sample(ctx_for(max_output_tokens=777), instructions(TASK))
    assert [(c.id, c.name) for c in resp.tool_calls] == [("c1", "run_tests")]
    assert resp.message["stop"] == "tool_use"
    assert resp.message["native"] == {"provider": "scripted", "content": {"x": 1}}
    assert resp.usage.tokens == 115
    assert resp.usage.usd_micros == 0
    assert resp.meta["usd_tracked"] is False
    assert resp.meta["tokens"] == {"input": 100, "output": 10, "thinking": 5, "cached": 0}
    assert resp.model == "scripted-1"
    c = client.calls[0]
    assert c["max_output_tokens"] == 777  # the budget's grant, exactly
    assert c["system"] == SYSTEM_PROMPT
    assert [t.name for t in c["tools"]] == [t.name for t in TOOL_DEFS]
    assert c["history"] == instructions(TASK)


async def test_without_a_grant_the_agents_own_cap_applies() -> None:
    client = ScriptedClient(turn(StopKind.END))
    await agent_for(client, max_output_tokens=4096).sample(ctx_for(), instructions(TASK))
    assert client.calls[0]["max_output_tokens"] == 4096


async def test_usd_only_when_pricing_is_given() -> None:
    t = turn(usage_=TurnUsage(input_tokens=1000, output_tokens=40, thinking_tokens=10))
    priced = agent_for(ScriptedClient(t), pricing=Pricing(2, 10))
    resp = await priced.sample(ctx_for(), instructions(TASK))
    assert resp.usage.usd_micros == 1000 * 2 + 50 * 10
    assert resp.meta["usd_tracked"] is True
    assert priced.pricing == Pricing(2, 10)
    assert agent_for(ScriptedClient()).pricing == Pricing(0, 0)


async def test_refusal_fails_the_job_with_a_clear_error() -> None:
    agent = agent_for(ScriptedClient(turn(StopKind.REFUSED)))
    with pytest.raises(Refused, match="refused"):
        await agent.sample(ctx_for(), instructions(TASK))


async def test_max_tokens_inside_a_tool_call_fails() -> None:
    t = turn(StopKind.MAX_TOKENS, calls=[ToolCall("c", "write_file", {"path": "a"})])
    with pytest.raises(Truncated):
        await agent_for(ScriptedClient(t)).sample(ctx_for(), instructions(TASK))


async def test_max_tokens_without_a_call_ends_with_a_summary(tmp_path: Path) -> None:
    agent = agent_for(ScriptedClient(turn(StopKind.MAX_TOKENS, "partial thoughts")))
    resp = await agent.sample(ctx_for(), instructions(TASK))
    result = await agent.act(ctx_for(tmp_path), resp)
    assert result.done
    assert result.summary == "partial thoughts"
    assert result.messages == ()


async def test_end_without_finish_is_nudged_then_ends(tmp_path: Path) -> None:
    ends = [turn(StopKind.END, f"thinking {i}") for i in range(4)]
    agent = agent_for(ScriptedClient(*ends), max_recovery_rounds=3)
    history = instructions(TASK)
    for _ in range(3):
        resp = await agent.sample(ctx_for(), history)
        res = await agent.act(ctx_for(tmp_path), resp)
        assert not res.done
        assert res.messages[0]["content"] == NUDGE
        history = [*history, resp.message, *res.messages]
    # the bound is counted from the history, so a fresh agent object sees it too
    resp = await agent_for(ScriptedClient(ends[3])).sample(ctx_for(), history)
    assert resp.message["final"] is True
    res = await agent.act(ctx_for(tmp_path), resp)
    assert res.done
    assert res.summary == "thinking 3"


async def test_progress_resets_the_recovery_count(tmp_path: Path) -> None:
    agent = agent_for(ScriptedClient(turn(StopKind.END)))
    history = [
        *instructions(TASK),
        turn(StopKind.END).message,
        user_message(NUDGE, recovery=True),
        turn(StopKind.TOOL_USE, calls=[ToolCall("c", "list_files", {})]).message,
        tool_result_message(("c", "list_files"), "a.py"),
    ]
    resp = await agent.sample(ctx_for(), history)
    assert "final" not in resp.message


async def test_malformed_turn_gets_an_error_message_back(tmp_path: Path) -> None:
    bad = turn(StopKind.MALFORMED, calls=[ToolCall("c", "write_file", {})])
    agent = agent_for(ScriptedClient(bad))
    resp = await agent.sample(ctx_for(), instructions(TASK))
    assert resp.tool_calls == ()
    assert "native" not in resp.message  # no dangling function call is replayed
    res = await agent.act(ctx_for(tmp_path), resp)
    assert not res.done
    assert res.messages[0]["recovery"] is True
    assert "could not be used" in res.messages[0]["content"]


async def test_malformed_beyond_the_bound_raises() -> None:
    history = [
        *instructions(TASK),
        turn(StopKind.MALFORMED).message,
        user_message("x", recovery=True),
        turn(StopKind.MALFORMED).message,
        user_message("x", recovery=True),
    ]
    agent = agent_for(ScriptedClient(turn(StopKind.MALFORMED)), max_recovery_rounds=2)
    with pytest.raises(AgentError, match="recovery rounds"):
        await agent.sample(ctx_for(), history)


async def test_empty_history_is_an_error() -> None:
    with pytest.raises(AgentError):
        await agent_for(ScriptedClient()).sample(ctx_for(), [])


async def test_client_errors_propagate() -> None:
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await agent_for(ScriptedClient(boom)).sample(ctx_for(), instructions(TASK))


# ---------------------------------------------------------------- act


@pytest.fixture
def work(tmp_path: Path) -> Path:
    w = tmp_path / "work"
    shutil.copytree(TOY, w, ignore=shutil.ignore_patterns("__pycache__"))
    return w


async def test_act_runs_calls_sequentially_and_reports_each(work: Path) -> None:
    calls = [
        ToolCall("a", "write_file", {"path": "n.txt", "contents": "1"}),
        ToolCall("b", "read_file", {"path": "n.txt"}),
        ToolCall("c", "read_file", {"path": "../escape"}),
        ToolCall("d", "finish", {"summary": "all done"}),
    ]
    agent = agent_for(ScriptedClient(turn(StopKind.TOOL_USE, calls=calls)))
    resp = await agent.sample(ctx_for(), instructions(TASK))
    res = await agent.act(ctx_for(work), resp)
    assert res.done
    assert res.summary == "all done"
    assert [(m["tool_call_id"], m["is_error"]) for m in res.messages] == [
        ("a", False),
        ("b", False),
        ("c", True),
        ("d", False),
    ]
    assert res.messages[1]["content"] == "1"  # saw the write of the call before it


async def test_act_needs_a_workdir() -> None:
    call = ToolCall("a", "list_files", {})
    agent = agent_for(ScriptedClient(turn(StopKind.TOOL_USE, calls=[call])))
    resp = await agent.sample(ctx_for(), instructions(TASK))
    with pytest.raises(RuntimeError, match="workdir"):
        await agent.act(ctx_for(), resp)


async def test_act_is_deterministic_for_a_replayed_response(work: Path) -> None:
    calls = [
        ToolCall("a", "run_tests", {}),
        ToolCall("b", "list_files", {}),
    ]
    agent = agent_for(ScriptedClient(turn(StopKind.TOOL_USE, calls=calls)))
    resp = await agent.sample(ctx_for(), instructions(TASK))
    first = await agent.act(ctx_for(work), resp)
    second = await agent.act(ctx_for(work), resp)
    assert first == second
    assert "exit code: 1" in first.messages[0]["content"]  # the toy repo has its bug


# ---------------------------------------------------------------- end to end through run_job

FIXED = (TOY / "textstats.py").read_text().replace("[: n - 1]", "[:n]")
SIGS = [bytes([i, 0xFF, 0xFE]) + SIGNATURE for i in range(6)]


def toy_script() -> list[Any]:
    """The model's side of a successful rollout: test, read, fix, test, finish."""
    steps = [
        ("run_tests", {"command": "pytest -q"}),
        ("read_file", {"path": "textstats.py"}),
        ("write_file", {"path": "textstats.py", "contents": FIXED}),
        ("run_tests", {"command": "pytest -q"}),
        ("finish", {"summary": "Fixed the off-by-one in top_words."}),
    ]
    return [tool_turn(n, a, signature=SIGS[i]) for i, (n, a) in enumerate(steps)]


def make_client(sdk: FakeSdk) -> GeminiClient:
    return GeminiClient(("m1", "m2"), sdk=sdk, policy=RetryPolicy(base_delay_s=0.0), rng=None)


def make_env(root: Path) -> DirEnv:
    return DirEnv(TOY, root, "j1")


class Crash(BaseException):
    """Stands in for kill -9."""


def by_type(events: list[Event], type_: EventType) -> list[Event]:
    return [e for e in events if e.type is type_]


async def run_toy(store: EventStore, env: DirEnv, agent: GeminiAgent, **kw: Any) -> Any:
    return await run_job(store, "j1", agent, env=env, **kw)


async def test_agent_fixes_the_toy_repo_through_run_job(store: EventStore, tmp_path: Path) -> None:
    sdk = FakeSdk(*toy_script())
    agent = agent_for(make_client(sdk))
    env = make_env(tmp_path / "envs")
    await create_job(store, instructions=instructions(TASK), config_hash=agent.config_hash, seed=1)
    job = await run_toy(store, env, agent)
    assert job.state is JobState.VERIFYING
    assert (env.workdir / "textstats.py").read_text() == FIXED
    assert (env.workdir / "tests" / "test_textstats.py").read_text() == (
        TOY / "tests" / "test_textstats.py"
    ).read_text()
    events = await store.read_job("j1")
    responded = by_type(events, EventType.LLM_RESPONDED)
    assert len(responded) == 5
    assert len(sdk.calls) == 5
    assert responded[0].payload["meta"]["usd_tracked"] is False
    assert responded[0].payload["meta"]["tokens"]["input"] == 100
    assert all(e.payload["config_hash"] == agent.config_hash for e in responded)
    # the 4th step re-ran the tests after the fix
    results = by_type(events, EventType.STEP_COMPLETED)[3].payload["delta"]
    assert results[-1]["content"].startswith("exit code: 0")
    assert "5 passed" in results[-1]["content"]
    # a signature stored in the event log came back byte-for-byte in the next request
    second_request = sdk.calls[1].contents
    model_turn = second_request[1]
    assert model_turn.role == "model"
    assert model_turn.parts is not None
    assert model_turn.parts[-1].thought_signature == SIGS[0]
    fifth = sdk.calls[4].contents
    assert fifth[-2].role == "model"
    assert fifth[-2].parts is not None
    assert fifth[-2].parts[-1].thought_signature == SIGS[3]


async def test_crash_after_llm_responded_replays_tools_without_a_new_call(
    store: EventStore, tmp_path: Path
) -> None:
    script = toy_script()
    crash_step = 2  # the write_file step
    sdk1 = FakeSdk(*script[: crash_step + 1])
    env = make_env(tmp_path / "envs")

    class CrashAfterWrite(GeminiAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            result = await super().act(ctx, response)
            if ctx.step == crash_step and ctx.attempt == 0:
                raise Crash  # the write has hit the disk, but nothing is committed
            return result

    agent1 = CrashAfterWrite(make_client(sdk1), plain_runner)
    await create_job(store, instructions=instructions(TASK), config_hash=agent1.config_hash, seed=1)
    with pytest.raises(Crash):
        await run_toy(store, env, agent1)
    assert len(sdk1.calls) == crash_step + 1
    assert (env.workdir / "textstats.py").read_text() == FIXED  # the uncommitted partial effect

    # A restarted process: new env object, new agent, new client that only knows steps 3 and 4.
    sdk2 = FakeSdk(*script[crash_step + 1 :])
    agent2 = agent_for(make_client(sdk2))
    job = await run_toy(store, make_env(tmp_path / "envs"), agent2)
    assert job.state is JobState.VERIFYING
    assert len(sdk2.calls) == 2  # steps 3 and 4 only: step 2's logged response was NOT re-sampled
    assert (make_env(tmp_path / "envs").workdir / "textstats.py").read_text() == FIXED
    events = await store.read_job("j1")
    assert len(by_type(events, EventType.LLM_RESPONDED)) == 5
    assert len(by_type(events, EventType.RESTORED)) == 1
    step2 = by_type(events, EventType.STEP_COMPLETED)[crash_step].payload["delta"]
    assert step2[-1]["content"].startswith("wrote ")  # the tool call was re-executed


async def test_changed_agent_config_refuses_to_resume(store: EventStore, tmp_path: Path) -> None:
    sdk = FakeSdk(*toy_script())
    agent = agent_for(make_client(sdk))
    await create_job(store, instructions=instructions(TASK), config_hash=agent.config_hash, seed=1)
    other = agent_for(make_client(FakeSdk()), system_prompt="different")
    job = await run_toy(store, make_env(tmp_path / "envs"), other)
    assert job.state is JobState.FAILED
    assert len(sdk.calls) == 0


async def test_refusal_fails_the_job_cleanly(store: EventStore, tmp_path: Path) -> None:
    refused = chunk([types.Part(text="")], finish=types.FinishReason.SAFETY, usage_md=usage())
    agent = agent_for(make_client(FakeSdk([refused])))
    await create_job(store, instructions=instructions(TASK), config_hash=agent.config_hash)
    job = await run_toy(store, make_env(tmp_path / "envs"), agent)
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    reasons = [e.payload.get("reason") for e in by_type(events, EventType.STATE_CHANGED)]
    assert any("Refused" in str(r) for r in reasons)


async def test_unavailability_is_recorded_in_the_log(store: EventStore, tmp_path: Path) -> None:
    from gemini_fakes import server_error

    script = toy_script()
    m1: list[Step] = [server_error(503)] * 3
    sdk = FakeSdk(by_model={"m1": m1, "m2": script})
    agent = agent_for(make_client(sdk))
    await create_job(store, instructions=instructions(TASK), config_hash=agent.config_hash)
    job = await run_toy(store, make_env(tmp_path / "envs"), agent)
    assert job.state is JobState.VERIFYING
    responded = by_type(await store.read_job("j1"), EventType.LLM_RESPONDED)
    first = responded[0].payload
    assert first["model"] == "m2"
    assert first["meta"]["fell_back"] is True
    assert [a["action"] for a in first["meta"]["failed_attempts"]] == [
        "retry",
        "retry",
        "next_model",
    ]
    # m1 now cools down: later calls go straight to m2, the retries are not paid at every step
    assert responded[1].payload["model"] == "m2"
    assert responded[1].payload["meta"]["failed_attempts"] == []
    assert len([c for c in sdk.calls if c.model == "m1"]) == 3


async def test_the_budget_grant_reaches_the_request_and_never_overshoots(
    store: EventStore, tmp_path: Path
) -> None:
    sdk = FakeSdk(*toy_script())
    agent = agent_for(make_client(sdk), max_output_tokens=16_384)
    await create_job(
        store,
        instructions=instructions(TASK),
        config_hash=agent.config_hash,
        budget={"tokens": 8_000},
    )
    bm = BudgetManager(Budget(tokens=1_000_000))
    job = await run_toy(store, make_env(tmp_path / "envs"), agent, budget=bm)
    assert job.state is JobState.VERIFYING
    first = sdk.calls[0].config.max_output_tokens
    assert first is not None
    assert 0 < first < 16_384  # capped to what the 8k job budget leaves after the input
    assert bm.overshoots == 0
    assert bm.overshoot_tokens == 0


# ---------------------------------------------------------------- live (real API)


def _load_key() -> str | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.is_file():
        for raw in env_file.read_text().splitlines():
            line = raw.strip().removeprefix("export ")
            key, _, value = line.partition("=")
            if key.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY") and value.strip():
                return value.strip().strip("\"'")
    return None


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("SB_LIVE_GEMINI") != "1", reason="set SB_LIVE_GEMINI=1")
@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="the live run executes model-written code in the macOS sandbox",
)
async def test_live_gemini_fixes_the_toy_repo(store: EventStore, tmp_path: Path) -> None:
    key = _load_key()
    if key is None:
        pytest.skip("no Gemini API key in the environment or .env")
    models = tuple(os.environ.get("SB_LIVE_MODELS", "gemini-3.6-flash").split(","))
    client = GeminiClient(models, api_key=key)
    agent = GeminiAgent(client, lambda ctx: LocalSandboxRunner(_wd(ctx)))
    env = make_env(tmp_path / "envs")
    await create_job(
        store,
        instructions=instructions(TASK),
        config_hash=agent.config_hash,
        budget={"tokens": 200_000},
    )
    bm = BudgetManager(Budget(tokens=200_000))
    t0 = time.monotonic()
    try:
        async with asyncio.timeout(600):
            job = await run_toy(store, env, agent, budget=bm)
    finally:
        await client.aclose()
    elapsed = time.monotonic() - t0
    events = await store.read_job("j1")
    responded = by_type(events, EventType.LLM_RESPONDED)
    served = sorted({str(e.payload["model"]) for e in responded})
    fell_back = [e.payload.get("meta", {}).get("failed_attempts") for e in responded]
    check = await ToolBox(env.workdir, LocalSandboxRunner(env.workdir)).execute("run_tests", {})
    print(
        f"\nLIVE: state={job.state.value} steps={len(responded)} tokens={job.usage.tokens} "
        f"time={elapsed:.1f}s models={served} failed_attempts={fell_back} "
        f"overshoot={bm.overshoot_tokens} final_tests={check.content.splitlines()[0]}"
    )
    print("LIVE transitions:", [e.payload for e in by_type(events, EventType.STATE_CHANGED)][-2:])
    failure = [str(e.payload.get("reason")) for e in by_type(events, EventType.STATE_CHANGED)][-1]
    if job.state is JobState.FAILED and "Unavailable:" in failure:
        pytest.skip(f"provider unavailable (not a code failure): {failure[:200]}")
    assert job.state is JobState.VERIFYING
    assert check.content.startswith("exit code: 0")
    assert (env.workdir / "tests" / "test_textstats.py").read_text() == (
        TOY / "tests" / "test_textstats.py"
    ).read_text()
    assert bm.overshoot_tokens == 0


def _wd(ctx: StepContext) -> Path:
    assert ctx.workdir is not None
    return ctx.workdir
