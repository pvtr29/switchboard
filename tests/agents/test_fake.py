import random

import pytest

from switchboard.agents.fake import FakeAgent, run_tool
from switchboard.core import Budget, Usage
from switchboard.runtime.agent import ReplayPolicy, StepContext


def ctx(step: int = 0, attempt: int = 0, seed: str = "s") -> StepContext:
    return StepContext(job_id="j", step=step, attempt=attempt, rng=random.Random(f"{seed}:{step}"))


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_sample_is_deterministic_and_attempt_independent() -> None:
    a, b = FakeAgent(), FakeAgent()
    r1 = await a.sample(ctx(attempt=0), [])
    r2 = await b.sample(ctx(attempt=3), [])
    assert r1 == r2
    assert a.sample_calls == [(0, 0)]
    assert b.sample_calls == [(0, 3)]


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_sample_varies_with_step_and_streams_tokens() -> None:
    agent = FakeAgent(token_interval=1.0)
    loop_time = __import__("asyncio").get_running_loop().time
    t0 = loop_time()
    r0 = await agent.sample(ctx(step=0), [])
    assert loop_time() - t0 >= 3.0  # at least 3 tokens at 1 s each (virtual time)
    r1 = await agent.sample(ctx(step=1), [])
    assert r0.message != r1.message
    assert r0.usage.tokens >= 53
    assert r0.usage.usd_micros == r0.usage.tokens * 3
    assert all(c.name in {t.name for t in agent.tools} for c in r0.tool_calls)
    assert all(t.replay is ReplayPolicy.SAFE for t in agent.tools)


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_act_runs_tools_and_marks_done_on_last_step() -> None:
    agent = FakeAgent(steps=2)
    for step, done in ((0, False), (1, True)):
        c = ctx(step=step)
        resp = await agent.sample(c, [])
        result = await agent.act(c, resp)
        assert result.done is done
        assert [m["tool_call_id"] for m in result.messages] == [t.id for t in resp.tool_calls]
        assert [m["content"] for m in result.messages] == [
            run_tool(t.name, t.args) for t in resp.tool_calls
        ]
    assert agent.act_calls == [(0, 0), (1, 0)]


@pytest.mark.asyncio
@pytest.mark.looptime
async def test_fail_at_step_raises_in_act() -> None:
    agent = FakeAgent(fail_at_step=1)
    c = ctx(step=1)
    resp = await agent.sample(c, [])
    with pytest.raises(RuntimeError, match="step 1"):
        await agent.act(c, resp)


def test_config_hash_depends_on_prompt_only_when_it_changes() -> None:
    assert FakeAgent().config_hash == FakeAgent(steps=9).config_hash
    assert FakeAgent().config_hash != FakeAgent(system_prompt="other").config_hash


def test_step_context_remaining() -> None:
    c = StepContext(
        job_id="j",
        step=0,
        attempt=0,
        rng=random.Random(0),
        budget=Budget(tokens=100),
        usage=Usage(tokens=30),
    )
    assert c.remaining.tokens == 70
    assert c.remaining.usd_micros is None
