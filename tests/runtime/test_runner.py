import asyncio
import random
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState, Usage
from switchboard.runtime.agent import ActResult, ModelResponse, StepContext
from switchboard.runtime.runner import JobNotRunnable, run_job
from switchboard.store import EventStore

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


class Crash(BaseException):
    """Stands in for kill -9: not an Exception, so nothing in the runner may swallow it."""


def by_type(events: list[Event], type_: EventType, step: int | None = None) -> list[Event]:
    return [e for e in events if e.type is type_ and (step is None or e.step_id == step)]


def deltas(events: list[Event]) -> list[Any]:
    return [e.payload["delta"] for e in by_type(events, T.STEP_COMPLETED)]


async def control_deltas(tmp_path_factory: pytest.TempPathFactory, seed: int, steps: int) -> Any:
    async with await EventStore.open(tmp_path_factory.mktemp("control") / "c.db") as s:
        await create_job(s, seed=seed)
        await run_job(s, "j1", FakeAgent(steps=steps))
        return deltas(await s.read_job("j1"))


async def test_happy_path(store: EventStore) -> None:
    await create_job(store, instructions=[{"role": "user", "content": "go"}])
    agent = FakeAgent(steps=5)
    job = await run_job(store, "j1", agent)
    assert job.state is JobState.VERIFYING
    assert job.last_step == 4
    events = await store.read_job("j1")
    for n in range(5):
        for t in (T.STEP_STARTED, T.LLM_RESPONDED, T.STEP_COMPLETED):
            assert len(by_type(events, t, n)) == 1
    assert len(by_type(events, T.USAGE_RECORDED)) == 5
    responses = by_type(events, T.LLM_RESPONDED)
    total = Usage()
    for e in responses:
        total = total + Usage(
            tokens=e.payload["usage"]["tokens"], usd_micros=e.payload["usage"]["usd_micros"]
        )
    assert job.usage == total
    states = [(e.payload["from"], e.payload["to"]) for e in by_type(events, T.STATE_CHANGED)]
    assert states == [
        ("queued", "provisioning"),
        ("provisioning", "running"),
        ("running", "verifying"),
    ]
    first = responses[0].payload
    assert all(c["replay"] == "safe" for c in first["tool_calls"])
    assert first["config_hash"] == agent.config_hash
    last = by_type(events, T.STEP_COMPLETED)[-1].payload
    assert last["done"] is True
    assert last["env_ref"] is None
    assert agent.sample_calls == [(n, 0) for n in range(5)]


async def test_history_passed_to_sample_grows_by_deltas(store: EventStore) -> None:
    await create_job(store, instructions=[{"role": "user", "content": "go"}])
    seen: list[int] = []

    class Spy(FakeAgent):
        async def sample(self, ctx: StepContext, history: Any) -> ModelResponse:
            seen.append(len(history))
            return await super().sample(ctx, history)

    await run_job(store, "j1", Spy(steps=3))
    events = await store.read_job("j1")
    expected = [1]
    for d in deltas(events)[:-1]:
        expected.append(expected[-1] + len(d))
    assert seen == expected


async def test_determinism_across_fresh_dbs(
    store: EventStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    await create_job(store, seed=7)
    await run_job(store, "j1", FakeAgent(steps=5))
    assert deltas(await store.read_job("j1")) == await control_deltas(tmp_path_factory, 7, 5)
    assert await control_deltas(tmp_path_factory, 8, 5) != await control_deltas(
        tmp_path_factory, 7, 5
    )


class CrashAgent(FakeAgent):
    def __init__(self, window: str, crash_step: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.window = window
        self.crash_step = crash_step

    def _armed(self, ctx: StepContext) -> bool:
        return ctx.step == self.crash_step and ctx.attempt == 0

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        if self._armed(ctx):
            if self.window == "W1":  # llm_responded committed, act not started
                raise Crash
            if self.window == "W2":  # halfway through act
                await asyncio.sleep(0.05)
                raise Crash
        result = await super().act(ctx, response)
        if self._armed(ctx) and self.window == "W3":  # act done, step not committed
            raise Crash
        return result


def crash_agent(window: str, n: int) -> FakeAgent:
    if window == "W0":

        def hook(ctx: StepContext, i: int) -> None:
            if ctx.step == n and ctx.attempt == 0 and i == 1:
                raise Crash

        return FakeAgent(on_token=hook)
    return CrashAgent(window, n)


@pytest.mark.parametrize("window", ["W0", "W1", "W2", "W3"])
async def test_crash_windows(
    window: str, store: EventStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    n = 2
    await create_job(store, seed=11)
    first = crash_agent(window, n)
    with pytest.raises(Crash):
        await run_job(store, "j1", first)
    mid = await store.read_job("j1")
    assert [e.step_id for e in by_type(mid, T.STEP_COMPLETED)] == [0, 1]
    assert len(by_type(mid, T.LLM_RESPONDED, n)) == (0 if window == "W0" else 1)

    second = FakeAgent()
    job = await run_job(store, "j1", second)
    assert job.state is JobState.VERIFYING
    events = await store.read_job("j1")
    assert deltas(events) == await control_deltas(tmp_path_factory, 11, 5)

    sampled_n = [c for c in first.sample_calls + second.sample_calls if c[0] == n]
    assert sampled_n == ([(n, 0), (n, 1)] if window == "W0" else [(n, 0)])
    assert second.act_calls[0] == (n, 1)
    assert len(by_type(events, T.STEP_STARTED, n)) == 2
    for t in (T.LLM_RESPONDED, T.STEP_COMPLETED):
        counts = Counter(e.step_id for e in by_type(events, t))
        assert counts == {k: 1 for k in range(5)}
    assert len(by_type(events, T.USAGE_RECORDED)) == 5  # no step was sampled twice


async def test_cancellation_mid_sample_commits_nothing(store: EventStore) -> None:
    await create_job(store)
    task = asyncio.create_task(run_job(store, "j1", FakeAgent(token_interval=10.0)))
    await asyncio.sleep(15)  # virtual: inside the first step's token stream
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    events = await store.read_job("j1")
    assert not by_type(events, T.LLM_RESPONDED)
    assert not by_type(events, T.STEP_COMPLETED)
    job = await run_job(store, "j1", FakeAgent())
    assert job.state is JobState.VERIFYING


async def test_restart_limit(store: EventStore) -> None:
    await create_job(store)
    for _ in range(3):
        with pytest.raises(Crash):
            await run_job(store, "j1", CrashAgentEveryTime(), max_attempts_per_step=3)
    job = await run_job(store, "j1", FakeAgent(), max_attempts_per_step=3)
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    assert by_type(events, T.STATE_CHANGED)[-1].payload["reason"] == "restart limit"
    assert len(by_type(events, T.STEP_STARTED, 0)) == 3
    assert len(by_type(events, T.LLM_RESPONDED, 0)) == 1


class CrashAgentEveryTime(FakeAgent):
    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        raise Crash


async def test_config_change_fails_job_and_appends_nothing_else(store: EventStore) -> None:
    await create_job(store)
    with pytest.raises(Crash):
        await run_job(store, "j1", CrashAgent("W3", 1))
    before = await store.read_job("j1")
    job = await run_job(store, "j1", FakeAgent(system_prompt="changed"))
    assert job.state is JobState.FAILED
    after = await store.read_job("j1")
    assert len(after) == len(before) + 1
    assert after[-1].payload["reason"] == "config changed"
    assert after[-1].type is T.STATE_CHANGED


async def test_job_created_config_hash_is_checked(store: EventStore) -> None:
    await create_job(store, config_hash="not-the-agents")
    job = await run_job(store, "j1", FakeAgent())
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    assert not by_type(events, T.STEP_STARTED)
    assert events[-1].payload["reason"] == "config changed"


async def test_job_created_matching_config_hash_runs(store: EventStore) -> None:
    agent = FakeAgent(steps=1)
    await create_job(store, config_hash=agent.config_hash)
    assert (await run_job(store, "j1", agent)).state is JobState.VERIFYING


async def test_agent_exception_fails_job(store: EventStore) -> None:
    await create_job(store)
    agent = FakeAgent(fail_at_step=2)
    job = await run_job(store, "j1", agent)
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    reason = by_type(events, T.STATE_CHANGED)[-1].payload["reason"]
    assert "RuntimeError" in reason
    assert "fake failure at step 2" in reason
    assert [e.step_id for e in by_type(events, T.STEP_COMPLETED)] == [0, 1]


async def test_sample_exception_fails_job(store: EventStore) -> None:
    await create_job(store)

    def hook(ctx: StepContext, i: int) -> None:
        raise ValueError("boom")

    job = await run_job(store, "j1", FakeAgent(on_token=hook))
    assert job.state is JobState.FAILED


@pytest.mark.parametrize("state", ["done", "failed", "verifying"])
async def test_non_runnable_states_raise(store: EventStore, state: str) -> None:
    await create_job(store)
    await run_job(store, "j1", FakeAgent(steps=1))  # -> verifying
    if state != "verifying":
        await store.append(
            Event(
                job_id="j1",
                epoch=0,
                ts=1.0,
                type=T.STATE_CHANGED,
                payload={"from": "verifying", "to": state, "reason": "x"},
            )
        )
    with pytest.raises(JobNotRunnable):
        await run_job(store, "j1", FakeAgent(steps=1))
    with pytest.raises(JobNotRunnable):
        await run_job(store, "nope", FakeAgent())


async def test_clock_is_used_for_timestamps(store: EventStore) -> None:
    await create_job(store)
    ticks = iter(range(100, 10_000))
    clock: Callable[[], float] = lambda: float(next(ticks))  # noqa: E731
    await run_job(store, "j1", FakeAgent(steps=1), clock=clock)
    ts = [e.ts for e in (await store.read_job("j1"))[1:]]
    assert ts == sorted(ts)
    assert ts[0] >= 100


async def test_duplicate_llm_responded_uses_logged_response(store: EventStore) -> None:
    """A zombie logged n's response between our step_started and our sample finishing."""
    await create_job(store)
    other = FakeAgent()

    class Racing(FakeAgent):
        async def sample(self, ctx: StepContext, history: Any) -> ModelResponse:
            resp = await super().sample(ctx, history)
            if ctx.step == 0 and not other.sample_calls:
                # the zombie wins: it logs a different response for step 0 first
                zombie_ctx = StepContext(
                    job_id=ctx.job_id, step=0, attempt=9, rng=random.Random("z")
                )
                zr = await other.sample(zombie_ctx, history)
                await store.append_group(
                    [
                        Event(
                            job_id="j1",
                            epoch=0,
                            step_id=0,
                            ts=1.0,
                            type=T.LLM_RESPONDED,
                            payload={
                                "message": zr.message,
                                "tool_calls": [
                                    {
                                        "id": c.id,
                                        "name": c.name,
                                        "args": dict(c.args),
                                        "replay": "safe",
                                    }
                                    for c in zr.tool_calls
                                ],
                                "usage": {"active_seconds": 0.0, "tokens": 1, "usd_micros": 3},
                                "model": zr.model,
                                "config_hash": self.config_hash,
                            },
                        )
                    ]
                )
            return resp

    agent = Racing(steps=1)
    job = await run_job(store, "j1", agent)
    events = await store.read_job("j1")
    logged = by_type(events, T.LLM_RESPONDED, 0)
    assert len(logged) == 1
    assert deltas(events)[0][0] == logged[0].payload["message"]
    assert job.state is JobState.VERIFYING
    # wasted sampling is still counted
    assert len(by_type(events, T.USAGE_RECORDED)) == 1


async def test_store_error_propagates_and_does_not_fail_job(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A store failure is an infrastructure fault for recovery to handle, not an agent error:
    # it must propagate and must not be recorded as the job failing.
    await create_job(store)
    real_append_group = store.append_group

    async def flaky(events: Any) -> Any:
        if any(e.type is T.LLM_RESPONDED for e in events):
            raise OSError("disk full")
        return await real_append_group(events)

    monkeypatch.setattr(store, "append_group", flaky)
    with pytest.raises(OSError, match="disk full"):
        await run_job(store, "j1", FakeAgent(steps=3))
    monkeypatch.undo()
    job = await store.get_job("j1")
    assert job is not None
    assert job.state is JobState.RUNNING
    # ...and a later run resumes the job normally.
    done = await run_job(store, "j1", FakeAgent(steps=3))
    assert done.state is JobState.VERIFYING
