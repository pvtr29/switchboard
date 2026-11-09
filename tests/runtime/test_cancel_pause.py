import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import ActResult, Message, ModelResponse, StepContext
from switchboard.runtime.runner import run_job
from switchboard.runtime.supervisor import (
    NotCancellable,
    NotPausable,
    NotResumable,
    Supervisor,
    UnknownJob,
)
from switchboard.store import EventStore

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]
STEPS = 5


class GatedAgent(FakeAgent):
    """Blocks in act (after sampling) until `gate` is set; `reached` is set on arrival."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.gate = asyncio.Event()
        self.reached = asyncio.Event()

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        self.reached.set()
        await self.gate.wait()
        return await super().act(ctx, response)


class SlowSampleAgent(FakeAgent):
    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        await asyncio.sleep(10)
        return await super().sample(ctx, history)


async def state_of(store: EventStore, job_id: str) -> JobState:
    job = await store.get_job(job_id)
    assert job is not None
    return job.state


async def assert_consistent(store: EventStore, job_id: str) -> list[Event]:
    """No half groups: every step_completed has its llm_responded, steps are contiguous,
    every committed step has its delta, and nothing follows the `cancelled` state change."""
    events = await store.read_job(job_id)
    responded = {e.step_id for e in events if e.type is T.LLM_RESPONDED}
    completed = [e.step_id for e in events if e.type is T.STEP_COMPLETED]
    assert completed == list(range(len(completed)))
    assert set(completed) <= responded
    for e in events:
        if e.type is T.STEP_COMPLETED:
            assert e.payload["delta"]
    changes = [e for e in events if e.type is T.STATE_CHANGED]
    if changes and changes[-1].payload["to"] == "cancelled":
        assert events[-1] is changes[-1]
    return events


def env_for(template: Path, root: Path, job_id: str) -> DirEnv:
    return DirEnv(template, root, job_id)


@pytest.fixture
def template(tmp_path: Path) -> Path:
    t = tmp_path / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


# ---- cancel -------------------------------------------------------------------------------


@pytest.mark.parametrize("with_env", [False, True])
async def test_cancel_random_awaits(
    store: EventStore, template: Path, tmp_path: Path, with_env: bool
) -> None:
    rng = random.Random(1234)
    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=STEPS, token_interval=0.01),
        (lambda job: env_for(template, tmp_path / "envs", job.job_id)) if with_env else None,
        max_running=3,
    )
    n = 40
    for i in range(n):
        job_id = f"j{i}"
        await create_job(store, job_id, seed=i)
        sup.start(job_id)
        # Mix of delays: before admission, mid provisioning, mid-sample, mid-act, after the end.
        await asyncio.sleep(rng.choice([0.0, rng.uniform(0, 0.5), rng.uniform(0, 0.5), 5.0]))
        await sup.cancel(job_id)
        assert await state_of(store, job_id) is JobState.CANCELLED
        await assert_consistent(store, job_id)
        assert sup.stats.running == 0  # cancel() returns only after the task released its slot
    await sup.aclose()
    assert (sup.stats.running, sup.stats.waiting) == (0, 0)


@pytest.mark.parametrize("where", ["sample", "act", "snapshot", "append_before", "append_after"])
async def test_cancel_in_every_phase(
    store: EventStore,
    template: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    env = env_for(template, tmp_path / "envs", "j1")
    agent: FakeAgent
    reached = asyncio.Event()
    never = asyncio.Event()
    match where:
        case "sample":
            agent = SlowSampleAgent(steps=STEPS)
            monkeypatch.setattr(agent, "sample", _signalling(agent.sample, reached))
        case "act":
            gated = GatedAgent(steps=STEPS)
            reached, agent = gated.reached, gated
        case "snapshot":
            agent = FakeAgent(steps=STEPS)
            real_snapshot = env.snapshot

            async def slow_snapshot(name: str) -> str:
                if name.startswith("s"):  # a step snapshot, not "initial"
                    reached.set()
                    await never.wait()
                return await real_snapshot(name)

            monkeypatch.setattr(env, "snapshot", slow_snapshot)
        case _:
            agent = FakeAgent(steps=STEPS)
            real_append = store.append_group

            async def gated_append(events: Sequence[Event]) -> list:  # type: ignore[type-arg]
                if any(e.type is T.STEP_COMPLETED for e in events):
                    reached.set()
                    if where == "append_before":
                        await never.wait()
                    result = await real_append(events)
                    await never.wait()  # cancelled while awaiting the ack
                    return result
                return await real_append(events)

            monkeypatch.setattr(store, "append_group", gated_append)
    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: agent, lambda job: env)
    sup.start("j1")
    await reached.wait()
    await sup.cancel("j1")
    assert await state_of(store, "j1") is JobState.CANCELLED
    events = await assert_consistent(store, "j1")
    reason = [e for e in events if e.type is T.STATE_CHANGED][-1].payload["reason"]
    assert reason == "cancelled by user"
    outcome = sup.outcome("j1")
    assert outcome is not None
    assert outcome.cancelled
    completed = [e for e in events if e.type is T.STEP_COMPLETED]
    if where == "append_after":
        assert len(completed) == 1  # the group was already queued: committed whole
    else:
        assert not completed  # a cancelled task never writes a checkpoint
    assert sup.stats.running == 0
    await sup.aclose()


def _signalling(
    fn: Callable[..., Awaitable[ModelResponse]], ev: asyncio.Event
) -> Callable[..., Awaitable[ModelResponse]]:
    async def inner(*a: object, **kw: object) -> ModelResponse:
        ev.set()
        return await fn(*a, **kw)

    return inner


async def test_cancel_mid_token_stream_writes_no_response(store: EventStore) -> None:
    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: FakeAgent(steps=3, token_interval=1.0))
    sup.start("j1")
    await asyncio.sleep(1.5)  # inside the first sample, after 1 token
    await sup.cancel("j1", "enough")
    events = await assert_consistent(store, "j1")
    assert not [e for e in events if e.type in (T.LLM_RESPONDED, T.STEP_COMPLETED)]
    assert [e for e in events if e.type is T.STATE_CHANGED][-1].payload["reason"] == "enough"
    await sup.aclose()


async def test_cancel_errors_and_idempotence(store: EventStore) -> None:
    sup = Supervisor(store, lambda job: FakeAgent(steps=2))
    with pytest.raises(UnknownJob):
        await sup.cancel("nope")
    await create_job(store, "done")
    sup.start("done")
    await sup.wait("done")
    await sup.cancel("done")  # verifying is non-terminal: cancellable
    assert await state_of(store, "done") is JobState.CANCELLED
    seq = await store.last_seq()
    await sup.cancel("done")  # second time: no-op
    assert await store.last_seq() == seq

    await create_job(store, "failed")
    failing = Supervisor(store, lambda job: FakeAgent(steps=2, fail_at_step=0))
    failing.start("failed")
    await failing.wait("failed")
    assert await state_of(store, "failed") is JobState.FAILED
    with pytest.raises(NotCancellable, match="failed"):
        await failing.cancel("failed")
    await failing.aclose()
    await sup.aclose()


async def test_concurrent_cancels_append_one_state_change(store: EventStore) -> None:
    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: SlowSampleAgent(steps=2))
    sup.start("j1")
    await asyncio.sleep(1)
    await asyncio.gather(sup.cancel("j1"), sup.cancel("j1"), sup.cancel("j1"))
    events = await store.read_job("j1")
    cancelled = [e for e in events if e.type is T.STATE_CHANGED and e.payload["to"] == "cancelled"]
    assert len(cancelled) == 1
    await sup.aclose()


# ---- pause / resume -----------------------------------------------------------------------


async def test_pause_mid_step_commits_then_pauses_and_resume_matches_control(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    seed = 21
    ctl_env = env_for(template, tmp_path / "control", "ctl")
    await create_job(store, "ctl", seed=seed)
    await run_job(store, "ctl", FakeAgent(steps=STEPS), env=ctl_env)

    env = env_for(template, tmp_path / "envs", "j1")
    await create_job(store, "j1", seed=seed)
    mid_step = asyncio.Event()

    def on_token(ctx: StepContext, i: int) -> None:
        if ctx.step == 1 and i == 1:
            mid_step.set()

    sup = Supervisor(
        store,
        lambda job: FakeAgent(steps=STEPS, on_token=on_token),
        lambda job: env,
        max_running=1,
    )
    sup.start("j1")
    await mid_step.wait()  # inside step 1's sample: step 0 is committed, step 1 is not
    events = await store.read_job("j1")
    assert [e.step_id for e in events if e.type is T.STEP_COMPLETED] == [0]
    await sup.pause("j1")
    await sup.wait("j1")
    assert await state_of(store, "j1") is JobState.PAUSED
    events = await assert_consistent(store, "j1")
    started = [e.step_id for e in events if e.type is T.STEP_STARTED]
    completed = [e.step_id for e in events if e.type is T.STEP_COMPLETED]
    assert completed == started == [0, 1]  # the in-flight step committed; none left half done
    assert [e for e in events if e.type is T.STATE_CHANGED][-1].payload["reason"] == (
        "paused by user"
    )
    assert sup.stats.running == 0  # no slot held
    seq = await store.last_seq()
    await asyncio.sleep(10_000)
    assert await store.last_seq() == seq  # no further steps

    await sup.resume("j1")
    await sup.wait("j1")
    job = await store.get_job("j1")
    assert job is not None
    assert job.state is JobState.VERIFYING
    events = await assert_consistent(store, "j1")
    assert [e.step_id for e in events if e.type is T.STEP_COMPLETED] == list(range(STEPS))
    assert [e for e in events if e.type is T.RESTORED]  # resumed from the log, env restored
    assert await env.hash() == await ctl_env.hash()
    await sup.aclose()


async def test_paused_jobs_do_not_count_against_the_limit(store: EventStore) -> None:
    for i in range(4):
        await create_job(store, f"j{i}", seed=i)
    sup = Supervisor(store, lambda job: FakeAgent(steps=500), max_running=2)
    for i in range(4):
        sup.start(f"j{i}")
    await asyncio.sleep(0.1)
    assert (await store.get_job("j2")) is not None
    assert await state_of(store, "j2") is JobState.QUEUED
    await sup.pause("j0")
    await sup.pause("j1")
    await sup.wait("j0")
    await sup.wait("j1")
    await asyncio.sleep(0.1)
    assert await state_of(store, "j0") is JobState.PAUSED
    assert await state_of(store, "j1") is JobState.PAUSED
    for j in ("j2", "j3"):
        assert await state_of(store, j) is JobState.RUNNING
        assert [e for e in await store.read_job(j) if e.type is T.STEP_STARTED]
    assert (sup.stats.running, sup.stats.waiting) == (2, 0)
    # resuming one paused job waits for a slot like any other
    await sup.resume("j0")
    await asyncio.sleep(0.1)
    assert await state_of(store, "j0") is JobState.QUEUED
    assert sup.stats.waiting == 1
    await sup.cancel("j2")
    await asyncio.sleep(0.1)
    assert await state_of(store, "j0") is JobState.RUNNING
    await sup.aclose()
    assert sup.stats.running == 0


async def test_pause_while_queued_takes_effect_at_first_boundary(store: EventStore) -> None:
    await create_job(store, "a", seed=1)
    await create_job(store, "b", seed=2)
    sup = Supervisor(store, lambda job: FakeAgent(steps=3), max_running=1)
    sup.start("a")
    sup.start("b")
    await asyncio.sleep(0.001)
    assert await state_of(store, "b") is JobState.QUEUED
    await sup.pause("b")  # remembered, not rejected
    await sup.wait("a")
    await sup.wait("b")
    assert await state_of(store, "b") is JobState.PAUSED
    events = await store.read_job("b")
    assert not [e for e in events if e.type is T.STEP_STARTED]  # went straight to paused
    await sup.resume("b")
    await sup.wait("b")
    assert await state_of(store, "b") is JobState.VERIFYING
    events = await assert_consistent(store, "b")
    assert [e.step_id for e in events if e.type is T.STEP_COMPLETED] == [0, 1, 2]
    await sup.aclose()


async def test_resume_before_pause_lands_withdraws_it(store: EventStore) -> None:
    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: FakeAgent(steps=3))
    sup.start("j1")
    await sup.pause("j1")
    await sup.resume("j1")
    await sup.wait("j1")
    assert await state_of(store, "j1") is JobState.VERIFYING
    await sup.aclose()


async def test_pause_requires_a_live_job(store: EventStore) -> None:
    sup = Supervisor(store, lambda job: FakeAgent(steps=2))
    with pytest.raises(UnknownJob):
        await sup.pause("nope")
    await create_job(store, "j1")
    with pytest.raises(NotPausable):  # created but never started
        await sup.pause("j1")
    sup.start("j1")
    await sup.wait("j1")
    with pytest.raises(NotPausable):  # verifying
        await sup.pause("j1")
    await sup.aclose()


async def test_cancel_paused_job_and_resume_errors(store: EventStore) -> None:
    await create_job(store, "j1")
    sup = Supervisor(store, lambda job: FakeAgent(steps=50))
    with pytest.raises(UnknownJob):
        await sup.resume("nope")
    sup.start("j1")
    await asyncio.sleep(0.05)
    with pytest.raises(NotResumable, match="running"):  # no pause pending: not paused
        await sup.resume("j1")
    await sup.pause("j1")
    await sup.wait("j1")
    assert await state_of(store, "j1") is JobState.PAUSED
    await sup.cancel("j1")
    assert await state_of(store, "j1") is JobState.CANCELLED
    with pytest.raises(NotResumable, match="cancelled"):
        await sup.resume("j1")
    await sup.aclose()


async def test_resumed_queued_job_is_not_started_over(store: EventStore) -> None:
    """run_job on a queued job with history resumes from the log (no re-provision)."""
    await create_job(store, "j1", seed=3)
    pause = {"on": False}
    agent = FakeAgent(steps=4)

    def boundary_hook(ctx: StepContext, i: int) -> None:
        if ctx.step == 1:
            pause["on"] = True

    agent.model.on_token = boundary_hook
    job = await run_job(store, "j1", agent, should_pause=lambda: pause["on"])
    assert job.state is JobState.PAUSED
    sup = Supervisor(store, lambda job: FakeAgent(steps=4))
    await sup.resume("j1")
    await sup.wait("j1")
    events = await assert_consistent(store, "j1")
    assert [e.step_id for e in events if e.type is T.STEP_COMPLETED] == [0, 1, 2, 3]
    assert [e.step_id for e in events if e.type is T.STEP_STARTED] == [0, 1, 2, 3]
    await sup.aclose()


async def test_cancel_retries_when_its_read_is_stale(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (seen on CI, py3.13): a cancelled task's last group (queued -> provisioning ->
    # running) can still be in the writer queue when cancel() re-reads the job, so cancel()
    # sees `queued` and appends `queued -> cancelled`, which is then illegal. cancel() must
    # re-read and retry instead of raising NotCancellable.
    await create_job(store, "j1")

    def sc(src: JobState, dst: JobState) -> Event:
        return Event(
            job_id="j1",
            epoch=0,
            ts=1.0,
            type=T.STATE_CHANGED,
            payload={"from": src.value, "to": dst.value, "reason": "test"},
        )

    queued = await store.get_job("j1")
    assert queued is not None
    assert queued.state is JobState.QUEUED
    await store.append_group(
        [sc(JobState.QUEUED, JobState.PROVISIONING), sc(JobState.PROVISIONING, JobState.RUNNING)]
    )
    real_get_job = store.get_job
    calls = 0

    async def stale_once(job_id: str) -> object:
        nonlocal calls
        calls += 1
        # cancel() reads twice before appending (existence check, then re-read): serve the
        # stale `queued` both times, then the truth.
        return queued if calls <= 2 else await real_get_job(job_id)

    sup = Supervisor(store, lambda job: FakeAgent())
    try:
        monkeypatch.setattr(store, "get_job", stale_once)
        await sup.cancel("j1", reason="stale read")
        assert calls >= 3  # the stale append was rejected and cancel() re-read
    finally:
        monkeypatch.undo()
        await sup.aclose()
    assert await state_of(store, "j1") is JobState.CANCELLED
    events = await assert_consistent(store, "j1")
    last = [e for e in events if e.type is T.STATE_CHANGED][-1]
    assert last.payload == {"from": "running", "to": "cancelled", "reason": "stale read"}
