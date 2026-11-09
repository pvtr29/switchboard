import asyncio
import gc
from collections.abc import Sequence
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, Job, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import ActResult, Message, ModelResponse, StepContext
from switchboard.runtime.runner import JobNotRunnable, run_job
from switchboard.runtime.supervisor import AlreadyRunning, Supervisor, SupervisorClosed
from switchboard.store import EventStore, StoreClosed

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]


class Crash(BaseException):
    """Not an Exception: the runner does not turn it into a failed job, the supervisor must."""


class CrashingAgent(FakeAgent):
    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        raise Crash("machinery down")


class SlowAgent(FakeAgent):
    async def sample(self, ctx: StepContext, history: Sequence[Message]) -> ModelResponse:
        await asyncio.sleep(3600)
        return await super().sample(ctx, history)


async def state_events(store: EventStore, job_id: str) -> list[Event]:
    return [e for e in await store.read_job(job_id) if e.type is T.STATE_CHANGED]


async def test_one_crash_among_twenty(store: EventStore) -> None:
    ids = [f"j{i}" for i in range(20)]
    for i, j in enumerate(ids):
        await create_job(store, j, seed=i)

    def factory(job: Job) -> FakeAgent:
        return CrashingAgent(steps=3) if job.job_id == "j7" else FakeAgent(steps=3)

    sup = Supervisor(store, factory)
    for j in ids:
        sup.start(j)
    assert sup.running["j0"].get_name() == "job:j0"
    for j in ids:
        await sup.wait(j)
    await sup.aclose()
    for j in ids:
        job = await store.get_job(j)
        assert job is not None
        if j == "j7":
            assert job.state is JobState.FAILED
            last = (await state_events(store, j))[-1]
            assert last.payload["reason"] == "supervisor: Crash: machinery down"
        else:
            assert job.state is JobState.VERIFYING
    assert sup.stats.started == 20
    assert sup.stats.finished == 19
    assert sup.stats.crashed == 1
    assert not sup.running
    assert not sup.errors
    outcome = sup.outcome("j7")
    assert outcome is not None
    assert isinstance(outcome.error, Crash)


async def test_agent_exception_is_run_jobs_business(store: EventStore) -> None:
    await create_job(store)
    sup = Supervisor(store, lambda job: FakeAgent(steps=3, fail_at_step=1))
    sup.start("j1")
    await sup.wait("j1")
    await sup.aclose()
    states = await state_events(store, "j1")
    failed = [e for e in states if e.payload["to"] == "failed"]
    assert len(failed) == 1
    assert failed[0].payload["reason"].startswith("agent error:")
    assert sup.stats.finished == 1
    assert sup.stats.crashed == 0
    outcome = sup.outcome("j1")
    assert outcome is not None
    assert outcome.job is not None
    assert outcome.job.state is JobState.FAILED


async def test_crash_of_queued_job_is_logged_not_appended(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await create_job(store)

    async def broken(*args: object, **kwargs: object) -> Job:
        raise RuntimeError("before anything ran")

    monkeypatch.setattr("switchboard.runtime.supervisor.run_job", broken)
    sup = Supervisor(store, lambda job: FakeAgent())
    sup.start("j1")
    await sup.wait("j1")
    await sup.aclose()
    job = await store.get_job("j1")
    assert job is not None
    assert job.state is JobState.QUEUED  # failed is not reachable from queued
    assert sup.stats.crashed == 1


async def test_unknown_job_crashes_without_events(store: EventStore) -> None:
    sup = Supervisor(store, lambda job: FakeAgent())
    sup.start("nope")
    await sup.wait("nope")
    await sup.aclose()
    outcome = sup.outcome("nope")
    assert outcome is not None
    assert isinstance(outcome.error, JobNotRunnable)


async def test_tasks_survive_gc(store: EventStore) -> None:
    n = 10
    for i in range(n):
        await create_job(store, f"j{i}", seed=i)
    sup = Supervisor(store, lambda job: FakeAgent(steps=4))
    for i in range(n):
        sup.start(f"j{i}")
    for _ in range(5):
        gc.collect()
        await asyncio.sleep(0.01)
    while sup.running:
        gc.collect()
        await asyncio.sleep(0.01)
    await sup.aclose()
    assert sup.stats.finished == n
    for i in range(n):
        job = await store.get_job(f"j{i}")
        assert job is not None
        assert job.state is JobState.VERIFYING


async def test_aclose_cancels_and_is_idempotent(store: EventStore) -> None:
    for i in range(3):
        await create_job(store, f"j{i}")
    sup = Supervisor(store, lambda job: SlowAgent(steps=2))
    for i in range(3):
        sup.start(f"j{i}")
    await asyncio.sleep(1)
    tasks = list(sup.running.values())
    assert len(tasks) == 3
    await sup.aclose()
    assert all(t.done() for t in tasks)
    assert not sup.running
    assert sup.stats.cancelled == 3
    before = await store.last_seq()
    await sup.aclose()
    assert await store.last_seq() == before
    for i in range(3):
        job = await store.get_job(f"j{i}")
        assert job is not None
        assert job.state is JobState.RUNNING  # aclose (shutdown) records nothing: stays resumable
    with pytest.raises(SupervisorClosed):
        sup.start("j0")


async def test_start_twice_while_live_raises(store: EventStore) -> None:
    await create_job(store)
    sup = Supervisor(store, lambda job: SlowAgent())
    sup.start("j1")
    with pytest.raises(AlreadyRunning):
        sup.start("j1")
    await sup.aclose()


async def test_store_closed_is_recorded_not_appended(tmp_path: Path) -> None:
    store = await EventStore.open(tmp_path / "sb.db")
    await create_job(store)
    sup = Supervisor(store, lambda job: FakeAgent(steps=5, token_interval=1.0))
    sup.start("j1")
    await asyncio.sleep(2)
    await store.aclose()
    await sup.wait("j1")
    await sup.aclose()
    assert len(sup.errors) == 1
    job_id, err = sup.errors[0]
    assert job_id == "j1"
    assert isinstance(err, StoreClosed)
    assert sup.stats.crashed == 1


async def test_dir_env_jobs_run_concurrently(store: EventStore, tmp_path: Path) -> None:
    template = tmp_path / "template"
    (template / "files").mkdir(parents=True)
    (template / "README").write_text("seed project\n")
    n, steps = 5, 4
    envs: dict[str, DirEnv] = {}

    def env_factory(job: Job) -> DirEnv:
        envs[job.job_id] = DirEnv(template, tmp_path / "envs", job.job_id)
        return envs[job.job_id]

    sup = Supervisor(store, lambda job: FakeAgent(steps=steps), env_factory)
    for i in range(n):
        await create_job(store, f"j{i}", seed=i)
        sup.start(f"j{i}")
    for i in range(n):
        await sup.wait(f"j{i}")
    await sup.aclose()
    assert len({e.workdir for e in envs.values()}) == n
    for i in range(n):
        control_env = DirEnv(template, tmp_path / "control", f"c{i}")
        await create_job(store, f"c{i}", seed=i)
        await run_job(store, f"c{i}", FakeAgent(steps=steps), env=control_env)
        assert await envs[f"j{i}"].hash() == await control_env.hash()
