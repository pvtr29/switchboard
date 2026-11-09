from pathlib import Path
from typing import Any

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState
from switchboard.env import dir_env
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import ActResult, ModelResponse, StepContext
from switchboard.runtime.runner import run_job
from switchboard.store import EventStore

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]
STEPS = 5
SEED = 11
CRASH_STEP = 2  # even step: the FakeAgent always appends to log.txt there


class Crash(BaseException):
    """Stands in for kill -9."""


def by_type(events: list[Event], type_: EventType) -> list[Event]:
    return [e for e in events if e.type is type_]


@pytest.fixture
def template(tmp_path: Path) -> Path:
    t = tmp_path / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


def make_env(template: Path, root: Path, job_id: str = "j1") -> DirEnv:
    return DirEnv(template, root, job_id)


async def control_hash(template: Path, root: Path) -> str:
    """Hash of an uninterrupted run with the same seed."""
    env = make_env(template, root / "control", "ctl")
    async with await EventStore.open(root / "control.db") as s:
        await create_job(s, job_id="ctl", seed=SEED)
        await run_job(s, "ctl", FakeAgent(steps=STEPS), env=env)
    return await env.hash()


def crashing_agent(window: str, *, after_snapshot: bool = False) -> FakeAgent:
    """Crash at CRASH_STEP on attempt 0 in the given window (W0..W3, as in test_runner.py)."""

    def armed(ctx: StepContext) -> bool:
        return ctx.step == CRASH_STEP and ctx.attempt == 0

    def token_hook(ctx: StepContext, i: int) -> None:
        if window == "W0" and armed(ctx) and i == 1:
            raise Crash

    def file_hook(ctx: StepContext, tool: str, path: Path) -> None:
        # W2: mid-act, after write_file but before append_file; W2b: after the append
        if armed(ctx) and (
            (window == "W2" and tool == "write_file") or (window == "W2b" and tool == "append_file")
        ):
            raise Crash

    class Agent(FakeAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            if window == "W1" and armed(ctx):
                raise Crash
            result = await super().act(ctx, response)
            if window == "W3" and armed(ctx):
                raise Crash
            return result

    return Agent(steps=STEPS, on_token=token_hook, on_file_write=file_hook)


@pytest.mark.parametrize("window", ["W0", "W1", "W2", "W2b", "W3"])
async def test_crash_windows_with_env(
    window: str, store: EventStore, template: Path, tmp_path: Path
) -> None:
    ctl = await control_hash(template, tmp_path)
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(store, "j1", crashing_agent(window), env=env)
    # a restarted process gets a fresh DirEnv object over the same directories
    env2 = make_env(template, tmp_path / "envs")
    job = await run_job(store, "j1", FakeAgent(steps=STEPS), env=env2)
    assert job.state is JobState.VERIFYING
    assert await env2.hash() == ctl
    events = await store.read_job("j1")
    restored = by_type(events, T.RESTORED)
    assert len(restored) == 1
    assert restored[0].step_id is None
    assert restored[0].payload == {
        "step": CRASH_STEP,
        "env_ref": f"s{CRASH_STEP}-a0",
        "reason": "resume",
    }


async def test_happy_path_with_env(store: EventStore, template: Path, tmp_path: Path) -> None:
    ctl = await control_hash(template, tmp_path)
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    job = await run_job(store, "j1", FakeAgent(steps=STEPS), env=env)
    assert job.state is JobState.VERIFYING
    assert await env.hash() == ctl
    events = await store.read_job("j1")
    states = [(e.payload["from"], e.payload["to"]) for e in by_type(events, T.STATE_CHANGED)]
    assert states[:2] == [("queued", "provisioning"), ("provisioning", "running")]
    assert by_type(events, T.STATE_CHANGED)[1].payload["env_ref"] == "initial"
    refs = [e.payload["env_ref"] for e in by_type(events, T.STEP_COMPLETED)]
    assert refs == [f"s{n + 1}-a0" for n in range(STEPS)]
    assert not by_type(events, T.RESTORED)
    # initial + the last K=2 committed snapshots only
    assert await env.snapshots() == ["initial", "s4-a0", "s5-a0"]


async def test_keep_last_is_a_parameter(store: EventStore, template: Path, tmp_path: Path) -> None:
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    await run_job(store, "j1", FakeAgent(steps=STEPS), env=env, keep_last=1)
    assert await env.snapshots() == ["initial", "s5-a0"]


async def test_no_env_behaviour_unchanged(store: EventStore) -> None:
    await create_job(store)
    seen: list[Any] = []

    class Spy(FakeAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            seen.append(ctx.workdir)
            return await super().act(ctx, response)

    job = await run_job(store, "j1", Spy(steps=2))
    assert job.state is JobState.VERIFYING
    assert seen == [None, None]
    events = await store.read_job("j1")
    assert all(e.payload["env_ref"] is None for e in by_type(events, T.STEP_COMPLETED))


async def test_restore_is_necessary(
    store: EventStore, template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same crash as the passing test, but with restore a no-op: the hash must differ."""
    ctl = await control_hash(template, tmp_path)
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(store, "j1", crashing_agent("W2b"), env=env)

    async def no_restore(self: DirEnv, env_ref: str) -> None:
        return None

    monkeypatch.setattr(DirEnv, "restore", no_restore)
    env2 = make_env(template, tmp_path / "envs")
    job = await run_job(store, "j1", FakeAgent(steps=STEPS), env=env2)
    assert job.state is JobState.VERIFYING
    assert await env2.hash() != ctl  # the append ran twice


async def test_crash_after_snapshot_before_commit_leaves_harmless_orphan(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    ctl = await control_hash(template, tmp_path)

    class CrashingEnv(DirEnv):
        async def snapshot(self, name: str) -> str:
            ref = await super().snapshot(name)
            if name == f"s{CRASH_STEP + 1}-a0":
                raise Crash
            return ref

    env = CrashingEnv(template, tmp_path / "envs", "j1")
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(store, "j1", FakeAgent(steps=STEPS), env=env)
    orphan = f"s{CRASH_STEP + 1}-a0"
    env2 = make_env(template, tmp_path / "envs")
    assert orphan in await env2.snapshots()
    events = await store.read_job("j1")
    assert [e.step_id for e in by_type(events, T.STEP_COMPLETED)] == [0, 1]
    assert all(orphan != e.payload["env_ref"] for e in by_type(events, T.STEP_COMPLETED))

    job = await run_job(store, "j1", FakeAgent(steps=STEPS), env=env2)
    assert job.state is JobState.VERIFYING
    assert await env2.hash() == ctl
    committed = [
        e.payload["env_ref"] for e in by_type(await store.read_job("j1"), T.STEP_COMPLETED)
    ]
    assert committed[CRASH_STEP] == f"s{CRASH_STEP + 1}-a1"  # re-run snapshot has a new name
    assert orphan not in await env2.snapshots()  # gc removed it


async def test_missing_snapshot_on_resume_fails_job(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(store, "j1", crashing_agent("W1"), env=env)
    await env.gc([])  # the snapshot S(n) is gone
    job = await run_job(
        store, "j1", FakeAgent(steps=STEPS), env=make_env(template, tmp_path / "envs")
    )
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    assert by_type(events, T.STATE_CHANGED)[-1].payload["reason"] == "snapshot missing"
    assert not by_type(events, T.RESTORED)


async def test_resume_without_recorded_snapshot_fails_job(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    """A job that started without an environment cannot be resumed with one."""
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(store, "j1", crashing_agent("W1"))
    job = await run_job(
        store, "j1", FakeAgent(steps=STEPS), env=make_env(template, tmp_path / "envs")
    )
    assert job.state is JobState.FAILED
    events = await store.read_job("j1")
    assert by_type(events, T.STATE_CHANGED)[-1].payload["reason"] == "snapshot missing"


async def test_crash_during_provisioning_is_retried(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    ctl = await control_hash(template, tmp_path)

    class CrashingEnv(DirEnv):
        async def snapshot(self, name: str) -> str:
            raise Crash

    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(
            store, "j1", FakeAgent(steps=STEPS), env=CrashingEnv(template, tmp_path / "envs", "j1")
        )
    job = await store.get_job("j1")
    assert job is not None
    assert job.state is JobState.PROVISIONING
    env = make_env(template, tmp_path / "envs")
    job = await run_job(store, "j1", FakeAgent(steps=STEPS), env=env)
    assert job.state is JobState.VERIFYING
    assert await env.hash() == ctl


# ---- skip-unchanged snapshots ----------------------------------------------------------------


class SparseWriter(FakeAgent):
    """Writes a NEW file on the steps in `write_on`; every other step only reads (and the tool
    calls of the fake model touch nothing)."""

    def __init__(self, write_on: set[int], steps: int = 6, crash_step: int | None = None) -> None:
        super().__init__(steps=steps)
        self.write_on = write_on
        self.crash_step = crash_step

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        if ctx.step == self.crash_step and ctx.attempt == 0:
            raise Crash
        assert ctx.workdir is not None
        (ctx.workdir / "README").read_text()
        if ctx.step in self.write_on:
            (ctx.workdir / f"out{ctx.step}.txt").write_text(f"step {ctx.step}\n")
        return ActResult(messages=(), done=ctx.step + 1 >= self.steps, summary="s")


@pytest.fixture
def trusting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dir_env, "RACY_NS", 0)  # trust fingerprints at once (see test_dir_env)


def refs_of(events: list[Event]) -> list[str]:
    return [e.payload["env_ref"] for e in by_type(events, T.STEP_COMPLETED)]


async def sparse_control_hash(template: Path, root: Path, write_on: set[int]) -> str:
    env = make_env(template, root / "control", "ctl")
    async with await EventStore.open(root / "control.db") as s:
        await create_job(s, job_id="ctl", seed=SEED)
        await run_job(s, "ctl", SparseWriter(write_on), env=env)
    return await env.hash()


async def test_read_only_steps_reuse_the_ref(
    store: EventStore, template: Path, tmp_path: Path, trusting: None
) -> None:
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    job = await run_job(store, "j1", SparseWriter(set()), env=env)
    assert job.state is JobState.VERIFYING
    assert refs_of(await store.read_job("j1")) == ["initial"] * 6
    assert await env.snapshots() == ["initial"]  # not a single extra snapshot


async def test_changed_steps_snapshot_and_unchanged_reuse(
    store: EventStore, template: Path, tmp_path: Path, trusting: None
) -> None:
    write_on = {0, 3}
    ctl = await sparse_control_hash(template, tmp_path, write_on)
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    await run_job(store, "j1", SparseWriter(write_on), env=env)
    # step 0 writes -> new snapshot; 1, 2 read only -> reuse it; 3 writes -> new; 4, 5 reuse
    assert refs_of(await store.read_job("j1")) == ["s1-a0"] * 3 + ["s4-a0"] * 3
    assert await env.hash() == ctl
    assert await env.snapshots() == ["initial", "s1-a0", "s4-a0"]


async def test_untrusted_fingerprint_always_snapshots(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    """Default racy window: the fast test's files are too fresh to trust, so nothing is skipped
    and the behaviour is exactly the old one."""
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    await run_job(store, "j1", SparseWriter(set(), steps=3), env=env)
    assert refs_of(await store.read_job("j1")) == ["s1-a0", "s2-a0", "s3-a0"]


async def test_env_without_change_detection_snapshots_every_step(
    store: EventStore, template: Path, tmp_path: Path, trusting: None
) -> None:
    inner = make_env(template, tmp_path / "envs")

    class Plain:
        """Only the base Environment protocol: no changed_since."""

        workdir = inner.workdir
        provision = inner.provision
        snapshot = inner.snapshot
        restore = inner.restore
        hash = inner.hash
        gc = inner.gc
        destroy = inner.destroy

    await create_job(store, seed=SEED)
    await run_job(store, "j1", SparseWriter(set(), steps=3), env=Plain())
    assert refs_of(await store.read_job("j1")) == ["s1-a0", "s2-a0", "s3-a0"]


@pytest.mark.parametrize("crash_step", [1, 2, 4])
async def test_recovery_after_reused_refs_is_correct(
    crash_step: int, store: EventStore, template: Path, tmp_path: Path, trusting: None
) -> None:
    write_on = {0, 3}
    ctl = await sparse_control_hash(template, tmp_path, write_on)
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(
            store,
            "j1",
            SparseWriter(write_on, crash_step=crash_step),
            env=make_env(template, tmp_path / "envs"),
        )
    env2 = make_env(template, tmp_path / "envs")  # a new process: no fingerprints
    job = await run_job(store, "j1", SparseWriter(write_on), env=env2)
    assert job.state is JobState.VERIFYING
    assert await env2.hash() == ctl
    events = await store.read_job("j1")
    (restored,) = by_type(events, T.RESTORED)
    expected_ref = {1: "s1-a0", 2: "s1-a0", 4: "s4-a0"}[crash_step]  # S(crash_step)
    assert restored.payload["env_ref"] == expected_ref  # the reused ref, and it still exists


async def test_keep_last_zero_never_reuses(
    store: EventStore, template: Path, tmp_path: Path, trusting: None
) -> None:
    env = make_env(template, tmp_path / "envs")
    await create_job(store, seed=SEED)
    await run_job(store, "j1", SparseWriter(set(), steps=3), env=env, keep_last=0)
    assert refs_of(await store.read_job("j1")) == ["s1-a0", "s2-a0", "s3-a0"]
