"""recover(): re-queue provisioning/running jobs, admit queued ones, leave the rest."""

import contextlib
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import Agent
from switchboard.runtime.recovery import REASON_RESTART, recover
from switchboard.runtime.runner import run_job
from switchboard.runtime.supervisor import Supervisor
from switchboard.store import EventStore

T = EventType
pytestmark = pytest.mark.asyncio


def state_change(job_id: str, src: JobState, dst: JobState) -> Event:
    return Event(
        job_id=job_id,
        epoch=0,
        ts=1.0,
        type=T.STATE_CHANGED,
        payload={"from": src.value, "to": dst.value, "reason": "test"},
    )


def make_supervisor(store: EventStore, template: Path | None, root: Path) -> Supervisor:
    def agent(job: object) -> Agent:
        return FakeAgent(steps=3, token_interval=0)

    def env(job: object) -> DirEnv | None:
        jid = getattr(job, "job_id")  # noqa: B009
        return DirEnv(template, root / "envs", jid) if template else None

    return Supervisor(store, agent, env)


async def test_every_state_is_handled(store: EventStore, tmp_path: Path) -> None:
    for jid in ("queued", "prov", "running", "paused", "over", "verifying", "failed"):
        await create_job(store, jid)
    await store.append_group([state_change("prov", JobState.QUEUED, JobState.PROVISIONING)])
    await store.append_group(
        [
            state_change("running", JobState.QUEUED, JobState.PROVISIONING),
            state_change("running", JobState.PROVISIONING, JobState.RUNNING),
        ]
    )
    chains = {
        "paused": [JobState.PAUSED],
        "over": [JobState.OVER_BUDGET],
        "verifying": [JobState.VERIFYING],
        "failed": [JobState.FAILED],
    }
    for jid, path in chains.items():
        moves, cur = [], JobState.QUEUED
        for nxt in [JobState.PROVISIONING, JobState.RUNNING, *path]:
            moves.append(state_change(jid, cur, nxt))
            cur = nxt
        await store.append_group(moves)

    sup = make_supervisor(store, None, tmp_path)  # no env: the log is synthetic
    async with sup_closer(sup):
        report = await recover(store, sup)
        assert report.ok
        assert (report.requeued, report.admitted) == (
            2,
            3,
        )  # prov and running re-queued; queued as is
        assert report.left == {"paused": 1, "over_budget": 1, "verifying": 1, "failed": 1}
        assert report.verified == 7
        for jid in ("queued", "prov", "running"):
            await sup.wait(jid)
        for jid in ("queued", "prov", "running"):
            job = await store.get_job(jid)
            assert job is not None
            assert job.state is JobState.VERIFYING
        for jid, state in (("paused", "paused"), ("over", "over_budget"), ("failed", "failed")):
            job = await store.get_job(jid)
            assert job is not None
            assert job.state.value == state
        reasons = [
            e.payload["reason"]
            for e in await store.read_job("running")
            if e.type is T.STATE_CHANGED and e.payload["to"] == "queued"
        ]
        assert reasons == [REASON_RESTART]


@contextlib.asynccontextmanager
async def sup_closer(sup: Supervisor) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await sup.aclose()


async def test_resumes_a_partly_run_job_from_its_checkpoint(
    store: EventStore, tmp_path: Path, template: Path
) -> None:
    await create_job(store, "j1")
    stop = {"n": 0}

    def should_stop() -> bool:
        stop["n"] += 1
        return stop["n"] > 2  # two steps commit, then stop at a boundary (like a drain)

    env = DirEnv(template, tmp_path / "envs", "j1")
    job = await run_job(
        store, "j1", FakeAgent(steps=3, token_interval=0), env=env, should_stop=should_stop
    )
    assert job.state is JobState.RUNNING
    assert job.last_step == 1
    sup = make_supervisor(store, template, tmp_path)
    async with sup_closer(sup):
        report = await recover(store, sup)
        assert (report.requeued, report.admitted) == (1, 1)
        await sup.wait("j1")
    done = await store.get_job("j1")
    assert done is not None
    assert done.state is JobState.VERIFYING
    assert done.last_step == 2
    events = await store.read_job("j1")
    assert [e.step_id for e in events if e.type is T.LLM_RESPONDED] == [0, 1, 2]
    assert [e for e in events if e.type is T.RESTORED]


async def test_prepare_is_called_and_failures_are_reported(
    store: EventStore, tmp_path: Path
) -> None:
    await create_job(store, "ok")
    await create_job(store, "bad")
    prepared: list[str] = []

    async def prepare(job_id: str) -> None:
        prepared.append(job_id)
        if job_id == "bad":
            raise LookupError("no spec")

    sup = make_supervisor(store, None, tmp_path)
    async with sup_closer(sup):
        report = await recover(store, sup, prepare=prepare)
        assert prepared == ["bad", "ok"]
        assert report.admitted == 1
        assert len(report.errors) == 1
        assert report.errors[0].startswith("bad:")
        assert not report.ok
        await sup.wait("ok")


async def test_projection_check_detects_a_corrupt_table(store: EventStore, tmp_path: Path) -> None:
    await create_job(store, "j1")
    await create_job(store, "j2")
    with closing(sqlite3.connect(tmp_path / "sb.db")) as conn:
        conn.execute("UPDATE jobs SET state = 'done' WHERE job_id = 'j2'")
        conn.commit()
    sup = make_supervisor(store, None, tmp_path)
    async with sup_closer(sup):
        report = await recover(store, sup, verify=True)
        assert report.mismatches == ("j2",)
        assert report.verified == 2
        skipped = await recover(store, make_supervisor(store, None, tmp_path), verify=False)
        assert skipped.verified == 0
        assert skipped.mismatches == ()
        await sup.wait("j1")
