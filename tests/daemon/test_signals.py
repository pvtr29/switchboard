"""SIGTERM against a real daemon process: stop at step boundaries, leave a resumable log."""

import asyncio
import signal
import sys
import time
from pathlib import Path

import pytest
from conftest import create_job, wait_for_socket, wait_state, wait_steps

from switchboard.agents.fake import FakeAgent
from switchboard.client.conn import Client
from switchboard.core import EventType, JobState
from switchboard.daemon.server import Daemon
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.runner import run_job
from switchboard.store import EventStore

T = EventType
pytestmark = pytest.mark.asyncio
STEPS = 40
JOBS = 3


async def start_daemon(
    sock: Path, db: Path, envs: Path, log: Path, *, shutdown_timeout: float = 10.0
) -> asyncio.subprocess.Process:
    with log.open("wb") as out:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "switchboard.client.cli",
            "daemon",
            "--socket",
            str(sock),
            "--db",
            str(db),
            "--envs",
            str(envs),
            "--max-running",
            "8",
            "--shutdown-timeout",
            str(shutdown_timeout),
            stdout=out,
            stderr=out,
        )
    await asyncio.to_thread(wait_for_socket, sock)
    return proc


async def test_sigterm_stops_at_boundaries_and_leaves_jobs_resumable(
    sock_dir: Path, tmp_path: Path, template: Path
) -> None:
    sock, db, envs = sock_dir / "s", tmp_path / "sb.db", tmp_path / "envs"
    proc = await start_daemon(sock, db, envs, tmp_path / "daemon.log")
    try:
        async with await Client.connect(str(sock)) as c:
            ids = [
                (
                    await c.request(
                        "submit",
                        steps=STEPS,
                        seed=i,
                        token_interval=0.03,
                        env_template=str(template),
                    )
                )["job_id"]
                for i in range(JOBS)
            ]
            await wait_steps(c, ids, 2)
            started = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            async with asyncio.timeout(10):
                code = await proc.wait()
            elapsed = time.monotonic() - started
    except BaseException:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    assert code == 0
    assert elapsed < 10
    assert not sock.exists()  # cleaned up

    async with await EventStore.open(db) as store:
        progress: dict[str, int] = {}
        for jid in ids:
            job = await store.get_job(jid)
            assert job is not None
            assert job.state is JobState.RUNNING  # not paused, failed or cancelled
            events = await store.read_job(jid)
            responded = {e.step_id for e in events if e.type is T.LLM_RESPONDED}
            completed = [e.step_id for e in events if e.type is T.STEP_COMPLETED]
            assert completed == list(range(len(completed)))  # contiguous: no hole, no partial
            assert set(completed) <= responded  # every committed step has its response
            assert all(e.payload["env_ref"] for e in events if e.type is T.STEP_COMPLETED)
            states = [e.payload["to"] for e in events if e.type is T.STATE_CHANGED]
            assert states == ["provisioning", "running"]
            assert not [e for e in events if e.type is T.EFFECT_INTENT]
            assert 2 <= len(completed) < STEPS
            progress[jid] = len(completed)

        # Resumable: the runner finishes each job from its log, and the environment ends up
        # byte-identical to an uninterrupted run with the same seed.
        for i, jid in enumerate(ids):
            env = DirEnv(template, envs, jid)
            job = await run_job(store, jid, FakeAgent(steps=STEPS, token_interval=0), env=env)
            assert job.state is JobState.VERIFYING
            assert job.last_step == STEPS - 1
            control_env = DirEnv(template, tmp_path / "control", f"c{i}")
            async with await EventStore.open(tmp_path / f"control{i}.db") as cs:
                await create_job(cs, f"c{i}", seed=i)
                await run_job(
                    cs, f"c{i}", FakeAgent(steps=STEPS, token_interval=0), env=control_env
                )
            assert await env.hash() == await control_env.hash()


async def test_sigterm_with_a_stuck_job_cancels_after_the_timeout(
    sock_dir: Path, tmp_path: Path, template: Path
) -> None:
    """A step that never reaches a boundary in time is cancelled, still writing nothing."""
    sock, db = sock_dir / "s", tmp_path / "sb.db"
    proc = await start_daemon(sock, db, tmp_path / "envs", tmp_path / "d.log", shutdown_timeout=0.5)
    async with await Client.connect(str(sock)) as c:
        # token_interval 10 s: the very first sample is far longer than the shutdown timeout
        job_id = (await c.request("submit", steps=3, token_interval=10.0))["job_id"]
        await wait_state(c, job_id, {"running"})
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    async with asyncio.timeout(10):
        assert await proc.wait() == 0
    assert time.monotonic() - started < 5
    async with await EventStore.open(db) as store:
        job = await store.get_job(job_id)
        assert job is not None
        assert job.state is JobState.RUNNING
        events = await store.read_job(job_id)
        assert [e.type for e in events if e.type in (T.STEP_STARTED, T.STEP_COMPLETED)] == [
            T.STEP_STARTED
        ]
        assert not [e for e in events if e.type is T.LLM_RESPONDED]


async def test_restarted_daemon_resumes_running_jobs(
    sock_dir: Path, tmp_path: Path, template: Path
) -> None:
    def make() -> Daemon:
        return Daemon(
            socket_path=sock_dir / "s",
            db_path=tmp_path / "sb.db",
            envs_dir=tmp_path / "envs",
            shutdown_timeout=5.0,
        )

    first = make()
    await first.start()
    async with await Client.connect(str(first.socket_path)) as c:
        job_id = (
            await c.request("submit", steps=6, token_interval=0.02, env_template=str(template))
        )["job_id"]
        await wait_steps(c, [job_id], 1)
    await first.shutdown()

    second = make()
    await second.start()
    try:
        async with await Client.connect(str(second.socket_path)) as c:
            await wait_state(c, job_id, {"verifying"}, within=1.0)
    finally:
        await second.shutdown()
