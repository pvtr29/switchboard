"""Daemon-level recovery: the report in `stats`, and the restart counter bounding a crash loop
(a FakeAgent that os._exit()s the daemon process at a given step, restarted repeatedly)."""

import asyncio
import sys
from pathlib import Path

import pytest
from conftest import wait_for_socket, wait_state, wait_steps

from switchboard.client.conn import Client
from switchboard.core import EventType, JobState
from switchboard.daemon.server import Daemon
from switchboard.store import EventStore

pytestmark = pytest.mark.asyncio
RESTART_LIMIT = 5  # runner default max_attempts_per_step


async def spawn(
    sock: Path, db: Path, envs: Path, sink: Path, log: Path, *, wait: bool = True
) -> asyncio.subprocess.Process:
    with log.open("ab") as out:
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
            "--sink-db",
            str(sink),
            stdout=out,
            stderr=out,
        )
    if wait:
        await asyncio.to_thread(wait_for_socket, sock)
    return proc


async def test_restart_report_is_exposed_in_stats(
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
    assert first.recovery is not None
    assert first.recovery.admitted == 0
    async with await Client.connect(str(first.socket_path)) as c:
        job_id = (
            await c.request("submit", steps=30, token_interval=0.02, env_template=str(template))
        )["job_id"]
        await wait_steps(c, [job_id], 1)
        paused = (await c.request("submit", steps=30, token_interval=0.02))["job_id"]
        await c.request("pause", job_id=paused)
        await wait_state(c, paused, {"paused"})
    await first.shutdown()

    second = make()
    await second.start()
    try:
        async with await Client.connect(str(second.socket_path)) as c:
            rec = (await c.request("stats"))["recovery"]
            assert rec["requeued"] == 1
            assert rec["admitted"] == 1
            assert rec["left"] == {"paused": 1}
            assert rec["mismatches"] == []
            assert rec["verified"] == 2  # small DB: every job is checked by default
            await c.request("cancel", job_id=job_id)
    finally:
        await second.shutdown()


async def test_crash_loop_ends_in_restart_limit(
    sock_dir: Path, tmp_path: Path, template: Path
) -> None:
    sock, db, envs = sock_dir / "s", tmp_path / "sb.db", tmp_path / "envs"
    sink, log = tmp_path / "sink.db", tmp_path / "daemon.log"
    proc = await spawn(sock, db, envs, sink, log)
    async with await Client.connect(str(sock)) as c:
        job_id = (
            await c.request(
                "submit",
                steps=4,
                token_interval=0.0,
                env_template=str(template),
                effect_every=1,
                exit_at_step=2,  # every attempt of step 2 kills the whole process
            )
        )["job_id"]
    crashes = 0
    try:
        while True:
            try:
                async with asyncio.timeout(3.0):
                    code = await proc.wait()
            except TimeoutError:
                break  # this daemon survived: it must have failed the job
            assert code == 86
            crashes += 1
            assert crashes <= RESTART_LIMIT + 1
            proc = await spawn(sock, db, envs, sink, log, wait=False)
        await asyncio.to_thread(wait_for_socket, sock)
        async with await Client.connect(str(sock)) as c:
            job = await wait_state(c, job_id, {"failed"}, within=5.0)
        assert crashes == RESTART_LIMIT
        assert job["last_step"] == 1  # steps 0 and 1 committed, step 2 never
    finally:
        if proc.returncode is None:
            proc.terminate()
            await proc.wait()
    async with await EventStore.open(db) as store:
        events = await store.read_job(job_id)
        attempts = [e.payload["attempt"] for e in events if e.type is EventType.STEP_STARTED]
        assert attempts[-RESTART_LIMIT:] == list(range(RESTART_LIMIT))  # step 2: attempts 0..4
        failed = [e for e in events if e.type is EventType.STATE_CHANGED][-1]
        assert failed.payload["to"] == JobState.FAILED.value
        assert failed.payload["reason"] == "restart limit"
