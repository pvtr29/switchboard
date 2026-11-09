"""ContainerEnv tests. They need a Docker daemon and the image, and skip cleanly without them.

Env vars for unusual setups (e.g. Colima with a broken credential helper):
SB_TEST_DOCKER_HOST, SB_TEST_DOCKER_CONFIG. The environment root is created under
~/.cache/switchboard-tests because the project/home directory is what Docker VMs share with
the host; the system temp dir often is not.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import JobState
from switchboard.env import (
    ContainerEnv,
    DockerError,
    MountError,
    docker_available,
    sweep_orphans,
)
from switchboard.env.container_env import _docker
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.runner import run_job
from switchboard.store import EventStore

pytestmark = pytest.mark.asyncio
IMAGE = "python:3.12-slim"
SHARED_BASE = Path.home() / ".cache" / "switchboard-tests"


def _docker_env() -> dict[str, str] | None:
    env: dict[str, str] = {}
    host = os.environ.get("SB_TEST_DOCKER_HOST")
    if host is None:
        colima = Path.home() / ".colima" / "default" / "docker.sock"
        if colima.exists() and "DOCKER_HOST" not in os.environ:
            host = f"unix://{colima}"
    if host:
        env["DOCKER_HOST"] = host
    if cfg := os.environ.get("SB_TEST_DOCKER_CONFIG"):
        env["DOCKER_CONFIG"] = cfg
    return env or None


def _probe(denv: Mapping[str, str] | None) -> str | None:
    """None when Docker and the image are usable, else the reason to skip."""
    full = None if denv is None else {**os.environ, **denv}
    try:
        info = subprocess.run(
            ["docker", "info"], capture_output=True, env=full, timeout=30, check=False
        )
        if info.returncode != 0:
            return "docker daemon not available"
        img = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            env=full,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"docker not usable: {e}"
    return None if img.returncode == 0 else f"image {IMAGE} not pulled"


@pytest.fixture(scope="session")
def docker_env() -> dict[str, str] | None:
    denv = _docker_env()
    reason = _probe(denv)
    if reason:
        pytest.skip(reason)
    return denv


@pytest.fixture
def mroot() -> Iterator[Path]:
    """An env root under a path shared with the Docker VM."""
    SHARED_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ce-", dir=SHARED_BASE))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def tmpl(mroot: Path) -> Path:
    t = mroot / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


def make(tmpl: Path, mroot: Path, docker_env: Mapping[str, str] | None, **kw: Any) -> ContainerEnv:
    return ContainerEnv(tmpl, mroot / "envs", "j1", docker_env=docker_env, **kw)


@pytest_asyncio.fixture
async def env(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> AsyncIterator[ContainerEnv]:
    e = make(tmpl, mroot, docker_env)
    await e.provision()
    try:
        yield e
    finally:
        await e.destroy()


async def containers(denv: Mapping[str, str] | None, job: str | None = None) -> list[str]:
    flt = "label=switchboard.job" + (f"={job}" if job else "")
    _, out = await _docker("docker", denv, "ps", "-aq", "--filter", flt)
    return out.split()


async def exec_names(e: ContainerEnv) -> list[str]:
    """command line of every process in the container (the scanner itself excluded)."""
    code = (
        "import os\n"
        "for p in os.listdir('/proc'):\n"
        "    if p.isdigit() and p != str(os.getpid()):\n"
        "        try: a = open(f'/proc/{p}/cmdline','rb').read().decode()\n"
        "        except OSError: continue\n"
        "        a = ' '.join(a.split('\\0')).strip()\n"
        "        if a: print(a)\n"
    )
    r = await e.run(["python", "-c", code], timeout_s=20)
    assert r.exit_code == 0, r.output
    return r.output.splitlines()


async def sleepers(env: ContainerEnv) -> list[str]:
    """The test sleeps (`sleep 30x`), not the container's own `sleep infinity`."""
    return [n for n in await exec_names(env) if n.startswith("sleep 3")]


async def wait_sleeping(env: ContainerEnv) -> None:
    async with asyncio.timeout(20):
        while not await sleepers(env):  # noqa: ASYNC110
            await asyncio.sleep(0.1)


async def test_run_captures_output_and_exit_code(env: ContainerEnv) -> None:
    r = await env.run(["python", "-c", "print(1)"], timeout_s=20)
    assert (r.exit_code, r.output.strip(), r.timed_out) == (0, "1", False)
    r = await env.run(
        ["python", "-c", "import sys; print('e', file=sys.stderr); sys.exit(3)"], timeout_s=20
    )
    assert (r.exit_code, r.output.strip(), r.timed_out) == (3, "e", False)
    r = await env.run(
        ["python", "-c", "import os; print(os.environ['X'], os.getcwd())"],
        timeout_s=20,
        env={"X": "v"},
    )
    assert r.output.split() == ["v", "/work"]


async def test_output_is_truncated_head_and_tail(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> None:
    e = make(tmpl, mroot, docker_env, max_output=1000)
    await e.provision()
    try:
        code = "print('A'*3000); print('B'*3000)"
        r = await e.run(["python", "-c", code], timeout_s=20)
        assert r.output.startswith("A" * 500)
        assert r.output.endswith("B" * 499 + "\n")
        assert "truncated" in r.output
        assert len(r.output) < 1200
    finally:
        await e.destroy()


async def test_isolation(env: ContainerEnv, docker_env: dict[str, str] | None) -> None:
    net = "import socket; socket.create_connection(('1.1.1.1',53),2)"
    r = await env.run(["python", "-c", net], timeout_s=20)
    assert r.exit_code != 0

    r = await env.run(["python", "-c", "open('/etc/pwn','w')"], timeout_s=20)
    assert r.exit_code != 0
    assert "Read-only" in r.output
    r = await env.run(["python", "-c", "open('/tmp/ok','w').write('x')"], timeout_s=20)
    assert r.exit_code == 0
    r = await env.run(["python", "-c", "open('/work/ok','w').write('x')"], timeout_s=20)
    assert r.exit_code == 0

    r = await env.run(["id", "-u"], timeout_s=20)
    assert r.output.strip() != "0"

    name = env.container
    assert name is not None
    fmt = "{{.HostConfig.PidsLimit}} {{.HostConfig.NetworkMode}} {{.HostConfig.ReadonlyRootfs}}"
    _, out = await _docker("docker", docker_env, "inspect", "-f", fmt, name)
    assert out == "256 none true"
    _, out = await _docker("docker", docker_env, "inspect", "-f", "{{.HostConfig.CapDrop}}", name)
    assert out == "[ALL]"


async def test_files_cross_the_mount_both_ways(env: ContainerEnv) -> None:
    r = await env.run(["python", "-c", "open('made_in_container','w').write('hi')"], timeout_s=20)
    assert r.exit_code == 0, r.output
    assert (env.workdir / "made_in_container").read_text() == "hi"
    (env.workdir / "made_on_host").write_text("yo")
    r = await env.run(["cat", "made_on_host"], timeout_s=20)
    assert r.output == "yo"
    assert not list(env.workdir.glob(".sb-mount-*"))


async def test_snapshot_restore_roundtrip_through_container(env: ContainerEnv) -> None:
    ref = await env.snapshot("a")
    h0 = await env.hash()
    await env.run(["python", "-c", "open('new','w').write('x'); open('README','w')"], timeout_s=20)
    assert await env.changed_since(ref)
    assert await env.hash() != h0
    await env.restore(ref)
    assert await env.hash() == h0
    r = await env.run(
        ["python", "-c", "print(sorted(__import__('os').listdir('.')))"], timeout_s=20
    )
    assert r.output.strip() == "['README', 'files']"  # container sees the restored dir
    await env.run(["python", "-c", "open('after','w')"], timeout_s=20)
    assert (env.workdir / "after").exists()
    assert await env.snapshots() == ["a"]
    await env.gc([])
    assert await env.snapshots() == []


async def test_restore_kills_earlier_processes(env: ContainerEnv) -> None:
    ref = await env.snapshot("a")
    # a daemonised background process that outlives its exec
    r = await env.run(["sh", "-c", "setsid sleep 301 >/dev/null 2>&1 & echo started"], timeout_s=20)
    assert r.output.strip() == "started"
    assert await sleepers(env) == ["sleep 301"]
    old = env.container
    await env.restore(ref)
    assert env.container != old
    assert await sleepers(env) == []


async def test_restore_missing_snapshot_raises(env: ContainerEnv) -> None:
    from switchboard.env.base import SnapshotMissing

    with pytest.raises(SnapshotMissing):
        await env.restore("nope")


async def test_timeout_kills_process_tree(env: ContainerEnv) -> None:
    script = "sleep 302 & setsid sleep 303 & nohup sleep 304 >/dev/null 2>&1 & wait"
    t0 = time.monotonic()
    r = await env.run(["sh", "-c", script], timeout_s=1.5)
    assert r.timed_out
    assert time.monotonic() - t0 < 15
    assert await sleepers(env) == []


async def test_cancel_kills_process(env: ContainerEnv) -> None:
    task = asyncio.create_task(env.run(["sh", "-c", "sleep 305 & wait"], timeout_s=120))
    await wait_sleeping(env)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await sleepers(env) == []


async def test_kill_spares_other_runs(env: ContainerEnv) -> None:
    other = asyncio.create_task(env.run(["sleep", "306"], timeout_s=60))
    await wait_sleeping(env)
    r = await env.run(["sleep", "307"], timeout_s=1)
    assert r.timed_out
    assert await sleepers(env) == ["sleep 306"]
    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await other


async def test_run_without_container_raises(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> None:
    e = make(tmpl, mroot, docker_env)
    with pytest.raises(RuntimeError, match="no running container"):
        await e.run(["true"], timeout_s=5)


async def test_provision_replaces_leftover_container_and_destroy_cleans(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> None:
    a = make(tmpl, mroot, docker_env)
    await a.provision()
    first = await containers(docker_env, "j1")
    assert len(first) == 1
    # a "restarted process": new object, same job, provisions again over the leftover
    b = make(tmpl, mroot, docker_env)
    await b.provision()
    second = await containers(docker_env, "j1")
    assert len(second) == 1
    assert second != first
    await b.destroy()
    assert await containers(docker_env, "j1") == []
    assert not (mroot / "envs" / "j1").exists()


async def test_sweep_orphans_removes_only_unknown_jobs(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> None:
    envs = [ContainerEnv(tmpl, mroot / "envs", j, docker_env=docker_env) for j in ("k1", "k2")]
    try:
        for e in envs:
            await e.provision()
        removed = await sweep_orphans({"k1"}, docker_env=docker_env)
        assert len(removed) == 1
        assert await containers(docker_env, "k1") != []
        assert await containers(docker_env, "k2") == []
    finally:
        for e in envs:
            await e.destroy()


async def test_unshared_root_raises_clear_mount_error(
    tmpl: Path, docker_env: dict[str, str] | None
) -> None:
    root = Path(tempfile.mkdtemp(prefix="sb-unshared-", dir="/private/tmp")).resolve()  # noqa: ASYNC240
    try:
        e = ContainerEnv(tmpl, root, "j1", docker_env=docker_env)
        try:
            await e.provision()
        except MountError as err:
            message = str(err)
        else:
            await e.destroy()
            pytest.skip("/private/tmp is shared with the Docker daemon here")
        assert str(e.workdir) in message
        assert await containers(docker_env, "j1") == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def test_docker_failures_are_typed(docker_env: dict[str, str] | None) -> None:
    assert await docker_available(docker_env=docker_env)
    assert not await docker_available("definitely-not-docker")
    with pytest.raises(DockerError):
        await _docker("docker", docker_env, "inspect", "no-such-container-xyz")


class Crash(BaseException):
    """Stands in for kill -9."""


async def test_runner_unchanged_on_container_env_with_crash(
    tmpl: Path, mroot: Path, docker_env: dict[str, str] | None
) -> None:
    steps, seed, crash_step = 5, 11, 2

    # control: DirEnv, uninterrupted
    ctl_env = DirEnv(tmpl, mroot / "control", "ctl")
    async with await EventStore.open(mroot / "control.db") as s:
        await create_job(s, job_id="ctl", seed=seed)
        await run_job(s, "ctl", FakeAgent(steps=steps), env=ctl_env)
    ctl = await ctl_env.hash()

    def hook(ctx: Any, tool: str, path: Path) -> None:
        if ctx.step == crash_step and ctx.attempt == 0 and tool == "write_file":
            raise Crash

    env = make(tmpl, mroot, docker_env)
    try:
        async with await EventStore.open(mroot / "sb.db") as store:
            await create_job(store, seed=seed)
            with pytest.raises(Crash):
                await run_job(store, "j1", FakeAgent(steps=steps, on_file_write=hook), env=env)
            env2 = make(tmpl, mroot, docker_env)  # restarted process
            job = await run_job(store, "j1", FakeAgent(steps=steps), env=env2)
            assert job.state is JobState.VERIFYING
            assert await env2.hash() == ctl
            assert len(await containers(docker_env, "j1")) == 1  # restore replaced, not leaked
            await env2.destroy()
    finally:
        await env.destroy()
    assert await containers(docker_env, "j1") == []


async def test_report_timings(tmpl: Path, mroot: Path, docker_env: dict[str, str] | None) -> None:
    e = make(tmpl, mroot, docker_env)
    t = time.perf_counter()
    await e.provision()
    provision = time.perf_counter() - t
    try:
        await e.snapshot("a")
        t = time.perf_counter()
        for _ in range(5):
            await e.run(["true"], timeout_s=20)
        run = (time.perf_counter() - t) / 5
        t = time.perf_counter()
        await e.restore("a")
        restore = time.perf_counter() - t
        print(f"\nTIMINGS provision={provision:.2f}s run={run:.3f}s restore={restore:.2f}s")
        assert max(provision, run, restore) < 30
    finally:
        await e.destroy()
