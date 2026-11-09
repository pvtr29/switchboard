"""ContainerEnv: a DirEnv whose workdir is bind-mounted into a locked-down Docker container.

Files, snapshots, hashes and change detection are the DirEnv's (cheap, crash-safe); the
container only adds process isolation for `run(argv)`. The host workdir is mounted read-write
at /work. The container has no network, a read-only root filesystem (plus a tmpfs /tmp), no
capabilities, no-new-privileges, a non-root user and cpu/memory/pids limits.

Killing: every `run` tags its process tree with an `SB_RUN=<id>` environment variable. On
timeout or cancellation a killer exec scans /proc for processes carrying that tag and SIGKILLs
them until none is left, so nothing started by `run` survives, whatever process group or
session it moved to (a process that deliberately scrubs its environment is not covered by
this; `restore` and `destroy` recreate the container, which is the hard guarantee).

The bind mount follows the workdir *directory*, and DirEnv.restore swaps that directory for a
new one, so `restore` removes the container first and starts a fresh one afterwards: the
processes of an interrupted step cannot survive into the re-run.

Docker is called through the `docker` CLI with `asyncio.create_subprocess_exec` only. The
caller chooses the environment (`docker_env` is merged over os.environ), nothing here knows
about Colima or DOCKER_HOST.
"""

import asyncio
import contextlib
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from switchboard.env.base import SnapshotMissing
from switchboard.env.dir_env import DirEnv

LABEL = "switchboard.job"
MOUNT_POINT = "/work"
DEFAULT_IMAGE = "python:3.12-slim"
DOCKER_TIMEOUT_S = 60.0
MAX_OUTPUT_BYTES = 64 * 1024
MOUNT_RETRY_PAUSES = (0.3, 0.7, 1.5)
_NAME_BAD = re.compile(r"[^A-Za-z0-9_.-]")
_KILL_SCRIPT = (
    'tag="SB_RUN=$1"; i=0; while [ $i -lt 20 ]; do found=0; '
    "for p in /proc/[0-9]*; do "
    'if [ "${p#/proc/}" != "$$" ] && tr "\\0" "\\n" < $p/environ 2>/dev/null '
    '| grep -qxF "$tag"; then kill -KILL "${p#/proc/}" 2>/dev/null && found=1; fi; done; '
    "[ $found = 0 ] && exit 0; i=$((i+1)); done; exit 1"
)


class DockerError(RuntimeError):
    """A docker CLI call failed (or timed out)."""


class MountError(DockerError):
    """The workdir is not visible inside the container (e.g. not shared with the Docker VM)."""


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    output: str  # stdout+stderr merged; head and tail kept when longer than the limit
    timed_out: bool


def _merged_env(docker_env: Mapping[str, str] | None) -> dict[str, str] | None:
    return None if docker_env is None else {**os.environ, **docker_env}


async def _docker(
    docker: str,
    docker_env: Mapping[str, str] | None,
    *args: str,
    limit_s: float = DOCKER_TIMEOUT_S,
    check: bool = True,
) -> tuple[int, str]:
    """Run `docker *args`, return (exit code, merged output). Raises DockerError on failure
    when `check`, always on timeout."""
    proc = await asyncio.create_subprocess_exec(
        docker,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=_merged_env(docker_env),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), limit_s)
    except BaseException as e:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await asyncio.shield(proc.wait())
        if isinstance(e, TimeoutError):
            raise DockerError(f"docker {args[0]} timed out after {limit_s}s") from e
        raise
    text = out.decode(errors="replace").strip()
    code = proc.returncode if proc.returncode is not None else -1
    if check and code != 0:
        raise DockerError(f"docker {' '.join(args[:2])} failed ({code}): {text}")
    return code, text


async def _job_containers(
    docker: str, docker_env: Mapping[str, str] | None, job_id: str
) -> list[str]:
    flt = f"label={LABEL}={job_id}"
    _, out = await _docker(docker, docker_env, "ps", "-aq", "--no-trunc", "--filter", flt)
    return out.split()


async def _remove(docker: str, docker_env: Mapping[str, str] | None, ids: Sequence[str]) -> None:
    if ids:
        await _docker(docker, docker_env, "rm", "-f", "-v", *ids, check=False)


async def docker_available(
    docker: str = "docker", docker_env: Mapping[str, str] | None = None
) -> bool:
    """True when `docker info` succeeds (a daemon is reachable)."""
    try:
        code, _ = await _docker(docker, docker_env, "info", limit_s=20, check=False)
    except (DockerError, OSError):
        return False
    return code == 0


async def sweep_orphans(
    keep: set[str],
    *,
    docker: str = "docker",
    docker_env: Mapping[str, str] | None = None,
) -> list[str]:
    """Remove every switchboard container whose job id is not in `keep`; returns the removed
    container ids. For daemon startup: `keep` is the set of jobs that may still resume."""
    _, out = await _docker(
        docker,
        docker_env,
        "ps",
        "-a",
        "--no-trunc",
        "--filter",
        f"label={LABEL}",
        "--format",
        '{{.ID}} {{.Label "' + LABEL + '"}}',
    )
    doomed = [
        cid for cid, _, job in (line.partition(" ") for line in out.splitlines()) if job not in keep
    ]
    await _remove(docker, docker_env, doomed)
    return doomed


class _Collector:
    """Reads a stream keeping the first and last `limit // 2` bytes."""

    def __init__(self, limit: int) -> None:
        self.half = max(limit // 2, 1)
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = self.half - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > self.half:
                extra = len(self.tail) - self.half
                self.dropped += extra
                del self.tail[:extra]

    def text(self) -> str:
        head = self.head.decode(errors="replace")
        if not self.dropped:
            return head + self.tail.decode(errors="replace")
        tail = self.tail.decode(errors="replace")
        return f"{head}\n[... {self.dropped} bytes truncated ...]\n{tail}"


async def _collect(stream: asyncio.StreamReader, collector: _Collector) -> None:
    while chunk := await stream.read(65536):
        collector.feed(chunk)


class ContainerEnv:
    def __init__(
        self,
        template: Path,
        root: Path,
        job_id: str,
        *,
        image: str = DEFAULT_IMAGE,
        docker: str = "docker",
        docker_env: Mapping[str, str] | None = None,
        cpus: float = 1.0,
        memory: str = "512m",
        pids: int = 256,
        network: bool = False,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self.job_id = job_id
        self.image = image
        self.docker = docker
        self.docker_env = dict(docker_env) if docker_env is not None else None
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.network = network
        self.max_output = max_output
        self._dir = DirEnv(template, root, job_id)
        self._container: str | None = None

    # -- files: all DirEnv -----------------------------------------------------------------
    @property
    def workdir(self) -> Path:
        return self._dir.workdir

    @property
    def container(self) -> str | None:
        """Name of the running container, if any."""
        return self._container

    async def snapshot(self, name: str) -> str:
        return await self._dir.snapshot(name)

    async def snapshots(self) -> list[str]:
        return await self._dir.snapshots()

    async def hash(self) -> str:
        return await self._dir.hash()

    async def gc(self, keep: Sequence[str]) -> None:
        await self._dir.gc(keep)

    async def changed_since(self, env_ref: str) -> bool:
        return await self._dir.changed_since(env_ref)

    # -- container lifecycle ---------------------------------------------------------------
    async def _d(self, *args: str, limit_s: float = DOCKER_TIMEOUT_S, check: bool = True) -> str:
        return (await _docker(self.docker, self.docker_env, *args, limit_s=limit_s, check=check))[1]

    async def _remove_job_containers(self) -> None:
        ids = await _job_containers(self.docker, self.docker_env, self.job_id)
        await _remove(self.docker, self.docker_env, ids)
        self._container = None

    def _user(self) -> str:
        uid, gid = os.getuid(), os.getgid()
        return f"{uid}:{gid}" if uid != 0 else "65534:65534"

    async def _start_container(self) -> None:
        await self._remove_job_containers()
        name = f"sb-{_NAME_BAD.sub('_', self.job_id)}-{uuid.uuid4().hex[:8]}"
        args = [
            "run",
            "-d",
            "--init",
            "--name",
            name,
            "--label",
            f"{LABEL}={self.job_id}",
            "--network",
            "bridge" if self.network else "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--pids-limit",
            str(self.pids),
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--user",
            self._user(),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-e",
            "HOME=/tmp",
            "-v",
            f"{self.workdir.resolve()}:{MOUNT_POINT}",
            "-w",
            MOUNT_POINT,
            self.image,
            "sleep",
            "infinity",
        ]
        await self._d(*args, limit_s=120)
        self._container = name

    async def _mount_visible(self) -> bool:
        marker = self.workdir / f".sb-mount-{uuid.uuid4().hex[:12]}"
        await asyncio.to_thread(marker.write_text, "ok")
        try:
            code, _ = await _docker(
                self.docker,
                self.docker_env,
                "exec",
                str(self._container),
                "test",
                "-f",
                f"{MOUNT_POINT}/{marker.name}",
                check=False,
            )
        finally:
            await asyncio.to_thread(marker.unlink, missing_ok=True)
        return code == 0

    async def _start_verified(self) -> None:
        """Start the container and prove the workdir is visible in it.

        Virtiofs (Colima/Docker Desktop) can show a directory that was just deleted and
        recreated at the same path as stale or empty for a moment, so a failed check is
        retried with a fresh container after a short pause before it is reported as an
        unshared path.
        """
        for pause in (*MOUNT_RETRY_PAUSES, None):
            await self._start_container()
            if await self._mount_visible():
                return
            await self._remove_job_containers()
            if pause is None:
                break
            await asyncio.sleep(pause)
        raise MountError(
            f"{self.workdir} is not visible inside the container: the host path is not "
            "shared with the Docker VM. Put the environment root under a mounted path "
            "(Colima mounts the home directory by default; /tmp usually is not mounted)."
        )

    async def provision(self) -> Path:
        await self._remove_job_containers()  # leftovers of a crashed run
        workdir = await self._dir.provision()
        await self._start_verified()
        return workdir

    async def restore(self, env_ref: str) -> None:
        await self._remove_job_containers()  # nothing may run or write during the swap
        try:
            await self._dir.restore(env_ref)
        except SnapshotMissing:
            raise
        else:
            await self._start_verified()

    async def destroy(self) -> None:
        await self._remove_job_containers()
        await self._dir.destroy()

    # -- running commands ------------------------------------------------------------------
    async def _kill_run(self, run_id: str) -> None:
        if self._container is None:
            return
        with contextlib.suppress(DockerError):
            await _docker(
                self.docker,
                self.docker_env,
                "exec",
                self._container,
                "sh",
                "-c",
                _KILL_SCRIPT,
                "sb-kill",
                run_id,
                check=False,
            )

    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        """Execute `argv` in the container (cwd /work, no shell unless `argv` is one).

        On timeout or cancellation every process of the run is killed before returning /
        propagating. Output is stdout+stderr merged, truncated to head+tail.
        """
        if self._container is None:
            raise RuntimeError("ContainerEnv has no running container (provision or restore first)")
        run_id = uuid.uuid4().hex
        args = ["exec", "-e", f"SB_RUN={run_id}", "-w", MOUNT_POINT]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        proc = await asyncio.create_subprocess_exec(
            self.docker,
            *args,
            self._container,
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_merged_env(self.docker_env),
        )
        assert proc.stdout is not None
        collector = _Collector(self.max_output)
        reader = asyncio.ensure_future(_collect(proc.stdout, collector))
        timed_out = False
        try:
            try:
                async with asyncio.timeout(timeout_s):
                    await proc.wait()
                    await reader
            except TimeoutError:
                timed_out = True
        except BaseException:
            await asyncio.shield(self._cleanup(proc, reader, run_id))
            raise
        if timed_out:
            await asyncio.shield(self._cleanup(proc, reader, run_id))
        code = proc.returncode if proc.returncode is not None else -1
        return RunResult(exit_code=code, output=collector.text(), timed_out=timed_out)

    async def _cleanup(
        self, proc: asyncio.subprocess.Process, reader: "asyncio.Future[None]", run_id: str
    ) -> None:
        await self._kill_run(run_id)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader
