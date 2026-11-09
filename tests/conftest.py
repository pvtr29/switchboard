import asyncio
import gc
import logging
import shutil
import socket
import tempfile
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
import pytest_asyncio

from switchboard.client.conn import Client
from switchboard.core import Event, EventType
from switchboard.daemon.server import Daemon
from switchboard.store import EventStore


@pytest.fixture(scope="session", autouse=True)
def _main_thread_event_loop() -> Iterator[None]:
    # looptime's pytest_pyfunc_call hook calls policy.get_event_loop() for every test. In a sync
    # test with no loop set, Python 3.12 emits a DeprecationWarning (fatal under
    # filterwarnings=error) or implicitly creates a loop that is never closed. Owning one
    # explicit loop for the session avoids both, whatever order the tests run in.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    asyncio.set_event_loop(None)
    loop.close()


class AsyncioErrorCapture(logging.Handler):
    """Collects ERROR records of the `asyncio` logger.

    "Task was destroyed but it is pending!" and "Task exception was never retrieved" are not
    warnings: asyncio reports them through `loop.call_exception_handler`, whose default handler
    logs to the `asyncio` logger. `filterwarnings = error` cannot see them, this can.
    """

    def __init__(self, *, consume: bool = False) -> None:
        """With `consume`, matching records are also hidden from other handlers (a logger
        filter runs before any handler), so a test can provoke errors on purpose without
        tripping the autouse guard."""
        super().__init__(level=logging.ERROR)
        self.consume = consume
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Nothing to do: `filter` already recorded the record, in handler mode and logger mode."""

    def messages(self) -> list[str]:
        out = []
        for r in self.records:
            text = r.getMessage()
            if r.exc_info and r.exc_info[1] is not None:
                text += f" [{type(r.exc_info[1]).__name__}: {r.exc_info[1]}]"
            out.append(text)
        return out

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            self.records.append(record)
            return not self.consume
        return True

    def __enter__(self) -> Self:
        logger = logging.getLogger("asyncio")
        if self.consume:
            logger.addFilter(self)
        else:
            logger.addHandler(self)
        return self

    def __exit__(self, *exc: object) -> None:
        logger = logging.getLogger("asyncio")
        if self.consume:
            logger.removeFilter(self)
        else:
            logger.removeHandler(self)


@pytest.fixture(autouse=True)
def _asyncio_error_guard() -> Iterator[None]:
    """Fail any test during which asyncio logged an error, including garbage found at teardown."""
    with AsyncioErrorCapture() as capture:
        yield
        gc.collect()  # destroyed-while-pending tasks and unretrieved exceptions report on __del__
        gc.collect()
    if capture.records:
        pytest.fail("asyncio reported errors:\n" + "\n".join(capture.messages()), pytrace=False)


async def create_job(
    store: EventStore,
    job_id: str = "j1",
    *,
    seed: int = 7,
    instructions: Sequence[Mapping[str, Any]] = (),
    config_hash: str = "",
    budget: Mapping[str, Any] | None = None,
) -> None:
    """Append job_created for `job_id` (state queued)."""
    await store.append(
        Event(
            job_id=job_id,
            epoch=0,
            ts=0.0,
            type=EventType.JOB_CREATED,
            payload={
                "kind": "fake",
                "task_ref": "t",
                "budget": dict(budget or {}),
                "seed": seed,
                "instructions": [dict(m) for m in instructions],
                "config_hash": config_hash,
            },
        )
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[EventStore]:
    async with await EventStore.open(tmp_path / "sb.db") as s:
        yield s


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    """A short directory for sockets: AF_UNIX paths are limited to ~104 bytes on macOS."""
    d = Path(tempfile.mkdtemp(prefix="sb", dir="/tmp"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def template(tmp_path: Path) -> Path:
    t = tmp_path / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


@pytest_asyncio.fixture
async def daemon(sock_dir: Path, tmp_path: Path) -> AsyncIterator[Daemon]:
    d = Daemon(
        socket_path=sock_dir / "s",
        db_path=tmp_path / "sb.db",
        envs_dir=tmp_path / "envs",
        shutdown_timeout=5.0,
    )
    await d.start()
    try:
        yield d
    finally:
        await d.shutdown()


@pytest_asyncio.fixture
async def client(daemon: Daemon) -> AsyncIterator[Client]:
    async with await Client.connect(str(daemon.socket_path)) as c:
        yield c


async def wait_state(
    client: Client, job_id: str, states: set[str], *, within: float = 10.0
) -> dict[str, Any]:
    """Poll `get` until the job is in one of `states`; returns the job dict."""
    async with asyncio.timeout(within):
        while True:
            job: dict[str, Any] = (await client.request("get", job_id=job_id))["job"]
            if job["state"] in states:
                return job
            await asyncio.sleep(0.02)


async def wait_steps(
    client: Client, job_ids: Sequence[str], steps: int, *, within: float = 15.0
) -> None:
    """Poll until every job has committed at least `steps` steps."""
    async with asyncio.timeout(within):
        while True:
            jobs = [(await client.request("get", job_id=j))["job"] for j in job_ids]
            if all(j["last_step"] is not None and j["last_step"] + 1 >= steps for j in jobs):
                return
            await asyncio.sleep(0.02)


def wait_for_socket(path: Path, timeout: float = 10.0) -> None:
    """Sync helper for subprocess tests: wait until something accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(str(path))
            return
        except OSError:
            time.sleep(0.05)
        finally:
            s.close()
    raise TimeoutError(f"nothing listening on {path}")
