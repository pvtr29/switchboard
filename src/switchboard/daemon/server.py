"""The daemon: one process, one event loop, one Unix socket speaking NDJSON (see protocol.py).

`Daemon` owns the EventStore, the Supervisor (with its BudgetManager), the EventBus and the
listening server. Ownership of every task:

- per-connection handler task: created by asyncio for each accepted connection, registered in
  `Daemon._handlers`, awaited (after cancel if needed) by `shutdown`;
- per-connection writer task and per-subscription pump tasks: owned by their `Connection`,
  cancelled and awaited in `Connection.close`.

Outbound queue policy (per connection, `OUT_QUEUE` chunks of already encoded lines): the
queue is BOUNDED and shared by responses and stream messages, and one writer coroutine drains it
with `await writer.drain()` after every chunk. When it is full, whoever wants to enqueue waits:
- a response waits, so a client that never reads its socket stalls only its own connection
  (its handler stops reading requests) and nobody else;
- a subscription pump waits; meanwhile that subscriber's bus queue overflows and drops the oldest
  events into a gap marker (bus.py). Streaming therefore never blocks the daemon.
When the peer is gone the writer task ends, `dead` is set and waiting producers are woken and
raise ConnectionClosed.

Socket: bound with umask 0o177 (mode 0600 from the first instant, no chmod race) plus an
explicit chmod. A leftover socket file of a crashed daemon is detected by trying to connect:
refused means stale and it is unlinked, accepted means another daemon lives and startup fails
with DaemonAlreadyRunning before the database is touched.

Graceful shutdown (SIGTERM/SIGINT or the `shutdown` command), see `Daemon.shutdown`: stop
accepting, `supervisor.drain()` (jobs stop at their next step boundary and write NOTHING, so
they stay `running` for recovery), wait up to `shutdown_timeout`, cancel what is left (still
writing nothing), close connections (flushing what is queued), close the store.

How a submitted job gets its agent and environment: the `job_created` payload carries the
extra keys `steps`, `token_interval` and `env_template` (the projection ignores unknown keys).
`JobSpec` decodes them; `agent_factory` builds a FakeAgent, `env_factory` a DirEnv under
`envs_dir` when a template was given. The supervisor's factories are synchronous, so specs are
cached in memory, filled at submit and (for jobs created by an earlier daemon process) loaded
from the log by `_ensure_spec` before a job is started again.

Recovery (runtime/recovery.py): `start` runs `recover` after the store and supervisor exist and
BEFORE the server starts listening, so no client can observe the half-recovered state. The
report is kept and exposed in `stats` ("recovery"). `verify_projection` (None = automatic, see
recovery.py) is the consistency check of the live projection against a fold of the log.

Real agents (phase 2): `agent="gemini"` with a `task` from the suite (switchboard.tasks)
makes a `rollout` job: a GeminiAgent over a shared LLM client (one per model list, closed at
shutdown), its `run_tests` tool in `runner_factory` (the macOS sandbox by default), a DirEnv
from the task's repo. Without an explicit token budget such a job gets
`default_rollout_tokens`.

Grading: a store listener sees every `* -> verifying` commit of a job that has a task and
starts a grading task (bounded by `max_running`): `verify.grader.grade` runs the visible and
hidden tests in a fresh copy, then `verifying -> done` + `verdict_recorded` are appended as one
group (a grader crash appends `verifying -> failed`). Grading tasks write nothing when
cancelled, so a job whose grading was interrupted (shutdown, kill -9) is still `verifying` and
is graded again at the next start (grading is a pure function of the final workdir).

External sink for the chaos harness: `sink_db` gives every job a persistent `SQLiteSink` (its
own SQLite file, so applications survive kill -9); `sink_naive` makes it non-idempotent. The
extra `job_created` keys `effect_every`, `effect_policy` and (testing only) `exit_at_step` make
the FakeAgent perform external effects / hard-exit the process.

Observability (commit 20): the daemon owns a `LoopLagMonitor` (started in `start`, stopped in
`_release`) whose p50/p99/max appear in `stats` ("loop_lag"), next to the store's write latency
("store_latency") and queue depth. `debug_asyncio=True` (CLI `--debug-asyncio`) or the standard
`PYTHONASYNCIODEBUG` turns on asyncio debug mode with `slow_callback_duration` = 50 ms, so a
callback that blocks the loop is logged as "Executing <...> took N seconds". `on_token` is a
TESTING-ONLY hook handed to every FakeAgent; a test makes it `time.sleep` to block the loop.
"""

import asyncio
import contextlib
import dataclasses
import errno
import logging
import os
import signal
import socket
import stat
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from switchboard.agents.fake import FakeAgent, TokenHook
from switchboard.agents.gemini import GeminiAgent
from switchboard.agents.gemini import instructions as task_instructions
from switchboard.agents.tools import CommandRunner, LocalSandboxRunner
from switchboard.core import (
    Budget,
    Event,
    EventType,
    Job,
    JobKind,
    JobState,
    ProjectionError,
    budget_from_dict,
    budget_to_dict,
)
from switchboard.daemon.bus import DEFAULT_QUEUE_SIZE, EventBus, Gap, Subscription
from switchboard.daemon.protocol import (
    BAD_ARGS,
    CONFLICT,
    INTERNAL,
    LINE_TOO_LONG,
    MAX_LINE_BYTES,
    NOT_FOUND,
    ProtocolError,
    Request,
    dumps_line,
    error_response,
    event_message,
    gap_message,
    job_to_dict,
    ok_response,
    parse_request,
)
from switchboard.env.base import Environment
from switchboard.env.dir_env import DirEnv
from switchboard.llm.client import LLMClient
from switchboard.obs import LoopLagMonitor
from switchboard.runtime.agent import Agent, ReplayPolicy, StepContext
from switchboard.runtime.recovery import RecoveryReport, recover
from switchboard.runtime.sinks import SQLiteSink
from switchboard.runtime.supervisor import (
    AlreadyRunning,
    NotCancellable,
    NotExtendable,
    NotPausable,
    NotResumable,
    Supervisor,
    SupervisorClosed,
    UnknownJob,
)
from switchboard.store import EventStore
from switchboard.tasks import UnknownTask, load_task
from switchboard.verify.grader import grade

logger = logging.getLogger("switchboard.daemon")

OUT_QUEUE = 32  # chunks per connection
BACKFILL_PAGE = 500
MAX_SUBSCRIPTIONS_PER_CONNECTION = 64
FLUSH_TIMEOUT_S = 2.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 10.0
MAX_STEPS = 1_000_000
AGENTS = ("fake", "gemini")
DEFAULT_ROLLOUT_TOKENS = 200_000
SLOW_CALLBACK_S = 0.05  # asyncio debug mode: log callbacks that hold the loop longer than this

type Result = dict[str, Any]


class DaemonError(Exception):
    """The daemon cannot start."""


class DaemonAlreadyRunning(DaemonError):
    """Another daemon answers on the socket."""


class ConnectionClosed(Exception):
    """The peer is gone; nothing more can be sent on this connection."""


@dataclass(frozen=True, slots=True)
class JobSpec:
    """What the agent and environment factories need beyond the Job projection."""

    steps: int = 5
    token_interval: float = 0.01
    env_template: str | None = None
    effect_every: int = 0
    effect_policy: str = ReplayPolicy.KEYED.value
    exit_at_step: int | None = None
    agent: str = "fake"
    task: str | None = None
    models: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JobSpec":
        exit_at = payload.get("exit_at_step")
        return cls(
            steps=int(payload.get("steps", 5)),
            token_interval=float(payload.get("token_interval", 0.01)),
            env_template=payload.get("env_template"),
            effect_every=int(payload.get("effect_every", 0)),
            effect_policy=str(payload.get("effect_policy", ReplayPolicy.KEYED.value)),
            exit_at_step=None if exit_at is None else int(exit_at),
            agent=str(payload.get("agent", "fake")),
            task=payload.get("task"),
            models=tuple(payload.get("models", ())),
        )


# ---- argument validation ---------------------------------------------------------------------


def _check_keys(args: Mapping[str, Any], allowed: set[str]) -> None:
    extra = set(args) - allowed
    if extra:
        raise ProtocolError(BAD_ARGS, f"unknown args {sorted(extra)}")


def _str(args: Mapping[str, Any], key: str, *, required: bool = True) -> str | None:
    v = args.get(key)
    if v is None:
        if required:
            raise ProtocolError(BAD_ARGS, f"{key} is required")
        return None
    if not isinstance(v, str) or not v:
        raise ProtocolError(BAD_ARGS, f"{key} must be a non-empty string")
    return v


def _job_id(args: Mapping[str, Any]) -> str:
    out = _str(args, "job_id")
    assert out is not None
    return out


def _int(args: Mapping[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    v = args.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ProtocolError(BAD_ARGS, f"{key} must be an integer in [{lo}, {hi}]")
    return v


def _num(args: Mapping[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = args.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi:
        raise ProtocolError(BAD_ARGS, f"{key} must be a number in [{lo}, {hi}]")
    return float(v)


# ---- connection ------------------------------------------------------------------------------


class Connection:
    def __init__(
        self, daemon: "Daemon", reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._daemon = daemon
        self._reader = reader
        self._writer = writer
        self._out: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=OUT_QUEUE)
        self._writer_task: asyncio.Task[None] | None = None
        self._subs: dict[str, tuple[Subscription, asyncio.Task[None]]] = {}
        self._closing = False
        self.dead = False

    # -- outbound ---------------------------------------------------------------------------
    async def send_bytes(self, data: bytes) -> None:
        if self.dead:
            raise ConnectionClosed
        await self._out.put(data)  # full: wait (see the module docstring)
        if self.dead:
            raise ConnectionClosed

    async def send(self, message: Mapping[str, Any]) -> None:
        await self.send_bytes(dumps_line(message))

    async def _write_loop(self) -> None:
        try:
            while True:
                chunk = await self._out.get()
                if chunk is None:
                    return
                self._writer.write(chunk)
                await self._writer.drain()
        except OSError:  # ConnectionError, BrokenPipeError: the peer went away
            pass
        finally:
            self.dead = True
            self._writer.close()
            while True:  # wake producers blocked in put(); they will see `dead`
                try:
                    self._out.get_nowait()
                except asyncio.QueueEmpty:
                    break

    # -- inbound ----------------------------------------------------------------------------
    async def run(self) -> None:
        self._writer_task = asyncio.create_task(self._write_loop(), name="sb-conn-writer")
        try:
            while not self.dead:
                try:
                    line = await self._reader.readline()
                except ValueError:  # LimitOverrunError surfaces as ValueError from readline
                    await self.send(
                        error_response(None, LINE_TOO_LONG, f"line exceeds {MAX_LINE_BYTES} bytes")
                    )
                    break
                except OSError:
                    break
                if not line:
                    break
                if line.strip():
                    await self._handle_line(line)
        except ConnectionClosed:
            pass
        finally:
            await self.close()

    async def _handle_line(self, line: bytes) -> None:
        try:
            req = parse_request(line)
        except ProtocolError as exc:
            await self.send(error_response(exc.req_id, exc.code, exc.message))
            return
        shutdown = req.cmd == "shutdown"
        try:
            if req.cmd == "subscribe":
                result = await self._subscribe(req)
            elif req.cmd == "unsubscribe":
                result = await self._unsubscribe(req)
            else:
                result = await self._daemon.execute(req)
        except ProtocolError as exc:
            await self.send(error_response(req.id, exc.code, exc.message))
            return
        await self.send(ok_response(req.id, result))
        if shutdown:
            self._daemon.request_shutdown()

    # -- subscriptions ----------------------------------------------------------------------
    async def _subscribe(self, req: Request) -> Result:
        args = req.args
        _check_keys(args, {"job_id", "from_seq"})
        job_id = _str(args, "job_id", required=False)
        from_seq = args.get("from_seq")
        if from_seq is not None and (
            isinstance(from_seq, bool) or not isinstance(from_seq, int) or from_seq < 0
        ):
            raise ProtocolError(BAD_ARGS, "from_seq must be an integer >= 0")
        if req.id in self._subs:
            raise ProtocolError(CONFLICT, f"subscription {req.id} already exists")
        if len(self._subs) >= MAX_SUBSCRIPTIONS_PER_CONNECTION:
            raise ProtocolError(CONFLICT, "too many subscriptions on this connection")
        if job_id is not None and await self._daemon.store.get_job(job_id) is None:
            raise ProtocolError(NOT_FOUND, f"unknown job {job_id}")
        if self._closing:
            raise ProtocolError(CONFLICT, "connection is closing")
        # Registered before anything is read from the store: an event committed from here on is
        # in the live queue, one committed earlier is in the store, and one that is both is
        # removed by the floor. Nothing awaits between registration and the task creation.
        floor = from_seq - 1 if from_seq else 0
        sub = self._daemon.bus.subscribe(job_id, floor=floor)
        task = asyncio.create_task(
            self._pump(req.id, sub, floor if from_seq else None), name=f"sb-pump:{req.id}"
        )
        task.add_done_callback(self._on_pump_done)
        self._subs[req.id] = (sub, task)
        return {"subscription": req.id}

    def _on_pump_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()  # retrieved: asyncio never logs "never retrieved"
        if exc is not None and not isinstance(exc, ConnectionClosed):
            logger.error("subscription pump failed: %r", exc)

    async def _unsubscribe(self, req: Request) -> Result:
        _check_keys(req.args, {"subscription"})
        sub_id = _str(req.args, "subscription")
        assert sub_id is not None
        entry = self._subs.pop(sub_id, None)
        if entry is None:
            raise ProtocolError(NOT_FOUND, f"no subscription {sub_id}")
        await self._stop_pump(*entry)
        return {"subscription": sub_id}

    @staticmethod
    async def _stop_pump(sub: Subscription, task: asyncio.Task[None]) -> None:
        sub.close()
        task.cancel()
        await asyncio.wait({task})

    async def _pump(self, sub_id: str, sub: Subscription, backfill_after: int | None) -> None:
        if backfill_after is not None:
            await self._backfill(sub_id, sub, backfill_after)
        while True:
            batch = await sub.next_batch()
            if not batch:
                return
            await self.send_bytes(
                b"".join(
                    dumps_line(
                        gap_message(sub_id, item.from_seq, item.to_seq)
                        if isinstance(item, Gap)
                        else event_message(sub_id, item)
                    )
                    for item in batch
                )
            )

    async def _backfill(self, sub_id: str, sub: Subscription, after: int) -> None:
        """Send every stored event with seq > `after` in pages, then hand over to the live
        queue. The floor makes the queue skip what was already sent."""
        while True:
            page = await self._daemon.store.read_since(after, BACKFILL_PAGE)
            if not page:
                return
            last = page[-1].seq
            assert last is not None
            after = last
            sub.floor = max(sub.floor, last)
            chunk = b"".join(
                dumps_line(event_message(sub_id, e))
                for e in page
                if sub.job_id is None or e.job_id == sub.job_id
            )
            if chunk:
                await self.send_bytes(chunk)
            if len(page) < BACKFILL_PAGE:
                return

    # -- teardown ---------------------------------------------------------------------------
    async def close(self) -> None:
        """Stop pumps, flush what is queued (bounded wait), close the socket. Idempotent."""
        if self._closing:
            return
        self._closing = True
        for entry in list(self._subs.values()):
            await self._stop_pump(*entry)
        self._subs.clear()
        task = self._writer_task
        if task is not None and not task.done():
            with contextlib.suppress(TimeoutError, ConnectionClosed):
                async with asyncio.timeout(FLUSH_TIMEOUT_S):
                    await self._out.put(None)
                    await asyncio.wait({task})
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
        if task is not None and not task.cancelled() and task.exception() is not None:
            logger.error("connection writer failed: %r", task.exception())
        self.dead = True
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()


# ---- daemon ----------------------------------------------------------------------------------


class Daemon:
    def __init__(
        self,
        *,
        socket_path: str | Path,
        db_path: str | Path,
        envs_dir: str | Path,
        max_running: int = 8,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
        bus_queue_size: int = DEFAULT_QUEUE_SIZE,
        clock: Callable[[], float] = time.time,
        sink_db: str | Path | None = None,
        sink_naive: bool = False,
        verify_projection: bool | None = None,
        debug_asyncio: bool = False,
        lag_interval: float = 0.1,
        on_token: TokenHook | None = None,
        store_linger: float = 0.0,
        api_key: str | None = None,
        llm_factory: Callable[[tuple[str, ...]], LLMClient] | None = None,
        runner_factory: Callable[[Path], CommandRunner] = LocalSandboxRunner,
        default_rollout_tokens: int = DEFAULT_ROLLOUT_TOKENS,
    ) -> None:
        self.store_linger = store_linger
        self._api_key = api_key
        self._llm_factory = llm_factory
        self.runner_factory = runner_factory
        self.default_rollout_tokens = default_rollout_tokens
        self._llms: dict[tuple[str, ...], LLMClient] = {}
        self._grading: dict[str, asyncio.Task[None]] = {}
        self._grade_slots = asyncio.Semaphore(max_running)
        self.debug_asyncio = debug_asyncio
        self.on_token = on_token
        self.lag = LoopLagMonitor(lag_interval)
        self.sink_db = Path(sink_db) if sink_db is not None else None
        self.sink_naive = sink_naive
        self.verify_projection = verify_projection
        self.recovery: RecoveryReport | None = None
        self._sink: SQLiteSink | None = None
        self.socket_path = Path(socket_path)
        self.db_path = Path(db_path)
        self.envs_dir = Path(envs_dir)
        self.max_running = max_running
        self.shutdown_timeout = shutdown_timeout
        self._bus_queue_size = bus_queue_size
        self._clock = clock
        self._store: EventStore | None = None
        self._supervisor: Supervisor | None = None
        self._bus: EventBus | None = None
        self._server: asyncio.Server | None = None
        self._conns: set[Connection] = set()
        self._handlers: set[asyncio.Task[Any]] = set()
        self._specs: dict[str, JobSpec] = {}
        self._shutdown_requested = asyncio.Event()
        self._closing = False
        self._stopped = False
        self._started_at = 0.0
        self._commands: dict[str, Callable[[Mapping[str, Any]], Awaitable[Result]]] = {
            "submit": self._submit,
            "list": self._list,
            "get": self._get,
            "cancel": self._cancel,
            "pause": self._pause,
            "resume": self._resume,
            "extend": self._extend,
            "stats": self._stats,
            "shutdown": self._shutdown_cmd,
        }

    @property
    def store(self) -> EventStore:
        assert self._store is not None, "daemon not started"
        return self._store

    @property
    def supervisor(self) -> Supervisor:
        assert self._supervisor is not None, "daemon not started"
        return self._supervisor

    @property
    def bus(self) -> EventBus:
        assert self._bus is not None, "daemon not started"
        return self._bus

    # ---- lifecycle ----------------------------------------------------------------------
    def _configure_debug(self) -> None:
        loop = asyncio.get_running_loop()
        if self.debug_asyncio:
            loop.set_debug(True)
        if self.debug_asyncio or (loop.get_debug() and os.environ.get("PYTHONASYNCIODEBUG")):
            loop.slow_callback_duration = SLOW_CALLBACK_S

    async def start(self) -> None:
        self._configure_debug()
        sock = await self._claim_socket()
        try:
            self.lag.start()
            self.envs_dir.mkdir(parents=True, exist_ok=True)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = await EventStore.open(self.db_path, linger=self.store_linger)
            self._bus = EventBus(queue_size=self._bus_queue_size)
            self._store.add_listener(self._bus.publish)
            self._store.add_listener(self._on_commit)
            if self.sink_db is not None:
                self._sink = await SQLiteSink.open(self.sink_db, idempotent=not self.sink_naive)
            self._supervisor = Supervisor(
                self._store,
                self._agent_factory,
                self._env_factory,
                self._sink_factory,
                clock=self._clock,
                max_running=self.max_running,
            )
            # Before the server listens: nobody sees the half-recovered state.
            self.recovery = await recover(
                self._store,
                self._supervisor,
                prepare=self._ensure_spec,
                verify=self.verify_projection,
                clock=self._clock,
            )
            for job in await self._store.list_jobs(JobState.VERIFYING):
                if job.kind is JobKind.ROLLOUT:  # fake jobs are never graded
                    self._schedule_grading(job.job_id)
            self._server = await asyncio.start_unix_server(
                self._on_connect, sock=sock, limit=MAX_LINE_BYTES
            )
        except BaseException:
            sock.close()
            await self._release()
            raise
        self._started_at = time.monotonic()
        logger.info("daemon listening on %s", self.socket_path)

    async def _claim_socket(self) -> socket.socket:
        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(mode):
                raise DaemonError(f"{path} exists and is not a socket")
            try:
                async with asyncio.timeout(2.0):
                    _, w = await asyncio.open_unix_connection(str(path))
            except ConnectionRefusedError:
                logger.warning("removing stale socket %s", path)
                path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except TimeoutError:
                raise DaemonAlreadyRunning(f"a daemon is not answering on {path}") from None
            else:
                w.close()
                with contextlib.suppress(OSError):
                    await w.wait_closed()
                raise DaemonAlreadyRunning(f"another daemon is running on {path}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)  # no await between umask and bind: nothing else is affected
        try:
            sock.bind(str(path))
        except OSError as exc:
            sock.close()
            if exc.errno == errno.EADDRINUSE:
                raise DaemonAlreadyRunning(f"another daemon is running on {path}") from exc
            raise
        finally:
            os.umask(old_umask)
        os.chmod(path, 0o600)
        return sock

    def request_shutdown(self) -> None:
        """Ask `run` to shut down (signal handlers and the `shutdown` command call this)."""
        self._shutdown_requested.set()

    async def run(self) -> None:
        """Start, serve until SIGTERM/SIGINT/`shutdown`, then shut down gracefully."""
        await self.start()
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for sig in signals:
            loop.add_signal_handler(sig, self.request_shutdown)
        try:
            await self._shutdown_requested.wait()
        finally:
            for sig in signals:
                loop.remove_signal_handler(sig)
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful stop, see the module docstring. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._closing = True
        logger.info("shutting down")
        if self._server is not None:
            self._server.close()  # stop accepting
        if self._supervisor is not None:
            self._supervisor.drain()
            if not await self._supervisor.wait_idle(self.shutdown_timeout):
                logger.warning("jobs still running after %.1fs, cancelling", self.shutdown_timeout)
            await self._supervisor.aclose()  # cancels leftovers; writes nothing
        await self._stop_grading()
        await asyncio.gather(*(c.close() for c in list(self._conns)))
        handlers = list(self._handlers)
        if handlers:
            _, pending = await asyncio.wait(handlers, timeout=FLUSH_TIMEOUT_S)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending)
        if self._server is not None:
            await self._server.wait_closed()
        await self._release()
        logger.info("shutdown complete")

    async def _release(self) -> None:
        await self._stop_grading()
        await self.lag.stop()
        llms, self._llms = list(self._llms.values()), {}
        for llm in llms:
            aclose = getattr(llm, "aclose", None)  # optional: not part of the LLMClient protocol
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
        if self._bus is not None:
            self._bus.close()
        if self._store is not None:
            self._store.remove_listener(self.bus.publish)
            self._store.remove_listener(self._on_commit)
            await self._store.aclose()
        if self._sink is not None:
            await self._sink.aclose()
        self.socket_path.unlink(missing_ok=True)

    # ---- connections --------------------------------------------------------------------
    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        conn = Connection(self, reader, writer)
        self._conns.add(conn)
        self._handlers.add(task)
        try:
            await conn.run()
        finally:
            self._conns.discard(conn)
            self._handlers.discard(task)

    # ---- factories ----------------------------------------------------------------------
    def _agent_factory(self, job: Job) -> Agent:
        spec = self._specs[job.job_id]
        if spec.agent == "gemini":
            return self._gemini_agent(spec.models)
        return FakeAgent(
            steps=spec.steps,
            token_interval=spec.token_interval,
            effect_every=spec.effect_every,
            effect_policy=ReplayPolicy(spec.effect_policy),
            exit_at_step=spec.exit_at_step,
            on_token=self.on_token,
        )

    def _sink_factory(self, job: Job) -> SQLiteSink | None:
        return self._sink

    def _env_factory(self, job: Job) -> Environment | None:
        spec = self._specs[job.job_id]
        template = str(load_task(spec.task).repo) if spec.task else spec.env_template
        return DirEnv(Path(template), self.envs_dir, job.job_id) if template else None

    # ---- real agents --------------------------------------------------------------------
    def _llm(self, models: tuple[str, ...]) -> LLMClient:
        llm = self._llms.get(models)
        if llm is None:
            if self._llm_factory is not None:
                llm = self._llm_factory(models)
            else:
                from switchboard.llm.gemini import DEFAULT_MODELS, GeminiClient

                if not self._api_key:
                    raise ProtocolError(BAD_ARGS, "the daemon has no Gemini API key")
                llm = GeminiClient(models or DEFAULT_MODELS, api_key=self._api_key)
            self._llms[models] = llm
        return llm

    def _gemini_agent(self, models: tuple[str, ...]) -> GeminiAgent:
        def runner(ctx: StepContext) -> CommandRunner:
            assert ctx.workdir is not None  # GeminiAgent checks before asking for a runner
            return self.runner_factory(ctx.workdir)

        return GeminiAgent(self._llm(models), runner)

    # ---- grading ------------------------------------------------------------------------
    def _on_commit(self, events: Sequence[Event]) -> None:
        """Store listener (sync, on the loop): start grading jobs that just became verifying."""
        if self._closing:
            return
        for e in events:
            if e.type is EventType.STATE_CHANGED and e.payload.get("to") == "verifying":
                spec = self._specs.get(e.job_id)
                if spec is None or spec.task is not None:
                    self._schedule_grading(e.job_id)

    def _schedule_grading(self, job_id: str) -> None:
        if job_id in self._grading or self._closing:
            return
        task = asyncio.create_task(self._grade(job_id), name=f"grade-{job_id}")
        self._grading[job_id] = task
        task.add_done_callback(lambda t: self._grading_done(job_id, t))

    def _grading_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        if self._grading.get(job_id) is task:
            del self._grading[job_id]
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error("grading of %s crashed", job_id, exc_info=exc)

    async def _stop_grading(self) -> None:
        tasks = list(self._grading.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks)

    async def _grade(self, job_id: str) -> None:
        await self._ensure_spec(job_id)
        spec = self._specs[job_id]
        if spec.task is None:
            return  # fake jobs stay verifying: nothing to grade them against
        async with self._grade_slots:
            job = await self.store.get_job(job_id)
            if job is None or job.state is not JobState.VERIFYING:
                return
            workdir = DirEnv(load_task(spec.task).repo, self.envs_dir, job_id).workdir
            try:
                result = await grade(load_task(spec.task), workdir, self.runner_factory)
            except Exception as exc:
                logger.exception("grading %s failed", job_id)
                group = [
                    self._event(
                        job,
                        EventType.STATE_CHANGED,
                        {
                            "from": "verifying",
                            "to": "failed",
                            "reason": f"grading failed: {type(exc).__name__}: {exc}",
                        },
                    )
                ]
            else:
                group = [
                    self._event(
                        job,
                        EventType.STATE_CHANGED,
                        {"from": "verifying", "to": "done", "reason": "graded"},
                    ),
                    self._event(job, EventType.VERDICT_RECORDED, result.to_payload()),
                ]
            try:
                await self.store.append_group(group)
            except ProjectionError:
                logger.warning("job %s moved on while it was graded; verdict dropped", job_id)

    def _event(self, job: Job, type_: EventType, payload: dict[str, Any]) -> Event:
        return Event(
            job_id=job.job_id, epoch=job.epoch, ts=self._clock(), type=type_, payload=payload
        )

    async def _ensure_spec(self, job_id: str) -> None:
        if job_id in self._specs:
            return
        for e in await self.store.read_job(job_id):
            if e.type is EventType.JOB_CREATED:
                self._specs[job_id] = JobSpec.from_payload(e.payload)
                return
        raise ProtocolError(NOT_FOUND, f"unknown job {job_id}")

    # ---- commands -----------------------------------------------------------------------
    async def execute(self, req: Request) -> Result:
        """Run a non-streaming command; raises ProtocolError for every failure."""
        handler = self._commands[req.cmd]
        try:
            return await handler(req.args)
        except ProtocolError:
            raise
        except UnknownJob as exc:
            raise ProtocolError(NOT_FOUND, str(exc)) from exc
        except (
            NotCancellable,
            NotPausable,
            NotResumable,
            NotExtendable,
            AlreadyRunning,
            SupervisorClosed,
        ) as exc:
            raise ProtocolError(CONFLICT, str(exc)) from exc
        except Exception as exc:
            logger.exception("command %s failed", req.cmd)
            raise ProtocolError(INTERNAL, f"{type(exc).__name__}: {exc}") from exc

    async def _job_result(self, job_id: str) -> Result:
        job = await self.store.get_job(job_id)
        if job is None:
            raise ProtocolError(NOT_FOUND, f"unknown job {job_id}")
        return {"job": job_to_dict(job) | {"live": job_id in self.supervisor.running}}

    async def _submit(self, a: Mapping[str, Any]) -> Result:
        _check_keys(
            a,
            {
                "kind",
                "task_ref",
                "budget",
                "seed",
                "steps",
                "env_template",
                "instructions",
                "token_interval",
                "effect_every",
                "effect_policy",
                "exit_at_step",
                "agent",
                "task",
                "models",
            },
        )
        if self._closing:
            raise ProtocolError(CONFLICT, "daemon is shutting down")
        agent = _str(a, "agent", required=False) or "fake"
        if agent not in AGENTS:
            raise ProtocolError(BAD_ARGS, f"agent must be one of {list(AGENTS)}")
        kind = JobKind.FAKE if agent == "fake" else JobKind.ROLLOUT
        if a.get("kind", kind.value) != kind.value:
            raise ProtocolError(BAD_ARGS, f"kind must be {kind.value!r} for agent {agent!r}")
        task_name = _str(a, "task", required=agent != "fake")
        models = a.get("models", [])
        if not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
            raise ProtocolError(BAD_ARGS, "models must be a list of model names")
        try:
            task = load_task(task_name) if task_name is not None else None
        except UnknownTask:
            raise ProtocolError(BAD_ARGS, f"unknown task {task_name!r}") from None
        task_ref = _str(a, "task_ref", required=False) or task_name or "fake"
        try:
            budget: Budget = budget_from_dict(a.get("budget", {}))
        except ProjectionError as exc:
            raise ProtocolError(BAD_ARGS, str(exc)) from exc
        if kind is JobKind.ROLLOUT and budget.tokens is None:
            budget = dataclasses.replace(budget, tokens=self.default_rollout_tokens)
        seed = _int(a, "seed", 0, -(2**53), 2**53)
        steps = _int(a, "steps", 5, 1, MAX_STEPS)
        interval = _num(a, "token_interval", 0.01, 0.0, 10.0)
        effect_every = _int(a, "effect_every", 0, 0, MAX_STEPS)
        policy = _str(a, "effect_policy", required=False) or ReplayPolicy.KEYED.value
        if policy not in {p.value for p in ReplayPolicy}:
            raise ProtocolError(BAD_ARGS, f"unknown effect_policy {policy!r}")
        if effect_every and self._sink is None:
            raise ProtocolError(BAD_ARGS, "effect_every needs a daemon started with --sink-db")
        exit_at = _int(a, "exit_at_step", -1, -1, MAX_STEPS)  # -1: never (testing only)
        template = _str(a, "env_template", required=False)
        if template is not None and not await asyncio.to_thread(Path(template).is_dir):
            raise ProtocolError(BAD_ARGS, f"env_template {template} is not a directory")
        instructions = a.get("instructions", [])
        if not isinstance(instructions, list) or not all(isinstance(m, dict) for m in instructions):
            raise ProtocolError(BAD_ARGS, "instructions must be a list of objects")
        if task is not None:
            if template is not None:
                raise ProtocolError(BAD_ARGS, "a task brings its own environment: no env_template")
            if not instructions:
                instructions = task_instructions(task.prompt)
        if agent == "fake":
            config_hash = FakeAgent(steps=steps).config_hash
        else:
            config_hash = self._gemini_agent(tuple(models)).config_hash
        spec = JobSpec(
            steps=steps,
            token_interval=interval,
            env_template=template,
            effect_every=effect_every,
            effect_policy=policy,
            exit_at_step=None if exit_at < 0 else exit_at,
            agent=agent,
            task=task_name,
            models=tuple(models),
        )
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        self._specs[job_id] = spec
        try:
            await self.store.append(
                Event(
                    job_id=job_id,
                    epoch=0,
                    ts=self._clock(),
                    type=EventType.JOB_CREATED,
                    payload={
                        "kind": kind.value,
                        "task_ref": task_ref,
                        "budget": budget_to_dict(budget),
                        "seed": seed,
                        "instructions": instructions,
                        "config_hash": config_hash,
                        "steps": steps,
                        "token_interval": interval,
                        "env_template": template,
                        "effect_every": effect_every,
                        "effect_policy": policy,
                        "exit_at_step": spec.exit_at_step,
                        "agent": agent,
                        "task": task_name,
                        "models": list(models),
                    },
                )
            )
            self.supervisor.start(job_id)
        except BaseException:
            self._specs.pop(job_id, None)
            raise
        return {"job_id": job_id}

    async def _list(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"state"})
        state: JobState | None = None
        raw = _str(a, "state", required=False)
        if raw is not None:
            try:
                state = JobState(raw)
            except ValueError:
                raise ProtocolError(BAD_ARGS, f"unknown state {raw!r}") from None
        return {"jobs": [job_to_dict(j) for j in await self.store.list_jobs(state)]}

    async def _get(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"job_id"})
        return await self._job_result(_job_id(a))

    async def _cancel(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"job_id", "reason"})
        job_id = _job_id(a)
        await self.supervisor.cancel(
            job_id, _str(a, "reason", required=False) or "cancelled by user"
        )
        return await self._job_result(job_id)

    async def _pause(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"job_id"})
        job_id = _job_id(a)
        await self.supervisor.pause(job_id)  # returns once recorded; the job pauses at a boundary
        return await self._job_result(job_id)

    async def _resume(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"job_id"})
        job_id = _job_id(a)
        await self._ensure_spec(job_id)
        await self.supervisor.resume(job_id)
        return await self._job_result(job_id)

    async def _extend(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, {"job_id", "active_seconds", "tokens", "usd_micros"})
        job_id = _job_id(a)
        secs = _num(a, "active_seconds", 0.0, 0.0, 1e12)
        tokens = _int(a, "tokens", 0, 0, 2**53)
        usd = _int(a, "usd_micros", 0, 0, 2**53)
        if not (secs or tokens or usd):
            raise ProtocolError(BAD_ARGS, "extend needs a positive delta")
        await self._ensure_spec(job_id)
        await self.supervisor.extend(job_id, active_seconds=secs, tokens=tokens, usd_micros=usd)
        return await self._job_result(job_id)

    async def _stats(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, set())
        by_state: dict[str, int] = {}
        for j in await self.store.list_jobs():
            by_state[j.state.value] = by_state.get(j.state.value, 0) + 1
        return {
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - self._started_at, 3),
            "connections": len(self._conns),
            "jobs": by_state,
            "supervisor": dataclasses.asdict(self.supervisor.stats),
            "store": dataclasses.asdict(self.store.stats),
            "store_latency": dataclasses.asdict(self.store.latency),
            "loop_lag": dataclasses.asdict(self.lag.stats()),
            "debug_asyncio": asyncio.get_running_loop().get_debug(),
            "bus": dataclasses.asdict(self.bus.stats),
            "recovery": (None if self.recovery is None else dataclasses.asdict(self.recovery)),
        }

    async def _shutdown_cmd(self, a: Mapping[str, Any]) -> Result:
        _check_keys(a, set())
        return {"shutting_down": True}  # Connection triggers request_shutdown after replying
