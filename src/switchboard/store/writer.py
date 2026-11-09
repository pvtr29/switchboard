"""Async facade over the SQLite event log: one writer task, group commit.

Why dedicated executors and not ``asyncio.to_thread``
-----------------------------------------------------
A sqlite3 connection is bound to the thread that created it (``check_same_thread=True``).
``asyncio.to_thread`` runs each call on *some* thread of the loop's default pool, so successive
calls land on different threads and sqlite3 would raise ``ProgrammingError``. (Switching the
check off would only trade the error for silent data races.) So the store owns two
``ThreadPoolExecutor(max_workers=1)``: one for the write connection, one for a read connection.
Each connection is opened inside its executor's thread and every later call goes through
``loop.run_in_executor`` on that same executor, so a connection only ever sees one thread. WAL
mode lets the reader run while the writer commits.

Group commit
------------
Callers enqueue *groups* of events with a future. The writer task takes one group, drains more
with ``get_nowait`` up to ``max_batch`` events, and commits them all in ONE transaction (one
WAL write instead of one per caller), then resolves each future. The jobs projection is
updated in that same transaction; a group whose events cannot be folded is rolled back alone
(savepoint) and only its caller gets the exception. A group is never split, so
``append_group`` is atomic. A group larger than ``max_batch`` is committed whole on its own.
Acks are sent only after COMMIT returned.

Optional `linger` (seconds, default 0 = off): when the previous batch held more than one group
and the queue is empty, the writer waits `linger` before committing so that more callers join
the batch. It trades up to `linger` of ack latency, only under load, for fewer transactions.

Backpressure: the queue is bounded; ``append`` awaits ``queue.put`` when it is full, so
producers slow down instead of memory growing.

Commit listeners: `add_listener(fn)` registers a plain function called on the event loop, after
each committed batch and before the callers' acks, with the events that batch INSERTED (seq
set, commit order). Duplicates and rejected groups are not reported. A listener must be quick
and must never await; an exception from one is logged and swallowed, so a faulty listener can
neither break the writer nor delay other listeners.

Cancellation: a caller cancelled while awaiting its ack may still have its events written
(they are already queued); only the ack is lost. The writer skips cancelled futures.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from switchboard.core import Event, Job, JobState
from switchboard.obs.stats import LatencySnapshot, LatencyWindow
from switchboard.store.interface import AppendResult, AppendStatus, EffectRecord, GroupResult
from switchboard.store.sqlite import Row, SQLiteEventLog, _with_seq, encode_event, open_db

logger = logging.getLogger("switchboard.store")

LATENCY_WINDOW = 8192  # samples kept per latency series
JOB_MIRROR_SIZE = 100_000  # job projections kept in memory (least recently changed evicted)

type CommitListener = Callable[[Sequence[Event]], None]


class StoreClosed(Exception):
    """The store is closed (or closing); no further appends are accepted."""


class WriterCrashed(StoreClosed):
    """The writer task died unexpectedly; the cause is chained as ``__cause__``.

    For the batch that was in flight when the writer died, the outcome is UNKNOWN: its
    transaction may already have committed in the writer thread. Callers must not treat this
    error as "not written"; recovery reads the log to find out (the same rule the effect
    journal applies to an intent without an outcome).
    """


@dataclass(frozen=True, slots=True)
class StoreStats:
    transactions: int
    events_written: int
    duplicates: int
    rejected_groups: int
    max_batch_seen: int
    queue_depth: int


@dataclass(frozen=True, slots=True)
class StoreLatency:
    """Write-path timings over the most recent batches/appends (bounded memory).

    `commit_s`: per committed batch, the time spent in the writer thread (BEGIN to COMMIT).
    `ack_s`: per append call, from entering the queue to the ack being resolved (queue wait +
    commit + listeners). `batch_events` / `batch_groups`: batch sizes (values are counts).
    """

    commit_s: LatencySnapshot
    ack_s: LatencySnapshot
    batch_events: LatencySnapshot
    batch_groups: LatencySnapshot


@dataclass(slots=True)
class _Item:
    group: list[Event]
    rows: list[Row]
    fut: asyncio.Future[list[AppendResult]]
    enqueued: float


def _commit_in_thread(
    log: SQLiteEventLog, groups: list[list[Event]], rows: list[list[Row]]
) -> tuple[list[GroupResult], float, dict[str, Job]]:
    t0 = time.perf_counter()
    outcomes = log.append_encoded(groups, rows)
    return outcomes, time.perf_counter() - t0, log.last_changed


def _close_in_thread(log: SQLiteEventLog) -> None:
    log.close()


class EventStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_batch: int = 512,
        queue_size: int = 10_000,
        linger: float = 0.0,
    ) -> None:
        if max_batch < 1 or queue_size < 1:
            raise ValueError("max_batch and queue_size must be >= 1")
        if linger < 0:
            raise ValueError("linger must be >= 0")
        self._linger = linger
        self._last_groups = 1
        self._path = path
        self._max_batch = max_batch
        self._queue: asyncio.Queue[_Item | None] = asyncio.Queue(maxsize=queue_size)
        self._wexec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sb-store-writer")
        self._rexec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sb-store-reader")
        self._wlog: SQLiteEventLog | None = None
        self._rlog: SQLiteEventLog | None = None
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._failure: WriterCrashed | None = None
        self._inflight: list[_Item] = []
        self._carry: _Item | None = None
        self._transactions = 0
        self._events_written = 0
        self._duplicates = 0
        self._rejected_groups = 0
        self._max_batch_seen = 0
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._commit_lat = LatencyWindow(LATENCY_WINDOW)
        self._ack_lat = LatencyWindow(LATENCY_WINDOW)
        self._batch_events = LatencyWindow(LATENCY_WINDOW)
        self._batch_groups = LatencyWindow(LATENCY_WINDOW)
        self._close_lock = asyncio.Lock()
        self._listeners: list[CommitListener] = []

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        max_batch: int = 512,
        queue_size: int = 10_000,
        linger: float = 0.0,
    ) -> Self:
        store = cls(path, max_batch=max_batch, queue_size=queue_size, linger=linger)
        loop = asyncio.get_running_loop()
        try:
            # Connections are created inside their owning threads. Writer first: it runs the
            # migration, so the reader never races it.
            store._wlog = await loop.run_in_executor(store._wexec, store._open_log)
            store._rlog = await loop.run_in_executor(store._rexec, store._open_log)
        except BaseException:
            await store._release_resources()
            raise
        store._task = asyncio.create_task(store._writer_loop(), name="sb-store-writer")
        store._task.add_done_callback(store._on_writer_done)
        return store

    def _open_log(self) -> SQLiteEventLog:
        return SQLiteEventLog(open_db(self._path))

    @property
    def _log(self) -> SQLiteEventLog:
        assert self._wlog is not None
        return self._wlog

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ---- writing -------------------------------------------------------------------------

    async def append(self, event: Event) -> AppendResult:
        return (await self.append_group([event]))[0]

    async def append_group(self, events: Sequence[Event]) -> list[AppendResult]:
        """Append events atomically (same transaction); results are in input order.

        Raises ValueError (for this caller only) if any event is invalid, ProjectionError (a
        ValueError; nothing from the group is written, other callers are unaffected) if the
        events cannot be folded into the job projection, StoreClosed after close,
        WriterCrashed if the writer died.
        """
        group = list(events)
        # Same rules as the log; fail before the group can poison a batch. The rows are encoded
        # exactly once, here, and travel with the group to the writer thread.
        rows = [encode_event(e) for e in group]
        if not group:
            return []
        self._check_open()
        fut: asyncio.Future[list[AppendResult]] = asyncio.get_running_loop().create_future()
        item = _Item(group, rows, fut, time.perf_counter())
        await self._queue.put(item)  # backpressure: waits while the queue is full
        if self._failure is not None and not fut.done():
            fut.set_exception(self._failure)
        return await fut

    def _check_open(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closing:
            raise StoreClosed("event store is closed")

    async def _writer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first: _Item | None
            if self._carry is not None:
                first, self._carry = self._carry, None
            else:
                first = await self._queue.get()
                if first is None:
                    return
            batch = [first]
            count = len(first.group)
            stop = False
            if self._linger and self._last_groups > 1 and self._queue.empty():
                # Under load (the previous batch had several groups) wait a moment so more
                # callers can join: fewer, bigger transactions. An idle store never lingers.
                await asyncio.sleep(self._linger)
            while count < self._max_batch:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                    break
                if count + len(item.group) > self._max_batch:
                    self._carry = item
                    break
                batch.append(item)
                count += len(item.group)
            self._inflight = batch
            groups = [item.group for item in batch]
            rows = [item.rows for item in batch]
            try:
                outcomes, commit_s, changed = await loop.run_in_executor(
                    self._wexec, _commit_in_thread, self._log, groups, rows
                )
            except Exception as exc:
                # e.g. disk error: the transaction rolled back, so nothing in it was written.
                for item in batch:
                    if not item.fut.done():
                        item.fut.set_exception(exc)
                self._inflight = []
                if stop:
                    return
                continue
            # An invariant violation here is a bug: let it kill the writer (see the callback).
            if len(outcomes) != len(batch):
                raise RuntimeError(
                    f"append_groups returned {len(outcomes)} for {len(batch)} groups"
                )
            self._mirror(changed)
            self._last_groups = len(batch)
            self._transactions += 1
            self._max_batch_seen = max(self._max_batch_seen, count)
            self._commit_lat.record(commit_s)
            self._batch_events.record(count)
            self._batch_groups.record(len(batch))
            committed: list[Event] = []
            acked: list[float] = []
            for item, outcome in zip(batch, outcomes, strict=True):
                group, fut = item.group, item.fut
                if isinstance(outcome, Exception):
                    # Projection-invalid group: nothing of it was written; others committed.
                    self._rejected_groups += 1
                    if not fut.done():
                        fut.set_exception(outcome)
                    continue
                if len(outcome) != len(group):
                    raise RuntimeError("append_groups returned a wrong number of results")
                if self._listeners:
                    committed.extend(
                        _with_seq(e, r.seq)
                        for e, r in zip(group, outcome, strict=True)
                        if r.seq is not None
                    )
                self._events_written += sum(r.status is AppendStatus.INSERTED for r in outcome)
                self._duplicates += sum(r.status is AppendStatus.DUPLICATE for r in outcome)
                if not fut.done():  # a cancelled caller no longer wants its ack
                    fut.set_result(outcome)
                    acked.append(item.enqueued)
            if committed:
                self._notify(committed)
            now = time.perf_counter()
            for enqueued in acked:
                self._ack_lat.record(now - enqueued)
            self._inflight = []
            if stop:
                return

    def add_listener(self, listener: CommitListener) -> None:
        """Call `listener(events)` on the loop after every committed batch (see module doc)."""
        self._listeners.append(listener)

    def remove_listener(self, listener: CommitListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, events: Sequence[Event]) -> None:
        for listener in list(self._listeners):
            try:
                listener(events)
            except Exception:
                logger.exception("commit listener %r failed", listener)

    def _on_writer_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            cause: BaseException | None = None
        else:
            cause = task.exception()  # retrieves it, so asyncio never logs "never retrieved"
        if cause is None and self._closing:
            return
        failure = WriterCrashed("store writer task stopped unexpectedly")
        failure.__cause__ = cause
        self._failure = failure
        pending = list(self._inflight)
        self._inflight = []
        if self._carry is not None:
            pending.append(self._carry)
            self._carry = None
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not None:
                pending.append(item)
        for item in pending:
            if not item.fut.done():
                item.fut.set_exception(failure)

    # ---- reading -------------------------------------------------------------------------

    async def _read[T](self, fn_name: str, *args: object) -> T:
        if self._closed:
            raise StoreClosed("event store is closed")
        assert self._rlog is not None
        fn = getattr(self._rlog, fn_name)
        result: T = await asyncio.get_running_loop().run_in_executor(self._rexec, fn, *args)
        return result

    async def read_since(self, after_seq: int, limit: int = 1000) -> list[Event]:
        return await self._read("read_since", after_seq, limit)

    async def read_job(self, job_id: str, after_seq: int = 0) -> list[Event]:
        return await self._read("read_job", job_id, after_seq)

    async def last_seq(self) -> int:
        return await self._read("last_seq")

    def _mirror(self, changed: dict[str, Job]) -> None:
        jobs = self._jobs
        for jid, job in changed.items():
            jobs[jid] = job
            jobs.move_to_end(jid)
        while len(jobs) > JOB_MIRROR_SIZE:
            jobs.popitem(last=False)

    async def get_job(self, job_id: str) -> Job | None:
        """The job's projection as of the last commit. Served from memory when the job changed
        recently (the writer mirrors every projection it writes, before acking the caller, so a
        caller always sees its own writes); otherwise read from SQLite. Never a thread hop on a
        hit, which matters when thousands of runners poll their job once per step."""
        if self._closed:
            raise StoreClosed("event store is closed")
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        return await self._read("get_job", job_id)

    async def get_effect(self, effect_id: str) -> EffectRecord | None:
        return await self._read("get_effect", effect_id)

    async def list_jobs(self, state: JobState | None = None) -> list[Job]:
        return await self._read("list_jobs", state)

    @property
    def stats(self) -> StoreStats:
        return StoreStats(
            transactions=self._transactions,
            events_written=self._events_written,
            duplicates=self._duplicates,
            rejected_groups=self._rejected_groups,
            max_batch_seen=self._max_batch_seen,
            queue_depth=self._queue.qsize(),
        )

    @property
    def latency(self) -> StoreLatency:
        return StoreLatency(
            commit_s=self._commit_lat.snapshot(),
            ack_s=self._ack_lat.snapshot(),
            batch_events=self._batch_events.snapshot(),
            batch_groups=self._batch_groups.snapshot(),
        )

    def reset_latency(self) -> None:
        """Forget the latency samples (a benchmark does this after ramp-up)."""
        for w in (self._commit_lat, self._ack_lat, self._batch_events, self._batch_groups):
            w.reset()

    # ---- shutdown ------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Reject new appends, drain and commit everything already queued, then close.

        Idempotent; concurrent calls wait for the first to finish.
        """
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            task = self._task
            if task is not None and not task.done():
                await self._queue.put(None)  # FIFO: after everything already queued
                await asyncio.wait([task])
            await self._release_resources()
            self._closed = True

    async def _release_resources(self) -> None:
        loop = asyncio.get_running_loop()
        # Each connection is closed on the thread that owns it.
        if self._wlog is not None:
            await loop.run_in_executor(self._wexec, _close_in_thread, self._wlog)
        if self._rlog is not None:
            await loop.run_in_executor(self._rexec, _close_in_thread, self._rlog)
        self._wexec.shutdown(wait=True)
        self._rexec.shutdown(wait=True)
