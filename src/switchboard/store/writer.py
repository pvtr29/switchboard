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
WAL write instead of one per caller), then resolves each future. A group is never split, so
``append_group`` is atomic. A group larger than ``max_batch`` is committed whole on its own.
Acks are sent only after COMMIT returned.

Backpressure: the queue is bounded; ``append`` awaits ``queue.put`` when it is full, so
producers slow down instead of memory growing.

Cancellation: a caller cancelled while awaiting its ack may still have its events written
(they are already queued); only the ack is lost. The writer skips cancelled futures.
"""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from switchboard.core import Event
from switchboard.store.interface import AppendResult, AppendStatus
from switchboard.store.sqlite import SQLiteEventLog, encode_event, open_db


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
    max_batch_seen: int
    queue_depth: int


type _Item = tuple[list[Event], asyncio.Future[list[AppendResult]]]


def _close_in_thread(log: SQLiteEventLog) -> None:
    log.close()


class EventStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_batch: int = 512,
        queue_size: int = 10_000,
    ) -> None:
        if max_batch < 1 or queue_size < 1:
            raise ValueError("max_batch and queue_size must be >= 1")
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
        self._max_batch_seen = 0
        self._close_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls, path: str | Path, *, max_batch: int = 512, queue_size: int = 10_000
    ) -> Self:
        store = cls(path, max_batch=max_batch, queue_size=queue_size)
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

        Raises ValueError (for this caller only) if any event is invalid, StoreClosed after
        close, WriterCrashed if the writer died.
        """
        group = list(events)
        for e in group:
            encode_event(e)  # same rules as the log; fail before the group can poison a batch
        if not group:
            return []
        self._check_open()
        fut: asyncio.Future[list[AppendResult]] = asyncio.get_running_loop().create_future()
        await self._queue.put((group, fut))  # backpressure: waits while the queue is full
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
            count = len(first[0])
            stop = False
            while count < self._max_batch:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                    break
                if count + len(item[0]) > self._max_batch:
                    self._carry = item
                    break
                batch.append(item)
                count += len(item[0])
            self._inflight = batch
            flat = [e for group, _ in batch for e in group]
            try:
                results = await loop.run_in_executor(self._wexec, self._log.append_many, flat)
            except Exception as exc:
                # e.g. disk error: the transaction rolled back, so nothing in it was written.
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                self._inflight = []
                if stop:
                    return
                continue
            # An invariant violation here is a bug: let it kill the writer (see the callback).
            if len(results) != len(flat):
                raise RuntimeError(f"append_many returned {len(results)} results for {len(flat)}")
            self._transactions += 1
            self._events_written += sum(r.status is AppendStatus.INSERTED for r in results)
            self._duplicates += sum(r.status is AppendStatus.DUPLICATE for r in results)
            self._max_batch_seen = max(self._max_batch_seen, count)
            pos = 0
            for group, fut in batch:
                part = results[pos : pos + len(group)]
                pos += len(group)
                if not fut.done():  # a cancelled caller no longer wants its ack
                    fut.set_result(part)
            self._inflight = []
            if stop:
                return

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
        for _, fut in pending:
            if not fut.done():
                fut.set_exception(failure)

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

    @property
    def stats(self) -> StoreStats:
        return StoreStats(
            transactions=self._transactions,
            events_written=self._events_written,
            duplicates=self._duplicates,
            max_batch_seen=self._max_batch_seen,
            queue_depth=self._queue.qsize(),
        )

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
