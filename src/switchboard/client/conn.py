"""Async client for the daemon socket (protocol: see daemon/protocol.py).

One reader task per client (owned by it, cancelled in `aclose`) reads lines and routes them:
a response to the future of the request with the same id, a stream message to the queue of its
subscription. Each subscription queue is BOUNDED (`STREAM_QUEUE` messages) and when it is full
the reader task waits, which stops reading the socket, which makes the daemon's per-subscriber
bus queue overflow into a gap marker: a slow consumer loses events visibly instead of growing
memory here. (The same wait also delays responses on that connection; use a second connection
for requests if a stream consumer is slow.)
"""

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

from switchboard.daemon.protocol import MAX_LINE_BYTES, PROTOCOL_VERSION, dumps_line

STREAM_QUEUE = 10_000


class ClientError(Exception):
    """The daemon answered with an error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ConnectionLost(ConnectionError):
    """The connection to the daemon ended before the request was answered."""


class Stream:
    """The messages (event or gap dicts) of one subscription, as an async iterator.

    Iteration ends when the connection closes or the subscription is cancelled.
    """

    def __init__(self, client: "Client", sub_id: str) -> None:
        self.id = sub_id
        self._client = client
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=STREAM_QUEUE)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, Any]]:
        while (msg := await self._queue.get()) is not None:
            yield msg

    async def unsubscribe(self) -> None:
        await self._client.request("unsubscribe", subscription=self.id)
        self._client._streams.pop(self.id, None)
        self.end_now()

    def end_now(self) -> None:
        """Put the end marker without waiting: if the queue is full the oldest message goes."""
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)


class Client:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._streams: dict[str, Stream] = {}
        self._write_lock = asyncio.Lock()
        self._lost: ConnectionLost | None = None
        self._closing = False
        self._task = asyncio.create_task(self._read_loop(), name="sb-client-reader")

    @classmethod
    async def connect(cls, socket_path: str) -> Self:
        reader, writer = await asyncio.open_unix_connection(socket_path, limit=MAX_LINE_BYTES)
        return cls(reader, writer)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _read_loop(self) -> None:
        try:
            while line := await self._reader.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if "ok" in msg:
                    fut = self._pending.pop(msg_id, None) if isinstance(msg_id, str) else None
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                elif isinstance(msg_id, str) and (stream := self._streams.get(msg_id)):
                    await stream._queue.put(msg)  # full: wait (see the module docstring)
        except (OSError, ValueError):  # reset, or a line longer than MAX_LINE_BYTES
            pass
        finally:
            self._lost = ConnectionLost("connection to the daemon closed")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(self._lost)
            self._pending.clear()
            for stream in list(self._streams.values()):
                if self._closing:
                    stream.end_now()  # aclose must not wait for a consumer that left
                else:
                    await stream._queue.put(None)  # keep every message, wait for the consumer

    async def _call(self, cmd: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self._lost is not None:
            raise self._lost
        req_id = str(next(self._ids))
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            async with self._write_lock:
                self._writer.write(
                    dumps_line({"v": PROTOCOL_VERSION, "id": req_id, "cmd": cmd, "args": args})
                )
                await self._writer.drain()
            return req_id, await fut
        except OSError as exc:
            raise ConnectionLost(str(exc)) from exc
        finally:
            self._pending.pop(req_id, None)

    async def request(self, cmd: str, **args: Any) -> dict[str, Any]:
        """Send one command and return its `result`; raises ClientError on an error response."""
        _, msg = await self._call(cmd, args)
        if msg["ok"]:
            result: dict[str, Any] = msg["result"]
            return result
        err = msg["error"]
        raise ClientError(err["code"], err["message"])

    async def subscribe(self, job_id: str | None = None, from_seq: int | None = None) -> Stream:
        """Start a subscription. The stream is registered before the request is sent so no
        message can arrive for an unknown id."""
        args: dict[str, Any] = {}
        if job_id is not None:
            args["job_id"] = job_id
        if from_seq is not None:
            args["from_seq"] = from_seq
        req_id = str(next(self._ids))
        stream = Stream(self, req_id)
        self._streams[req_id] = stream
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            if self._lost is not None:
                raise self._lost
            async with self._write_lock:
                self._writer.write(
                    dumps_line(
                        {"v": PROTOCOL_VERSION, "id": req_id, "cmd": "subscribe", "args": args}
                    )
                )
                await self._writer.drain()
            msg = await fut
            if not msg["ok"]:
                raise ClientError(msg["error"]["code"], msg["error"]["message"])
        except BaseException:
            self._streams.pop(req_id, None)
            raise
        finally:
            self._pending.pop(req_id, None)
        return stream

    async def aclose(self) -> None:
        self._closing = True
        self._task.cancel()
        await asyncio.wait({self._task})
        self._writer.close()
        with contextlib.suppress(OSError):
            await self._writer.wait_closed()
