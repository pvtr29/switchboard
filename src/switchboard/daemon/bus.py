"""Event bus: fan committed events out to subscribers without ever waiting for one.

Every subscriber owns ONE bounded queue (`queue_size`, default 1000 events). Full policy:
DROP THE OLDEST queued event and remember its seq in a gap marker `[from_seq, to_seq]` that is
delivered before everything still queued. Dropped events are always older than the queued ones,
so the marker is a single range at the head of the queue and consecutive drops simply widen it
(coalescing): the queue never holds more than `queue_size` events plus one marker, however far
behind the consumer is. `publish` is synchronous, never awaits and costs O(subscribers) per
event, so a stalled subscriber cannot slow the producer or other subscribers down.

A subscriber that lost events can fetch them again from the store (`subscribe` with
`from_seq`), which is what the gap marker tells it to do.

`floor` (set by backfill, see server.py) is the highest seq the subscriber already received by
another route: queued events at or below it are skipped on delivery and gaps are trimmed to
what lies above it. That is how backfill from the store switches to the live queue without
losing or duplicating an event.
"""

import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from switchboard.core import Event

DEFAULT_QUEUE_SIZE = 1000


@dataclass(frozen=True, slots=True)
class Gap:
    from_seq: int
    to_seq: int


@dataclass(frozen=True, slots=True)
class BusStats:
    subscribers: int
    published: int
    dropped: int
    max_queue_depth: int


class Subscription:
    def __init__(self, bus: "EventBus", job_id: str | None, queue_size: int) -> None:
        self._bus = bus
        self.job_id = job_id
        self.queue_size = queue_size
        self.floor = 0
        self.dropped = 0
        self.max_depth = 0
        self._events: deque[Event] = deque()
        self._gap: tuple[int, int] | None = None
        self._wake = asyncio.Event()
        self._closed = False

    @property
    def depth(self) -> int:
        """Events queued right now (the gap marker, if any, is not counted)."""
        return len(self._events)

    def offer(self, event: Event) -> None:
        """Queue `event`, dropping the oldest into the gap marker if full. Never awaits."""
        if self._closed or (self.job_id is not None and event.job_id != self.job_id):
            return
        assert event.seq is not None
        if len(self._events) >= self.queue_size:
            oldest = self._events.popleft()
            assert oldest.seq is not None
            lo = self._gap[0] if self._gap else oldest.seq
            self._gap = (lo, oldest.seq)
            self.dropped += 1
            self._bus._dropped += 1
        self._events.append(event)
        if len(self._events) > self.max_depth:
            self.max_depth = len(self._events)
            if self.max_depth > self._bus._max_depth:
                self._bus._max_depth = self.max_depth
        self._wake.set()

    def _take(self, limit: int) -> list[Event | Gap]:
        out: list[Event | Gap] = []
        if self._gap is not None:
            lo, hi = self._gap
            self._gap = None
            lo = max(lo, self.floor + 1)
            if lo <= hi:
                out.append(Gap(lo, hi))
        while self._events and len(out) < limit:
            e = self._events.popleft()
            assert e.seq is not None
            if e.seq > self.floor:
                out.append(e)
        return out

    async def next_batch(self, limit: int = 256) -> list[Event | Gap]:
        """Up to `limit` items (a gap first, then events) in order; [] once closed."""
        while True:
            out = self._take(limit)
            if out:
                return out
            if self._closed:
                return []
            self._wake.clear()
            await self._wake.wait()

    def close(self) -> None:
        self._closed = True
        self._bus._subs.discard(self)
        self._events.clear()
        self._gap = None
        self._wake.set()


class EventBus:
    def __init__(self, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self._queue_size = queue_size
        self._subs: set[Subscription] = set()
        self._published = 0
        self._dropped = 0
        self._max_depth = 0

    def subscribe(self, job_id: str | None = None, *, floor: int = 0) -> Subscription:
        sub = Subscription(self, job_id, self._queue_size)
        sub.floor = floor
        self._subs.add(sub)
        return sub

    def publish(self, events: Sequence[Event]) -> None:
        """Offer committed events (in seq order) to every subscriber. Never awaits."""
        self._published += len(events)
        subs = tuple(self._subs)
        for event in events:
            for sub in subs:
                sub.offer(event)

    @property
    def subscribers(self) -> tuple[Subscription, ...]:
        return tuple(self._subs)

    def close(self) -> None:
        for sub in tuple(self._subs):
            sub.close()

    @property
    def stats(self) -> BusStats:
        return BusStats(len(self._subs), self._published, self._dropped, self._max_depth)
