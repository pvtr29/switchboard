"""Concurrency slots: an explicit pool, not a semaphore held across a job's whole task.

Only `provisioning` and `running` jobs hold a slot (docs/plan.md "Concurrency slots"). A job
that pauses gives its slot back and must acquire a new one when it is resumed. A plain
`async with semaphore:` around the entire job task would be wrong: the task would keep the
slot while the job sits in `paused`, so idle jobs would starve active ones. Here the holder
owns a `Slot` token and releases it at the moment it stops being active.

Guarantees:

- FIFO: waiters are admitted in the order they asked. A newcomer never overtakes a waiter,
  even if a slot is momentarily free.
- Exactly-once release: `Slot.release` is idempotent, so `finally` blocks and error paths can
  all call it without double-freeing a slot.
- Cancellation while waiting never leaks: if the grant raced with the cancel (the future
  already carries a slot), the slot is released before CancelledError is re-raised.
"""

import asyncio
import contextlib
from collections import deque


class Slot:
    """A token for one held slot. `release` is idempotent."""

    __slots__ = ("_pool", "_released")

    def __init__(self, pool: "SlotPool") -> None:
        self._pool = pool
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._free()


class SlotPool:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._held = 0
        self._max_seen = 0
        self._waiters: deque[asyncio.Future[Slot]] = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def held(self) -> int:
        return self._held

    @property
    def waiting(self) -> int:
        return sum(1 for f in self._waiters if not f.done())

    @property
    def max_seen(self) -> int:
        return self._max_seen

    def _grant(self) -> Slot:
        self._held += 1
        self._max_seen = max(self._max_seen, self._held)
        return Slot(self)

    def _free(self) -> None:
        self._held -= 1
        while self._waiters and self._held < self._capacity:
            fut = self._waiters.popleft()
            if not fut.done():  # skip waiters that were cancelled
                fut.set_result(self._grant())

    async def acquire(self) -> Slot:
        if self._held < self._capacity and not self._waiters:
            return self._grant()
        fut: asyncio.Future[Slot] = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            return await fut
        except BaseException:
            # Cancelled (or otherwise interrupted) while waiting. If the grant already landed
            # in the future, hand the slot back; otherwise just drop our place in the line.
            if fut.done() and not fut.cancelled():
                fut.result().release()
            else:
                fut.cancel()
                with contextlib.suppress(ValueError):
                    self._waiters.remove(fut)
            raise
