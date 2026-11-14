"""Garbage collection driven by the event loop instead of by allocation counts.

Why: CPython's cyclic collector stops every thread while it runs. With 10k agents in one
process the automatic schedule caused the event-loop stalls (reports/density-10k.md):

- A young (generation 0) collection is triggered when allocations minus deallocations since the
  last one exceed a threshold, and it walks every tracked object allocated since. Each event
  replaces a job projection whose previous version had usually been promoted already, and
  freeing an OLD object decrements the young count too, so the young list grew far past the
  threshold before a collection fired: 50-240 ms pauses, at any threshold tried.
- Middle and old collections walk everything that survived, which at this scale is mostly
  conversation history and job projections: data that reference counting frees and that never
  forms cycles. Walking it again and again found nothing (5 cyclic objects in a 40 s run).

What this does instead (`GcScheduler`, one task on the loop):

1. Automatic collection is turned off (`gc.disable()`).
2. Every `young_s` the loop runs `gc.collect(0)`: the young list only holds what was allocated in
   that window, so the pause is short and it always falls BETWEEN callbacks, never inside a
   store commit on the writer thread.
3. Right after, `gc.freeze()` moves the survivors to the permanent generation, so no later
   collection walks them again. Reference counting still frees frozen objects as usual; only
   the cycle detector ignores them.
4. Every `full_every_s` a full sweep (`gc.unfreeze()`, `gc.collect()`, `gc.freeze()`) finds any
   cycle that formed among frozen objects (e.g. a task that ended with an exception). That
   sweep walks the whole heap: about 0.5 s with 10k agents, so it is rare, and its duration is
   reported.

`stop()` unfreezes everything and restores automatic collection as it was.
Measured (benchmarks/density.py, 10k agents): loop lag p99 ~90 ms -> ~6-10 ms, max ~120 ms ->
~10-12 ms, and events/s up from ~18.2k to ~19.4k because no collection walks the heap any more.
"""

import asyncio
import gc
import logging
import time
from dataclasses import dataclass

from switchboard.obs.stats import LatencySnapshot, LatencyWindow

logger = logging.getLogger("switchboard.gc")

DEFAULT_YOUNG_S = 0.05
DEFAULT_FULL_EVERY_S = 600.0


@dataclass(frozen=True, slots=True)
class GcStats:
    young: LatencySnapshot  # pause per scheduled young collection (seconds)
    full_sweeps: int
    last_full_s: float  # pause of the most recent full sweep
    frozen: int  # objects in the permanent generation now
    garbage_found: int  # unreachable objects the collections found, all time


class GcScheduler:
    def __init__(
        self, young_s: float = DEFAULT_YOUNG_S, full_every_s: float = DEFAULT_FULL_EVERY_S
    ) -> None:
        if young_s <= 0 or full_every_s <= 0:
            raise ValueError("young_s and full_every_s must be > 0")
        self.young_s = young_s
        self.full_every_s = full_every_s
        self._young = LatencyWindow(8192)
        self._full_sweeps = 0
        self._last_full_s = 0.0
        self._garbage = 0
        self._task: asyncio.Task[None] | None = None
        self._was_enabled = True

    def start(self) -> None:
        """Take over collection; call from the running loop. Idempotent."""
        if self._task is not None:
            return
        self._was_enabled = gc.isenabled()
        gc.disable()
        self._task = asyncio.get_running_loop().create_task(self._run(), name="sb-gc")

    async def stop(self) -> None:
        """Stop the task, unfreeze, and restore automatic collection. Idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        gc.unfreeze()
        if self._was_enabled:
            gc.enable()

    def stats(self) -> GcStats:
        return GcStats(
            young=self._young.snapshot(),
            full_sweeps=self._full_sweeps,
            last_full_s=self._last_full_s,
            frozen=gc.get_freeze_count(),
            garbage_found=self._garbage,
        )

    def collect_young(self) -> None:
        t = time.perf_counter()
        self._garbage += gc.collect(0)
        gc.freeze()
        self._young.record(time.perf_counter() - t)

    def sweep(self) -> None:
        t = time.perf_counter()
        gc.unfreeze()
        found = gc.collect()
        gc.freeze()
        self._garbage += found
        self._full_sweeps += 1
        self._last_full_s = time.perf_counter() - t
        logger.info(
            "gc full sweep found=%d pause_s=%.3f frozen=%d",
            found,
            self._last_full_s,
            gc.get_freeze_count(),
        )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        last_full = loop.time()
        while True:
            await asyncio.sleep(self.young_s)
            self.collect_young()
            if loop.time() - last_full >= self.full_every_s:
                self.sweep()
                last_full = loop.time()


__all__ = ["DEFAULT_FULL_EVERY_S", "DEFAULT_YOUNG_S", "GcScheduler", "GcStats"]
