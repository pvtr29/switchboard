"""Process tuning for a daemon that runs thousands of jobs on one event loop.

Garbage collection: CPython's default generation-0 threshold (700 allocations) means a
collection every few hundred events once thousands of tasks and projections are alive, and each
older-generation collection walks the whole heap while the loop is stopped. In the density
benchmark (benchmarks/density.py, 5000 agents) the default thresholds gave an event-loop lag p99
of ~270 ms; (100000, 50, 1000) gave ~40 ms at the same event rate. The price is up to 100k
young container objects (tens of MB) waiting for collection.
"""

import gc

GC_THRESHOLD: tuple[int, int, int] = (100_000, 50, 1000)


def tune_gc(threshold: tuple[int, int, int] = GC_THRESHOLD) -> None:
    gc.set_threshold(*threshold)
