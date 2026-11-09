"""Percentiles over a bounded window of samples.

`percentile` uses the nearest-rank definition: the p-th percentile of n samples is the value at
1-based rank ceil(p/100 * n) of the sorted samples, so it is always an observed value and is
exact on known data (p50 of 1..100 is 50, p99 is 99).

`LatencyWindow` keeps the most recent `size` samples in a ring buffer (memory is bounded
however long the process runs) plus all-time count, sum and max, which a window forgets.
Recording is O(1); sorting happens only when a snapshot is taken.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (0 < p <= 100) of `samples`; 0.0 for an empty sequence."""
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1]


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """Summary of a window. p50/p99/max cover the window; count/mean/max_ever cover all time."""

    count: int  # samples ever recorded
    window: int  # samples the percentiles are computed from
    p50: float
    p99: float
    max: float
    mean: float
    max_ever: float


class LatencyWindow:
    def __init__(self, size: int = 8192) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self._samples: deque[float] = deque(maxlen=size)
        self._count = 0
        self._sum = 0.0
        self._max_ever = 0.0

    def record(self, value: float) -> None:
        self._samples.append(value)
        self._count += 1
        self._sum += value
        if value > self._max_ever:
            self._max_ever = value

    def reset(self) -> None:
        self._samples.clear()
        self._count = 0
        self._sum = 0.0
        self._max_ever = 0.0

    @property
    def count(self) -> int:
        return self._count

    def snapshot(self) -> LatencySnapshot:
        window = list(self._samples)
        window.sort()
        n = len(window)
        if n == 0:
            return LatencySnapshot(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return LatencySnapshot(
            count=self._count,
            window=n,
            p50=window[max(math.ceil(0.50 * n), 1) - 1],
            p99=window[max(math.ceil(0.99 * n), 1) - 1],
            max=window[-1],
            mean=self._sum / self._count,
            max_ever=self._max_ever,
        )
