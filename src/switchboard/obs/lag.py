"""Event-loop lag monitor.

A background task sleeps `interval` seconds and records how late it woke up
(`actual - expected`, never negative). Anything that blocks the loop (a stray `time.sleep`, a
huge synchronous computation, GIL contention with a worker thread) delays that wake-up by the
same amount, so the samples are a direct measure of how long other tasks had to wait.

The samples live in a bounded ring (`window` most recent samples, 600 by default = one minute
at the default interval). p50/p99/max cover the window; `max_ever` and `samples` cover the whole
life of the monitor. The monitor owns its task (strong reference, cancelled and awaited by
`stop`). It uses the loop's clock, so it also works, trivially, under a virtual-time loop.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from switchboard.obs.stats import LatencyWindow

DEFAULT_INTERVAL_S = 0.1
DEFAULT_WINDOW = 600


@dataclass(frozen=True, slots=True)
class LagStats:
    interval_s: float
    samples: int  # total samples taken
    window: int  # samples the percentiles cover
    p50_s: float
    p99_s: float
    max_s: float  # largest lag in the window
    max_ever_s: float  # largest lag since start (or reset)


class LoopLagMonitor:
    def __init__(self, interval: float = DEFAULT_INTERVAL_S, window: int = DEFAULT_WINDOW) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = interval
        self._window = LatencyWindow(window)
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start sampling on the running loop. Idempotent."""
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="sb-loop-lag")

    async def stop(self) -> None:
        """Cancel the sampler and wait for it. Idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        interval = self.interval
        record = self._window.record
        while True:
            expected = loop.time() + interval
            await asyncio.sleep(interval)
            record(max(loop.time() - expected, 0.0))

    def reset(self) -> None:
        """Forget all samples (e.g. after a benchmark's ramp-up)."""
        self._window.reset()

    def stats(self) -> LagStats:
        s = self._window.snapshot()
        return LagStats(
            interval_s=self.interval,
            samples=s.count,
            window=s.window,
            p50_s=s.p50,
            p99_s=s.p99,
            max_s=s.max,
            max_ever_s=s.max_ever,
        )

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
