"""Event store throughput: concurrent producers appending through the group-commit writer.

Run: uv run python benchmarks/store_throughput.py
"""

import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from switchboard.core import Event, EventType
from switchboard.store import EventStore

CONFIGS = [
    # (producers, events each, max_batch)
    (1, 2_000, 1),
    (1, 2_000, 512),
    (10, 500, 1),
    (10, 500, 512),
    (200, 50, 1),
    (200, 50, 512),
]


async def run(producers: int, per_producer: int, max_batch: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with await EventStore.open(Path(tmp) / "bench.db", max_batch=max_batch) as store:
            latencies: list[float] = []
            # Every event needs its job to exist (projection), so create them untimed first.
            await store.append_group(
                [
                    Event(
                        job_id=f"job{i}",
                        epoch=0,
                        step_id=None,
                        ts=0.0,
                        type=EventType.JOB_CREATED,
                        payload={"kind": "fake", "task_ref": "bench", "budget": {}, "seed": i},
                    )
                    for i in range(producers)
                ]
            )
            base = store.stats

            async def producer(i: int) -> None:
                for n in range(per_producer):
                    event = Event(
                        job_id=f"job{i}",
                        epoch=0,
                        step_id=None,
                        ts=0.0,
                        type=EventType.USAGE_RECORDED,
                        payload={"n": n, "tokens": 128},
                    )
                    t0 = time.perf_counter()
                    await store.append(event)
                    latencies.append(time.perf_counter() - t0)

            start = time.perf_counter()
            await asyncio.gather(*(producer(i) for i in range(producers)))
            elapsed = time.perf_counter() - start
            st = store.stats
    events = st.events_written - base.events_written
    txns = st.transactions - base.transactions
    q = statistics.quantiles(latencies, n=100)
    print(
        f"{producers:>4} producers  max_batch={max_batch:>3}  "
        f"{events / elapsed:>9,.0f} events/s  txns={txns:>6}  "
        f"mean_batch={events / txns:>6.1f}  "
        f"p50={q[49] * 1000:>7.2f}ms  p99={q[98] * 1000:>7.2f}ms"
    )


async def main() -> None:
    for config in CONFIGS:
        await run(*config)


if __name__ == "__main__":
    asyncio.run(main())
