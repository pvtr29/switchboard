import asyncio
import json
import time
from typing import Any

import pytest
from conftest import create_job

from switchboard.client.conn import Client
from switchboard.core import Event, EventType
from switchboard.daemon.bus import EventBus, Gap
from switchboard.daemon.protocol import MAX_LINE_BYTES
from switchboard.daemon.server import Daemon

pytestmark = pytest.mark.asyncio

TOTAL_EVENTS = 50_000
BATCH = 250
PACE_S = 0.005  # pause between batches so one shared loop can keep up


def ev(seq: int, job: str = "a") -> Event:
    return Event(seq=seq, job_id=job, epoch=0, ts=0.0, type=EventType.USAGE_RECORDED, payload={})


def usage(job: str, i: int) -> Event:
    return Event(
        job_id=job,
        epoch=0,
        ts=time.time(),
        type=EventType.USAGE_RECORDED,
        payload={"tokens_in": 1, "i": i},
    )


def seqs(batch: list[Event | Gap]) -> list[Any]:
    return [(x.from_seq, x.to_seq) if isinstance(x, Gap) else x.seq for x in batch]


# ---- unit: the queue policy ------------------------------------------------------------------


async def test_full_queue_drops_oldest_into_one_coalesced_gap() -> None:
    bus = EventBus(queue_size=5)
    sub = bus.subscribe()
    bus.publish([ev(i) for i in range(1, 13)])
    assert sub.depth == 5
    assert sub.dropped == 7
    assert seqs(await sub.next_batch()) == [(1, 7), 8, 9, 10, 11, 12]
    bus.publish([ev(i) for i in range(13, 15)])
    assert seqs(await sub.next_batch()) == [13, 14]  # nothing was lost this time: no gap


async def test_gap_widens_while_the_consumer_stays_away() -> None:
    bus = EventBus(queue_size=3)
    sub = bus.subscribe()
    bus.publish([ev(i) for i in range(1, 6)])  # gap 1..2, queue 3,4,5
    assert seqs(await sub.next_batch(limit=2)) == [(1, 2), 3]
    bus.publish([ev(i) for i in range(6, 10)])  # queue 4,5 + 6,7,8,9 -> drops 4,5,6 -> gap 4..6
    assert seqs(await sub.next_batch()) == [(4, 6), 7, 8, 9]


async def test_publish_never_waits_and_a_stalled_subscriber_does_not_slow_others() -> None:
    bus = EventBus(queue_size=10)
    stalled, fast = bus.subscribe(), bus.subscribe()
    for start in range(1, 1001, 10):
        bus.publish([ev(i) for i in range(start, start + 10)])
        batch = await fast.next_batch(limit=100)
        assert all(not isinstance(x, Gap) for x in batch)
    assert stalled.depth == 10
    assert stalled.max_depth == 10
    assert bus.stats.dropped == 990


async def test_job_filter_and_floor() -> None:
    bus = EventBus(queue_size=100)
    only_b = bus.subscribe("b")
    bus.publish([ev(1, "a"), ev(2, "b"), ev(3, "a")])
    assert seqs(await only_b.next_batch()) == [2]
    sub = bus.subscribe(floor=10)
    bus.publish([ev(9), ev(10), ev(11), ev(12)])
    assert seqs(await sub.next_batch()) == [11, 12]  # already delivered by backfill: skipped


async def test_floor_trims_gap_to_what_is_not_yet_delivered() -> None:
    bus = EventBus(queue_size=2)
    sub = bus.subscribe()
    bus.publish([ev(i) for i in range(1, 9)])  # gap 1..6, queue 7,8
    sub.floor = 3
    assert seqs(await sub.next_batch()) == [(4, 6), 7, 8]
    sub.floor = 100
    bus.publish([ev(i) for i in range(101, 106)])
    assert seqs(await sub.next_batch())[0] == (101, 103)


async def test_close_wakes_a_waiting_consumer() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    waiter = asyncio.create_task(sub.next_batch())
    await asyncio.sleep(0)
    sub.close()
    assert await waiter == []
    assert bus.stats.subscribers == 0


# ---- integration: the stalled-client test ------------------------------------------------------


def covered(messages: list[dict[str, Any]]) -> list[int]:
    """Every seq an event or a gap marker accounts for, in delivery order."""
    out: list[int] = []
    for m in messages:
        if "gap" in m:
            out.extend(range(m["gap"]["from_seq"], m["gap"]["to_seq"] + 1))
        else:
            out.append(m["seq"])
    return out


async def test_stalled_client_adds_no_lag_for_others_and_memory_stays_flat(
    daemon: Daemon, capsys: pytest.CaptureFixture[str]
) -> None:
    store = daemon.store
    await create_job(store, "load")
    start_seq = await store.last_seq()

    # A client that subscribes and then never reads again.
    s_reader, s_writer = await asyncio.open_unix_connection(
        str(daemon.socket_path), limit=MAX_LINE_BYTES
    )
    s_writer.write(
        json.dumps({"v": 1, "id": "stall", "cmd": "subscribe", "args": {}}).encode() + b"\n"
    )
    await s_writer.drain()
    assert json.loads(await s_reader.readline())["ok"] is True

    lags: list[float] = []
    fast_seqs: list[int] = []
    gaps_seen = 0

    async with await Client.connect(str(daemon.socket_path)) as fast_client:
        fast = await fast_client.subscribe()

        async def consume() -> None:
            nonlocal gaps_seen
            async for msg in fast:
                if "gap" in msg:
                    gaps_seen += 1
                    continue
                fast_seqs.append(msg["seq"])
                lags.append(time.time() - msg["event"]["ts"])
                if msg["seq"] == start_seq + TOTAL_EVENTS:
                    return

        consumer = asyncio.create_task(consume())
        max_loop_lag = 0.0
        max_depths: list[int] = []

        async def sample() -> None:
            nonlocal max_loop_lag
            while True:
                t0 = time.perf_counter()
                await asyncio.sleep(0.01)
                max_loop_lag = max(max_loop_lag, time.perf_counter() - t0 - 0.01)
                max_depths.append(max(s.depth for s in daemon.bus.subscribers))

        sampler = asyncio.create_task(sample())
        for i in range(0, TOTAL_EVENTS, BATCH):
            await store.append_group([usage("load", j) for j in range(i, i + BATCH)])
            await asyncio.sleep(PACE_S)
        async with asyncio.timeout(30):
            await consumer
        sampler.cancel()
        await asyncio.wait({sampler})

        assert fast_seqs == list(range(start_seq + 1, start_seq + TOTAL_EVENTS + 1))
        assert gaps_seen == 0
        lags.sort()
        p50, p99, worst = lags[len(lags) // 2], lags[int(len(lags) * 0.99)], lags[-1]
        stalled = max(daemon.bus.subscribers, key=lambda s: s.dropped)
        bound = stalled.queue_size
        assert stalled.max_depth <= bound
        assert max(max_depths) <= bound
        assert daemon.bus.stats.max_queue_depth <= bound
        assert stalled.dropped > 0
        assert p99 < 1.0
        assert max_loop_lag < 0.5

        # The stalled client wakes up: its gaps must account for exactly what it missed.
        got: list[dict[str, Any]] = []
        async with asyncio.timeout(30):
            while True:
                m = json.loads(await s_reader.readline())
                got.append(m)
                last = m["gap"]["to_seq"] if "gap" in m else m["seq"]
                if last == start_seq + TOTAL_EVENTS:
                    break
        ids = covered(got)
        assert ids == list(range(start_seq + 1, start_seq + TOTAL_EVENTS + 1))
        n_gap = sum(1 for m in got if "gap" in m)
        assert n_gap >= 1
        s_writer.close()
        await s_writer.wait_closed()
    with capsys.disabled():
        print(
            f"\n[bus] events={TOTAL_EVENTS} fast_lag_p50={p50 * 1000:.1f}ms "
            f"p99={p99 * 1000:.1f}ms max={worst * 1000:.1f}ms "
            f"loop_lag_max={max_loop_lag * 1000:.1f}ms "
            f"stalled_max_queue={stalled.max_depth}/{bound} stalled_dropped={stalled.dropped} "
            f"stalled_gap_markers={n_gap} stalled_events_received={len(got) - n_gap}"
        )


async def test_backfill_while_events_keep_arriving_loses_and_duplicates_nothing(
    daemon: Daemon, client: Client
) -> None:
    store = daemon.store
    await create_job(store, "load")
    for i in range(0, 3000, 500):  # history that exists before the subscription
        await store.append_group([usage("load", j) for j in range(i, i + 500)])

    async def produce() -> None:
        for i in range(3000, 8000, 100):
            await store.append_group([usage("load", j) for j in range(i, i + 100)])
            await asyncio.sleep(0.001)

    producer = asyncio.create_task(produce())
    stream = await client.subscribe(from_seq=1)
    await producer
    last = await store.last_seq()
    got: list[dict[str, Any]] = []
    async with asyncio.timeout(20):
        async for msg in stream:
            got.append(msg)
            if (msg["gap"]["to_seq"] if "gap" in msg else msg["seq"]) == last:
                break
    assert covered(got) == list(range(1, last + 1))  # every seq exactly once, in order
    assert not any("gap" in m for m in got)  # a fast consumer needs no gap here


async def test_subscribe_from_seq_backfills_then_goes_live(daemon: Daemon, client: Client) -> None:
    await create_job(daemon.store, "j")
    await daemon.store.append(usage("j", 0))
    stream = await client.subscribe(job_id="j", from_seq=2)
    first = await asyncio.wait_for(anext(aiter(stream)), 5)
    assert first["seq"] == 2  # backfilled; seq 1 is below from_seq
    await daemon.store.append(usage("j", 1))
    second = await asyncio.wait_for(anext(aiter(stream)), 5)
    assert second["seq"] == 3  # live
