"""Single-node density benchmark: N FakeAgent jobs in ONE process, real store + supervisor + bus.

Run: uv run python benchmarks/density.py --agents 1000,5000,10000

What it loads (docs/plan.md, success metric "Single-node density"):

- the real `EventStore` (SQLite WAL, group commit, jobs projection) on a temp file;
- the real `Supervisor` with `max_running = N`, so every agent is a live task at once;
- the real `EventBus` fed by the store's commit listener, with subscribed clients that encode every
  event to NDJSON exactly like a daemon connection pump does (`event_message` + `dumps_line`);
- FakeAgent jobs: every token is an `asyncio.sleep` in real time (no virtual time), a step is
  step_started + llm_responded + usage_recorded + step_completed = 4 events, and the token
  interval is chosen so an agent emits ~2 events/s. No environment (DirEnv is measured by
  benchmarks/env_snapshot.py), no sockets (the daemon's socket handling is not the subject).

Per tier: ramp-up (agents are started gradually), `--settle` seconds, then `--duration` seconds of
steady state during which everything is measured. The achieved event rate is MEASURED and
reported, never assumed. Results go to benchmarks/results/density-<timestamp>.json.
"""

import argparse
import asyncio
import cProfile
import gc
import json
import os
import pstats
import resource
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, Job
from switchboard.daemon.bus import EventBus, Gap, Subscription
from switchboard.daemon.protocol import dumps_line, event_line, gap_message
from switchboard.obs import LoopLagMonitor, tune_gc
from switchboard.runtime.supervisor import Supervisor
from switchboard.store import EventStore

RESULTS_DIR = Path(__file__).parent / "results"
AVG_TOKENS = 5.5  # FakeModel streams randint(3, 8) tokens per step
EVENTS_PER_STEP = 4  # step_started, llm_responded, usage_recorded, step_completed


@dataclass
class TierResult:
    agents: int
    label: str
    duration_s: float = 0.0
    events: int = 0
    events_per_s: float = 0.0
    events_per_s_per_agent: float = 0.0
    target_events_per_s: float = 0.0
    lag_p50_ms: float = 0.0
    lag_p99_ms: float = 0.0
    lag_max_ms: float = 0.0
    lag_samples: int = 0
    transactions: int = 0
    mean_batch_events: float = 0.0
    commit_p50_ms: float = 0.0
    commit_p99_ms: float = 0.0
    ack_p50_ms: float = 0.0
    ack_p99_ms: float = 0.0
    writer_thread_busy: float = 0.0
    rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    tasks_alive: int = 0
    threads: int = 0
    bus_published: int = 0
    bus_dropped: int = 0
    queue_depth_max: int = 0
    queue_depth_mean: float = 0.0
    cpu_util: float = 0.0
    loop_cpu_util: float = 0.0
    ramp_s: float = 0.0
    wall_s: float = 0.0
    steps_committed: int = 0
    failed_jobs: int = 0
    subscriber_events: int = 0
    notes: list[str] = field(default_factory=list)


def current_rss_mb() -> float:
    """Resident set size now (not the peak): /proc on Linux, `ps` elsewhere."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
        ).stdout
        return int(out.strip()) / 1024


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes vs KB


async def consume(sub: Subscription, encode: bool, counter: list[int]) -> None:
    """A subscribed client: take batches, serialise like Connection._pump does."""
    while True:
        batch = await sub.next_batch()
        if not batch:
            return
        if encode:
            data = b"".join(
                dumps_line(gap_message("s", item.from_seq, item.to_seq))
                if isinstance(item, Gap)
                else event_line("s", item)
                for item in batch
            )
            counter[1] += len(data)
        counter[0] += len(batch)


async def sample_queue(store: EventStore, out: list[int]) -> None:
    while True:
        out.append(store.stats.queue_depth)
        await asyncio.sleep(0.05)


async def run_tier(
    n: int,
    *,
    duration: float,
    settle: float,
    ramp: float,
    event_rate: float,
    subscribers: int,
    encode: bool,
    label: str,
    profile: bool,
    lag_interval: float,
    freeze_gc: bool = False,
    jitter: float = 0.5,
    linger: float = 0.0,
) -> TierResult:
    # events/s/agent = EVENTS_PER_STEP / (AVG_TOKENS * token_interval)
    token_interval = EVENTS_PER_STEP / (AVG_TOKENS * event_rate)
    res = TierResult(agents=n, label=label, target_events_per_s=n * event_rate, ramp_s=ramp)
    wall0 = time.perf_counter()
    config_hash = FakeAgent(steps=1).config_hash
    with tempfile.TemporaryDirectory(prefix="sb-density-") as tmp:
        store = await EventStore.open(Path(tmp) / "density.db", linger=linger)
        bus = EventBus()
        store.add_listener(bus.publish)

        def agent_factory(job: Job) -> FakeAgent:
            return FakeAgent(steps=10_000_000, token_interval=token_interval, token_jitter=jitter)

        sup = Supervisor(store, agent_factory, None, None, max_running=n)
        ids = [f"d{i:05d}" for i in range(n)]
        for lo in range(0, n, 500):
            await store.append_group(
                [
                    Event(
                        job_id=jid,
                        epoch=0,
                        ts=time.time(),
                        type=EventType.JOB_CREATED,
                        payload={
                            "kind": "fake",
                            "task_ref": "density",
                            "budget": {},
                            "seed": i,
                            "instructions": [],
                            "config_hash": config_hash,
                        },
                    )
                    for i, jid in enumerate(ids[lo : lo + 500], start=lo)
                ]
            )
        counters = [[0, 0] for _ in range(subscribers)]
        subs = [bus.subscribe(None) for _ in range(subscribers)]
        consumers = [
            asyncio.create_task(consume(s, encode, c), name=f"sb-bench-consumer-{i}")
            for i, (s, c) in enumerate(zip(subs, counters, strict=True))
        ]
        lag = LoopLagMonitor(lag_interval, window=int((duration + 5) / lag_interval) + 1000)
        lag.start()
        depths: list[int] = []
        sampler = asyncio.create_task(sample_queue(store, depths), name="sb-bench-queue-sampler")

        # Ramp: start jobs evenly over `ramp` seconds so the first steps are not synchronised.
        t_ramp = time.perf_counter()
        slice_s = 0.02
        per_slice = max(1, round(n * slice_s / ramp)) if ramp > 0 else n
        for lo in range(0, n, per_slice):
            for jid in ids[lo : lo + per_slice]:
                sup.start(jid)
            await asyncio.sleep(slice_s if ramp > 0 else 0)
        await asyncio.sleep(settle)
        res.ramp_s = time.perf_counter() - t_ramp

        # Steady state.
        gc.collect()
        if freeze_gc:
            gc.freeze()
        # gc.collect() blocks the loop: let the monitor take (and we discard) the sample it ruined.
        await asyncio.sleep(lag_interval * 3)
        lag.reset()
        store.reset_latency()
        depths.clear()
        for c in counters:
            c[0] = c[1] = 0
        s0, b0 = store.stats, bus.stats
        cpu0, lcpu0, t0 = time.process_time(), time.thread_time(), time.perf_counter()
        prof = cProfile.Profile() if profile else None
        if prof is not None:
            prof.enable()
        await asyncio.sleep(duration)
        if prof is not None:
            prof.disable()
        t1, cpu1, lcpu1 = time.perf_counter(), time.process_time(), time.thread_time()
        s1, b1 = store.stats, bus.stats
        lagstats = lag.stats()
        lat = store.latency
        res.duration_s = t1 - t0
        res.events = s1.events_written - s0.events_written
        res.events_per_s = res.events / res.duration_s
        res.events_per_s_per_agent = res.events_per_s / n
        res.lag_p50_ms = lagstats.p50_s * 1000
        res.lag_p99_ms = lagstats.p99_s * 1000
        res.lag_max_ms = lagstats.max_s * 1000
        res.lag_samples = lagstats.samples
        res.transactions = s1.transactions - s0.transactions
        res.mean_batch_events = lat.batch_events.mean if lat.batch_events.count else 0.0
        res.commit_p50_ms = lat.commit_s.p50 * 1000
        res.commit_p99_ms = lat.commit_s.p99 * 1000
        res.ack_p50_ms = lat.ack_s.p50 * 1000
        res.ack_p99_ms = lat.ack_s.p99 * 1000
        res.writer_thread_busy = lat.commit_s.mean * lat.commit_s.count / res.duration_s
        res.rss_mb = await asyncio.to_thread(current_rss_mb)
        res.peak_rss_mb = peak_rss_mb()
        res.tasks_alive = len(asyncio.all_tasks())
        res.threads = threading.active_count()
        res.bus_published = b1.published - b0.published
        res.bus_dropped = b1.dropped - b0.dropped
        res.queue_depth_max = max(depths, default=0)
        res.queue_depth_mean = sum(depths) / len(depths) if depths else 0.0
        res.cpu_util = (cpu1 - cpu0) / res.duration_s
        res.loop_cpu_util = (lcpu1 - lcpu0) / res.duration_s
        res.subscriber_events = counters[0][0] if counters else 0
        sstats = sup.stats
        res.failed_jobs = sstats.crashed
        res.steps_committed = res.events // EVENTS_PER_STEP
        if prof is not None:
            st = pstats.Stats(prof)
            st.sort_stats("tottime").print_stats(18)

        # Teardown: nothing is written on cancel (jobs stay `running`; this DB is thrown away).
        sampler.cancel()
        await lag.stop()
        await sup.aclose()
        for s in subs:
            s.close()
        await asyncio.gather(*consumers, sampler, return_exceptions=True)
        await store.aclose()
    res.wall_s = time.perf_counter() - wall0
    return res


def format_table(rows: list[TierResult]) -> str:
    head = (
        f"{'agents':>7} {'ev/s':>8} {'ev/s/ag':>8} | {'lag p50':>8} {'p99':>8} {'max':>8} | "
        f"{'commit p50':>10} {'p99':>8} | {'ack p50':>8} {'p99':>8} | {'txns':>6} {'batch':>6} "
        f"{'wrtr%':>5} | {'RSS MB':>7} {'tasks':>6} {'drop':>6} {'qmax':>6} {'cpu%':>5} "
        f"{'loop%':>5} {'wall':>6}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.agents:>7} {r.events_per_s:>8,.0f} {r.events_per_s_per_agent:>8.2f} | "
            f"{r.lag_p50_ms:>8.2f} {r.lag_p99_ms:>8.2f} {r.lag_max_ms:>8.2f} | "
            f"{r.commit_p50_ms:>10.2f} {r.commit_p99_ms:>8.2f} | "
            f"{r.ack_p50_ms:>8.2f} {r.ack_p99_ms:>8.2f} | "
            f"{r.transactions:>6} {r.mean_batch_events:>6.1f} {r.writer_thread_busy * 100:>5.0f} | "
            f"{r.rss_mb:>7.0f} {r.tasks_alive:>6} {r.bus_dropped:>6} {r.queue_depth_max:>6} "
            f"{r.cpu_util * 100:>5.0f} {r.loop_cpu_util * 100:>5.0f} {r.wall_s:>6.1f}"
        )
    lines.append("(lag/commit/ack in ms; lag = loop wake-up lateness sampled every interval)")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    tiers = [int(x) for x in args.agents.split(",")]
    rows: list[TierResult] = []
    for n in tiers:
        ramp = args.ramp if args.ramp is not None else max(3.0, n / 1000 * 1.0)
        print(f"== tier {n} agents (ramp {ramp:.0f}s, settle {args.settle}s, {args.duration}s) ==")
        row = await run_tier(
            n,
            duration=args.duration,
            settle=args.settle,
            ramp=ramp,
            event_rate=args.event_rate,
            subscribers=args.subscribers,
            encode=not args.no_encode,
            label=args.label,
            profile=args.profile,
            lag_interval=args.lag_interval,
            freeze_gc=args.freeze_gc,
            jitter=args.jitter,
            linger=args.linger,
        )
        rows.append(row)
        print(format_table([row]))
        gc.collect()
    print()
    print(format_table(rows))
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"density-{stamp}.json"
    doc: dict[str, Any] = {
        "benchmark": "density",
        "label": args.label,
        "timestamp": stamp,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cpus": os.cpu_count(),
        "args": vars(args),
        "tiers": [asdict(r) for r in rows],
    }
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agents", default="1000,5000,10000", help="comma-separated tier sizes")
    ap.add_argument("--duration", type=float, default=20.0, help="steady-state seconds per tier")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds after ramp before measuring")
    ap.add_argument(
        "--ramp", type=float, default=None, help="ramp-up seconds (default N/1000, >=3)"
    )
    ap.add_argument("--event-rate", type=float, default=2.0, help="target events/s per agent")
    ap.add_argument("--subscribers", type=int, default=1, help="subscribed clients (all events)")
    ap.add_argument("--no-encode", action="store_true", help="subscribers do not serialise events")
    ap.add_argument("--lag-interval", type=float, default=0.02, help="lag monitor interval (s)")
    ap.add_argument(
        "--switch-interval", type=float, default=None, help="sys.setswitchinterval (seconds)"
    )
    ap.add_argument(
        "--gc-threshold",
        default=None,
        help="gc.set_threshold, e.g. 50000,20,100 (default: the daemon's tuning, obs.tuning)",
    )
    ap.add_argument("--no-gc-tuning", action="store_true", help="CPython's default GC thresholds")
    ap.add_argument("--freeze-gc", action="store_true", help="gc.freeze() before measuring")
    ap.add_argument("--jitter", type=float, default=0.5, help="token pause jitter fraction")
    ap.add_argument(
        "--linger",
        type=float,
        default=0.002,
        help="EventStore group-commit linger (s), as sb daemon",
    )
    ap.add_argument("--label", default="run", help="free-text label stored in the result")
    ap.add_argument(
        "--profile", action="store_true", help="cProfile the loop thread (steady state)"
    )
    parsed = ap.parse_args()
    if parsed.switch_interval is not None:
        sys.setswitchinterval(parsed.switch_interval)
    if parsed.gc_threshold is not None:
        gc.set_threshold(*[int(x) for x in parsed.gc_threshold.split(",")])
    elif not parsed.no_gc_tuning:
        tune_gc()  # what `sb daemon` does by default
    asyncio.run(main(parsed))
