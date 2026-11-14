"""Soak benchmark: hours of continuous job churn in ONE process, sampled every minute.

Run: uv run python benchmarks/soak.py --hours 24 --agents 1000 --out reports/soak

What it loads: the real `EventStore` (SQLite WAL), `Supervisor`, `EventBus` with one encoding
subscriber, `GcScheduler` and `LoopLagMonitor`, i.e. what `sb daemon` runs, minus sockets.
Workload: `--agents` FakeAgent jobs are kept alive at all times. Each job runs a random number
of steps (`--min-steps`..`--max-steps`) and finishes (-> verifying); a refill loop starts a new job
for every one that finished, so jobs, tasks and their objects are created and released all day.
Every 5th job emits an external effect per step through an in-memory sink.

Per sample (JSON lines in <out>/soak-<stamp>.jsonl): uptime, RSS, open file descriptors,
threads, asyncio tasks, child processes, jobs started/finished/crashed, events written and the
event rate over the last interval, loop lag p50/p99/max, store ack p99, GC young p99, full
sweeps, frozen objects, and DB size. The summary at the end (and `--report` on an existing file)
checks the plan's soak targets (docs/plan.md "24-hour soak"):

- RSS growth < 5% between the first sample after warm-up and the last one;
- no leaked tasks, file descriptors, threads or child processes (last vs first after warm-up).

A free-disk guard stops the run early (and says so) if the disk gets below `--min-free-gb`.
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, Job
from switchboard.daemon.bus import EventBus, Gap, Subscription
from switchboard.daemon.protocol import dumps_line, event_line, gap_message
from switchboard.obs import GcScheduler, LoopLagMonitor
from switchboard.runtime.sinks import FakeExternalSink
from switchboard.runtime.supervisor import Supervisor
from switchboard.store import EventStore

AVG_TOKENS = 5.5
EVENTS_PER_STEP = 4


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True
        ).stdout
        return int(out.strip()) / 1024


def open_fds() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(d))
        except OSError:
            continue
    return -1


def children() -> int:
    out = subprocess.run(
        ["pgrep", "-P", str(os.getpid())], capture_output=True, text=True, check=False
    ).stdout
    return len([x for x in out.split() if x.strip()]) - 1  # minus the pgrep itself


def system_sample(db: Path) -> dict[str, Any]:
    """The blocking parts of a sample (ps, pgrep, stat, statvfs); run in a thread."""
    return {
        "rss_mb": round(rss_mb(), 1),
        "fds": open_fds(),
        "children": children(),
        "db_mb": round(db.stat().st_size / 1e6, 1) if db.exists() else 0.0,
        "free_gb": round(shutil.disk_usage(db.parent).free / 1e9, 1),
    }


def append_line(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


@dataclass
class Counters:
    started: int = 0
    seq: int = 0


async def consume(sub: Subscription, out: list[int]) -> None:
    while True:
        batch = await sub.next_batch()
        if not batch:
            return
        data = b"".join(
            dumps_line(gap_message("s", i.from_seq, i.to_seq))
            if isinstance(i, Gap)
            else event_line("s", i)
            for i in batch
        )
        out[0] += len(batch)
        out[1] += len(data)


async def soak(args: argparse.Namespace, out: Path, db: Path) -> Path:
    token_interval = EVENTS_PER_STEP / (AVG_TOKENS * args.event_rate)
    rng = random.Random(args.seed)
    config_hash = FakeAgent(steps=1).config_hash
    counters = Counters()
    steps_of: dict[str, int] = {}

    gcs = GcScheduler(full_every_s=args.full_sweep_s)
    gcs.start()
    lag = LoopLagMonitor(0.05, window=4096)
    lag.start()
    store = await EventStore.open(db, linger=0.002)
    bus = EventBus()
    store.add_listener(bus.publish)
    sub_counts = [0, 0]
    sub = bus.subscribe(None)
    consumer = asyncio.create_task(consume(sub, sub_counts), name="soak-consumer")
    sink = FakeExternalSink()

    def agent_factory(job: Job) -> FakeAgent:
        n = int(job.task_ref.split(":")[1])
        return FakeAgent(
            steps=steps_of.pop(job.job_id, 10),
            token_interval=token_interval,
            token_jitter=0.5,
            effect_every=1 if n % 5 == 0 else 0,
        )

    sup = Supervisor(
        store, agent_factory, None, lambda job: sink, max_running=args.agents, worker_id="soak"
    )

    async def start_jobs(k: int) -> None:
        events = []
        ids = []
        for _ in range(k):
            n = counters.started
            counters.started += 1
            jid = f"s{n:09d}"
            steps_of[jid] = rng.randint(args.min_steps, args.max_steps)
            ids.append(jid)
            events.append(
                Event(
                    job_id=jid,
                    epoch=0,
                    ts=time.time(),
                    type=EventType.JOB_CREATED,
                    payload={
                        "kind": "fake",
                        "task_ref": f"soak:{n}",
                        "budget": {},
                        "seed": n,
                        "instructions": [],
                        "config_hash": config_hash,
                    },
                )
            )
        for lo in range(0, len(events), 500):
            await store.append_group(events[lo : lo + 500])
        for jid in ids:
            sup.start(jid)

    # Ramp up gradually so first steps are not synchronised.
    ramp_slices = max(1, int(args.ramp_s / 0.1))
    per = max(1, args.agents // ramp_slices)
    while counters.started < args.agents:
        await start_jobs(min(per, args.agents - counters.started))
        await asyncio.sleep(0.1)

    t0 = time.monotonic()
    deadline = t0 + args.hours * 3600
    last_sample = t0 - args.interval_s  # sample immediately
    prev_events, prev_t = store.stats.events_written, t0
    stopped_reason = "duration reached"
    print(f"soak: {args.agents} agents, {args.hours} h, db {db}, samples -> {out}", flush=True)
    while True:
        await asyncio.sleep(1.0)
        # Refill: one new job for every job no longer running.
        live = sup.stats.running + sup.stats.waiting
        if live < args.agents:
            await start_jobs(args.agents - live)
        now = time.monotonic()
        if now - last_sample >= args.interval_s:
            last_sample = now
            st, ls, gst = store.stats, lag.stats(), gcs.stats()
            ev = st.events_written
            rate = (ev - prev_events) / max(now - prev_t, 1e-9)
            prev_events, prev_t = ev, now
            lag.reset()
            sstats = sup.stats
            sysinfo = await asyncio.to_thread(system_sample, db)
            sample: dict[str, Any] = {
                "t_s": round(now - t0, 1),
                **sysinfo,
                "threads": threading.active_count(),
                "tasks": len(asyncio.all_tasks()),
                "jobs_started": counters.started,
                "jobs_finished": sstats.finished,
                "jobs_crashed": sstats.crashed,
                "events": ev,
                "events_per_s": round(rate, 1),
                "lag_p50_ms": round(ls.p50_s * 1000, 2),
                "lag_p99_ms": round(ls.p99_s * 1000, 2),
                "lag_max_ms": round(ls.max_s * 1000, 2),
                "ack_p99_ms": round(store.latency.ack_s.p99 * 1000, 2),
                "gc_young_p99_ms": round(gst.young.p99 * 1000, 2),
                "gc_full_sweeps": gst.full_sweeps,
                "gc_last_full_ms": round(gst.last_full_s * 1000, 1),
                "gc_frozen": gst.frozen,
                "gc_garbage_found": gst.garbage_found,
                "sink_applications": sink.counts().applications,
                "sink_duplicates": sink.counts().duplicates,
                "bus_dropped": bus.stats.dropped,
            }
            await asyncio.to_thread(append_line, out, sample)
            print(
                f"[{sample['t_s'] / 3600:6.2f} h] rss={sample['rss_mb']}MB "
                f"fds={sample['fds']} tasks={sample['tasks']} ev/s={sample['events_per_s']} "
                f"lag p99={sample['lag_p99_ms']}ms jobs={sample['jobs_finished']} "
                f"db={sample['db_mb']}MB",
                flush=True,
            )
            if sample["free_gb"] < args.min_free_gb:
                stopped_reason = f"free disk below {args.min_free_gb} GB"
                break
        if now >= deadline:
            break

    await lag.stop()
    await sup.aclose()
    sub.close()
    await asyncio.gather(consumer, return_exceptions=True)
    await store.aclose()
    await gcs.stop()
    await asyncio.to_thread(append_line, out, {"end": stopped_reason, "db": str(db)})
    print(f"soak ended: {stopped_reason}", flush=True)
    return out


def report(path: Path, warmup_s: float) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    end = next((r["end"] for r in rows if "end" in r), "still running (or killed)")
    samples = [r for r in rows if "t_s" in r]
    after = [s for s in samples if s["t_s"] >= warmup_s] or samples
    if not after:
        return {"file": str(path), "samples": 0}
    first, last = after[0], after[-1]
    growth = (last["rss_mb"] - first["rss_mb"]) / first["rss_mb"] * 100

    def delta(k: str) -> int:
        return int(last[k] - first[k])

    res = {
        "file": str(path),
        "end": end,
        "hours": round(last["t_s"] / 3600, 2),
        "samples": len(samples),
        "rss_first_mb": first["rss_mb"],
        "rss_last_mb": last["rss_mb"],
        "rss_max_mb": max(s["rss_mb"] for s in after),
        "rss_growth_pct": round(growth, 2),
        "fds_delta": delta("fds"),
        "threads_delta": delta("threads"),
        "tasks_delta": delta("tasks"),
        "children_delta": delta("children"),
        "jobs_finished": last["jobs_finished"],
        "jobs_crashed": last["jobs_crashed"],
        "sink_duplicates": last["sink_duplicates"],
        "lag_p99_ms_worst_sample": max(s["lag_p99_ms"] for s in after),
        "lag_max_ms_worst_sample": max(s["lag_max_ms"] for s in after),
        "events_per_s_mean": round(sum(s["events_per_s"] for s in after) / len(after), 1),
        "gc_full_sweeps": last["gc_full_sweeps"],
        "gc_last_full_ms": last["gc_last_full_ms"],
        "targets": {
            "rss_growth_below_5pct": growth < 5,
            "no_leaked_fds": delta("fds") <= 0,
            "no_leaked_threads": delta("threads") <= 0,
            "no_leaked_tasks": delta("tasks") <= 2,  # sampling jitter: refill in flight
            "no_leaked_children": delta("children") <= 0,
            "no_crashed_jobs": last["jobs_crashed"] == 0,
            "no_duplicate_effects": last["sink_duplicates"] == 0,
        },
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--agents", type=int, default=1000, help="jobs kept running at all times")
    ap.add_argument("--event-rate", type=float, default=0.5, help="target events/s per agent")
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--interval-s", type=float, default=60.0, help="seconds between samples")
    ap.add_argument("--ramp-s", type=float, default=10.0)
    ap.add_argument("--full-sweep-s", type=float, default=600.0, help="GcScheduler full sweep")
    ap.add_argument("--warmup-s", type=float, default=3600.0, help="ignored by the RSS check")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/soak")
    ap.add_argument("--db-dir", default=None, help="where the soak DB lives (default: a temp dir)")
    ap.add_argument("--keep-db", action="store_true")
    ap.add_argument("--report", metavar="JSONL", help="only summarise an existing soak file")
    a = ap.parse_args()
    if a.report:
        print(json.dumps(report(Path(a.report), a.warmup_s), indent=2))
        sys.exit(0)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    db_dir = Path(a.db_dir) if a.db_dir else Path(tempfile.mkdtemp(prefix="sb-soak-"))
    db_file = db_dir / f"soak-{stamp}.db"
    path = asyncio.run(soak(a, out_dir / f"soak-{stamp}.jsonl", db_file))
    if not a.keep_db:
        for p in db_dir.glob(f"{db_file.name}*"):
            p.unlink()
    print(json.dumps(report(path, a.warmup_s), indent=2))
