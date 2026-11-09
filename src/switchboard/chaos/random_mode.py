"""Random chaos mode: SIGKILL a real daemon subprocess at random times, again and again.

Flow: references first (each seed run uninterrupted in-process; the environment hash and the
external actions depend only on seed and steps, never on the job id, so one reference per seed
is enough), then a daemon subprocess (fresh DB, env root, sink DB, short socket) gets `jobs`
FakeAgent jobs (DirEnv + one keyed effect per step); then `kills` times: sleep a seeded random
interval, SIGKILL, start it again (recovery runs before it listens). After the last restart
everything is left to finish (global timeout), the daemon is stopped with SIGTERM, and every
invariant is checked by reading the DBs directly:

- all jobs finished (`verifying`: there is no verifier yet), none failed;
- steps contiguous, one llm_responded and one step_completed per step;
- the environment hash of every job equals the reference of its own seed;
- at the sink every effect key was applied exactly once, the keys are exactly the effect
  intents of the log (all with outcomes), and the multiset of logical actions equals the
  references' (nothing lost, duplicated or extra);
- every restart's projection check (`recovery.mismatches`) was clean;
- no daemon (or child) process of this run is left behind.

Results measure the journal against a fake sink, not a real external system.
"""

import asyncio
import contextlib
import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from switchboard.agents.fake import FakeAgent
from switchboard.chaos.results import LABEL
from switchboard.client.conn import Client
from switchboard.core import Event, EventType, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.runner import run_job
from switchboard.runtime.sinks import SQLiteSink, read_applications
from switchboard.store import EventStore

T = EventType
FINISHED = {JobState.VERIFYING.value, JobState.DONE.value}
STOPPED = FINISHED | {JobState.FAILED.value, JobState.CANCELLED.value}


@dataclass(frozen=True, slots=True)
class RandomConfig:
    jobs: int = 200
    kills: int = 20
    seed: int = 0
    steps: int = 10
    token_interval: float = 0.02
    max_running: int = 16
    min_interval: float = 0.3
    max_interval: float = 0.9
    timeout: float = 180.0  # global, for everything after the references
    keep: bool = False


class ChaosFailure(RuntimeError):
    pass


def _template(root: Path) -> Path:
    t = root / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


async def _references(cfg: RandomConfig, root: Path, template: Path) -> dict[int, dict[str, Any]]:
    """seed -> {"hash", "actions"} from uninterrupted in-process runs."""
    d = root / "ref"
    await asyncio.to_thread(d.mkdir)
    store = await EventStore.open(d / "sb.db")
    sink = await SQLiteSink.open(d / "sink.db")
    gate = asyncio.Semaphore(8)
    out: dict[int, dict[str, Any]] = {}

    async def one(seed: int) -> None:
        job_id = f"ref-{seed}"
        async with gate:
            await store.append(
                Event(
                    job_id=job_id,
                    epoch=0,
                    ts=0.0,
                    type=T.JOB_CREATED,
                    payload={
                        "kind": "fake",
                        "task_ref": "chaos",
                        "budget": {},
                        "seed": seed,
                        "instructions": [],
                        "config_hash": FakeAgent(steps=cfg.steps).config_hash,
                    },
                )
            )
            env = DirEnv(template, d / "envs", job_id)
            agent = FakeAgent(steps=cfg.steps, token_interval=0.0, effect_every=1)
            job = await run_job(store, job_id, agent, env=env, sink=sink)
            if job.state is not JobState.VERIFYING:
                raise ChaosFailure(f"reference {seed} ended {job.state.value}")
            out[seed] = {"hash": await env.hash(), "job_id": job_id}

    try:
        await asyncio.gather(*(one(s) for s in range(cfg.jobs)))
        actions: dict[str, list[str]] = {}
        for a in read_applications(d / "sink.db"):
            actions.setdefault(a.key, []).append(a.action)
        for ref in out.values():
            keys = [
                e.payload["effect_id"]
                for e in await store.read_job(ref["job_id"])
                if e.type is T.EFFECT_INTENT
            ]
            ref["actions"] = [actions[k][0] for k in keys]
    finally:
        await store.aclose()
        await sink.aclose()
    return out


class DaemonProc:
    """The daemon subprocess: spawn (waits until it listens, i.e. recovery is done) and kill."""

    def __init__(self, cfg: RandomConfig, root: Path) -> None:
        self.cfg = cfg
        self.root = root
        self.socket = root / "s"
        self.proc: asyncio.subprocess.Process | None = None
        self.starts = 0

    async def spawn(self) -> None:
        self.starts += 1
        with (self.root / "daemon.log").open("ab") as log:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "switchboard.client.cli",
                "daemon",
                "--socket",
                str(self.socket),
                "--db",
                str(self.root / "sb.db"),
                "--envs",
                str(self.root / "envs"),
                "--sink-db",
                str(self.root / "sink.db"),
                "--max-running",
                str(self.cfg.max_running),
                "--shutdown-timeout",
                "20",
                stdout=log,
                stderr=log,
            )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                raise ChaosFailure(f"daemon exited with {self.proc.returncode} on start")
            try:
                _, w = await asyncio.open_unix_connection(str(self.socket))
            except OSError:
                await asyncio.sleep(0.02)
            else:
                w.close()
                with contextlib.suppress(OSError):
                    await w.wait_closed()
                return
        raise ChaosFailure("daemon did not start listening in 60 s")

    async def kill9(self) -> None:
        assert self.proc is not None
        self.proc.send_signal(signal.SIGKILL)
        await self.proc.wait()

    async def stop(self) -> int:
        """SIGTERM, wait; SIGKILL if it hangs. Returns the exit code."""
        assert self.proc is not None
        if self.proc.returncode is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                async with asyncio.timeout(30):
                    await self.proc.wait()
            except TimeoutError:
                await self.kill9()
        assert self.proc.returncode is not None
        return self.proc.returncode

    async def recovery_stats(self) -> dict[str, Any]:
        async with await Client.connect(str(self.socket)) as c:
            stats = await c.request("stats")
        rec: dict[str, Any] = stats["recovery"]
        return rec


def _states(db: Path) -> Counter[str]:
    uri = f"file:{db}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10.0)) as conn:
        return Counter(r[0] for r in conn.execute("SELECT state FROM jobs"))


def _read_events(db: Path) -> dict[str, list[tuple[str, int | None, dict[str, Any]]]]:
    uri = f"file:{db}?mode=ro"
    out: dict[str, list[tuple[str, int | None, dict[str, Any]]]] = {}
    with closing(sqlite3.connect(uri, uri=True, timeout=10.0)) as conn:
        for job_id, type_, step, payload in conn.execute(
            "SELECT job_id, type, step_id, payload FROM events ORDER BY seq"
        ):
            out.setdefault(job_id, []).append((type_, step, json.loads(payload)))
    return out


def _processes_mentioning(text: str) -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    me = str(os.getpid())
    return [
        line.strip()
        for line in ps.stdout.splitlines()
        if text in line and line.split()[0] != me and "ps -axo" not in line
    ]


async def _check(
    cfg: RandomConfig,
    root: Path,
    template: Path,
    refs: dict[int, dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    events = await asyncio.to_thread(_read_events, root / "sb.db")
    states = await asyncio.to_thread(_states, root / "sb.db")
    apps = await asyncio.to_thread(read_applications, root / "sink.db")

    jobs = {j: ev for j, ev in events.items() if any(t == T.JOB_CREATED for t, _, _ in ev)}
    if len(jobs) != cfg.jobs:
        failures.append(f"{len(jobs)} jobs in the log, expected {cfg.jobs}")
    if states.get(JobState.VERIFYING.value, 0) != cfg.jobs:
        failures.append(f"job states: {dict(states)}")

    intents: set[str] = set()
    outcomes: set[str] = set()
    reruns = 0
    max_attempts = 0
    expected_actions: Counter[str] = Counter()
    seeds: dict[str, int] = {}
    for job_id, ev in jobs.items():
        seed = next(p["seed"] for t, _, p in ev if t == T.JOB_CREATED)
        seeds[job_id] = seed
        expected_actions.update(refs[seed]["actions"])
        for kind in (T.LLM_RESPONDED, T.STEP_COMPLETED):
            steps = [s for t, s, _ in ev if t == kind]
            if steps != list(range(cfg.steps)):
                failures.append(f"{job_id}: {kind.value} steps {steps}")
        started = Counter(s for t, s, _ in ev if t == T.STEP_STARTED)
        max_attempts = max(max_attempts, *started.values())
        reruns += sum(n - 1 for n in started.values())
        intents |= {p["effect_id"] for t, _, p in ev if t == T.EFFECT_INTENT}
        outcomes |= {p["effect_id"] for t, _, p in ev if t == T.EFFECT_OUTCOME}
    if intents != outcomes:
        failures.append(f"{len(intents - outcomes)} effect intents without an outcome")

    gate = asyncio.Semaphore(16)

    async def env_hash(job_id: str) -> tuple[str, str]:
        async with gate:
            return job_id, await DirEnv(template, root / "envs", job_id).hash()

    mismatched = 0
    for job_id, h in await asyncio.gather(*(env_hash(j) for j in jobs)):
        if h != refs[seeds[job_id]]["hash"]:
            mismatched += 1
            failures.append(f"{job_id}: environment hash differs from its reference")

    per_key = Counter(a.key for a in apps)
    dup_keys = {k: n for k, n in per_key.items() if n > 1}
    if dup_keys:
        failures.append(f"{len(dup_keys)} keys applied more than once")
    if set(per_key) != intents:
        failures.append(
            f"sink keys differ from intents: {len(set(per_key) - intents)} extra, "
            f"{len(intents - set(per_key))} missing"
        )
    got_actions = Counter(a.action for a in apps)
    if got_actions != expected_actions:
        failures.append("logical actions at the sink differ from the references'")
    mismatches = sum(len(r["mismatches"]) for r in recoveries)
    if mismatches:
        failures.append(f"{mismatches} projection mismatches reported by recovery")

    metrics = {
        "jobs": len(jobs),
        "states": dict(states),
        "env_mismatches": mismatched,
        "sink_applications": len(apps),
        "sink_distinct_keys": len(per_key),
        "duplicate_keys": len(dup_keys),
        "extra_applications": sum(n - 1 for n in per_key.values()),
        "effect_intents": len(intents),
        "reran_steps": reruns,
        "max_attempts_on_a_step": max_attempts,
        "projection_mismatches": mismatches,
    }
    return metrics, failures


async def run_random(cfg: RandomConfig, *, base: Path | None = None) -> dict[str, Any]:
    """Run the full random mode; returns the result dict (`ok` says whether it passed)."""
    t0 = time.monotonic()
    root = Path(
        await asyncio.to_thread(tempfile.mkdtemp, prefix="sbchaos", dir=str(base or "/tmp"))
    )
    rng = random.Random(cfg.seed)
    daemon = DaemonProc(cfg, root)
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    recoveries: list[dict[str, Any]] = []
    kills_done = 0
    orphans: list[str] = []
    exit_code: int | None = None
    ref_s = 0.0
    try:
        template = _template(root)
        refs = await _references(cfg, root, template)
        ref_s = time.monotonic() - t0
        async with asyncio.timeout(cfg.timeout):
            await daemon.spawn()
            async with await Client.connect(str(daemon.socket)) as c:
                for seed in range(cfg.jobs):
                    await c.request(
                        "submit",
                        seed=seed,
                        steps=cfg.steps,
                        token_interval=cfg.token_interval,
                        env_template=str(template),
                        effect_every=1,
                        effect_policy="keyed",
                    )
            for _ in range(cfg.kills):
                await asyncio.sleep(rng.uniform(cfg.min_interval, cfg.max_interval))
                await daemon.kill9()
                kills_done += 1
                await daemon.spawn()
                recoveries.append(await daemon.recovery_stats())
            while True:
                states = await asyncio.to_thread(_states, root / "sb.db")
                if sum(n for s, n in states.items() if s in STOPPED) >= cfg.jobs:
                    break
                await asyncio.sleep(0.25)
            exit_code = await daemon.stop()
        if exit_code != 0:
            failures.append(f"final daemon exited with {exit_code}")
        metrics, check_failures = await _check(cfg, root, template, refs, recoveries)
        failures += check_failures
        for _ in range(50):  # cp children of a killed daemon may need a moment
            orphans = await asyncio.to_thread(_processes_mentioning, str(root))
            if not orphans:
                break
            await asyncio.sleep(0.1)
        if orphans:
            failures.append(f"{len(orphans)} leftover processes: {orphans[:3]}")
    except TimeoutError:
        failures.append(f"global timeout of {cfg.timeout}s hit")
    except ChaosFailure as exc:
        failures.append(str(exc))
    finally:
        if daemon.proc is not None and daemon.proc.returncode is None:
            await daemon.kill9()
    ok = not failures
    result = {
        "mode": "random",
        "label": LABEL,
        "config": {
            "jobs": cfg.jobs,
            "kills": cfg.kills,
            "seed": cfg.seed,
            "steps": cfg.steps,
            "max_running": cfg.max_running,
        },
        "ok": ok,
        "kills_done": kills_done,
        "daemon_starts": daemon.starts,
        "recovery_requeued_total": sum(r["requeued"] for r in recoveries),
        "recovery_requeued_per_restart": [r["requeued"] for r in recoveries],
        "metrics": metrics,
        "failures": failures,
        "reference_s": round(ref_s, 3),
        "wall_s": round(time.monotonic() - t0, 3),
        "workdir": str(root) if (cfg.keep or not ok) else None,
    }
    if not (cfg.keep or not ok):
        await asyncio.to_thread(shutil.rmtree, root, True)
    return result


def format_random(result: dict[str, Any]) -> str:
    m = result["metrics"]
    c = result["config"]
    lines = [
        f"random chaos: {c['jobs']} jobs, {result['kills_done']} kill -9, seed {c['seed']}, "
        f"{c['steps']} steps each, wall {result['wall_s']}s",
        f"jobs finished (verifying): {m.get('states', {}).get('verifying', 0)}/{c['jobs']}",
        f"env hash mismatches:       {m.get('env_mismatches')}",
        f"sink: {m.get('sink_applications')} applications, {m.get('sink_distinct_keys')} keys, "
        f"{m.get('duplicate_keys')} duplicated keys, {m.get('extra_applications')} extra",
        f"jobs re-queued by recovery (total over restarts): {result['recovery_requeued_total']}",
        f"steps re-run: {m.get('reran_steps')}, max attempts on one step: "
        f"{m.get('max_attempts_on_a_step')}",
        f"projection mismatches:     {m.get('projection_mismatches')}",
        "RESULT: " + ("PASS" if result["ok"] else "FAIL: " + "; ".join(result["failures"])),
        "",
        result["label"],
    ]
    return "\n".join(lines)
