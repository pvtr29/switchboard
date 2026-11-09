"""Exhaustive chaos mode: crash a deterministic run once at EVERY commit window of EVERY step.

For each (step n, window) a fresh run is crashed exactly there (see hooks.py), then a restart is
simulated the way a dead daemon leaves things: the store and sink are closed, the SAME event
DB, environment root and sink DB are reopened, `recover()` runs, and the job finishes under a
Supervisor. The result is deterministic: no randomness, no timing, no machine dependence.

Durable invariants checked per crash point:

- the job finished (state `verifying`; there is no verifier yet);
- the final environment hash equals the uninterrupted reference run;
- at the sink every reference key was applied exactly once and no other key was applied, and
  no logical action (kind + args) was applied twice;
- the log has exactly one llm_responded and one step_completed per step, contiguous, and the
  whole conversation (every step's delta, which embeds the response and the effect receipt)
  equals the reference run's;
- exactly one effect_intent per reference key, each with an outcome;
- logged responses are not re-sampled: the restarted agent samples exactly the steps without a
  logged response (n for W0, n+1.. for every later window).

Naive baseline over the same crash points: on restart the job's progress is thrown away and it
re-runs from scratch as a NEW job with the same seed, into the SAME environment directory with
no restore (provisioning keeps whatever the dead process left), and the agent calls a
non-idempotent sink directly, with no journal. Duplicates are counted by logical action, since
a new job id means new keys. Results measure the journal against a fake sink only.
"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from switchboard.chaos.hooks import (
    DESCRIPTIONS,
    ChaosAgent,
    Crash,
    CrashingStore,
    CrashPlan,
    NoProvisionEnv,
    Window,
)
from switchboard.chaos.results import LABEL, table
from switchboard.core import Event, EventType, Job, JobState
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.recovery import recover
from switchboard.runtime.runner import run_job
from switchboard.runtime.sinks import SQLiteSink, counts_of
from switchboard.runtime.supervisor import Supervisor
from switchboard.store import EventStore

JOB_ID = "chaos-job"
RESTART_JOB_ID = "chaos-job-restart"  # the naive baseline's from-scratch job

T = EventType


@dataclass(frozen=True, slots=True)
class ExhaustiveConfig:
    steps: int = 3
    seed: int = 11
    windows: tuple[Window, ...] = tuple(Window)
    concurrency: int = 4  # crash points run in parallel (they are independent)


@dataclass(slots=True)
class Reference:
    env_hash: str
    keys: list[str]  # effect keys in step order
    actions: list[str]  # logical actions in step order
    deltas: list[Any]  # conversation delta per step
    steps: int


@dataclass(slots=True)
class DurablePoint:
    window: str
    step: int
    ok: bool = False
    failures: list[str] = field(default_factory=list)
    env_match: bool = False
    extra_applications: int = 0  # duplicate applications, by key and by action
    missing_effects: int = 0
    resampled_logged: int = 0  # steps with a logged response that were sampled again
    final_state: str = ""


@dataclass(slots=True)
class NaivePoint:
    window: str
    step: int
    env_match: bool = False
    extra_applications: int = 0  # duplicate applications by logical action
    missing_effects: int = 0
    final_state: str = ""


async def _create_job(store: EventStore, job_id: str, seed: int, steps: int) -> None:
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
                "config_hash": ChaosAgent(steps=steps).config_hash,
            },
        )
    )


def _make_template(root: Path) -> Path:
    t = root / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


def _sink_hooks(sink: SQLiteSink, agent: ChaosAgent) -> None:
    sink.before_apply = agent.before_apply
    sink.after_apply = agent.after_apply


def _expected_sampled(plan: CrashPlan, steps: int) -> list[int]:
    first = plan.step if plan.window is Window.W0 else plan.step + 1
    return list(range(first, steps))


async def _reference(cfg: ExhaustiveConfig, root: Path, template: Path) -> Reference:
    d = root / "reference"
    await asyncio.to_thread(d.mkdir, parents=True)
    store = await EventStore.open(d / "sb.db")
    sink = await SQLiteSink.open(d / "sink.db")
    try:
        await _create_job(store, JOB_ID, cfg.seed, cfg.steps)
        env = DirEnv(template, d / "envs", JOB_ID)
        job = await run_job(store, JOB_ID, ChaosAgent(steps=cfg.steps), env=env, sink=sink)
        if job.state is not JobState.VERIFYING:
            raise RuntimeError(f"reference run ended {job.state.value}")
        events = await store.read_job(JOB_ID)
        apps = sink.ledger()
        return Reference(
            env_hash=await env.hash(),
            keys=[e.payload["effect_id"] for e in events if e.type is T.EFFECT_INTENT],
            actions=[a.action for a in apps],
            deltas=[e.payload["delta"] for e in events if e.type is T.STEP_COMPLETED],
            steps=cfg.steps,
        )
    finally:
        await store.aclose()
        await sink.aclose()


async def _crash_once(
    cfg: ExhaustiveConfig, d: Path, template: Path, plan: CrashPlan, *, naive: bool
) -> None:
    """Run the job until the planned crash fires, then close everything like a dead process."""
    await asyncio.to_thread(d.mkdir, parents=True, exist_ok=True)
    store = await CrashingStore.open(d / "sb.db")
    sink = await SQLiteSink.open(d / "sink.db", idempotent=not naive)
    try:
        agent = ChaosAgent(steps=cfg.steps, plan=plan, naive_sink=sink if naive else None)
        _sink_hooks(sink, agent)
        if plan.window is Window.W4:
            store.crash_after_step_commit = plan.step
        await _create_job(store, JOB_ID, cfg.seed, cfg.steps)
        env = DirEnv(template, d / "envs", JOB_ID)
        try:
            await run_job(store, JOB_ID, agent, env=env, sink=None if naive else sink)
        except Crash:
            pass
        else:
            raise RuntimeError(f"no crash fired at {plan}")
    finally:
        await store.aclose()
        await sink.aclose()


async def _durable_point(
    cfg: ExhaustiveConfig, ref: Reference, root: Path, template: Path, plan: CrashPlan
) -> DurablePoint:
    point = DurablePoint(plan.window.value, plan.step)
    d = root / f"durable-{plan.window.value}-{plan.step}"
    await _crash_once(cfg, d, template, plan, naive=False)

    # ---- restart: same DB, env root and sink DB; recovery; let the job finish ----------------
    store = await EventStore.open(d / "sb.db")
    sink = await SQLiteSink.open(d / "sink.db")
    agents: list[ChaosAgent] = []

    def agent_factory(job: Job) -> ChaosAgent:
        agents.append(ChaosAgent(steps=cfg.steps))
        return agents[-1]

    sup = Supervisor(
        store,
        agent_factory,
        lambda job: DirEnv(template, d / "envs", job.job_id),
        lambda job: sink,
        max_running=1,
    )
    try:
        report = await recover(store, sup)
        if not report.ok:
            point.failures.append(f"recovery reported problems: {report}")
        await sup.wait(JOB_ID)
        out = sup.outcome(JOB_ID)
        if out is not None and out.error is not None:
            point.failures.append(f"job task crashed: {out.error!r}")
        job = await store.get_job(JOB_ID)
        assert job is not None
        point.final_state = job.state.value
        if job.state is not JobState.VERIFYING:
            point.failures.append(f"job ended {job.state.value}, not verifying")

        env_hash = await DirEnv(template, d / "envs", JOB_ID).hash()
        point.env_match = env_hash == ref.env_hash
        if not point.env_match:
            point.failures.append("environment hash differs from the reference")

        apps = sink.ledger()
        by_key = counts_of(apps)
        by_action = counts_of(apps, by_action=True)
        point.extra_applications = by_key.extra_applications + by_action.extra_applications
        keys = {a.key for a in apps}
        point.missing_effects = len(set(ref.keys) - keys)
        if point.extra_applications:
            point.failures.append(f"{point.extra_applications} duplicate applications")
        if point.missing_effects:
            point.failures.append(f"{point.missing_effects} reference keys never applied")
        if keys - set(ref.keys):
            point.failures.append(f"unexpected keys applied: {sorted(keys - set(ref.keys))}")
        if sorted(a.action for a in apps) != sorted(ref.actions):
            point.failures.append("applied actions differ from the reference")

        events = await store.read_job(JOB_ID)
        for kind in (T.LLM_RESPONDED, T.STEP_COMPLETED):
            steps = [e.step_id for e in events if e.type is kind]
            if steps != list(range(cfg.steps)):
                point.failures.append(f"{kind.value} steps {steps} are not 0..{cfg.steps - 1}")
        deltas = [e.payload["delta"] for e in events if e.type is T.STEP_COMPLETED]
        if deltas != ref.deltas:
            point.failures.append("conversation differs from the reference")
        intents = [e.payload["effect_id"] for e in events if e.type is T.EFFECT_INTENT]
        if intents != ref.keys:
            point.failures.append("effect intents differ from the reference keys")
        for key in intents:
            rec = await store.get_effect(key)
            if rec is None or not rec.has_outcome:
                point.failures.append(f"effect {key[:12]} has no outcome")

        expected = _expected_sampled(plan, cfg.steps)
        sampled = [s for a in agents for s, _ in a.sample_calls]
        if sampled != expected:
            point.failures.append(f"restart sampled steps {sampled}, expected {expected}")
            point.resampled_logged = len(set(sampled) - set(expected))
    finally:
        await sup.aclose()
        await store.aclose()
        await sink.aclose()
    point.ok = not point.failures
    return point


async def _naive_point(
    cfg: ExhaustiveConfig, ref: Reference, root: Path, template: Path, plan: CrashPlan
) -> NaivePoint:
    point = NaivePoint(plan.window.value, plan.step)
    d = root / f"naive-{plan.window.value}-{plan.step}"
    await _crash_once(cfg, d, template, plan, naive=True)

    # ---- naive restart: from scratch, new job id, same seed, same env dir, no restore --------
    store = await EventStore.open(d / "sb.db")
    sink = await SQLiteSink.open(d / "sink.db", idempotent=False)
    try:
        await _create_job(store, RESTART_JOB_ID, cfg.seed, cfg.steps)
        env = NoProvisionEnv(template, d / "envs", JOB_ID)
        job = await run_job(
            store, RESTART_JOB_ID, ChaosAgent(steps=cfg.steps, naive_sink=sink), env=env
        )
        point.final_state = job.state.value
        point.env_match = await env.hash() == ref.env_hash
        apps = sink.ledger()
        point.extra_applications = counts_of(apps, by_action=True).extra_applications
        point.missing_effects = len(set(ref.actions) - {a.action for a in apps})
    finally:
        await store.aclose()
        await sink.aclose()
    return point


async def run_exhaustive(cfg: ExhaustiveConfig, workdir: Path) -> dict[str, Any]:
    """Run every (step, window) crash point, durable and naive; return the result dict."""
    started = time.monotonic()
    template = _make_template(workdir)
    ref = await _reference(cfg, workdir, template)
    plans = [CrashPlan(w, n) for n in range(cfg.steps) for w in cfg.windows]
    gate = asyncio.Semaphore(cfg.concurrency)

    async def both(plan: CrashPlan) -> tuple[DurablePoint, NaivePoint]:
        async with gate:
            durable = await _durable_point(cfg, ref, workdir, template, plan)
            naive = await _naive_point(cfg, ref, workdir, template, plan)
            return durable, naive

    pairs = await asyncio.gather(*(both(p) for p in plans))
    durable = [d for d, _ in pairs]
    naive = [n for _, n in pairs]

    n_points = len(plans)
    ok = sum(1 for p in durable if p.ok)
    durable_dups = sum(p.extra_applications for p in durable)
    naive_dups = sum(p.extra_applications for p in naive)
    naive_dup_points = sum(1 for p in naive if p.extra_applications)
    summary = {
        "crash_points": n_points,
        "durable_correct": ok,
        "durable_correct_pct": round(100.0 * ok / n_points, 2),
        "durable_duplicate_effects": durable_dups,
        "naive_duplicate_effects": naive_dups,
        "naive_points_with_duplicates": naive_dup_points,
        "durable_env_mismatches": sum(1 for p in durable if not p.env_match),
        "naive_env_mismatches": sum(1 for p in naive if not p.env_match),
        "durable_missing_effects": sum(p.missing_effects for p in durable),
        "naive_missing_effects": sum(p.missing_effects for p in naive),
        "durable_resampled_logged_responses": sum(p.resampled_logged for p in durable),
    }
    headline = (
        f"{n_points} crash points: durable resume {summary['durable_correct_pct']:g}% correct, "
        f"{durable_dups} duplicate effects; naive resume duplicated {naive_dups} effects "
        f"across {naive_dup_points} crash points"
    )
    return {
        "mode": "exhaustive",
        "label": LABEL,
        "config": {
            "steps": cfg.steps,
            "seed": cfg.seed,
            "windows": [w.value for w in cfg.windows],
        },
        "reference": {
            "env_hash": ref.env_hash,
            "effect_keys": len(ref.keys),
            "steps": ref.steps,
        },
        "headline": headline,
        "summary": summary,
        "durable": [vars_of(p) for p in durable],
        "naive": [vars_of(p) for p in naive],
        "failures": [f"{p.window}@{p.step}: {f}" for p in durable for f in p.failures],
        "wall_s": round(time.monotonic() - started, 3),
    }


def vars_of(point: DurablePoint | NaivePoint) -> dict[str, Any]:
    return {name: getattr(point, name) for name in point.__slots__}


def format_exhaustive(result: dict[str, Any]) -> str:
    """The per-window table (durable vs naive, summed over steps) plus the headline."""
    rows: list[list[object]] = []
    windows: Sequence[str] = result["config"]["windows"]
    for w in windows:
        d = [p for p in result["durable"] if p["window"] == w]
        n = [p for p in result["naive"] if p["window"] == w]
        rows.append(
            [
                w,
                DESCRIPTIONS[Window(w)],
                len(d),
                f"{sum(p['ok'] for p in d)}/{len(d)}",
                sum(p["extra_applications"] for p in d),
                sum(p["extra_applications"] for p in n),
                sum(not p["env_match"] for p in d),
                sum(not p["env_match"] for p in n),
            ]
        )
    s = result["summary"]
    rows.append(
        [
            "all",
            "",
            s["crash_points"],
            f"{s['durable_correct']}/{s['crash_points']}",
            s["durable_duplicate_effects"],
            s["naive_duplicate_effects"],
            s["durable_env_mismatches"],
            s["naive_env_mismatches"],
        ]
    )
    head = [
        "window",
        "crash point",
        "points",
        "durable ok",
        "dups dur",
        "dups naive",
        "env dur",
        "env naive",
    ]
    return "\n".join([table(head, rows), "", result["headline"], "", result["label"]])
