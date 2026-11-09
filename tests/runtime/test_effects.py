"""Effect journal: exactly-once at the sink across every crash window, reconciliation, the
naive baseline, stored-policy rules, and the effects projection."""

import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest
from conftest import create_job

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType, JobState, ProjectionError
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import ActResult, ModelResponse, ReplayPolicy, StepContext
from switchboard.runtime.effects import canonical_json, effect_key
from switchboard.runtime.runner import run_job
from switchboard.runtime.sinks import EffectCounts, FakeExternalSink
from switchboard.store import AppendStatus, EffectRecord, EventStore, SQLiteEventLog, open_db

T = EventType
pytestmark = [pytest.mark.asyncio, pytest.mark.looptime]
STEPS = 4
SEED = 5
CRASH_STEP = 2
WINDOWS = ["before_intent", "before_apply", "after_apply", "after_outcome"]
AFTER_EXTERNAL_ACTION = ["after_apply", "after_outcome"]


class Crash(BaseException):
    """Stands in for kill -9."""


def agent_for(
    window: str | None, sink: FakeExternalSink, *, policy: ReplayPolicy, naive: bool = False
) -> FakeAgent:
    """An agent with one effect per step that crashes once in `window` at CRASH_STEP.

    The sink hooks fire on the third real application (the CRASH_STEP effect), once; the two
    agent-side windows mirror W1/W3 of test_runner.py."""
    fired = False

    def hook_for(name: str, extra: int) -> Callable[[str], None]:
        def hook(key: str) -> None:
            nonlocal fired
            if window == name and not fired and len(sink.ledger) == CRASH_STEP + extra:
                fired = True
                raise Crash

        return hook

    sink.before_apply = hook_for("before_apply", 0)
    sink.after_apply = hook_for("after_apply", 1)

    def armed(ctx: StepContext) -> bool:
        return ctx.step == CRASH_STEP and ctx.attempt == 0

    class Agent(FakeAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            if window == "before_intent" and armed(ctx):
                raise Crash
            result = await super().act(ctx, response)
            if window == "after_outcome" and armed(ctx):
                raise Crash
            return result

    return Agent(
        steps=STEPS,
        effect_every=1,
        effect_policy=policy,
        naive_sink=sink if naive else None,
    )


@pytest.fixture
def template(tmp_path: Path) -> Path:
    t = tmp_path / "template"
    (t / "files").mkdir(parents=True)
    (t / "README").write_text("seed project\n")
    return t


async def intent_ids(store: EventStore, job_id: str = "j1") -> list[str]:
    return [
        e.payload["effect_id"] for e in await store.read_job(job_id) if e.type is T.EFFECT_INTENT
    ]


async def control(template: Path, root: Path) -> tuple[str, list[str]]:
    """Uninterrupted run: env hash and the effect keys it produced."""
    env = DirEnv(template, root / "control", "j1")
    sink = FakeExternalSink()
    async with await EventStore.open(root / "control.db") as s:
        await create_job(s, seed=SEED)
        await run_job(s, "j1", agent_for(None, sink, policy=ReplayPolicy.KEYED), env=env, sink=sink)
        assert sink.counts() == EffectCounts(STEPS, STEPS, 0, 0)
        return await env.hash(), await intent_ids(s)


async def crash_and_resume(
    store: EventStore,
    template: Path,
    root: Path,
    window: str,
    sink: FakeExternalSink,
    *,
    policy: ReplayPolicy = ReplayPolicy.KEYED,
    resume_policy: ReplayPolicy | None = None,
    naive: bool = False,
) -> tuple[FakeAgent, DirEnv]:
    await create_job(store, seed=SEED)
    use = None if naive else sink
    with pytest.raises(Crash):
        await run_job(
            store,
            "j1",
            agent_for(window, sink, policy=policy, naive=naive),
            env=DirEnv(template, root / "envs", "j1"),
            sink=use,
        )
    resumed = agent_for(None, sink, policy=resume_policy or policy, naive=naive)
    env = DirEnv(template, root / "envs", "j1")  # a restarted process: fresh objects
    job = await run_job(store, "j1", resumed, env=env, sink=use)
    assert job.state is JobState.VERIFYING
    return resumed, env


@pytest.mark.parametrize("window", WINDOWS)
async def test_keyed_exactly_once_in_every_window(
    window: str, store: EventStore, template: Path, tmp_path: Path
) -> None:
    ctl_hash, ctl_keys = await control(template, tmp_path)
    sink = FakeExternalSink()
    _, env = await crash_and_resume(store, template, tmp_path, window, sink)
    # asserted at the sink, not the log
    assert sink.duplicates() == {}
    assert sink.counts() == EffectCounts(STEPS, STEPS, 0, 0)
    assert all(sink.applications(k) == 1 for k in ctl_keys)
    assert [k for k, _, _ in sink.ledger] == ctl_keys  # same keys as the uninterrupted run
    assert await env.hash() == ctl_hash
    assert await intent_ids(store) == ctl_keys  # re-runs never wrote a second intent
    for k in ctl_keys:
        rec = await rec_of(store, k)
        assert rec.has_outcome
        assert rec.policy == "keyed"


@pytest.mark.parametrize("window", WINDOWS)
async def test_naive_baseline_duplicates(
    window: str, store: EventStore, template: Path, tmp_path: Path
) -> None:
    """Same crash points, journal bypassed, sink applies every call: the contrast."""
    sink = FakeExternalSink(idempotent=False)
    await crash_and_resume(store, template, tmp_path, window, sink, naive=True)
    counts = sink.counts()
    if window in AFTER_EXTERNAL_ACTION:
        assert counts.duplicates > 0
        assert counts.extra_applications > 0
        assert len(sink.duplicates()) == counts.duplicates
    else:
        assert counts.duplicates == 0  # nothing had been applied yet when it crashed
    assert not [e for e in await store.read_job("j1") if e.type is T.EFFECT_INTENT]


async def test_unsafe_with_lookup_reconciles(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    sink = FakeExternalSink()
    await crash_and_resume(
        store, template, tmp_path, "after_apply", sink, policy=ReplayPolicy.UNSAFE
    )
    assert sink.duplicates() == {}
    assert sink.counts().applications == STEPS
    statuses = [(await rec_of(store, k)).status for k in await intent_ids(store)]
    assert statuses == ["applied", "applied", "reconciled", "applied"]


async def test_unsafe_with_lookup_not_applied_is_applied_now(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    sink = FakeExternalSink()
    await crash_and_resume(
        store, template, tmp_path, "before_apply", sink, policy=ReplayPolicy.UNSAFE
    )
    assert sink.counts() == EffectCounts(STEPS, STEPS, 0, 0)
    assert (await rec_of(store, (await intent_ids(store))[CRASH_STEP])).status == "applied"


async def test_unsafe_without_lookup_is_interrupted(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    sink = FakeExternalSink(supports_lookup=False)  # lookup() would raise if it were called
    resumed, _ = await crash_and_resume(
        store, template, tmp_path, "after_apply", sink, policy=ReplayPolicy.UNSAFE
    )
    keys = await intent_ids(store)
    assert sink.applications(keys[CRASH_STEP]) == 1  # the original, not repeated
    assert sink.duplicates() == {}
    assert (await rec_of(store, keys[CRASH_STEP])).status == "interrupted"
    seen = [r for r in resumed.effect_results if r.interrupted]
    assert [r.effect_id for r in seen] == [keys[CRASH_STEP]]
    # replaying the interrupted effect again returns the recorded outcome, not a new one
    done = [e for e in await store.read_job("j1") if e.type is T.STEP_COMPLETED]
    assert done[CRASH_STEP].payload["delta"][-1]["content"] == "interrupted"


async def test_stored_policy_is_never_upgraded(
    store: EventStore, template: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # stored unsafe, caller now says keyed. A keyed re-apply would duplicate on this sink
    # (not idempotent), so "interrupted + one application" proves the stored policy won.
    sink = FakeExternalSink(idempotent=False, supports_lookup=False)
    with caplog.at_level(logging.WARNING, logger="switchboard.runtime.effects"):
        await crash_and_resume(
            store,
            template,
            tmp_path,
            "after_apply",
            sink,
            policy=ReplayPolicy.UNSAFE,
            resume_policy=ReplayPolicy.KEYED,
        )
    keys = await intent_ids(store)
    rec = await rec_of(store, keys[CRASH_STEP])
    assert rec.policy == "unsafe"
    assert rec.status == "interrupted"
    assert sink.duplicates() == {}
    assert "stored policy unsafe wins over requested keyed" in caplog.text


async def test_effect_key_is_deterministic_and_canonical() -> None:
    base = {"b": 1, "a": "é"}
    a = effect_key("j", 3, 1, base)
    assert a == effect_key("j", 3, 1, {"a": "é", "b": 1})  # key order is irrelevant
    assert len(a) == 64
    assert canonical_json(base) == '{"a":"é","b":1}'  # compact, not ASCII-escaped
    others = {
        effect_key("k", 3, 1, base),
        effect_key("j", 4, 1, base),
        effect_key("j", 3, 2, base),
        effect_key("j", 3, 1, {"a": "x", "b": 1}),
    }
    assert a not in others
    assert len(others) == 4


async def test_multiple_effects_per_step_use_call_index(store: EventStore) -> None:
    await create_job(store, seed=SEED)
    sink = FakeExternalSink()

    class Two(FakeAgent):
        async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
            for _ in range(2):  # identical args, distinguished only by call_index
                await ctx.effect("ping", {"x": 1}, policy=ReplayPolicy.KEYED)
            return await super().act(ctx, response)

    await run_job(store, "j1", Two(steps=1), sink=sink)
    keys = await intent_ids(store)
    assert keys == [effect_key("j1", 0, i, {"x": 1}) for i in range(2)]
    assert sink.counts() == EffectCounts(2, 2, 0, 0)


async def test_ctx_effect_without_sink_raises(store: EventStore) -> None:
    await create_job(store)
    job = await run_job(store, "j1", FakeAgent(steps=1, effect_every=1))
    assert job.state is JobState.FAILED  # RuntimeError from the agent fails the job


async def rec_of(store: EventStore, effect_id: str) -> EffectRecord:
    rec = await store.get_effect(effect_id)
    assert rec is not None
    return rec


def effects_rows(path: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute("SELECT * FROM effects ORDER BY effect_id").fetchall()


async def test_effects_table_matches_events_after_rebuild(
    store: EventStore, template: Path, tmp_path: Path
) -> None:
    sink = FakeExternalSink(supports_lookup=False)
    await create_job(store, seed=SEED)
    with pytest.raises(Crash):
        await run_job(
            store, "j1", agent_for("after_apply", sink, policy=ReplayPolicy.UNSAFE), sink=sink
        )
    db = tmp_path / "sb.db"
    before = effects_rows(db)
    assert len(before) == CRASH_STEP + 1
    assert [r[5] for r in before if r[0] == (await intent_ids(store))[CRASH_STEP]] == ["intent"]
    await run_job(store, "j1", agent_for(None, sink, policy=ReplayPolicy.UNSAFE), sink=sink)
    live = effects_rows(db)
    assert len(live) == STEPS
    with closing(open_db(db)) as conn:
        SQLiteEventLog(conn).rebuild()
    assert effects_rows(db) == live
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("DELETE FROM effects")  # even a wiped table comes back
        conn.commit()
    with closing(open_db(db)) as conn:
        SQLiteEventLog(conn).rebuild()
    assert effects_rows(db) == live


def intent(effect_id: str = "e1", **over: object) -> Event:
    payload: dict[str, object] = {
        "effect_id": effect_id,
        "kind": "send",
        "args_hash": "h",
        "args": {},
        "policy": "keyed",
        "step": 0,
        "call_index": 0,
    }
    return Event(job_id="j1", epoch=0, ts=1.0, type=T.EFFECT_INTENT, payload={**payload, **over})


def outcome(effect_id: str, status: str, receipt: str) -> Event:
    payload = {"effect_id": effect_id, "outcome": {"r": receipt}, "status": status}
    return Event(job_id="j1", epoch=0, ts=2.0, type=T.EFFECT_OUTCOME, payload=payload)


async def test_duplicate_intent_and_outcome_are_reported_and_ignored(store: EventStore) -> None:
    await create_job(store)
    first, second = await store.append(intent()), await store.append(intent(kind="other"))
    assert first.status is AppendStatus.INSERTED
    assert second.status is AppendStatus.DUPLICATE
    assert second.seq is None
    # in a group the duplicate is reported per event and the rest of the group still lands
    group = await store.append_group([intent(), intent("e2")])
    assert [r.status for r in group] == [AppendStatus.DUPLICATE, AppendStatus.INSERTED]
    rec = await rec_of(store, "e1")
    assert (rec.kind, rec.status) == ("send", "intent")  # the first intent wins
    assert [e.type for e in await store.read_job("j1")].count(T.EFFECT_INTENT) == 2

    assert (await store.append(outcome("e1", "applied", "a"))).status is AppendStatus.INSERTED
    assert (await store.append(outcome("e1", "reconciled", "b"))).status is AppendStatus.DUPLICATE
    rec = await rec_of(store, "e1")
    assert (rec.status, rec.outcome) == ("applied", {"r": "a"})
    assert await store.get_effect("nope") is None


async def test_invalid_effect_events_reject_their_group(store: EventStore) -> None:
    await create_job(store)
    with pytest.raises(ProjectionError):
        await store.append(outcome("ghost", "applied", "x"))  # no intent
    with pytest.raises(ProjectionError):
        await store.append(intent(step=-1))
    with pytest.raises(ProjectionError):
        await store.append_group([intent("ok"), outcome("ok", "bogus", "x")])
    assert await store.get_effect("ok") is None  # the whole group rolled back
