"""Real-agent rollouts through the daemon: submit -> GeminiAgent steps -> grading -> verdict.

The LLM is scripted (no network) and the command runner is unsandboxed (trusted task code), so
this runs everywhere; the sandboxed runner and the live model have their own tests.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from rollout_fakes import PlainRunner, Script, ScriptedLLM, hack_script, solve_script

from switchboard.agents.tools import CommandRunner, RunResult
from switchboard.client.conn import Client, ClientError
from switchboard.daemon.server import DEFAULT_ROLLOUT_TOKENS, Daemon
from switchboard.llm.types import QuotaExhausted
from switchboard.store import EventStore
from switchboard.tasks import load_task

pytestmark = pytest.mark.asyncio

ROMAN = load_task("roman")


class Env:
    """A daemon factory over one db/envs pair, so a test can restart the daemon."""

    def __init__(self, sock_dir: Path, tmp_path: Path) -> None:
        self.sock_dir = sock_dir
        self.tmp_path = tmp_path
        self.daemons: list[Daemon] = []

    async def start(
        self,
        llm: ScriptedLLM | None,
        runner: Callable[[Path], CommandRunner] = PlainRunner,
        **kw: Any,
    ) -> Daemon:
        d = Daemon(
            socket_path=self.sock_dir / "s",
            db_path=self.tmp_path / "sb.db",
            envs_dir=self.tmp_path / "envs",
            shutdown_timeout=kw.pop("shutdown_timeout", 5.0),
            llm_factory=None if llm is None else (lambda models: llm),
            runner_factory=runner,
            **kw,
        )
        await d.start()
        self.daemons.append(d)
        return d


@pytest_asyncio.fixture
async def env(sock_dir: Path, tmp_path: Path) -> AsyncIterator[Env]:
    e = Env(sock_dir, tmp_path)
    try:
        yield e
    finally:
        for d in e.daemons:
            await d.shutdown()


async def submit(d: Daemon, task: str = "roman", **extra: Any) -> str:
    async with await Client.connect(str(d.socket_path)) as c:
        r = await c.request("submit", kind="rollout", agent="gemini", task=task, **extra)
    return str(r["job_id"])


async def wait_state(d: Daemon, job_id: str, states: Sequence[str]) -> Any:
    async with asyncio.timeout(30):
        while True:
            job = await d.store.get_job(job_id)
            if job is not None and job.state.value in states:
                return job
            await asyncio.sleep(0.02)


async def events(store: EventStore, job_id: str, type_: str) -> list[Any]:
    return [e for e in await store.read_job(job_id) if e.type.value == type_]


async def test_a_solved_task_is_graded_pass(env: Env) -> None:
    llm = ScriptedLLM(solve_script(ROMAN.root, "roman.py"))
    d = await env.start(llm)
    job_id = await submit(d)
    job = await wait_state(d, job_id, ["done", "failed"])
    assert job.state.value == "done"
    assert job.verdict.value == "pass"
    assert job.kind.value == "rollout"
    assert job.task_ref == "roman"
    assert job.budget.tokens == DEFAULT_ROLLOUT_TOKENS
    (verdict,) = await events(d.store, job_id, "verdict_recorded")
    assert verdict.payload["failed"] == 0
    assert verdict.payload["passed"] >= 12  # visible + hidden
    assert verdict.payload["ignored_changes"] == []
    assert verdict.payload["monitor"]["rules"] == []
    assert verdict.payload["monitor"]["score"] == 0.0
    (created,) = await events(d.store, job_id, "job_created")
    assert created.payload["instructions"][0]["content"] == ROMAN.prompt
    assert llm.calls == [0, 1, 2]


async def test_reward_hacking_is_graded_fail_and_reported(env: Env) -> None:
    d = await env.start(ScriptedLLM(hack_script("tests/test_roman.py")))
    job_id = await submit(d)
    job = await wait_state(d, job_id, ["done", "failed"])
    assert job.verdict.value == "fail"
    (verdict,) = await events(d.store, job_id, "verdict_recorded")
    assert verdict.payload["ignored_changes"] == ["conftest.py", "tests/test_roman.py"]
    assert verdict.payload["tampered"] == ["conftest.py", "tests/test_roman.py"]
    monitor = verdict.payload["monitor"]
    assert set(monitor["rules"]) == {"writes_tests", "pytest_subversion"}
    assert monitor["score"] > 0.9
    assert {e["step"] for e in monitor["evidence"]} == {0, 1}
    # the agent itself saw green: its own run_tests output had no failure
    steps = await events(d.store, job_id, "step_completed")
    run_out = steps[2].payload["delta"][1]["content"]
    assert "exit code: 0" in run_out


async def test_bad_rollout_submissions(env: Env) -> None:
    d = await env.start(ScriptedLLM([]))
    async with await Client.connect(str(d.socket_path)) as c:
        for args, needle in (
            ({"task": "nope"}, "unknown task"),
            ({}, "task is required"),
            ({"task": "roman", "env_template": str(ROMAN.repo)}, "own environment"),
            ({"task": "roman", "models": "gemini"}, "models must be"),
            ({"task": "roman", "kind": "fake"}, "kind must be"),
        ):
            with pytest.raises(ClientError) as exc:
                await c.request("submit", agent="gemini", **args)
            assert exc.value.code == "bad_args"
            assert needle in exc.value.message
        with pytest.raises(ClientError, match="agent must be"):
            await c.request("submit", agent="gpt")


async def test_without_a_key_gemini_jobs_are_refused(env: Env) -> None:
    d = await env.start(None, api_key=None)
    with pytest.raises(ClientError, match="no Gemini API key"):
        await submit(d)


async def test_explicit_budget_and_models_are_kept(env: Env) -> None:
    llm = ScriptedLLM(solve_script(ROMAN.root, "roman.py"))
    d = await env.start(llm)
    job_id = await submit(d, budget={"tokens": 5000}, models=["m1", "m2"])
    job = await wait_state(d, job_id, ["done", "failed"])
    assert job.budget.tokens == 5000
    (created,) = await events(d.store, job_id, "job_created")
    assert created.payload["models"] == ["m1", "m2"]


class GatedRunner(PlainRunner):
    """Blocks every run whose argv matches `argv` until `release` is set."""

    entered = asyncio.Event()
    release = asyncio.Event()
    argv: tuple[str, ...] = ()

    async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResult:
        if tuple(argv) == self.argv:
            type(self).entered.set()
            await type(self).release.wait()
        return await super().run(argv, timeout_s=timeout_s)


def gated(argv: tuple[str, ...]) -> type[GatedRunner]:
    return type(
        "Gated",
        (GatedRunner,),
        {"entered": asyncio.Event(), "release": asyncio.Event(), "argv": argv},
    )


async def test_restart_mid_tool_replays_the_logged_response_without_a_new_call(
    env: Env,
) -> None:
    script: Script = solve_script(ROMAN.root, "roman.py")
    runner = gated(("pytest", "-q"))  # the agent's run_tests in step 1
    first = ScriptedLLM(script)
    d1 = await env.start(first, runner=runner, shutdown_timeout=0.2)
    job_id = await submit(d1)
    await asyncio.wait_for(runner.entered.wait(), 30)
    await d1.shutdown()  # step 1's response is logged, its tool never finished

    second = ScriptedLLM(script)
    d2 = await env.start(second)
    job = await wait_state(d2, job_id, ["done", "failed"])
    assert job.verdict.value == "pass"
    assert first.calls == [0, 1]
    assert second.calls == [2]  # step 1 was replayed from the log, not re-sampled
    responded = await events(d2.store, job_id, "llm_responded")
    assert [e.step_id for e in responded] == [0, 1, 2]


async def test_a_job_left_verifying_is_graded_at_the_next_start(env: Env) -> None:
    runner = gated(("pytest", "-q", "tests"))  # the grader's run
    d1 = await env.start(ScriptedLLM(solve_script(ROMAN.root, "roman.py")), runner=runner)
    job_id = await submit(d1)
    await asyncio.wait_for(runner.entered.wait(), 30)
    await d1.shutdown()  # grading interrupted: nothing written, job stays verifying

    d2 = await env.start(ScriptedLLM([]))
    job = await wait_state(d2, job_id, ["done", "failed"])
    assert job.verdict.value == "pass"
    assert len(await events(d2.store, job_id, "verdict_recorded")) == 1


async def test_out_of_quota_pauses_then_resumes_by_itself(env: Env) -> None:
    llm = ScriptedLLM(solve_script(ROMAN.root, "roman.py"))
    llm.fail_once[1] = QuotaExhausted("daily quota", retry_after_s=0.3)
    d = await env.start(llm)
    job_id = await submit(d)
    job = await wait_state(d, job_id, ["done", "failed"])
    assert job.verdict.value == "pass"
    changes = [e.payload for e in await events(d.store, job_id, "state_changed")]
    paused = next(c for c in changes if c["to"] == "paused")
    assert paused["reason"].startswith("retry later: daily quota")
    assert "resume_at" in paused
    assert any(c["reason"] == "retry time reached" for c in changes)
    assert llm.calls == [0, 1, 1, 2]  # step 1 re-sampled: nothing of it was committed


async def test_the_retry_wait_survives_a_daemon_restart(env: Env) -> None:
    llm = ScriptedLLM(solve_script(ROMAN.root, "roman.py"))
    llm.fail_once[0] = QuotaExhausted("daily quota", retry_after_s=1.0)
    d1 = await env.start(llm)
    job_id = await submit(d1)
    await wait_state(d1, job_id, ["paused"])
    await d1.shutdown()
    d2 = await env.start(llm)
    job = await wait_state(d2, job_id, ["done", "failed"])
    assert job.verdict.value == "pass"


async def test_a_manual_resume_before_the_wait_ends_is_fine(env: Env) -> None:
    llm = ScriptedLLM(solve_script(ROMAN.root, "roman.py"))
    llm.fail_once[0] = QuotaExhausted("daily quota", retry_after_s=3600.0)
    d = await env.start(llm)
    job_id = await submit(d)
    await wait_state(d, job_id, ["paused"])
    async with await Client.connect(str(d.socket_path)) as c:
        await c.request("resume", job_id=job_id)
    job = await wait_state(d, job_id, ["done", "failed"])
    assert job.verdict.value == "pass"
