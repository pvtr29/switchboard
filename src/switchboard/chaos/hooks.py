"""Crash injection: where in a step a simulated kill -9 lands.

`Crash` derives from BaseException so nothing in the runtime treats it as an agent error: it
unwinds like a kill (nothing is written after it), and `run_job` called directly (no
supervisor, which would record a crash as a failure) leaves the log exactly as a real kill
would. The windows, per step n (docs/plan.md, commit 19):

    W0   mid-sample: during the token stream, before llm_responded(n)
    W1   after llm_responded(n) was committed, before act starts
    W2   mid-act, after the first file tool wrote (a partial act)
    W2b  mid-act, after every file tool wrote, before the external effect
    W3a  before the effect_intent is written
    W3b  after effect_intent, before the sink applied anything
    W3c  after the sink applied, before effect_outcome is recorded
    W3d  after effect_outcome, before the step commit
    W4   after step_completed(n) was committed, before step n+1 starts
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from switchboard.agents.fake import FakeAgent
from switchboard.core import Event, EventType
from switchboard.env.dir_env import DirEnv
from switchboard.runtime.agent import ActResult, ModelResponse, ReplayPolicy, StepContext
from switchboard.runtime.effects import ExternalSink
from switchboard.store import AppendResult, EventStore


class Crash(BaseException):
    """Stands in for kill -9 (BaseException: no `except Exception` in the runtime catches it)."""


class Window(StrEnum):
    W0 = "W0"
    W1 = "W1"
    W2 = "W2"
    W2B = "W2b"
    W3A = "W3a"
    W3B = "W3b"
    W3C = "W3c"
    W3D = "W3d"
    W4 = "W4"


DESCRIPTIONS: dict[Window, str] = {
    Window.W0: "mid-sample (token stream)",
    Window.W1: "after llm_responded, before act",
    Window.W2: "mid-act, after the first file write",
    Window.W2B: "mid-act, after all file writes",
    Window.W3A: "before effect_intent",
    Window.W3B: "after effect_intent, before sink apply",
    Window.W3C: "after sink apply, before effect_outcome",
    Window.W3D: "after effect_outcome, before step commit",
    Window.W4: "after step commit, before next step",
}


@dataclass(frozen=True, slots=True)
class CrashPlan:
    window: Window
    step: int


class CrashingStore(EventStore):
    """An EventStore that can raise Crash right AFTER a group was committed (W4)."""

    crash_after_step_commit: int | None = None

    async def append_group(self, events: Sequence[Event]) -> list[AppendResult]:
        results = await super().append_group(events)
        step = self.crash_after_step_commit
        if step is not None and any(
            e.type is EventType.STEP_COMPLETED and e.step_id == step for e in events
        ):
            self.crash_after_step_commit = None
            raise Crash
        return results


class NoProvisionEnv(DirEnv):
    """The naive baseline's environment: `provision` keeps whatever the dead process left in
    the workdir (a restart from scratch with no environment restore)."""

    async def provision(self) -> Path:
        await self._prepare()
        return self.workdir


class ChaosAgent(FakeAgent):
    """A FakeAgent that raises `Crash` once, in `plan.window` of `plan.step`, on attempt 0.

    W3b and W3c fire from the sink's before_apply / after_apply hooks (this class's methods of the
    same names), which see
    `effect_armed` while this agent's external effect for the planned step is in flight. W4 is
    the store's job (`CrashingStore`). The same hooks work against the naive baseline, where the
    agent calls the sink directly and no intent or outcome exists: the windows keep their
    position in the step, so the baseline is crashed at the same points.
    """

    def __init__(
        self,
        *,
        steps: int,
        plan: CrashPlan | None = None,
        token_interval: float = 0.0,
        effect_every: int = 1,
        effect_policy: ReplayPolicy = ReplayPolicy.KEYED,
        naive_sink: ExternalSink | None = None,
    ) -> None:
        super().__init__(
            steps=steps,
            token_interval=token_interval,
            on_token=self._on_token,
            on_file_write=self._on_file_write,
            effect_every=effect_every,
            effect_policy=effect_policy,
            naive_sink=naive_sink,
        )
        self.plan = plan
        self.effect_armed = False
        self.fired = False

    def _armed(self, ctx: StepContext, window: Window) -> bool:
        p = self.plan
        return (
            p is not None
            and p.window is window
            and ctx.step == p.step
            and ctx.attempt == 0
            and not self.fired
        )

    def _fire(self) -> None:
        self.fired = True
        raise Crash

    def _on_token(self, ctx: StepContext, i: int) -> None:
        if i == 1 and self._armed(ctx, Window.W0):
            self._fire()

    def _on_file_write(self, ctx: StepContext, tool: str, path: Path) -> None:
        if tool == "write_file" and self._armed(ctx, Window.W2):
            self._fire()

    async def act(self, ctx: StepContext, response: ModelResponse) -> ActResult:
        if self._armed(ctx, Window.W1):
            self.act_calls.append((ctx.step, ctx.attempt))
            self._fire()
        return await super().act(ctx, response)

    async def _file_effects(
        self, ctx: StepContext, response: ModelResponse
    ) -> list[dict[str, Any]]:
        out = await super()._file_effects(ctx, response)
        if self._armed(ctx, Window.W2B):
            self._fire()
        return out

    async def _external_effect(self, ctx: StepContext, response: ModelResponse) -> dict[str, Any]:
        if self._armed(ctx, Window.W3A):
            self._fire()
        self.effect_armed = any(self._armed(ctx, w) for w in (Window.W3B, Window.W3C))
        try:
            out = await super()._external_effect(ctx, response)
        finally:
            self.effect_armed = False
        if self._armed(ctx, Window.W3D):
            self._fire()
        return out

    def before_apply(self, key: str) -> None:
        """Install as the sink's before_apply: window W3b."""
        if self.effect_armed and self.plan is not None and self.plan.window is Window.W3B:
            self._fire()

    def after_apply(self, key: str) -> None:
        """Install as the sink's after_apply: window W3c."""
        if self.effect_armed and self.plan is not None and self.plan.window is Window.W3C:
            self._fire()
