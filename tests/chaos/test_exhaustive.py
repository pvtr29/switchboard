"""Exhaustive chaos mode as a regular test: every window of every step, durable vs naive."""

import json
from pathlib import Path

import pytest

from switchboard.chaos.exhaustive import ExhaustiveConfig, format_exhaustive, run_exhaustive
from switchboard.chaos.hooks import Window

pytestmark = pytest.mark.asyncio
STEPS = 3


def stable(result: dict[str, object]) -> str:
    """Everything except timing and paths."""
    return json.dumps({k: v for k, v in result.items() if k != "wall_s"}, sort_keys=True)


async def test_every_crash_point_recovers_exactly(tmp_path: Path) -> None:
    result = await run_exhaustive(ExhaustiveConfig(steps=STEPS), tmp_path / "a")
    assert result["failures"] == []
    s = result["summary"]
    assert s["crash_points"] == STEPS * len(Window)
    assert s["durable_correct"] == s["crash_points"]
    assert s["durable_correct_pct"] == 100.0
    assert s["durable_duplicate_effects"] == 0
    assert s["durable_env_mismatches"] == 0
    assert s["durable_missing_effects"] == 0
    assert s["durable_resampled_logged_responses"] == 0
    assert result["headline"].startswith(f"{s['crash_points']} crash points: durable resume 100%")
    # the naive baseline shows what the journal and the restore are for
    assert s["naive_duplicate_effects"] > 0
    assert s["naive_env_mismatches"] > 0
    assert s["naive_missing_effects"] == 0
    for row in result["durable"]:
        assert row["final_state"] == "verifying"  # no verifier yet
    table = format_exhaustive(result)
    assert "FAKE external sink" in table
    assert "naive resume duplicated" in table


async def test_exhaustive_is_deterministic(tmp_path: Path) -> None:
    cfg = ExhaustiveConfig(steps=2, windows=(Window.W0, Window.W3C, Window.W4))
    first = await run_exhaustive(cfg, tmp_path / "a")
    second = await run_exhaustive(cfg, tmp_path / "b")
    assert stable(first) == stable(second)
