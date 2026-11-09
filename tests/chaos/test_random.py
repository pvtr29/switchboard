"""Random chaos mode (real kill -9 of a daemon subprocess) at a small scale.

Runs in the default suite because it takes a few seconds. The plan's scale (200 jobs, 20 kills)
is `uv run python scripts/chaos.py random --jobs 200 --kills 20`; `pytest -m chaos` runs only
the tests marked chaos.
"""

from pathlib import Path

import pytest

from switchboard.chaos.random_mode import RandomConfig, run_random

pytestmark = [pytest.mark.asyncio, pytest.mark.chaos]


async def test_small_random_run_keeps_every_invariant(sock_dir: Path) -> None:
    cfg = RandomConfig(
        jobs=10,
        kills=3,
        seed=3,
        steps=12,
        token_interval=0.02,
        min_interval=0.2,
        max_interval=0.5,
        timeout=60.0,
    )
    result = await run_random(cfg, base=sock_dir)
    assert result["failures"] == []
    assert result["ok"]
    assert result["kills_done"] == 3
    m = result["metrics"]
    assert m["states"] == {"verifying": 10}
    assert m["env_mismatches"] == 0
    assert m["duplicate_keys"] == 0
    assert m["sink_applications"] == m["sink_distinct_keys"] == 10 * 12
    assert result["recovery_requeued_total"] > 0  # the kills really interrupted running jobs
