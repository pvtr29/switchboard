"""Snapshot/restore latency of DirEnv: detected copy method vs the copytree fallback.

Usage: uv run python benchmarks/env_snapshot.py [--iterations N]
"""

import argparse
import asyncio
import os
import statistics
import tempfile
import time
from pathlib import Path

from switchboard.env import dir_env
from switchboard.env.dir_env import COPY_COPYTREE, DirEnv

CASES: list[tuple[str, int, int]] = [  # (label, files, bytes per file)
    ("100 x 1 KB", 100, 1024),
    ("1,000 x 1 KB", 1_000, 1024),
    ("10,000 x 1 KB", 10_000, 1024),
    ("1 x 50 MB", 1, 50 * 1024 * 1024),
]


def build_template(path: Path, files: int, size: int) -> None:
    payload = os.urandom(min(size, 1 << 20))
    for i in range(files):
        d = path / f"d{i % 50}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"f{i}.bin", "wb") as f:
            for _ in range(max(1, size // len(payload))):
                f.write(payload[: min(size, len(payload))])


def pct(samples: list[float], q: float) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


async def measure(env: DirEnv, iterations: int) -> tuple[list[float], list[float]]:
    await env.provision()
    snaps: list[float] = []
    restores: list[float] = []
    for i in range(iterations):
        t = time.perf_counter()
        await env.snapshot(f"s{i}")
        snaps.append((time.perf_counter() - t) * 1000)
        (env.workdir / "dirty").write_text(str(i))
        t = time.perf_counter()
        await env.restore(f"s{i}")
        restores.append((time.perf_counter() - t) * 1000)
    return snaps, restores


async def measure_skip(env: DirEnv, iterations: int) -> tuple[list[float], list[float]]:
    """Cost of proving "unchanged" (`changed_since`) next to the snapshot it replaces."""
    await env.provision()
    snaps: list[float] = []
    checks: list[float] = []
    for i in range(iterations):
        t = time.perf_counter()
        await env.snapshot(f"k{i}")
        snaps.append((time.perf_counter() - t) * 1000)
        await asyncio.sleep(0.01)
        t = time.perf_counter()
        assert await env.changed_since(f"k{i}") is False
        checks.append((time.perf_counter() - t) * 1000)
    return snaps, checks


async def main(iterations: int) -> None:
    rows: list[str] = []
    with tempfile.TemporaryDirectory(prefix="sb-bench-") as tmp:
        root = Path(tmp)
        probe = DirEnv(root, root / "probe", "p")  # detect once, on this filesystem
        (root / "probe").mkdir()
        await probe.provision()
        detected = dir_env._detected
        for label, files, size in CASES:
            template = root / f"t-{files}-{size}"
            template.mkdir()
            await asyncio.to_thread(build_template, template, files, size)
            n = max(3, iterations // 5) if size > 1 << 20 or files >= 10_000 else iterations
            for method in ("detected", COPY_COPYTREE):
                if method == COPY_COPYTREE:
                    dir_env._detected = COPY_COPYTREE
                else:
                    dir_env._detected = detected
                env = DirEnv(template, root / "envs", f"j-{files}-{method[:3]}")
                snaps, restores = await measure(env, n)
                used = env.copy_method
                rows.append(
                    f"{label:<15}{used:<22}{n:>4}"
                    f"{statistics.median(snaps):>12.1f}{pct(snaps, 0.99):>10.1f}"
                    f"{statistics.median(restores):>13.1f}{pct(restores, 0.99):>10.1f}"
                )
                await env.destroy()
    print(
        f"{'case':<15}{'method':<22}{'n':>4}{'snap p50':>12}{'p99':>10}"
        f"{'restore p50':>13}{'p99':>10}  (ms)"
    )
    print("\n".join(rows))
    print()
    print(
        "skip-unchanged snapshots: a step that changed nothing pays changed_since, not a snapshot"
    )
    print(f"{'case':<15}{'snapshot p50':>14}{'check p50':>12}{'check p99':>11}{'saved/step':>12}")
    dir_env.RACY_NS = 0  # fresh template files are older than the scan only after a moment
    with tempfile.TemporaryDirectory(prefix="sb-bench-") as tmp:
        root = Path(tmp)
        for label, files, size in CASES[:3]:
            template = root / f"t-{files}-{size}"
            template.mkdir()
            await asyncio.to_thread(build_template, template, files, size)
            env = DirEnv(template, root / "envs", f"k-{files}")
            snaps, checks = await measure_skip(env, max(5, iterations // 2))
            s50, c50 = statistics.median(snaps), statistics.median(checks)
            print(
                f"{label:<15}{s50:>14.1f}{c50:>12.1f}{pct(checks, 0.99):>11.1f}"
                f"{(1 - c50 / s50) * 100:>11.0f}%"
            )
            await env.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=20)
    asyncio.run(main(ap.parse_args().iterations))
