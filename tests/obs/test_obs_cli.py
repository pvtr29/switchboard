"""`sb daemon --log-format json --debug-asyncio` and `sb stats`, as a user runs them."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import wait_for_socket

from switchboard.client.cli import build_parser

SB = [sys.executable, "-m", "switchboard.client.cli"]


def test_daemon_flags_parse() -> None:
    args = build_parser().parse_args(
        ["daemon", "--log-format", "json", "--log-level", "DEBUG", "--debug-asyncio"]
    )
    assert (args.log_format, args.log_level, args.debug_asyncio) == ("json", "debug", True)
    defaults = build_parser().parse_args(["daemon"])
    assert (defaults.log_format, defaults.log_level, defaults.debug_asyncio) == (
        "text",
        "info",
        False,
    )


def test_daemon_json_logs_and_stats_output(sock_dir: Path, tmp_path: Path) -> None:
    sock = sock_dir / "s"
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("SB_SOCKET", None)
    log = tmp_path / "daemon.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [
                *SB,
                "daemon",
                "--socket",
                str(sock),
                "--db",
                str(tmp_path / "sb.db"),
                "--envs",
                str(tmp_path / "envs"),
                "--log-format",
                "json",
                "--debug-asyncio",
            ],
            stdout=out,
            stderr=out,
            env=env,
        )
        try:
            wait_for_socket(sock)

            def sb(*args: str) -> str:
                done = subprocess.run(
                    [*SB, "--socket", str(sock), *args],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                    check=True,
                )
                return done.stdout

            job = sb("submit", "--steps", "2", "--token-interval", "0.001").strip()
            deadline = time.monotonic() + 10
            while "state: verifying" not in sb("get", job):
                assert time.monotonic() < deadline
                time.sleep(0.05)
            stats = sb("stats")
        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert lines, "the daemon logged nothing"
    for doc in lines:
        assert {"ts", "level", "logger", "msg"} <= set(doc)
    admitted = [d for d in lines if d["msg"].startswith("job admitted")]
    assert admitted
    assert admitted[0]["job_id"] == job  # `extra` fields become JSON keys
    for needle in (
        "pid: ",
        "jobs.verifying: 1",
        "slots: running=",
        "loop_lag: p50=",
        "[asyncio debug on]",
        "store_commit: p50=",
        "store_ack: p50=",
        "bus: subscribers=",
        "recovery: ",
    ):
        assert needle in stats, (needle, stats)
