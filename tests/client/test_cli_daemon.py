"""`sb` as a user runs it: subprocesses against a daemon subprocess. Every command gets an
explicit --socket and HOME points into tmp, so nothing here can touch the real ~/.switchboard."""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import wait_for_socket

from switchboard.client.cli import default_socket

SB = [sys.executable, "-m", "switchboard.client.cli"]


class Sb:
    def __init__(self, sock: Path, home: Path) -> None:
        self.sock = sock
        self.env = {**os.environ, "HOME": str(home)}
        self.env.pop("SB_SOCKET", None)

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [*SB, "--socket", str(self.sock), *args],
            capture_output=True,
            text=True,
            timeout=30,
            env=self.env,
            check=False,
        )
        if check:
            assert proc.returncode == 0, proc.stderr
        return proc

    def state(self, job: str) -> str:
        for line in self.run("get", job).stdout.splitlines():
            if line.startswith("state: "):
                return line.removeprefix("state: ")
        raise AssertionError("no state line")

    def wait_state(self, job: str, wanted: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.state(job) == wanted:
                return
            time.sleep(0.05)
        raise AssertionError(f"{job} never reached {wanted}, is {self.state(job)}")


@pytest.fixture
def sb(sock_dir: Path, tmp_path: Path) -> Iterator[Sb]:
    sock = sock_dir / "s"
    home = tmp_path / "home"
    home.mkdir()
    log = (tmp_path / "daemon.log").open("wb")
    tool = Sb(sock, home)
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
        ],
        stdout=log,
        stderr=log,
        env=tool.env,
    )
    try:
        wait_for_socket(sock)
        yield tool
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        assert not (home / ".switchboard").exists()  # the real default location was never used


def test_submit_ps_get(sb: Sb, template: Path) -> None:
    job = sb.run("submit", "--steps", "3", "--seed", "1", "--env-template", str(template)).stdout
    job = job.strip()
    assert job.startswith("job-")
    sb.wait_state(job, "verifying")

    ps = sb.run("ps").stdout.splitlines()
    assert ps[0].split() == ["JOB", "STATE", "STEP", "TOKENS", "USD", "ACTIVE_S", "UPDATED"]
    row = ps[1].split()
    assert row[0] == job
    assert row[1] == "verifying"
    assert row[2] == "3"
    assert int(row[3]) > 0
    assert len(ps) == 2
    empty = sb.run("ps", "--state", "queued").stdout.splitlines()
    assert len(empty) == 1  # header only
    assert empty[0].split() == ps[0].split()
    assert "last_step: 2" in sb.run("get", job).stdout


def test_closing_the_terminal_loses_nothing(sb: Sb) -> None:
    jobs = [sb.run("submit", "--steps", "2").stdout.strip() for _ in range(3)]
    for j in jobs:
        sb.wait_state(j, "verifying")
    first = sb.run("ps").stdout  # each invocation is a new process and a new connection
    second = sb.run("ps").stdout
    assert first == second
    assert {line.split()[0] for line in first.splitlines()[1:]} == set(jobs)


def test_attach_prints_events_until_verifying(sb: Sb) -> None:
    job = sb.run("submit", "--steps", "4", "--token-interval", "0.02").stdout.strip()
    out = sb.run("attach", job).stdout.splitlines()  # exits by itself at verifying
    assert out[0].split()[2:4] == [job, "job_created"]
    assert out[-1].endswith("running -> verifying (agent finished)")
    assert sum("step_completed" in line for line in out) == 4
    seqs = [int(line.split()[0]) for line in out]
    assert seqs == sorted(set(seqs))  # in order, no duplicates
    # attaching to a finished job replays its history and exits again
    again = sb.run("attach", job).stdout.splitlines()
    assert again == out


def test_ctrl_c_detaches_without_affecting_the_job(sb: Sb) -> None:
    job = sb.run("submit", "--steps", "10000", "--token-interval", "0.01").stdout.strip()
    sb.wait_state(job, "running")
    proc = subprocess.Popen(
        [*SB, "--socket", str(sb.sock), "attach", job],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=sb.env,
    )
    assert proc.stdout is not None
    assert "job_created" in proc.stdout.readline()  # it is attached and streaming
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=15)
    assert proc.returncode == 130
    assert "detached" in err
    assert sb.state(job) == "running"
    time.sleep(0.2)
    assert sb.state(job) == "running"  # still going after the detach
    assert sb.run("cancel", job).stdout.strip() == f"{job}: cancelled"


def test_pause_resume_extend_cancel_stats(sb: Sb) -> None:
    job = sb.run("submit", "--steps", "5000", "--token-interval", "0.005").stdout.strip()
    sb.wait_state(job, "running")
    sb.run("pause", job)
    sb.wait_state(job, "paused")
    assert sb.run("resume", job).stdout.strip().startswith(f"{job}:")
    sb.wait_state(job, "running")
    assert sb.run("extend", job, "--tokens", "500", "--seconds", "30").returncode == 0
    stats = sb.run("stats").stdout
    assert "jobs.running: 1" in stats
    assert "pid: " in stats
    sb.run("cancel", job)
    assert sb.state(job) == "cancelled"


def test_budget_flags_and_extend_recover_an_over_budget_job(sb: Sb) -> None:
    job = sb.run("submit", "--steps", "6", "--budget-tokens", "120").stdout.strip()
    sb.wait_state(job, "over_budget")
    sb.run("extend", job, "--tokens", "100000")
    sb.wait_state(job, "verifying")
    sb.run("submit", "--steps", "1", "--budget-seconds", "60", "--budget-usd-micros", "5000000")


def test_errors_have_exit_codes(sb: Sb, sock_dir: Path) -> None:
    missing = sb.run("get", "nope", check=False)
    assert missing.returncode == 1
    assert "not_found" in missing.stderr
    assert sb.run("submit", "--steps", "0", check=False).returncode == 1
    gone = subprocess.run(
        [*SB, "--socket", str(sock_dir / "nobody"), "ps"],
        capture_output=True,
        text=True,
        env=sb.env,
        check=False,
        timeout=15,
    )
    assert gone.returncode == 2
    assert "cannot reach the daemon" in gone.stderr


def test_shutdown_command_stops_the_daemon(sb: Sb) -> None:
    assert "shutting down" in sb.run("shutdown").stdout
    deadline = time.monotonic() + 10
    while sb.sock.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not sb.sock.exists()
    assert sb.run("ps", check=False).returncode == 2


def test_socket_option_works_after_the_subcommand_too(sb: Sb) -> None:
    proc = subprocess.run(
        [*SB, "ps", "--socket", str(sb.sock)],
        capture_output=True,
        text=True,
        env=sb.env,
        check=False,
        timeout=15,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("JOB")


def test_default_socket_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SB_SOCKET", "/somewhere/else.sock")
    assert default_socket() == "/somewhere/else.sock"
    monkeypatch.delenv("SB_SOCKET")
    monkeypatch.setenv("HOME", "/h")
    assert default_socket() == "/h/.switchboard/sb.sock"
