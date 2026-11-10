"""LocalSandboxRunner against the real macOS sandbox. Skipped elsewhere."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from switchboard.agents import tools
from switchboard.agents.tools import CommandRejected, LocalSandboxRunner, ToolBox

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
        reason="needs macOS sandbox-exec",
    ),
]

# The probe lives in the workdir like any test the agent could write.
PROBE = """
import os, pathlib, socket

def test_network_blocked():
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect(("1.1.1.1", 443))
    except OSError as e:
        assert e.errno in (1, 13), e  # EPERM/EACCES from the sandbox, not a timeout
    else:
        raise AssertionError("network allowed")

def test_write_outside_blocked():
    try:
        pathlib.Path("__OUTSIDE__").write_text("x")
    except PermissionError:
        pass
    else:
        raise AssertionError("write outside workdir allowed")

def test_home_read_blocked():
    try:
        pathlib.Path("__HOME_FILE__").read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("home read allowed")

def test_no_secrets():
    secrets = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "SB_PROBE_SECRET"}
    assert not secrets & set(os.environ)

def test_workdir_write_ok():
    pathlib.Path("ok.txt").write_text("fine")
"""


def make_probe(work: Path, outside: Path, home_file: Path) -> None:
    (work / "test_probe.py").write_text(
        PROBE.replace("__OUTSIDE__", str(outside)).replace("__HOME_FILE__", str(home_file))
    )


@pytest.fixture
def home_file() -> Path:
    # a real file under the user's home, outside every allowed path
    return Path.home() / ".sb_sandbox_probe_target"


OUTSIDE = Path("/tmp/sb_escape_probe.txt")


def arm_probe(work: Path, home_file: Path) -> None:
    OUTSIDE.unlink(missing_ok=True)
    work.mkdir()
    make_probe(work, OUTSIDE, home_file)
    try:
        home_file.write_text("private")
    except OSError:
        pytest.skip("cannot create a probe file in the home directory")


def disarm_probe(home_file: Path) -> None:
    home_file.unlink(missing_ok=True)
    OUTSIDE.unlink(missing_ok=True)


def probe_findings(work: Path) -> tuple[str, bool, list[str]]:
    return (
        (work / "ok.txt").read_text(),
        OUTSIDE.exists(),
        sorted(p.name for p in work.iterdir()),
    )


async def test_sandbox_blocks_network_outside_writes_home_reads_and_secrets(
    tmp_path: Path, home_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SB_PROBE_SECRET", "hunter2")
    monkeypatch.setenv("GEMINI_API_KEY", "hunter2")
    work = tmp_path / "work"
    arm_probe(work, home_file)
    try:
        result = await LocalSandboxRunner(work).run(["pytest", "-q"], timeout_s=120)
        assert not result.timed_out
        assert result.exit_code == 0, result.output
        assert "5 passed" in result.output
        written, escaped, names = probe_findings(work)
        assert written == "fine"
        assert not escaped
        # the sandbox leaves no noise in the workdir
        assert names == ["ok.txt", "test_probe.py"]
    finally:
        disarm_probe(home_file)


async def test_runner_refuses_other_programs(tmp_path: Path) -> None:
    runner = LocalSandboxRunner(tmp_path)
    with pytest.raises(CommandRejected):
        await runner.run(["sh", "-c", "id"], timeout_s=5)
    with pytest.raises(CommandRejected):
        await runner.run([], timeout_s=5)


async def test_timeout_kills_the_process_group(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(
        "import subprocess, sys, time\ndef test_slow():\n    time.sleep(60)\n"
    )
    result = await LocalSandboxRunner(tmp_path).run(["pytest", "-q"], timeout_s=2)
    assert result.timed_out


async def test_toolbox_runs_real_pytest_in_the_sandbox(tmp_path: Path) -> None:
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 1 + 1 == 3\n")
    out = await ToolBox(tmp_path, LocalSandboxRunner(tmp_path)).execute("run_tests", {})
    assert not out.is_error
    assert out.content.startswith("exit code: 1")
    assert "assert (1 + 1) == 3" in out.content or "assert 2 == 3" in out.content
    assert str(tmp_path) not in out.content


async def test_runner_refuses_without_sandbox_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "SANDBOX_EXEC", None)
    with pytest.raises(CommandRejected, match="no OS sandbox"):
        await LocalSandboxRunner(tmp_path).run(["pytest"], timeout_s=5)
    out = await ToolBox(tmp_path, LocalSandboxRunner(tmp_path)).execute("run_tests", {})
    assert out.is_error
    assert "no OS sandbox" in out.content


async def test_environment_is_minimal(tmp_path: Path) -> None:
    env = tools.sandbox_env(tmp_path)
    assert not {"GEMINI_API_KEY", "GOOGLE_API_KEY"} & set(env)
    assert set(env) <= {
        "PATH", "HOME", "TMPDIR", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
        "PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }  # fmt: skip
    assert os.environ.get("HOME") != env["HOME"]
