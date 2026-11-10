import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from switchboard.agents.tools import (
    MAX_OUTPUT_CHARS,
    TOOL_DEFS,
    TOOL_SPECS,
    CommandRejected,
    RunResult,
    ToolBox,
    parse_pytest_command,
    resolve_in_workdir,
    truncate,
    validate_args,
)
from switchboard.runtime.agent import ReplayPolicy

pytestmark = pytest.mark.asyncio


class RecordingRunner:
    def __init__(self, result: RunResult | None = None) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.result = result or RunResult(0, "1 passed in 0.01s\n")

    async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResult:
        self.calls.append((list(argv), timeout_s))
        return self.result


@pytest.fixture
def work(tmp_path: Path) -> Path:
    w = tmp_path / "work"
    (w / "tests").mkdir(parents=True)
    (w / "a.py").write_text("x = 1\n")
    (w / "tests" / "test_a.py").write_text("def test(): pass\n")
    return w


# ---------------------------------------------------------------- definitions


async def test_every_tool_is_safe_and_has_a_schema() -> None:
    assert {s.name for s in TOOL_SPECS} == {
        "run_tests",
        "read_file",
        "write_file",
        "list_files",
        "finish",
    }
    assert all(s.replay is ReplayPolicy.SAFE for s in TOOL_SPECS)
    assert all(t.parameters["type"] == "object" for t in TOOL_DEFS)


# ---------------------------------------------------------------- allowlist (ported from the spike)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pytest", []),
        ("pytest -q", ["-q"]),
        ("python -m pytest -q", ["-q"]),
        ("python3 -m pytest -x -v", ["-x", "-v"]),
        ("pytest -q tests/test_a.py", ["-q", "tests/test_a.py"]),
        ("pytest tests/test_a.py::test", ["tests/test_a.py::test"]),
        ("pytest -q --tb=short", ["-q", "--tb=short"]),
        ("pytest -k 'foo bar'", ["-k", "foo bar"]),
        ("pytest -ra -l -s", ["-ra", "-l", "-s"]),
    ],
)
async def test_allowlist_accepts(command: str, expected: list[str], work: Path) -> None:
    assert parse_pytest_command(command, work) == expected


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "rm -rf /",
        "python -c 'print(1)'",
        "python script.py",
        "pytest; rm -rf /",
        "pytest && echo hi",
        "pytest | cat",
        "pytest $(whoami)",
        "pytest `id`",
        "pytest > out.txt",
        "pytest *.py",
        "pytest -p evil_plugin",
        "pytest -c /etc/passwd",
        "pytest --rootdir=/",
        "pytest --tb=weird",
        "pytest -k",
        "pytest -k 'a or __import__(\"os\")'",
        "pytest /etc/passwd",
        "pytest ../outside",
        "pytest tests/../../outside.py",
        "pytest 'unterminated",
        "",
    ],
)
async def test_allowlist_rejects(command: str, work: Path) -> None:
    with pytest.raises(CommandRejected):
        parse_pytest_command(command, work)


async def test_allowlist_rejects_symlink_out_of_the_workdir(work: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "link").symlink_to(outside)
    with pytest.raises(CommandRejected, match="outside"):
        parse_pytest_command("pytest link", work)


# ---------------------------------------------------------------- schema validation


@pytest.mark.parametrize(
    ("tool", "args", "fragment"),
    [
        ("read_file", {}, "missing required"),
        ("read_file", {"path": 3}, "must be string"),
        ("read_file", {"path": "a", "extra": 1}, "unexpected keys"),
        ("write_file", {"path": "a"}, "missing required"),
        ("run_tests", {"timeout_s": "5"}, "must be integer"),
        ("run_tests", {"timeout_s": True}, "must be integer"),
        ("run_tests", {"timeout_s": 0}, ">= 1"),
        ("run_tests", {"timeout_s": 10_000}, "<= 600"),
        ("finish", {"summary": None}, "must be string"),
    ],
)
async def test_invalid_input_is_rejected_before_running(
    tool: str, args: dict[str, object], fragment: str, work: Path
) -> None:
    runner = RecordingRunner()
    out = await ToolBox(work, runner).execute(tool, args)
    assert out.is_error
    assert fragment in out.content
    assert runner.calls == []


async def test_unknown_tool_is_an_error_not_a_crash(work: Path) -> None:
    out = await ToolBox(work, RecordingRunner()).execute("rm_rf", {})
    assert out.is_error
    assert "unknown tool" in out.content


async def test_validate_args_subset() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "number"},
            "xs": {"type": "array", "items": {"type": "string"}},
            "mode": {"enum": ["a", "b"]},
        },
    }
    assert validate_args(schema, {"n": 1.5, "xs": ["a"], "mode": "a"}) is None
    assert validate_args(schema, {"xs": [1]}) == "'xs'[0] must be string, got int"
    assert validate_args(schema, {"mode": "c"}) is not None
    assert validate_args(schema, []) is not None


# ---------------------------------------------------------------- files


async def test_read_write_list_roundtrip(work: Path) -> None:
    box = ToolBox(work, RecordingRunner())
    assert (await box.execute("read_file", {"path": "a.py"})).content == "x = 1\n"
    out = await box.execute("write_file", {"path": "pkg/new.py", "contents": "y = 2\n"})
    assert not out.is_error
    assert (work / "pkg" / "new.py").read_text() == "y = 2\n"
    listing = (await box.execute("list_files", {})).content.splitlines()
    assert listing == ["a.py", "pkg/new.py", "tests/test_a.py"]
    sub = (await box.execute("list_files", {"path": "tests"})).content
    assert sub == "tests/test_a.py"


async def test_list_files_skips_noise_and_does_not_follow_symlinks(
    work: Path, tmp_path: Path
) -> None:
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "x.pyc").write_text("")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    (work / "linkdir").symlink_to(outside)
    (work / "linkfile").symlink_to(outside / "secret.txt")
    listing = (await ToolBox(work, RecordingRunner()).execute("list_files", {})).content
    assert "secret.txt" not in listing
    assert "__pycache__" not in listing
    assert "linkfile -> (symlink)" in listing


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../../x", "", "a\0b", "tests/../.."])
async def test_path_escapes_are_rejected(path: str, work: Path) -> None:
    box = ToolBox(work, RecordingRunner())
    for tool, args in (
        ("read_file", {"path": path}),
        ("write_file", {"path": path, "contents": "x"}),
        ("list_files", {"path": path}),
    ):
        out = await box.execute(tool, args)
        assert out.is_error, (tool, path)


async def test_symlink_escapes_are_rejected(work: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (work / "link.txt").symlink_to(secret)
    (work / "linkdir").symlink_to(tmp_path)
    (work / "dangling").symlink_to(tmp_path / "nonexistent.txt")
    box = ToolBox(work, RecordingRunner())
    assert (await box.execute("read_file", {"path": "link.txt"})).is_error
    assert (await box.execute("read_file", {"path": "linkdir/secret.txt"})).is_error
    assert (await box.execute("write_file", {"path": "link.txt", "contents": "pwn"})).is_error
    assert (await box.execute("write_file", {"path": "linkdir/new.txt", "contents": "p"})).is_error
    assert (await box.execute("write_file", {"path": "dangling", "contents": "p"})).is_error
    assert secret.read_text() == "top secret"
    assert not (tmp_path / "new.txt").exists()
    assert not (tmp_path / "nonexistent.txt").exists()


async def test_resolve_allows_an_inner_symlink(work: Path) -> None:
    (work / "inner").symlink_to(work / "tests")
    assert resolve_in_workdir(work, "inner/test_a.py") == (work / "tests" / "test_a.py").resolve()


async def test_missing_file_and_directory_errors_are_reported(work: Path) -> None:
    box = ToolBox(work, RecordingRunner())
    assert (await box.execute("read_file", {"path": "nope.py"})).is_error
    assert (await box.execute("read_file", {"path": "tests"})).is_error
    assert (await box.execute("list_files", {"path": "a.py"})).is_error
    assert (await box.execute("write_file", {"path": ".", "contents": "x"})).is_error


async def test_oversized_write_is_refused(work: Path) -> None:
    out = await ToolBox(work, RecordingRunner()).execute(
        "write_file", {"path": "big", "contents": "x" * 1_000_001}
    )
    assert out.is_error
    assert not (work / "big").exists()


# ---------------------------------------------------------------- run_tests and output


async def test_run_tests_goes_through_the_runner_without_a_shell(work: Path) -> None:
    runner = RecordingRunner()
    out = await ToolBox(work, runner).execute(
        "run_tests", {"command": "pytest -q tests/test_a.py", "timeout_s": 5}
    )
    assert runner.calls == [(["pytest", "-q", "tests/test_a.py"], 5)]
    assert out.content == "exit code: 0\n1 passed\n"  # timing scrubbed
    assert not out.is_error


async def test_run_tests_defaults(work: Path) -> None:
    runner = RecordingRunner()
    await ToolBox(work, runner).execute("run_tests", {})
    assert runner.calls == [(["pytest", "-q"], 60)]


async def test_rejected_command_never_reaches_the_runner(work: Path) -> None:
    runner = RecordingRunner()
    out = await ToolBox(work, runner).execute("run_tests", {"command": "rm -rf /"})
    assert out.is_error
    assert out.content.startswith("command rejected")
    assert runner.calls == []


async def test_timeout_is_an_error_outcome(work: Path) -> None:
    runner = RecordingRunner(RunResult(-9, "", timed_out=True))
    out = await ToolBox(work, runner).execute("run_tests", {"timeout_s": 3})
    assert out.is_error
    assert "timed out after 3s" in out.content


async def test_runner_output_is_scrubbed_and_truncated(work: Path) -> None:
    noisy = f"{work}/a.py:1: boom\n" + "x" * 100_000 + "\n== 1 failed in 12.34s ==\n"
    runner = RecordingRunner(RunResult(1, noisy))
    out = await ToolBox(work, runner).execute("run_tests", {})
    assert str(work) not in out.content
    assert "./a.py:1: boom" in out.content  # head kept
    assert out.content.endswith("== 1 failed ==\n")  # tail kept, timing gone
    assert "chars truncated" in out.content
    assert len(out.content) < MAX_OUTPUT_CHARS + 200


async def test_truncate_keeps_head_and_tail() -> None:
    text = "H" * 50 + "M" * 1000 + "T" * 50
    out = truncate(text, 100)
    assert out.startswith("H" * 50)
    assert out.endswith("T" * 50)
    assert "1000 chars truncated" in out
    assert truncate("short", 100) == "short"


async def test_finish_sets_done(work: Path) -> None:
    out = await ToolBox(work, RecordingRunner()).execute("finish", {"summary": "ok"})
    assert out.done
    assert not out.is_error


async def test_execute_is_a_pure_function_of_call_and_environment(work: Path) -> None:
    a = ToolBox(work, RecordingRunner())
    first = [await a.execute("list_files", {}), await a.execute("read_file", {"path": "a.py"})]
    second = [await a.execute("list_files", {}), await a.execute("read_file", {"path": "a.py"})]
    assert first == second


async def test_cancellation_propagates_from_a_tool(work: Path) -> None:
    class Slow:
        async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResult:
            await asyncio.sleep(3600)
            return RunResult(0, "")

    task = asyncio.ensure_future(ToolBox(work, Slow()).execute("run_tests", {}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
