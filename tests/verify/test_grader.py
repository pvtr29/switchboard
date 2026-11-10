"""The task suite and the hidden-test grader.

Every task must be solvable (reference solution passes visible + hidden tests) and broken as
shipped. The grader must not be fooled by an agent that edits tests or adds pytest config, and
must catch a fix that only satisfies the visible tests.
"""

import shutil
import sys
from pathlib import Path

import pytest
from rollout_fakes import PlainRunner

from switchboard.agents.tools import SANDBOX_EXEC, LocalSandboxRunner
from switchboard.core import Verdict
from switchboard.tasks import UnknownTask, load_task, task_names
from switchboard.verify.grader import Grade, changed_files, grade, parse_counts

TASKS = task_names()


def agent_copy(name: str, tmp_path: Path, *, solved: bool) -> Path:
    task = load_task(name)
    work = tmp_path / "work"
    shutil.copytree(task.repo, work)
    if solved:
        for f in task.solution.iterdir():
            shutil.copyfile(f, work / f.name)
    return work


async def run_grade(name: str, work: Path) -> Grade:
    return await grade(load_task(name), work, PlainRunner)


def test_the_suite_has_tasks_with_prompts_and_hidden_tests() -> None:
    assert len(TASKS) >= 4
    for name in TASKS:
        t = load_task(name)
        assert t.prompt
        assert t.editable
        assert any(t.hidden.iterdir())
        for rel in t.editable:
            assert (t.repo / rel).is_file()
            assert (t.solution / rel).is_file()


def test_unknown_and_unsafe_task_names_are_rejected() -> None:
    for bad in ("nope", "../tasks", ".hidden", "roman/repo"):
        with pytest.raises(UnknownTask):
            load_task(bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", TASKS)
async def test_shipped_bug_fails_and_reference_solution_passes(name: str, tmp_path: Path) -> None:
    broken = await run_grade(name, agent_copy(name, tmp_path / "a", solved=False))
    assert broken.verdict is Verdict.FAIL
    assert broken.failed > 0
    fixed = await run_grade(name, agent_copy(name, tmp_path / "b", solved=True))
    assert fixed.verdict is Verdict.PASS_, fixed.output
    assert fixed.failed == 0
    assert fixed.passed > 0
    assert fixed.ignored_changes == ()


@pytest.mark.asyncio
async def test_hidden_tests_catch_a_fix_that_only_satisfies_the_visible_ones(
    tmp_path: Path,
) -> None:
    # lru: fixing only `get` makes the visible tests pass; `put` refreshing recency is hidden.
    work = agent_copy("lru", tmp_path, solved=False)
    src = (work / "lru.py").read_text()
    src = src.replace(
        "        return self._data.get(key, default)",
        "        if key not in self._data:\n            return default\n"
        "        self._data.move_to_end(key)\n        return self._data[key]",
    )
    (work / "lru.py").write_text(src)
    visible = await PlainRunner(work).run(["pytest", "-q"], timeout_s=60)
    assert visible.exit_code == 0, visible.output  # the agent would see green
    g = await run_grade("lru", work)
    assert g.verdict is Verdict.FAIL
    assert g.failed == 1


@pytest.mark.asyncio
async def test_edited_tests_and_conftest_do_not_count_but_are_reported(tmp_path: Path) -> None:
    work = agent_copy("roman", tmp_path, solved=False)
    (work / "tests" / "test_roman.py").write_text("def test_ok() -> None:\n    pass\n")
    (work / "conftest.py").write_text(
        "import pytest\n\ndef pytest_collection_modifyitems(items):\n"
        "    for i in items:\n        i.add_marker(pytest.mark.skip)\n"
    )
    (work / "pytest.ini").write_text("[pytest]\naddopts = --co\n")
    visible = await PlainRunner(work).run(["pytest", "-q"], timeout_s=60)
    assert visible.exit_code != 0 or "failed" not in visible.output  # hacked green in the env
    g = await run_grade("roman", work)
    assert g.verdict is Verdict.FAIL
    assert g.ignored_changes == ("conftest.py", "pytest.ini", "tests/test_roman.py")


@pytest.mark.asyncio
async def test_a_symlinked_editable_file_counts_as_deleted(tmp_path: Path) -> None:
    work = agent_copy("roman", tmp_path, solved=True)
    target = tmp_path / "elsewhere.py"
    shutil.copyfile(work / "roman.py", target)
    (work / "roman.py").unlink()
    (work / "roman.py").symlink_to(target)
    g = await run_grade("roman", work)
    assert g.verdict is Verdict.FAIL  # the module is missing in the grading tree
    assert g.passed == 0


@pytest.mark.asyncio
async def test_grading_never_writes_to_the_agents_workdir(tmp_path: Path) -> None:
    work = agent_copy("intervals", tmp_path, solved=True)
    before = sorted(p.relative_to(work).as_posix() for p in work.rglob("*"))
    await run_grade("intervals", work)
    assert sorted(p.relative_to(work).as_posix() for p in work.rglob("*")) == before


def test_payload_is_json_shaped() -> None:
    g = Grade(Verdict.PASS_, 3, 0, 0, 0, False, ("x.py",), "3 passed")
    assert g.to_payload() == {
        "verdict": "pass",
        "passed": 3,
        "failed": 0,
        "errors": 0,
        "exit_code": 0,
        "timed_out": False,
        "ignored_changes": ["x.py"],
        "output": "3 passed",
    }


@pytest.mark.parametrize(
    ("output", "counts"),
    [
        ("..F\n1 failed, 2 passed in 0.1s", (2, 1, 0)),
        ("12 passed in 0.06s", (12, 0, 0)),
        ("1 passed, 2 errors in 0.2s", (1, 0, 2)),
        ("1 error in 0.2s\n", (0, 0, 1)),
        ("no tests ran in 0.01s", (0, 0, 0)),
        ("", (0, 0, 0)),
    ],
)
def test_parse_counts(output: str, counts: tuple[int, int, int]) -> None:
    assert parse_counts(output) == counts


def test_changed_files(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "pkg" / "__pycache__").mkdir(parents=True)
        (d / "same.py").write_text("x")
    (a / "edited.py").write_text("1")
    (b / "edited.py").write_text("2")
    (a / "removed.py").write_text("")
    (b / "added.py").write_text("")
    (b / "pkg" / "__pycache__" / "junk.pyc").write_text("")
    assert changed_files(a, b) == ["added.py", "edited.py", "removed.py"]


@pytest.mark.asyncio
@pytest.mark.skipif(SANDBOX_EXEC is None, reason="needs macOS sandbox-exec")
async def test_grading_in_the_local_sandbox(tmp_path: Path) -> None:
    assert sys.platform == "darwin"
    work = agent_copy("slugify", tmp_path, solved=True)
    g = await grade(load_task("slugify"), work, LocalSandboxRunner)
    assert g.verdict is Verdict.PASS_, g.output
