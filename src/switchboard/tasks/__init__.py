"""The SWE task suite: small buggy repositories with visible and hidden tests.

Each task is a directory under `suite/`:

- `task.json`: `prompt` (the agent's first message) and `editable` (the files the fix may
  touch, relative to the repo root);
- `repo/`: what the agent gets as its environment: the buggy source plus the VISIBLE tests;
- `hidden/`: extra tests the agent never sees, run only by the grader. They catch fixes that
  satisfy the visible tests without fixing the bug (special-casing, partial fixes);
- `solution/`: a reference fix (copied over `repo/`), used by the suite's own tests to prove
  every task is solvable and that the buggy version fails.

The agent's sandbox cannot read this directory (it lives under the home directory, which the
sandbox denies), and the agent's file tools are confined to its workdir.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SUITE_DIR = Path(__file__).parent / "suite"


class UnknownTask(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    prompt: str
    editable: tuple[str, ...]
    root: Path

    @property
    def repo(self) -> Path:
        return self.root / "repo"

    @property
    def hidden(self) -> Path:
        return self.root / "hidden"

    @property
    def solution(self) -> Path:
        return self.root / "solution"


def task_names() -> list[str]:
    return sorted(p.name for p in SUITE_DIR.iterdir() if (p / "task.json").is_file())


def load_task(name: str) -> Task:
    root = SUITE_DIR / name
    meta_path = root / "task.json"
    if "/" in name or name.startswith(".") or not meta_path.is_file():
        raise UnknownTask(name)
    meta = json.loads(meta_path.read_text("utf-8"))
    editable = tuple(meta["editable"])
    for rel in editable:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"task {name}: editable path {rel!r} escapes the repo")
    return Task(name=name, prompt=str(meta["prompt"]), editable=editable, root=root)


__all__ = ["SUITE_DIR", "Task", "UnknownTask", "load_task", "task_names"]
