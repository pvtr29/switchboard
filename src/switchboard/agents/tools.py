"""Tools for an LLM agent working on a repository: definitions, validation, execution.

Five tools, all of them local to the job's environment, so all `ReplayPolicy.SAFE`: recovery
restores the environment to S(n) and re-runs the step, which re-executes them identically.

- `run_tests`: only `pytest [safe flags] [in-repo test paths]`, split with shlex and run with
  exec, never a shell. An allowlist alone is not enough (the agent can write a module that
  pytest then imports), so execution goes through a `CommandRunner`, which supplies the
  isolation: `LocalSandboxRunner` (macOS sandbox-exec, below) or, later, a container.
- `read_file` / `write_file` / `list_files`: paths are relative to the workdir; absolute paths,
  `..`, and symlinks that lead outside are rejected.
- `finish`: ends the rollout.

Every tool input is validated against its JSON Schema before anything runs. Output is
truncated head+tail and scrubbed of what varies between runs (timings, the workdir's path), so
`ToolBox.execute` stays a pure function of (call, environment state).
"""

import asyncio
import contextlib
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from switchboard.llm.types import ToolDef
from switchboard.runtime.agent import ReplayPolicy, ToolSpec

MAX_OUTPUT_CHARS = 20_000
MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARS = 1_000_000
MAX_LIST_ENTRIES = 500
DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 600

# --------------------------------------------------------------------------- definitions

TOOL_DEFS: tuple[ToolDef, ...] = (
    ToolDef(
        "run_tests",
        "Run the test suite. Only `pytest [flags] [test paths]` is allowed (flags: -q -v -x -s "
        "-l -ra --tb=short -k EXPR); there is no shell. Returns exit code and output.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "e.g. `pytest -q` (the default) or `pytest -q tests/test_x.py`.",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT_S}).",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_S,
                },
            },
            "additionalProperties": False,
        },
    ),
    ToolDef(
        "read_file",
        "Read a UTF-8 text file. Path is relative to the working directory.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        "write_file",
        "Create or overwrite a file. Path is relative to the working directory.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "contents": {"type": "string"}},
            "required": ["path", "contents"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        "list_files",
        "List files under a directory (default: the working directory), recursively.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory, default `.`."}},
            "additionalProperties": False,
        },
    ),
    ToolDef(
        "finish",
        "Declare the task done. Call this when the tests pass.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
    ),
)
TOOL_SCHEMAS: Mapping[str, Mapping[str, Any]] = {t.name: t.parameters for t in TOOL_DEFS}
TOOL_SPECS: tuple[ToolSpec, ...] = tuple(ToolSpec(t.name, ReplayPolicy.SAFE) for t in TOOL_DEFS)


# --------------------------------------------------------------------------- validation


def validate_args(schema: Mapping[str, Any], value: Any, *, where: str = "input") -> str | None:
    """Check `value` against a small JSON Schema subset (object/string/integer/number/boolean/
    array, properties, required, additionalProperties, minimum, maximum, enum). Returns an error
    text for the model, or None when valid."""
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{where} must be one of {list(enum)}"
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, Mapping):
            return f"{where} must be an object, got {type(value).__name__}"
        props: Mapping[str, Any] = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                return f"unexpected keys: {extra}"
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"missing required keys: {missing}"
        for key, v in value.items():
            if key in props and (err := validate_args(props[key], v, where=repr(key))):
                return err
        return None
    if kind == "array":
        if not isinstance(value, list):
            return f"{where} must be an array, got {type(value).__name__}"
        item = schema.get("items")
        if item is not None:
            for i, v in enumerate(value):
                if err := validate_args(item, v, where=f"{where}[{i}]"):
                    return err
        return None
    is_bool = isinstance(value, bool)
    ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not is_bool,
        "number": isinstance(value, int | float) and not is_bool,
        "boolean": is_bool,
        None: True,
    }.get(kind, False)
    if not ok:
        return f"{where} must be {kind}, got {type(value).__name__}"
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where} must be >= {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where} must be <= {schema['maximum']}"
    return None


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and the tail (the start and the summary line both matter)."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n...[{omitted} chars truncated]...\n{text[-half:]}"


# --------------------------------------------------------------------------- pytest allowlist


class CommandRejected(ValueError):
    """The command is not allowed (allowlist) or cannot run safely here (no sandbox)."""


_FORBIDDEN_CHARS = set(";&|$`<>(){}[]*?!~\\\n\r")
_SIMPLE_FLAGS = {"-q", "-qq", "-v", "-vv", "-x", "-s", "-l", "-ra", "-rA", "--no-header"}
_TB_VALUES = {"auto", "long", "short", "line", "native", "no"}
_K_EXPR = re.compile(r"^[A-Za-z0-9_ ]+$")


def parse_pytest_command(command: str, workdir: Path) -> list[str]:
    """The pytest arguments of an allowlisted command (without `pytest`), or CommandRejected."""
    bad = sorted(_FORBIDDEN_CHARS & set(command))
    if bad:
        raise CommandRejected(f"characters not allowed: {''.join(bad)!r} (no shell is used)")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise CommandRejected(f"cannot parse command: {exc}") from None
    if argv[:1] == ["pytest"]:
        rest = argv[1:]
    elif argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        rest = argv[3:]
    else:
        raise CommandRejected("only `pytest [flags] [test paths]` is allowed")

    out: list[str] = []
    root = workdir.resolve()
    it = iter(rest)
    for arg in it:
        if arg in _SIMPLE_FLAGS:
            out.append(arg)
        elif arg.startswith("--tb="):
            if arg.removeprefix("--tb=") not in _TB_VALUES:
                raise CommandRejected(f"unsupported --tb value in {arg!r}")
            out.append(arg)
        elif arg == "-k":
            expr = next(it, None)
            if expr is None or not _K_EXPR.match(expr):
                raise CommandRejected("-k needs a simple expression (letters, digits, _, spaces)")
            out += ["-k", expr]
        elif arg.startswith("-"):
            raise CommandRejected(f"flag not allowed: {arg}")
        else:
            node, _, _test = arg.partition("::")
            target = (root / node).resolve()
            if target != root and root not in target.parents:
                raise CommandRejected(f"path outside the working directory: {arg}")
            out.append(arg)
    return out


# --------------------------------------------------------------------------- command runners


class RunResultLike(Protocol):
    """What a runner returns. A protocol, so the container environment's own RunResult type
    satisfies it without this module importing it."""

    @property
    def exit_code(self) -> int: ...
    @property
    def output(self) -> str: ...
    @property
    def timed_out(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class RunResult:
    exit_code: int
    output: str
    timed_out: bool = False


class CommandRunner(Protocol):
    """Runs one argv inside the job's isolated environment, in its working directory.

    A runner kills what it started on timeout AND on cancellation. `argv[0]` is a bare program
    name such as `pytest`; resolving it is the runner's business.
    """

    async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResultLike: ...


SANDBOX_EXEC = shutil.which("sandbox-exec") if sys.platform == "darwin" else None


def _sb_quote(p: Path | str) -> str:
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(workdir: Path, scratch: Path) -> str:
    """SBPL profile; later rules win, so broad denies come first and narrow allows after.

    No network at all; writes only in the workdir, the per-run scratch directory and a few
    device nodes; no reads under the user's home except the Python runtime and venv."""
    work = Path(os.path.realpath(workdir))
    scr = Path(os.path.realpath(scratch))
    home = Path(os.path.realpath(Path.home()))
    runtime = {Path(os.path.realpath(p)) for p in (sys.prefix, sys.base_prefix, sys.executable)}
    reads = "\n".join(f"  (subpath {_sb_quote(p)})" for p in sorted(runtime | {work, scr}))
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
  (subpath {_sb_quote(work)})
  (subpath {_sb_quote(scr)})
  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty") (literal "/dev/dtracehelper"))
(deny file-read* (subpath {_sb_quote(home)}))
(allow file-read-metadata)
(allow file-read*
{reads})
"""


def sandbox_env(scratch: Path) -> dict[str, str]:
    """A minimal environment: nothing inherited except what is needed to run Python."""
    return {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


class LocalSandboxRunner:
    """Runs `pytest ...` as a child of the current Python under macOS `sandbox-exec`.

    Refuses to run when sandbox-exec is missing (no unsandboxed fallback), and refuses any
    program but `pytest`. The child gets a minimal environment (no inherited secrets) and a
    private scratch directory outside the workdir, so the workdir stays free of noise."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def run(self, argv: Sequence[str], *, timeout_s: float) -> RunResult:
        if SANDBOX_EXEC is None:
            raise CommandRejected("refusing to run: no OS sandbox (sandbox-exec) on this platform")
        if not argv or argv[0] != "pytest":
            raise CommandRejected("only `pytest` can run in the local sandbox")
        scratch = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="sb-run-"))
        try:
            proc = await asyncio.create_subprocess_exec(
                SANDBOX_EXEC,
                "-p",
                sandbox_profile(self.workdir, scratch),
                sys.executable,
                "-m",
                "pytest",
                *argv[1:],
                cwd=self.workdir,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=sandbox_env(scratch),
            )
            try:
                data, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except TimeoutError:
                _kill_group(proc)
                await proc.wait()
                return RunResult(-signal.SIGKILL, "", timed_out=True)
            except BaseException:  # includes CancelledError: clean up, then re-raise
                _kill_group(proc)
                await asyncio.shield(proc.wait())
                raise
            return RunResult(proc.returncode or 0, data.decode(errors="replace"))
        finally:
            await asyncio.to_thread(shutil.rmtree, scratch, True)


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)


# --------------------------------------------------------------------------- the tool box

_TIMING = re.compile(r" in \d+(?:\.\d+)?s(?: \(\d+:\d{2}:\d{2}\))?")
_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".tmp", ".venv", "node_modules"}


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str
    is_error: bool = False
    done: bool = False


def resolve_in_workdir(workdir: Path, path: str) -> Path:
    """Resolve a model-supplied relative path; reject escapes, symlinks included."""
    if not path or "\0" in path:
        raise ValueError("invalid path")
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path escapes working directory: {path!r}")
    root = workdir.resolve()
    full = (root / p).resolve()  # follows symlinks, so a link out of the tree lands outside
    if not full.is_relative_to(root):
        raise ValueError(f"path escapes working directory: {path!r}")
    return full


class ToolBox:
    """Executes validated tool calls against `workdir` through `runner`."""

    def __init__(
        self, workdir: Path, runner: CommandRunner, *, max_output_chars: int = MAX_OUTPUT_CHARS
    ) -> None:
        self.workdir = workdir
        self.runner = runner
        self.max_output_chars = max_output_chars

    async def execute(self, name: str, args: Mapping[str, Any]) -> ToolOutcome:
        """Never raises for a bad call: validation and tool failures come back as error
        outcomes for the model to read. CancelledError propagates."""
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            return ToolOutcome(f"unknown tool {name!r}", is_error=True)
        if err := validate_args(schema, args):
            return ToolOutcome(f"invalid arguments for {name}: {err}", is_error=True)
        try:
            match name:
                case "run_tests":
                    return await self._run_tests(args)
                case "read_file":
                    return await self._read_file(args["path"])
                case "write_file":
                    return await self._write_file(args["path"], args["contents"])
                case "list_files":
                    return await self._list_files(args.get("path", "."))
                case _:  # finish
                    return ToolOutcome("ok", done=True)
        except Exception as exc:
            return ToolOutcome(f"{type(exc).__name__}: {exc}", is_error=True)

    async def _run_tests(self, args: Mapping[str, Any]) -> ToolOutcome:
        timeout_s = int(args.get("timeout_s", DEFAULT_TIMEOUT_S))
        try:
            pytest_args = parse_pytest_command(args.get("command", "pytest -q"), self.workdir)
            result = await self.runner.run(["pytest", *pytest_args], timeout_s=timeout_s)
        except CommandRejected as exc:
            return ToolOutcome(f"command rejected: {exc}", is_error=True)
        if result.timed_out:
            return ToolOutcome(f"command timed out after {timeout_s}s and was killed", True)
        return ToolOutcome(f"exit code: {result.exit_code}\n{self._clean(result.output)}")

    def _clean(self, text: str) -> str:
        """Truncate, and drop what differs between identical runs: timings, the workdir path."""
        for root in {str(self.workdir), str(self.workdir.resolve())}:
            text = text.replace(root, ".")
        return truncate(_TIMING.sub("", text), self.max_output_chars)

    async def _read_file(self, path: str) -> ToolOutcome:
        p = resolve_in_workdir(self.workdir, path)

        def read() -> str:
            with p.open("rb") as f:
                data = f.read(MAX_READ_BYTES + 1)
            return data[:MAX_READ_BYTES].decode("utf-8", errors="replace")

        text = await asyncio.to_thread(read)
        return ToolOutcome(truncate(text, self.max_output_chars))

    async def _write_file(self, path: str, contents: str) -> ToolOutcome:
        if len(contents) > MAX_WRITE_CHARS:
            return ToolOutcome(f"contents too large (max {MAX_WRITE_CHARS} chars)", True)
        p = resolve_in_workdir(self.workdir, path)
        if p == self.workdir.resolve():
            return ToolOutcome("path is the working directory", True)

        def write() -> bool:
            existed = p.exists()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(contents, "utf-8")
            return existed

        verb = "overwrote" if await asyncio.to_thread(write) else "created"
        return ToolOutcome(f"{verb} {path} ({len(contents)} chars)")

    async def _list_files(self, path: str) -> ToolOutcome:
        base = resolve_in_workdir(self.workdir, path)
        root = self.workdir.resolve()

        def walk() -> list[str]:
            if not base.is_dir():
                raise NotADirectoryError(path)
            out: list[str] = []
            for dirpath, dirs, files in os.walk(base, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in _IGNORED_DIRS)
                for f in sorted(files):
                    rel = (Path(dirpath) / f).relative_to(root).as_posix()
                    out.append(rel + (" -> (symlink)" if (Path(dirpath) / f).is_symlink() else ""))
                    if len(out) >= MAX_LIST_ENTRIES:
                        out.append(f"...[listing stopped at {MAX_LIST_ENTRIES} entries]")
                        return out
            return out

        lines = await asyncio.to_thread(walk)
        return ToolOutcome("\n".join(lines) if lines else "(no files)")
