"""DirEnv: a working directory snapshotted by copy-on-write directory clones.

Layout under `root/<job_id>/`: `work/`, `snapshots/<name>/` and `.tmp-*` scratch directories.

Crash safety: every mutation builds into a `.tmp-*` directory and is published by `os.rename`.
- snapshot: clone work -> `.tmp-snap-*`, rename into `snapshots/<name>` (an existing snapshot is
  first renamed aside to `.tmp-aside-snap-<name>-*`, then deleted).
- restore: clone snapshot -> `.tmp-restore-*`, rename work aside to `.tmp-aside-work-*`, rename
  the clone to `work`, delete the aside.
Leftovers are cleaned on provision and before the first other operation of an instance. A
leftover aside whose target is missing (crash between the two renames) is renamed back, so
neither work nor a snapshot is ever lost; anything else under `.tmp-*` is deleted.

Copy method (detected once per process, verified by a real probe on the actual filesystem):
macOS `cp -c -R` (APFS clonefile), Linux `cp -a --reflink=auto`, else `shutil.copytree`.
"""

import asyncio
import hashlib
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from switchboard.env.base import SnapshotMissing

DEFAULT_IGNORE: tuple[str, ...] = ("__pycache__", ".pytest_cache")
_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_UID_LEN = 12

COPY_CLONEFILE = "cp -c (clonefile)"
COPY_REFLINK = "cp --reflink=auto"
COPY_COPYTREE = "copytree"

_detected: str | None = None


def _cp_args(method: str, src: Path, dst: Path) -> list[str]:
    if method == COPY_CLONEFILE:
        return ["cp", "-c", "-R", str(src), str(dst)]
    return ["cp", "-a", "--reflink=auto", str(src), str(dst)]


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True)


async def _cp(method: str, src: Path, dst: Path) -> bool:
    proc = await asyncio.create_subprocess_exec(
        *_cp_args(method, src, dst),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait() == 0


def _write_probe(src: Path) -> None:
    src.mkdir(parents=True)
    (src / "f").write_text("probe")
    (src / "sub").mkdir()
    (src / "sub" / "g").write_text("x")


def _probe_ok(dst: Path) -> bool:
    try:
        return (dst / "f").read_text() == "probe" and (dst / "sub" / "g").read_text() == "x"
    except OSError:
        return False


async def _detect(scratch: Path) -> str:
    """Pick the copy method by trying it once in `scratch` (same filesystem as the env)."""
    global _detected
    if _detected is not None:
        return _detected
    candidates = {"darwin": [COPY_CLONEFILE], "linux": [COPY_REFLINK]}.get(sys.platform, [])
    chosen = COPY_COPYTREE
    probe = scratch / f".tmp-probe-{uuid.uuid4().hex[:_UID_LEN]}"
    try:
        for method in candidates:
            await asyncio.to_thread(_write_probe, probe / "a")
            try:
                if await _cp(method, probe / "a", probe / "b") and await asyncio.to_thread(
                    _probe_ok, probe / "b"
                ):
                    chosen = method
                    break
            except OSError:  # cp binary missing
                pass
            await asyncio.to_thread(_rmtree, probe)
    finally:
        await asyncio.to_thread(_rmtree, probe)
    _detected = chosen
    return chosen


def _hash_tree(root: Path, ignore: Sequence[str]) -> str:
    h = hashlib.sha256()

    def feed(tag: bytes, *parts: bytes) -> None:
        h.update(tag)
        for p in parts:
            h.update(len(p).to_bytes(8, "big"))
            h.update(p)

    def walk(directory: Path, rel: str) -> None:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if any(Path(e.name).match(p) for p in ignore):
                continue
            path = f"{rel}{e.name}"
            st = e.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                feed(b"L", path.encode(), os.readlink(e.path).encode())
            elif stat.S_ISDIR(st.st_mode):
                feed(b"D", path.encode())
                walk(Path(e.path), path + "/")
            else:
                feed(b"F", path.encode(), b"x" if st.st_mode & 0o100 else b"-")
                fh = hashlib.sha256()
                with open(e.path, "rb") as f:
                    while chunk := f.read(1 << 20):
                        fh.update(chunk)
                feed(b"C", str(st.st_size).encode(), fh.digest())

    walk(root, "")
    return h.hexdigest()


class DirEnv:
    def __init__(
        self,
        template: Path,
        root: Path,
        job_id: str,
        *,
        ignore: Sequence[str] = DEFAULT_IGNORE,
    ) -> None:
        self.template = template
        self.job_dir = root / job_id
        self.ignore = tuple(ignore)
        self._cleaned = False

    # -- layout ---------------------------------------------------------------------------
    @property
    def workdir(self) -> Path:
        return self.job_dir / "work"

    @property
    def snapshots_dir(self) -> Path:
        return self.job_dir / "snapshots"

    @property
    def copy_method(self) -> str:
        """The detected copy method, or "undetected" before the first copy."""
        return _detected or "undetected"

    def _snap(self, name: str) -> Path:
        if not _NAME_RE.fullmatch(name):
            raise ValueError(f"invalid snapshot name {name!r}")
        return self.snapshots_dir / name

    def _tmp(self, kind: str) -> Path:
        return self.job_dir / f".tmp-{kind}-{uuid.uuid4().hex[:_UID_LEN]}"

    # -- helpers --------------------------------------------------------------------------
    async def _copy(self, src: Path, dst: Path) -> None:
        """Copy directory `src` to the non-existing `dst`."""
        method = await _detect(self.job_dir)
        if method != COPY_COPYTREE:
            try:
                if await _cp(method, src, dst):
                    return
            except OSError:
                pass
            await asyncio.to_thread(_rmtree, dst)
        await asyncio.to_thread(_copytree, src, dst)

    async def _prepare(self) -> None:
        await asyncio.to_thread(self.job_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self.snapshots_dir.mkdir, exist_ok=True)
        if not self._cleaned:
            await asyncio.to_thread(self._clean_leftovers)
            self._cleaned = True

    def _clean_leftovers(self) -> None:
        if not self.job_dir.is_dir():
            return
        for p in sorted(self.job_dir.glob(".tmp-*")):
            m = re.fullmatch(rf"\.tmp-aside-(work|snap-(.+))-[0-9a-f]{{{_UID_LEN}}}", p.name)
            if m is not None:
                target = self.workdir if m.group(1) == "work" else self.snapshots_dir / m.group(2)
                if not target.exists():
                    os.rename(p, target)
                    continue
            _rmtree(p)

    # -- Environment ----------------------------------------------------------------------
    async def provision(self) -> Path:
        await asyncio.to_thread(_rmtree, self.job_dir)  # leftovers of a crashed provision
        await asyncio.to_thread(self.snapshots_dir.mkdir, parents=True)
        tmp = self._tmp("provision")
        await self._copy(self.template, tmp)
        await asyncio.to_thread(os.rename, tmp, self.workdir)
        self._cleaned = True
        return self.workdir

    async def snapshot(self, name: str) -> str:
        final = self._snap(name)
        await self._prepare()
        tmp = self._tmp("snap")
        await self._copy(self.workdir, tmp)
        await asyncio.to_thread(self._publish, tmp, final, name)
        return name

    def _publish(self, tmp: Path, final: Path, name: str) -> None:
        aside: Path | None = None
        if final.exists():
            aside = self._tmp(f"aside-snap-{name}")
            os.rename(final, aside)
        os.rename(tmp, final)
        if aside is not None:
            _rmtree(aside)

    async def restore(self, env_ref: str) -> None:
        src = self._snap(env_ref)
        await self._prepare()
        if not await asyncio.to_thread(src.is_dir):
            raise SnapshotMissing(env_ref)
        tmp = self._tmp("restore")
        await self._copy(src, tmp)
        await asyncio.to_thread(self._swap_work, tmp)

    def _swap_work(self, tmp: Path) -> None:
        aside: Path | None = None
        if self.workdir.exists():
            aside = self._tmp("aside-work")
            os.rename(self.workdir, aside)
        os.rename(tmp, self.workdir)
        if aside is not None:
            _rmtree(aside)

    async def hash(self) -> str:
        return await asyncio.to_thread(_hash_tree, self.workdir, self.ignore)

    async def gc(self, keep: Sequence[str]) -> None:
        await self._prepare()
        await asyncio.to_thread(self._gc, set(keep))

    def _gc(self, keep: set[str]) -> None:
        for p in self.snapshots_dir.iterdir():
            if p.name not in keep:
                _rmtree(p)

    async def destroy(self) -> None:
        await asyncio.to_thread(_rmtree, self.job_dir)

    async def snapshots(self) -> list[str]:
        """Names of the snapshots currently stored (for tests and reporting)."""
        return await asyncio.to_thread(
            lambda: (
                sorted(p.name for p in self.snapshots_dir.iterdir())
                if self.snapshots_dir.is_dir()
                else []
            )
        )
