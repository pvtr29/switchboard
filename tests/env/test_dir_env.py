import os
import shutil
from pathlib import Path

import pytest

from switchboard.env import dir_env
from switchboard.env.base import SnapshotMissing
from switchboard.env.dir_env import DirEnv

pytestmark = pytest.mark.asyncio


def make_template(root: Path) -> Path:
    t = root / "template"
    (t / "a" / "b").mkdir(parents=True)
    (t / "empty").mkdir()
    (t / "top.txt").write_text("top")
    (t / "a" / "b" / "deep.txt").write_text("deep")
    run = t / "run.sh"
    run.write_text("#!/bin/sh\n")
    run.chmod(0o755)
    os.symlink("top.txt", t / "link")
    return t


@pytest.fixture
def env(tmp_path: Path) -> DirEnv:
    return DirEnv(make_template(tmp_path), tmp_path / "envs", "j1")


async def test_provision_from_template(env: DirEnv) -> None:
    work = await env.provision()
    assert work == env.workdir
    assert (work / "a" / "b" / "deep.txt").read_text() == "deep"
    assert (work / "empty").is_dir()
    assert env.copy_method in {dir_env.COPY_CLONEFILE, dir_env.COPY_REFLINK, dir_env.COPY_COPYTREE}
    # the template itself is untouched by edits to the workdir
    (work / "top.txt").write_text("changed")
    assert (env.template / "top.txt").read_text() == "top"


async def test_provision_wipes_leftovers(env: DirEnv) -> None:
    await env.provision()
    (env.workdir / "junk").write_text("x")
    (env.job_dir / ".tmp-provision-deadbeef0000").mkdir()
    await env.provision()
    assert not (env.workdir / "junk").exists()
    assert not list(env.job_dir.glob(".tmp-*"))


async def test_snapshot_restore_round_trip(env: DirEnv) -> None:
    work = await env.provision()
    h0 = await env.hash()
    ref = await env.snapshot("s0")
    shutil.rmtree(work / "a")
    (work / "empty").rmdir()
    (work / "run.sh").chmod(0o644)
    (work / "link").unlink()
    os.symlink("elsewhere", work / "link")
    (work / "new.txt").write_text("n")
    assert await env.hash() != h0
    await env.restore(ref)
    assert await env.hash() == h0
    assert (work / "empty").is_dir()
    assert os.access(work / "run.sh", os.X_OK)
    assert os.readlink(work / "link") == "top.txt"
    assert (work / "a" / "b" / "deep.txt").read_text() == "deep"
    assert not (work / "new.txt").exists()
    assert not list(env.job_dir.glob(".tmp-*"))


async def test_hash_ignores_mtime_and_ignored_dirs(env: DirEnv) -> None:
    work = await env.provision()
    h = await env.hash()
    os.utime(work / "top.txt", (1, 1))
    os.utime(work / "a", (5, 5))
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "x.pyc").write_text("junk")
    assert await env.hash() == h


async def test_hash_is_order_independent(tmp_path: Path) -> None:
    names = ["b", "a", "c", "d"]
    hashes = []
    for i, order in enumerate([names, names[::-1]]):
        t = tmp_path / f"t{i}"
        t.mkdir()
        for n in order:
            (t / n).write_text(n)
            (t / f"{n}dir").mkdir()
        e = DirEnv(t, tmp_path / "envs", f"j{i}")
        await e.provision()
        hashes.append(await e.hash())
    assert hashes[0] == hashes[1]


async def test_hash_changes_on_content_mode_and_structure(env: DirEnv) -> None:
    work = await env.provision()
    seen = {await env.hash()}
    for mutate in (
        lambda: (work / "top.txt").write_text("top2"),
        lambda: (work / "top.txt").chmod(0o755),
        lambda: (work / "newdir").mkdir(),
        lambda: (work / "link").unlink(),
    ):
        mutate()
        seen.add(await env.hash())
    assert len(seen) == 5


async def test_hash_distinguishes_file_boundaries(tmp_path: Path) -> None:
    hs = []
    for i, files in enumerate([{"a": "bc", "d": ""}, {"a": "b", "d": "c"}]):
        t = tmp_path / f"t{i}"
        t.mkdir()
        for k, v in files.items():
            (t / k).write_text(v)
        e = DirEnv(t, tmp_path / "envs", f"j{i}")
        await e.provision()
        hs.append(await e.hash())
    assert hs[0] != hs[1]


async def test_snapshot_replaces_existing_name(env: DirEnv) -> None:
    await env.provision()
    await env.snapshot("s")
    (env.workdir / "top.txt").write_text("v2")
    await env.snapshot("s")
    (env.workdir / "top.txt").write_text("v3")
    await env.restore("s")
    assert (env.workdir / "top.txt").read_text() == "v2"
    assert await env.snapshots() == ["s"]
    assert not list(env.job_dir.glob(".tmp-*"))


async def test_crashed_snapshot_leaves_no_half_snapshot(env: DirEnv) -> None:
    await env.provision()
    await env.snapshot("good")
    # a crash mid-snapshot leaves only a partial temp dir, never the final name
    partial = env.job_dir / ".tmp-snap-0123456789ab"
    partial.mkdir()
    (partial / "half").write_text("x")
    assert await env.snapshots() == ["good"]
    # a new instance is the process restart: its first operation cleans the leftovers
    fresh = DirEnv(env.template, env.job_dir.parent, "j1")
    await fresh.snapshot("next")
    assert not partial.exists()
    assert await env.snapshots() == ["good", "next"]


async def test_crash_between_snapshot_renames_is_recovered(env: DirEnv) -> None:
    await env.provision()
    await env.snapshot("s")
    # crash after renaming the old snapshot aside, before the new one was moved in
    os.rename(env.snapshots_dir / "s", env.job_dir / ".tmp-aside-snap-s-0123456789ab")
    fresh = DirEnv(env.template, env.job_dir.parent, "j1")
    await fresh.gc(["s"])
    assert await fresh.snapshots() == ["s"]
    assert (fresh.snapshots_dir / "s" / "top.txt").read_text() == "top"


async def test_restore_after_corruption(env: DirEnv) -> None:
    work = await env.provision()
    h = await env.hash()
    await env.snapshot("s")
    shutil.rmtree(work)
    work.mkdir()
    (work / "garbage").write_text("g")
    await env.restore("s")
    assert await env.hash() == h


async def test_crashed_restore_leftovers_cleaned(env: DirEnv) -> None:
    work = await env.provision()
    h = await env.hash()
    await env.snapshot("s")
    # crash between "rename work aside" and "rename temp into place"
    os.rename(work, env.job_dir / ".tmp-aside-work-0123456789ab")
    shutil.copytree(env.snapshots_dir / "s", env.job_dir / ".tmp-restore-0123456789ab")
    fresh = DirEnv(env.template, env.job_dir.parent, "j1")
    await fresh.restore("s")
    assert await fresh.hash() == h
    assert not list(env.job_dir.glob(".tmp-*"))
    # crash after the swap, before the old work was deleted
    (env.job_dir / ".tmp-aside-work-aaaaaaaaaaaa").mkdir()
    await DirEnv(env.template, env.job_dir.parent, "j1").gc(["s"])
    assert not list(env.job_dir.glob(".tmp-*"))
    assert work.is_dir()


async def test_gc_keeps_only_requested(env: DirEnv) -> None:
    await env.provision()
    for n in ("a", "b", "c"):
        await env.snapshot(n)
    await env.gc(["a", "c", "never-existed"])
    assert await env.snapshots() == ["a", "c"]
    await env.gc([])
    assert await env.snapshots() == []


async def test_restore_missing_and_bad_names(env: DirEnv) -> None:
    await env.provision()
    with pytest.raises(SnapshotMissing):
        await env.restore("nope")
    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="invalid snapshot name"):
            await env.snapshot(bad)


async def test_destroy(env: DirEnv) -> None:
    await env.provision()
    await env.snapshot("s")
    await env.destroy()
    assert not env.job_dir.exists()


async def test_copy_method_detected_once_and_fallback_works(
    env: DirEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    await env.provision()
    detected = env.copy_method
    assert detected != "undetected"
    assert dir_env._detected == detected
    # forcing the fallback must give identical results
    monkeypatch.setattr(dir_env, "_detected", dir_env.COPY_COPYTREE)
    h = await env.hash()
    await env.snapshot("s")
    (env.workdir / "top.txt").write_text("zzz")
    await env.restore("s")
    assert await env.hash() == h
    assert env.copy_method == dir_env.COPY_COPYTREE


async def test_failing_cp_falls_back_to_copytree(
    env: DirEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(method: str, src: Path, dst: Path) -> bool:
        dst.mkdir()  # noqa: ASYNC240  (a failed cp may leave a partial destination)
        return False

    monkeypatch.setattr(dir_env, "_detected", dir_env.COPY_REFLINK)
    monkeypatch.setattr(dir_env, "_cp", failing)
    await env.provision()
    assert (env.workdir / "a" / "b" / "deep.txt").read_text() == "deep"
