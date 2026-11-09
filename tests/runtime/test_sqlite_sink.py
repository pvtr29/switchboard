"""SQLiteSink: persistent ledger, idempotent and naive modes, hooks, counting."""

from pathlib import Path

import pytest

from switchboard.runtime.sinks import EffectCounts, SQLiteSink, counts_of, read_applications

pytestmark = pytest.mark.asyncio


async def test_idempotent_applies_once_per_key_and_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "sink.db"
    async with await SQLiteSink.open(db) as sink:
        first = await sink.apply("k1", "send", {"x": 1})
        again = await sink.apply("k1", "send", {"x": 2})  # same key, different bytes
        await sink.apply("k2", "send", {"x": 1})  # same bytes, different key
        assert again == first
        assert await sink.lookup("k1") == first
        assert await sink.lookup("nope") is None
        assert sink.counts() == EffectCounts(2, 2, 0, 0)
    async with await SQLiteSink.open(db) as reopened:  # a restarted process
        assert await reopened.apply("k1", "send", {"x": 1}) == first
        assert reopened.counts() == EffectCounts(2, 2, 0, 0)
        assert reopened.applications("k1") == 1


async def test_naive_applies_every_call(tmp_path: Path) -> None:
    async with await SQLiteSink.open(tmp_path / "s.db", idempotent=False) as sink:
        for _ in range(3):
            await sink.apply("k", "send", {"x": 1})
        assert sink.counts() == EffectCounts(3, 1, 1, 2)
        assert sink.duplicates() == {"k": 3}
        apps = read_applications(sink.path)
        assert [a.n for a in apps] == [1, 2, 3]
        # by action identity, two keys with the same action are a duplicate too
        await sink.apply("other", "send", {"x": 1})
        assert counts_of(sink.ledger(), by_action=True).extra_applications == 3


async def test_hooks_fire_only_around_real_applications(tmp_path: Path) -> None:
    seen: list[str] = []
    async with await SQLiteSink.open(
        tmp_path / "s.db",
        before_apply=lambda k: seen.append(f"before {k}"),
        after_apply=lambda k: seen.append(f"after {k}"),
    ) as sink:
        await sink.apply("k", "send", {})
        await sink.apply("k", "send", {})
    assert seen == ["before k", "after k"]


async def test_crash_before_apply_leaves_nothing(tmp_path: Path) -> None:
    class Boom(BaseException):
        pass

    def before(key: str) -> None:
        raise Boom

    async with await SQLiteSink.open(tmp_path / "s.db", before_apply=before) as sink:
        with pytest.raises(Boom):
            await sink.apply("k", "send", {})
        assert sink.counts().applications == 0
        sink.before_apply = None
        await sink.apply("k", "send", {})
        assert sink.counts().applications == 1


async def test_lookup_unsupported(tmp_path: Path) -> None:
    async with await SQLiteSink.open(tmp_path / "s.db", supports_lookup=False) as sink:
        with pytest.raises(RuntimeError, match="lookup"):
            await sink.lookup("k")
