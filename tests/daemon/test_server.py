import asyncio
import json
import os
import socket
import stat
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from conftest import wait_state

from switchboard.client.conn import Client, ClientError
from switchboard.daemon.protocol import MAX_LINE_BYTES
from switchboard.daemon.server import Daemon, DaemonAlreadyRunning, DaemonError

pytestmark = pytest.mark.asyncio


class Raw:
    """A bare socket speaking raw lines, to send what the real client never would."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer

    async def send(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def recv(self) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), 5)
        msg: dict[str, Any] = json.loads(line)
        return msg

    async def ask(self, obj: object) -> dict[str, Any]:
        await self.send(json.dumps(obj).encode() + b"\n")
        return await self.recv()

    async def closed(self) -> bool:
        return await asyncio.wait_for(self.reader.read(1), 5) == b""


@pytest_asyncio.fixture
async def raw(daemon: Daemon) -> AsyncIterator[Raw]:
    reader, writer = await asyncio.open_unix_connection(
        str(daemon.socket_path), limit=MAX_LINE_BYTES
    )
    yield Raw(reader, writer)
    writer.close()
    await writer.wait_closed()


def req(cmd: str, **args: Any) -> dict[str, Any]:
    return {"v": 1, "id": "r1", "cmd": cmd, "args": args}


# ---- every command over a real socket ---------------------------------------------------------


async def test_submit_list_get_stats(client: Client) -> None:
    job_id = (await client.request("submit", task_ref="t1", steps=3, seed=5))["job_id"]
    job = await wait_state(client, job_id, {"verifying"})
    assert job["last_step"] == 2
    assert job["task_ref"] == "t1"
    assert job["usage"]["tokens"] > 0

    listed = (await client.request("list"))["jobs"]
    assert [j["job_id"] for j in listed] == [job_id]
    assert (await client.request("list", state="verifying"))["jobs"][0]["job_id"] == job_id
    assert (await client.request("list", state="queued"))["jobs"] == []
    got = (await client.request("get", job_id=job_id))["job"]
    assert got["state"] == "verifying"
    assert got["live"] is False

    stats = await client.request("stats")
    assert stats["jobs"] == {"verifying": 1}
    assert stats["connections"] == 1
    assert stats["store"]["events_written"] > 5
    assert stats["supervisor"]["finished"] == 1


async def test_submit_with_env_template_provisions_a_workdir(
    client: Client, daemon: Daemon, template: Path
) -> None:
    job_id = (await client.request("submit", steps=2, env_template=str(template)))["job_id"]
    await wait_state(client, job_id, {"verifying"})
    work = daemon.envs_dir / job_id / "work"
    assert (work / "README").read_text() == "seed project\n"
    assert (work / "log.txt").exists()  # the fake agent edited it


async def test_submit_defaults_and_instructions(client: Client, daemon: Daemon) -> None:
    job_id = (
        await client.request("submit", instructions=[{"role": "user", "content": "hi"}], steps=1)
    )["job_id"]
    await wait_state(client, job_id, {"verifying"})
    first = (await daemon.store.read_job(job_id))[0]
    assert first.payload["instructions"] == [{"role": "user", "content": "hi"}]
    assert first.payload["steps"] == 1


async def test_pause_resume_cancel(client: Client) -> None:
    job_id = (await client.request("submit", steps=500, token_interval=0.005))["job_id"]
    await wait_state(client, job_id, {"running"})
    await client.request("pause", job_id=job_id)
    paused = await wait_state(client, job_id, {"paused"})
    step_at_pause = paused["last_step"]
    await asyncio.sleep(0.2)
    assert (await client.request("get", job_id=job_id))["job"]["last_step"] == step_at_pause
    await client.request("resume", job_id=job_id)
    job = await wait_state(client, job_id, {"running"})
    assert job["live"] is True
    res = await client.request("cancel", job_id=job_id)
    assert res["job"]["state"] == "cancelled"
    # cancelling again is a no-op, and a cancelled job cannot be paused
    assert (await client.request("cancel", job_id=job_id))["job"]["state"] == "cancelled"
    with pytest.raises(ClientError) as exc:
        await client.request("pause", job_id=job_id)
    assert exc.value.code == "conflict"


async def test_extend_requeues_an_over_budget_job(client: Client) -> None:
    job_id = (await client.request("submit", steps=6, budget={"tokens": 120}))["job_id"]
    stuck = await wait_state(client, job_id, {"over_budget"})
    assert stuck["last_step"] is not None
    res = await client.request("extend", job_id=job_id, tokens=100_000)
    assert res["job"]["budget"]["tokens"] == 100_120
    done = await wait_state(client, job_id, {"verifying"})
    assert done["last_step"] == 5


async def test_subscribe_streams_events_and_unsubscribe_stops_them(client: Client) -> None:
    stream = await client.subscribe()
    job_id = (await client.request("submit", steps=2))["job_id"]
    types = []
    async with asyncio.timeout(10):
        async for msg in stream:
            assert msg["id"] == stream.id
            assert msg["event"]["job_id"] == job_id
            types.append(msg["event"]["type"])
            if msg["event"]["type"] == "state_changed" and (
                msg["event"]["payload"]["to"] == "verifying"
            ):
                break
    assert types[0] == "job_created"
    assert types.count("step_completed") == 2
    await stream.unsubscribe()
    other = (await client.request("submit", steps=1))["job_id"]
    await wait_state(client, other, {"verifying"})
    with pytest.raises(ClientError) as exc:
        await client.request("unsubscribe", subscription=stream.id)
    assert exc.value.code == "not_found"


async def test_subscribe_job_filter_and_unknown_job(client: Client) -> None:
    with pytest.raises(ClientError) as exc:
        await client.subscribe(job_id="nope")
    assert exc.value.code == "not_found"
    a = (await client.request("submit", steps=1))["job_id"]
    b = (await client.request("submit", steps=1))["job_id"]
    await wait_state(client, a, {"verifying"})
    await wait_state(client, b, {"verifying"})
    stream = await client.subscribe(job_id=b, from_seq=1)  # backfill, filtered
    seen: set[str] = set()
    async with asyncio.timeout(5):
        async for msg in stream:
            seen.add(msg["event"]["job_id"])
            if msg["event"]["type"] == "state_changed" and (
                msg["event"]["payload"]["to"] == "verifying"
            ):
                break
    assert seen == {b}


async def test_shutdown_command_stops_the_daemon(sock_dir: Path, tmp_path: Path) -> None:
    d = Daemon(
        socket_path=sock_dir / "s",
        db_path=tmp_path / "sb.db",
        envs_dir=tmp_path / "envs",
        shutdown_timeout=5.0,
    )
    runner = asyncio.create_task(d.run())
    for _ in range(200):
        if d.socket_path.exists():
            break
        await asyncio.sleep(0.01)
    async with await Client.connect(str(d.socket_path)) as c:
        job_id = (await c.request("submit", steps=300, token_interval=0.005))["job_id"]
        await wait_state(c, job_id, {"running"})
        assert await c.request("shutdown") == {"shutting_down": True}  # the reply is delivered
    async with asyncio.timeout(10):
        await runner
    assert not d.socket_path.exists()


# ---- errors ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "code", "echo_id"),
    [
        (b"not json\n", "bad_json", None),
        (b"[1, 2]\n", "bad_json", None),
        (b'{"v": 2, "id": "x", "cmd": "list"}\n', "bad_version", "x"),
        (b'{"id": "x", "cmd": "list"}\n', "bad_version", "x"),
        (b'{"v": 1, "cmd": "list"}\n', "bad_args", None),
        (b'{"v": 1, "id": "x", "cmd": "frobnicate"}\n', "unknown_cmd", "x"),
        (b'{"v": 1, "id": "x", "cmd": 7}\n', "unknown_cmd", "x"),
        (b'{"v": 1, "id": "x", "cmd": "list", "args": [1]}\n', "bad_args", "x"),
        (b"\xff\xfe\n", "bad_json", None),
    ],
)
async def test_malformed_requests_get_error_codes(
    raw: Raw, line: bytes, code: str, echo_id: str | None
) -> None:
    await raw.send(line)
    msg = await raw.recv()
    assert msg["ok"] is False
    assert msg["error"]["code"] == code
    assert msg["id"] == echo_id
    # the connection survives a bad request
    assert (await raw.ask(req("list")))["ok"] is True


@pytest.mark.parametrize(
    "args",
    [
        {"steps": 0},
        {"steps": "3"},
        {"steps": True},
        {"kind": "rollout"},
        {"budget": {"tokens": -1}},
        {"budget": {"tokens": "lots"}},
        {"budget": {"bogus": 1}},
        {"budget": [1]},
        {"seed": 1.5},
        {"env_template": "/definitely/not/here"},
        {"instructions": "hi"},
        {"token_interval": -1},
        {"surprise": 1},
    ],
)
async def test_submit_validates_args(raw: Raw, args: dict[str, Any]) -> None:
    msg = await raw.ask(req("submit", **args))
    assert msg["ok"] is False
    assert msg["error"]["code"] == "bad_args"
    assert (await raw.ask(req("list")))["result"] == {"jobs": []}  # nothing was created


async def test_bad_args_and_not_found_and_conflict(client: Client) -> None:
    cases: list[tuple[str, dict[str, Any], str]] = [
        ("get", {}, "bad_args"),
        ("get", {"job_id": 5}, "bad_args"),
        ("get", {"job_id": "nope"}, "not_found"),
        ("cancel", {"job_id": "nope"}, "not_found"),
        ("pause", {"job_id": "nope"}, "not_found"),
        ("resume", {"job_id": "nope"}, "not_found"),
        ("extend", {"job_id": "nope", "tokens": 1}, "not_found"),
        ("extend", {"job_id": "x"}, "bad_args"),
        ("extend", {"job_id": "x", "tokens": -5}, "bad_args"),
        ("list", {"state": "sleeping"}, "bad_args"),
        ("stats", {"x": 1}, "bad_args"),
        ("subscribe", {"from_seq": -1}, "bad_args"),
        ("subscribe", {"from_seq": "1"}, "bad_args"),
    ]
    for cmd, args, code in cases:
        with pytest.raises(ClientError) as exc:
            await client.request(cmd, **args)
        assert exc.value.code == code, (cmd, args)

    job_id = (await client.request("submit", steps=1))["job_id"]
    await wait_state(client, job_id, {"verifying"})
    for cmd in ("pause", "resume"):
        with pytest.raises(ClientError) as exc:
            await client.request(cmd, job_id=job_id)
        assert exc.value.code == "conflict"
    with pytest.raises(ClientError) as exc:
        await client.request("subscribe", job_id=job_id, bogus=1)
    assert exc.value.code == "bad_args"


async def test_duplicate_subscription_id_is_a_conflict(raw: Raw) -> None:
    sub = {"v": 1, "id": "s1", "cmd": "subscribe", "args": {}}
    assert (await raw.ask(sub))["ok"] is True
    again = await raw.ask(sub)
    assert again["error"]["code"] == "conflict"


async def test_overlong_line_closes_only_that_connection(
    daemon: Daemon, raw: Raw, client: Client
) -> None:
    await raw.send(b'{"v":1,"id":"x","cmd":"list","args":{"pad":"' + b"a" * (MAX_LINE_BYTES + 10))
    msg = await raw.recv()
    assert msg["error"]["code"] == "line_too_long"
    assert await raw.closed()
    # a line right below the limit is fine (and then rejected for its args, not its size)
    reader, writer = await asyncio.open_unix_connection(
        str(daemon.socket_path), limit=MAX_LINE_BYTES
    )
    ok = Raw(reader, writer)
    pad = "a" * (MAX_LINE_BYTES - 200)
    assert (await ok.ask(req("list", pad=pad)))["error"]["code"] == "bad_args"
    writer.close()
    await writer.wait_closed()
    assert (await client.request("list"))["jobs"] == []  # everyone else is unaffected


async def test_one_bad_client_does_not_affect_others(daemon: Daemon, client: Client) -> None:
    job_id = (await client.request("submit", steps=40, token_interval=0.005))["job_id"]
    stream = await client.subscribe(job_id=job_id)
    # 1) garbage, 2) a half line then a hard close, 3) a subscriber that vanishes mid-stream
    for payload in (b"\x00\x01garbage\n", b'{"v":1,"id":"half'):
        r, w = await asyncio.open_unix_connection(str(daemon.socket_path))
        w.write(payload)
        await w.drain()
        w.close()
        await w.wait_closed()
    r, w = await asyncio.open_unix_connection(str(daemon.socket_path))
    w.write(b'{"v":1,"id":"s","cmd":"subscribe","args":{}}\n')
    await w.drain()
    await r.readline()
    w.close()
    await w.wait_closed()
    async with asyncio.timeout(10):
        async for msg in stream:
            if msg["event"]["type"] == "state_changed" and (
                msg["event"]["payload"]["to"] == "verifying"
            ):
                break
    for _ in range(100):  # the daemon notices the closed connections
        if (await client.request("stats"))["connections"] == 1:
            break
        await asyncio.sleep(0.02)
    assert (await client.request("stats"))["connections"] == 1
    assert daemon.bus.stats.subscribers == 1


async def test_a_client_that_never_reads_responses_only_stalls_itself(
    daemon: Daemon, client: Client
) -> None:
    _, writer = await asyncio.open_unix_connection(str(daemon.socket_path))
    blob = b"".join(
        json.dumps({"v": 1, "id": f"i{i}", "cmd": "stats", "args": {}}).encode() + b"\n"
        for i in range(2000)
    )
    writer.write(blob)  # thousands of responses that nobody reads
    await writer.drain()
    for _ in range(20):  # the daemon stays responsive for everyone else meanwhile
        assert (await client.request("list"))["jobs"] == []
        await asyncio.sleep(0.01)
    writer.close()
    await writer.wait_closed()


# ---- socket ------------------------------------------------------------------------------------


async def test_socket_is_private(daemon: Daemon) -> None:
    mode = stat.S_IMODE(daemon.socket_path.stat().st_mode)
    assert mode == 0o600


async def test_umask_is_restored_after_binding(sock_dir: Path, tmp_path: Path) -> None:
    before = os.umask(0o022)
    try:
        d = Daemon(socket_path=sock_dir / "s", db_path=tmp_path / "a.db", envs_dir=tmp_path / "e")
        await d.start()
        try:
            assert os.umask(0o022) == 0o022
        finally:
            await d.shutdown()
    finally:
        os.umask(before)


async def test_stale_socket_file_is_replaced(sock_dir: Path, tmp_path: Path) -> None:
    path = sock_dir / "s"
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))  # bound but never listening: connect is refused, like a crashed daemon
    dead.close()
    assert path.exists()
    d = Daemon(socket_path=path, db_path=tmp_path / "a.db", envs_dir=tmp_path / "e")
    await d.start()
    try:
        async with await Client.connect(str(path)) as c:
            assert (await c.request("list"))["jobs"] == []
    finally:
        await d.shutdown()
    assert not path.exists()


async def test_second_daemon_refuses_to_start(daemon: Daemon, tmp_path: Path) -> None:
    second = Daemon(
        socket_path=daemon.socket_path, db_path=tmp_path / "other.db", envs_dir=tmp_path / "e2"
    )
    with pytest.raises(DaemonAlreadyRunning):
        await second.start()
    assert not (tmp_path / "other.db").exists()  # it never touched its database
    # the first one is untouched and still owns its socket
    async with await Client.connect(str(daemon.socket_path)) as c:
        assert (await c.request("list"))["jobs"] == []


async def test_regular_file_at_socket_path_is_never_deleted(sock_dir: Path, tmp_path: Path) -> None:
    path = sock_dir / "s"
    path.write_text("precious")
    d = Daemon(socket_path=path, db_path=tmp_path / "a.db", envs_dir=tmp_path / "e")
    with pytest.raises(DaemonError, match="not a socket"):
        await d.start()
    assert path.read_text() == "precious"
