"""Wire protocol: versioned newline-delimited JSON (NDJSON), one message per line.

Client -> daemon, a request::

    {"v": 1, "id": "<str>", "cmd": "<str>", "args": {...}}

Daemon -> client, the response with the same id::

    {"v": 1, "id": "<id>", "ok": true,  "result": {...}}
    {"v": 1, "id": "<id>", "ok": false, "error": {"code": "<code>", "message": "<text>"}}

Streamed messages of a subscription (`id` is the id of the `subscribe` request)::

    {"v": 1, "id": "<sub id>", "event": {...}, "seq": 17}
    {"v": 1, "id": "<sub id>", "gap": {"from_seq": 18, "to_seq": 40}}

A gap says "events with seq in [from_seq, to_seq] were not delivered" (the subscriber's queue
overflowed, see bus.py); the client can fetch them again by subscribing with `from_seq`.

Error codes: bad_json (not JSON / not an object), bad_version, unknown_cmd, bad_args,
not_found, conflict (valid request, wrong state or shutting down), internal, line_too_long
(the line exceeded MAX_LINE_BYTES; the daemon then closes that connection, only that one).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from switchboard.core import Event, Job, budget_to_dict, usage_to_dict

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1024 * 1024

BAD_JSON = "bad_json"
BAD_VERSION = "bad_version"
UNKNOWN_CMD = "unknown_cmd"
BAD_ARGS = "bad_args"
NOT_FOUND = "not_found"
CONFLICT = "conflict"
INTERNAL = "internal"
LINE_TOO_LONG = "line_too_long"

COMMANDS = frozenset(
    {
        "submit",
        "list",
        "get",
        "cancel",
        "pause",
        "resume",
        "extend",
        "subscribe",
        "unsubscribe",
        "stats",
        "shutdown",
    }
)


class ProtocolError(Exception):
    """A request that must be answered with an error response."""

    def __init__(self, code: str, message: str, req_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.req_id = req_id


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    cmd: str
    args: Mapping[str, Any] = field(default_factory=dict)


def _default(obj: object) -> object:
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, set | frozenset | tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def dumps_line(message: Mapping[str, Any]) -> bytes:
    """One NDJSON line (with the trailing newline)."""
    return json.dumps(message, separators=(",", ":"), default=_default).encode() + b"\n"


def parse_request(line: bytes) -> Request:
    """Validate one line; raises ProtocolError with the id when it could be recovered."""
    try:
        msg = json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError(BAD_JSON, f"not valid JSON: {exc}") from None
    if not isinstance(msg, dict):
        raise ProtocolError(BAD_JSON, "a message must be a JSON object")
    raw_id = msg.get("id")
    req_id = raw_id if isinstance(raw_id, str) else None
    if msg.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(
            BAD_VERSION, f"unsupported protocol version {msg.get('v')!r}, want 1", req_id
        )
    if req_id is None or not req_id:
        raise ProtocolError(BAD_ARGS, "id must be a non-empty string")
    cmd = msg.get("cmd")
    if not isinstance(cmd, str) or cmd not in COMMANDS:
        raise ProtocolError(UNKNOWN_CMD, f"unknown command {cmd!r}", req_id)
    args = msg.get("args", {})
    if not isinstance(args, dict):
        raise ProtocolError(BAD_ARGS, "args must be an object", req_id)
    return Request(req_id, cmd, args)


def ok_response(req_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": req_id, "ok": True, "result": result}


def error_response(req_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "id": req_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }


def event_to_dict(e: Event) -> dict[str, Any]:
    return {
        "seq": e.seq,
        "job_id": e.job_id,
        "epoch": e.epoch,
        "step_id": e.step_id,
        "ts": e.ts,
        "type": e.type.value,
        "v": e.v,
        "payload": e.payload,
    }


def event_message(sub_id: str, e: Event) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": sub_id, "event": event_to_dict(e), "seq": e.seq}


def gap_message(sub_id: str, from_seq: int, to_seq: int) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "id": sub_id,
        "gap": {"from_seq": from_seq, "to_seq": to_seq},
    }


def job_to_dict(j: Job) -> dict[str, Any]:
    return {
        "job_id": j.job_id,
        "kind": j.kind.value,
        "task_ref": j.task_ref,
        "state": j.state.value,
        "verdict": j.verdict.value if j.verdict else None,
        "budget": budget_to_dict(j.budget),
        "usage": usage_to_dict(j.usage),
        "epoch": j.epoch,
        "last_step": j.last_step,
        "last_seq": j.last_seq,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
    }
