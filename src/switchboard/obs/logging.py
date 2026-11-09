"""Structured logging: one JSON object per line.

`JsonFormatter` emits `ts` (ISO 8601 UTC, millisecond precision), `level`, `logger`, `msg`, the
fields passed through `extra=` (e.g. `job_id`), and `exc` (the formatted traceback) when an
exception is attached. Values that JSON cannot represent are stringified, so a log call can
never raise because of its arguments.

SECRETS: never put credentials, tokens or API keys in a log message or in `extra`. As a safety
net, `extra` fields whose name looks sensitive (see `REDACTED_KEYS`) are replaced by
"[redacted]", but the rule is not to log them in the first place.
"""

import json
import logging
import sys
import time
from typing import IO, Any

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime"}
)
REDACTED_KEYS = ("password", "secret", "token", "api_key", "apikey", "authorization", "credential")
# Counts and intervals that merely contain the word "token".
_SAFE_KEYS = frozenset({"tokens", "tokens_in", "tokens_out", "token_interval"})
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _redact(key: str) -> bool:
    lowered = key.lower()
    return lowered not in _SAFE_KEYS and any(k in lowered for k in REDACTED_KEYS)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        secs = int(record.created)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs))
        doc: dict[str, Any] = {
            "ts": f"{ts}.{int((record.created - secs) * 1000):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_") or key in doc:
                continue
            doc[key] = "[redacted]" if _redact(key) else value
        if record.exc_info and record.exc_info[0] is not None:
            doc["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            doc["exc"] = record.exc_text
        return json.dumps(doc, default=str, separators=(",", ":"), ensure_ascii=False)


class _SwitchboardHandler(logging.StreamHandler[IO[str]]):
    """Marker type so `configure_logging` can find and replace its own handler."""


def configure_logging(
    *, json: bool = True, level: int | str = logging.INFO, stream: IO[str] | None = None
) -> logging.Handler:
    """Install ONE handler on the root logger (replacing ours from an earlier call).

    Returns the handler. `level` is a number or a name such as "debug".
    """
    root = logging.getLogger()
    for old in [h for h in root.handlers if isinstance(h, _SwitchboardHandler)]:
        root.removeHandler(old)
    handler = _SwitchboardHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if json else logging.Formatter(TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)
    return handler
