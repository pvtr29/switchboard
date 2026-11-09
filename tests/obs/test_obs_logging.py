import io
import json
import logging
import re
from collections.abc import Iterator

import pytest

from switchboard.obs import JsonFormatter, configure_logging


def record(msg: str = "hello", *args: object, **extra: object) -> logging.LogRecord:
    r = logging.LogRecord("switchboard.test", logging.WARNING, __file__, 1, msg, args or None, None)
    for k, v in extra.items():
        setattr(r, k, v)
    return r


def test_json_line_has_the_fixed_fields() -> None:
    line = JsonFormatter().format(record("job %s admitted", "j1"))
    doc = json.loads(line)
    assert "\n" not in line
    assert list(doc) == ["ts", "level", "logger", "msg"]
    assert doc["level"] == "WARNING"
    assert doc["logger"] == "switchboard.test"
    assert doc["msg"] == "job j1 admitted"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", doc["ts"])


def test_extra_fields_are_included_and_odd_values_stringified() -> None:
    doc = json.loads(
        JsonFormatter().format(record(job_id="j7", steps=3, obj=object, nested={"a": 1}))
    )
    assert doc["job_id"] == "j7"
    assert doc["steps"] == 3
    assert doc["nested"] == {"a": 1}
    assert doc["obj"].startswith("<class")  # not serialisable: str(), never an exception


def test_exception_is_included() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        r = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    doc = json.loads(JsonFormatter().format(r))
    assert "RuntimeError: boom" in doc["exc"]
    assert doc["exc"].startswith("Traceback")


def test_secret_looking_extras_are_redacted_counts_are_not() -> None:
    doc = json.loads(
        JsonFormatter().format(
            record(api_key="sk-123", Authorization="Bearer x", password="p", tokens=5, tokens_in=2)
        )
    )
    assert doc["api_key"] == doc["Authorization"] == doc["password"] == "[redacted]"
    assert doc["tokens"] == 5
    assert doc["tokens_in"] == 2
    assert "sk-123" not in json.dumps(doc)


def test_unicode_is_kept() -> None:
    assert json.loads(JsonFormatter().format(record("naïve \u2713")))["msg"] == "naïve \u2713"


@pytest.fixture
def clean_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


def test_configure_logging_json_text_and_replacement(clean_root: logging.Logger) -> None:
    out = io.StringIO()
    configure_logging(json=True, level="debug", stream=out)
    logging.getLogger("switchboard.x").debug("one", extra={"job_id": "j1"})
    doc = json.loads(out.getvalue())
    assert (doc["msg"], doc["job_id"], doc["level"]) == ("one", "j1", "DEBUG")
    assert clean_root.level == logging.DEBUG

    out2 = io.StringIO()
    configure_logging(json=False, level=logging.WARNING, stream=out2)  # replaces, not stacks
    logging.getLogger("switchboard.x").info("hidden")
    logging.getLogger("switchboard.x").warning("shown")
    assert out.getvalue().count("\n") == 1  # the first handler is gone
    assert "shown" in out2.getvalue()
    assert "hidden" not in out2.getvalue()
    assert not out2.getvalue().startswith("{")
    ours = [h for h in clean_root.handlers if type(h).__name__ == "_SwitchboardHandler"]
    assert len(ours) == 1
