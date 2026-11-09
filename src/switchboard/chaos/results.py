"""Result files and tables shared by both chaos modes."""

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LABEL = (
    "Measures the effect journal and recovery against a FAKE external sink (SQLiteSink), "
    "not against a real external system. No verifier exists yet: 'finished' means the job "
    "reached 'verifying'."
)


def write_result(mode: str, result: Mapping[str, Any], out_dir: Path) -> Path:
    """Write `result` to `<out_dir>/chaos-<mode>-<UTC timestamp>.json` and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out_dir / f"chaos-{mode}-{stamp}.json"
    n = 1
    while path.exists():  # two runs within one second
        n += 1
        path = out_dir / f"chaos-{mode}-{stamp}-{n}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A plain-text table with left-aligned columns."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]

    def line(parts: Sequence[str]) -> str:
        return "  ".join(p.ljust(w) for p, w in zip(parts, widths, strict=True)).rstrip()

    return "\n".join([line(headers), line(["-" * w for w in widths]), *map(line, cells)])
