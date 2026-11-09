"""The store's durability claim, tested against a real SIGKILL rather than a mocked crash."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import pytest

from switchboard.store import SQLiteEventLog, open_db

BATCH = 10

WRITER = textwrap.dedent(
    f"""
    import sys
    from switchboard.core import Event, EventType
    from switchboard.store import SQLiteEventLog, open_db

    log = SQLiteEventLog(open_db(sys.argv[1]))
    batch = 0
    while True:
        batch += 1
        log.append_many([
            Event(job_id="j", epoch=0, ts=0.0, type=EventType.STATE_CHANGED,
                  payload={{"batch": batch, "i": i}})
            for i in range({BATCH})
        ])
        # Acknowledge only after append_many returned, i.e. after COMMIT.
        print(batch, flush=True)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL semantics")
def test_acked_batches_survive_kill9_and_no_batch_is_partial(tmp_path: Path) -> None:
    db = tmp_path / "events.sqlite"
    proc = subprocess.Popen(
        [sys.executable, "-c", WRITER, str(db)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    acked = 0
    deadline = time.monotonic() + 10
    # Kill while the writer is mid-stream, after a reasonable number of commits.
    while acked < 200 and time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        acked = int(line)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    # The writer may have committed and acked more batches than we had read: drain the pipe.
    for line in proc.stdout:
        acked = int(line)
    proc.stdout.close()
    assert acked >= 200, "writer did not make progress"

    log = SQLiteEventLog(open_db(db))
    try:
        per_batch = Counter(e.payload["batch"] for e in log.read_since(0, limit=1_000_000))
    finally:
        log.close()

    # Every acknowledged batch is fully present...
    assert all(per_batch[b] == BATCH for b in range(1, acked + 1))
    # ...and nothing (including the batch in flight at the kill) is half-written.
    assert all(count == BATCH for count in per_batch.values())
    assert max(per_batch) <= acked + 1
