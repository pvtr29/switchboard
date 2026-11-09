"""The Environment protocol: a snapshot-able working directory a step can be rolled back to.

A step is a transaction over the environment (docs/plan.md "Execution model"): the runner
restores S(n) before re-running step n and snapshots S(n+1) before committing step n.
An `env_ref` is an opaque string the environment hands out and later accepts back.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class SnapshotMissing(Exception):
    """`restore` was given an env_ref that does not exist (never taken, or garbage collected)."""


class Environment(Protocol):
    @property
    def workdir(self) -> Path: ...

    async def provision(self) -> Path:
        """Create a fresh workdir from the template. Wipes leftovers of a crashed provision."""
        ...

    async def snapshot(self, name: str) -> str:
        """Capture the workdir atomically under `name` and return its env_ref.

        A crash mid-snapshot never leaves a half snapshot under the final name. Snapshotting
        an existing name replaces it.
        """
        ...

    async def restore(self, env_ref: str) -> None:
        """Make the workdir exactly equal to the snapshot. Raises SnapshotMissing."""
        ...

    async def hash(self) -> str:
        """Deterministic content hash of the workdir (independent of mtimes and ordering)."""
        ...

    async def gc(self, keep: Sequence[str]) -> None:
        """Delete every snapshot whose env_ref is not in `keep`."""
        ...

    async def destroy(self) -> None:
        """Remove the workdir and all snapshots."""
        ...
