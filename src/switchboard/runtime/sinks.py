"""FakeExternalSink: an in-memory outside world that counts applications per key.

Duplicates are counted by action identity (the effect key), not by bytes: two applications of
the same key are a duplicate even if their payloads differ, and two keys with identical bytes
are not.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EffectCounts:
    applications: int  # total external actions performed
    keys: int  # distinct keys applied
    duplicates: int  # keys applied more than once
    extra_applications: int  # applications beyond the first, summed over keys


class FakeExternalSink:
    """`idempotent=True`: a known key returns its stored outcome without applying again.
    `idempotent=False`: every call applies (the naive world). `supports_lookup=False` makes
    `lookup` unusable, so unsafe effects cannot be reconciled. `before_apply` / `after_apply`
    run (with the key) right before / after a REAL application is recorded; tests raise from
    them to crash on either side of the external action."""

    def __init__(
        self,
        *,
        idempotent: bool = True,
        supports_lookup: bool = True,
        before_apply: Callable[[str], None] | None = None,
        after_apply: Callable[[str], None] | None = None,
    ) -> None:
        self.idempotent = idempotent
        self.supports_lookup = supports_lookup
        self.before_apply = before_apply
        self.after_apply = after_apply
        self.ledger: list[tuple[str, str, Mapping[str, Any]]] = []
        self._outcomes: dict[str, Mapping[str, Any]] = {}

    async def apply(self, key: str, kind: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.idempotent and key in self._outcomes:
            return self._outcomes[key]
        if self.before_apply is not None:
            self.before_apply(key)
        self.ledger.append((key, kind, dict(args)))
        outcome = {"receipt": f"{key[:12]}-{self.applications(key)}", "kind": kind}
        # idempotent sinks keep the first outcome; a naive one just overwrites
        self._outcomes.setdefault(key, outcome)
        if self.after_apply is not None:
            self.after_apply(key)
        return self._outcomes[key] if self.idempotent else outcome

    async def lookup(self, key: str) -> Mapping[str, Any] | None:
        if not self.supports_lookup:
            raise RuntimeError("this sink does not support lookup")
        return self._outcomes.get(key)

    def applications(self, key: str) -> int:
        return sum(1 for k, _, _ in self.ledger if k == key)

    def duplicates(self) -> dict[str, int]:
        """Keys applied more than once, with their application counts."""
        return {k: n for k, n in Counter(k for k, _, _ in self.ledger).items() if n > 1}

    def counts(self) -> EffectCounts:
        per_key = Counter(k for k, _, _ in self.ledger)
        return EffectCounts(
            applications=len(self.ledger),
            keys=len(per_key),
            duplicates=sum(1 for n in per_key.values() if n > 1),
            extra_applications=sum(n - 1 for n in per_key.values()),
        )
