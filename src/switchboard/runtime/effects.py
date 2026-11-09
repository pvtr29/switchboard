"""Effect journal: external effects as intent -> apply -> outcome (docs/plan.md "Execution model").

`ctx.effect(kind, args, policy=...)` runs through `EffectJournal.effect`:

1. An outcome already recorded for the key is returned as is, nothing is applied (replay of a
   step that crashed after the outcome but before its commit).
2. No intent yet: append `effect_intent` (awaiting the commit), `sink.apply`, append
   `effect_outcome(applied)`.
3. An intent without an outcome (a crash somewhere after the intent) is reconciled by the
   policy STORED with the intent, never the caller's:
   - safe: re-applied. Safe effects only touch the environment, so they normally should not go
     through the journal at all; if one does, recovery simply runs it again.
   - keyed: `sink.apply` again with the same key; the sink dedupes by key.
   - unsafe: `sink.lookup(key)` when the sink supports it: found -> outcome `reconciled`; not
     found -> it never happened, apply now. Without lookup support the outcome is recorded as
     `interrupted` and the effect is NOT applied: the agent gets `interrupted=True` and decides.
     (If the crash came before the external action, that effect is lost, not repeated: the
     journal prefers losing to duplicating for unsafe effects.)

Effect key = sha256(job_id|step|call_index|canonical_json(args)); call_index counts
`ctx.effect` calls within the step, so a re-run derives the same keys. The journal writes
events outside the step transaction; the `effects` table is their projection (see the store).
Exactly-once is a property of journal + sink together: keyed needs a sink that dedupes.

Restore does not undo the outside world: restoring S(n) only rolls back the environment.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from switchboard.core import Event, EventType
from switchboard.runtime.agent import EffectResult, ReplayPolicy
from switchboard.store import AppendStatus, EffectRecord, EventStore

logger = logging.getLogger(__name__)


def _default(obj: object) -> object:
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_default,
    )


def args_hash(args: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(args).encode()).hexdigest()


def effect_key(job_id: str, step: int, call_index: int, args: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{job_id}|{step}|{call_index}|{canonical_json(args)}".encode()
    ).hexdigest()


class ExternalSink(Protocol):
    """The outside world. `supports_lookup` says whether `lookup` is meaningful."""

    supports_lookup: bool

    async def apply(self, key: str, kind: str, args: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def lookup(self, key: str) -> Mapping[str, Any] | None:
        """Reconciliation: the outcome if `key` was applied, else None."""
        ...


class EffectJournal:
    """One per step attempt (it holds the call counter); built by the runner."""

    def __init__(
        self,
        store: EventStore,
        sink: ExternalSink,
        *,
        job_id: str,
        step: int,
        epoch: int,
        clock: Callable[[], float],
    ) -> None:
        self._store = store
        self._sink = sink
        self._job_id = job_id
        self._step = step
        self._epoch = epoch
        self._clock = clock
        self._calls = 0

    def _event(self, type_: EventType, payload: Mapping[str, Any]) -> Event:
        return Event(
            job_id=self._job_id, epoch=self._epoch, ts=self._clock(), type=type_, payload=payload
        )

    async def effect(
        self, kind: str, args: Mapping[str, Any], *, policy: ReplayPolicy
    ) -> EffectResult:
        index = self._calls
        self._calls += 1
        key = effect_key(self._job_id, self._step, index, args)
        record = await self._store.get_effect(key)
        if record is None:
            (res,) = await self._store.append_group(
                [
                    self._event(
                        EventType.EFFECT_INTENT,
                        {
                            "effect_id": key,
                            "kind": kind,
                            "args_hash": args_hash(args),
                            "args": args,
                            "policy": policy.value,
                            "step": self._step,
                            "call_index": index,
                        },
                    )
                ]
            )
            if res.status is AppendStatus.DUPLICATE:
                record = await self._store.get_effect(key)  # another runner got there first
            else:
                outcome = await self._sink.apply(key, kind, args)
                return await self._record(key, "applied", outcome)
        assert record is not None
        if record.has_outcome:
            return _result(record)
        stored = ReplayPolicy(record.policy)
        if stored is not policy:
            logger.warning(
                "effect %s: stored policy %s wins over requested %s",
                key,
                stored.value,
                policy.value,
            )
        if stored is ReplayPolicy.UNSAFE:
            if not self._sink.supports_lookup:
                return await self._record(key, "interrupted", {})
            found = await self._sink.lookup(key)
            if found is not None:
                return await self._record(key, "reconciled", found)
        # safe, keyed, or unsafe-but-provably-not-applied
        return await self._record(key, "applied", await self._sink.apply(key, kind, args))

    async def _record(self, key: str, status: str, outcome: Mapping[str, Any]) -> EffectResult:
        (res,) = await self._store.append_group(
            [
                self._event(
                    EventType.EFFECT_OUTCOME,
                    {"effect_id": key, "outcome": outcome, "status": status},
                )
            ]
        )
        if res.status is AppendStatus.DUPLICATE:
            # An outcome is already recorded (a racing runner): the first one is of record.
            record = await self._store.get_effect(key)
            assert record is not None
            return _result(record)
        return EffectResult(effect_id=key, status=status, outcome=outcome)


def _result(record: EffectRecord) -> EffectResult:
    return EffectResult(
        effect_id=record.effect_id, status=record.status, outcome=record.outcome or {}
    )


__all__ = [
    "EffectJournal",
    "ExternalSink",
    "args_hash",
    "canonical_json",
    "effect_key",
]
