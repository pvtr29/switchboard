import dataclasses

from switchboard.core import Budget, Job, JobKind, JobState, Usage
from switchboard.core.projection import _replaced


def test_replaced_matches_dataclasses_replace_for_every_field() -> None:
    """`_replaced` lists the Job fields by hand; this fails when a field is added to Job."""
    job = Job(
        job_id="j",
        kind=JobKind.ROLLOUT,
        task_ref="t",
        state=JobState.DONE,
        verdict=None,
        budget=Budget(tokens=5),
        usage=Usage(tokens=2),
        owner="w",
        lease_until=9.0,
        epoch=3,
        last_step=4,
        last_seq=1,
        created_at=1.0,
        updated_at=2.0,
    )
    assert {f.name for f in dataclasses.fields(Job)} == {
        "job_id", "kind", "task_ref", "state", "verdict", "budget", "usage", "owner",
        "lease_until", "epoch", "last_step", "last_seq", "created_at", "updated_at",
    }  # fmt: skip
    for changes in ({}, {"usage": Usage(tokens=7)}, {"state": JobState.FAILED, "owner": None}):
        assert _replaced(job, 8, 3.0, changes) == dataclasses.replace(
            job, last_seq=8, updated_at=3.0, **changes
        )
