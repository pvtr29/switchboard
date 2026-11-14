
# switchboard

A small, from-scratch runtime for running LLM agents for a long time without losing work.

An agent rollout is a loop: ask the model what to do, run the tools it asked for, repeat, maybe
for hours. Things go wrong in the middle all the time. The process gets killed, the machine
restarts, the provider rate-limits you, or a "dead" worker turns out to be only frozen and
wakes up again. Switchboard is my attempt to make that loop crash-proof, and to understand
exactly where each guarantee comes from, so I wrote the event log, the effect journal, the
snapshots and the fencing myself instead of using Temporal or DBOS.

The idea fits in one paragraph. Every step is a little transaction. Before any tool runs, the
model's response is written to an append-only log. After the tools ran, the environment (a
directory or a container) is snapshotted and the step commits. If anything dies, you restore
the last snapshot and replay from the log: the model is never asked the same question twice
(no tokens paid twice), the environment is rolled back rather than half-changed, and an
external side effect happens exactly once. That's it. Everything else is making this fast and
making sure it's true.

## quick start

You need [uv](https://docs.astral.sh/uv/) and a Mac (the local tool sandbox uses macOS
`sandbox-exec`; the container environment needs Docker, e.g. Colima).

```sh
uv sync
uv run sb daemon &              # the daemon; talks over a Unix socket
uv run sb submit --steps 20     # a fake agent, no API key needed
uv run sb attach <job-id>       # watch it; Ctrl-C detaches, the job keeps going
```

Now `kill -9` the daemon in the middle and start it again. The job picks up where it was.

For a real agent, put a `GEMINI_API_KEY` in `.env`, then:

```sh
uv run sb tasks                                     # 20 small buggy repos to fix
uv run sb submit --agent gemini --task intervals    # graded by hidden tests at the end
```

## does it work

I tried to break it on purpose:

- Crash at every single commit point of a job: it resumes correctly 27/27 times, with zero
  duplicated side effects. A naive "just restart the job" resume duplicates 36 of them.
- `kill -9` a real daemon 20 times while it runs 200 jobs: everything finishes, nothing is
  duplicated, every environment matches an uninterrupted run.
- A real Gemini rollout, killed mid-way, then rate-limited twice by the free tier: it resumed
  each time and passed the hidden tests, with every step's model call made exactly once.
- 10,000 concurrent agents in one Python process, at ~19.5k events/s with the event loop staying
  responsive (p99 lag 6–12 ms).

Getting the last one was the most fun. It took finding that SQLite was doing random B-tree
writes for every event, that the GIL was being handed back and forth on every row, and that
Python's garbage collector was pausing everything for 100 ms at a time because of a quirk in how
it counts young objects. The full story, with numbers, is in
[docs/benchmarks.md](docs/benchmarks.md).

## reward hacking

When agents are graded by tests, some of them learn to cheat: edit the tests, skip them, fake
the test runner, hard-code the answers. So rollouts are graded in a fresh copy with hidden tests,
any tampering is reported, and a cheap rule-based monitor reads what the agent *did* (from the
log, which the agent can't edit), not what it says.

On a set of planted honest and hacked runs that I wrote before tuning anything, every hack is
caught by at least one of those layers. On the external Terminal Wrench dataset a small learned
monitor scores AUC 0.945, but a control experiment shows that mostly measures "was the agent told
to attack", so I don't read too much into it. Details in the benchmarks doc.

## code

It's plain Python asyncio, about 13k lines with tests.

- `core/` the data model and the fold from events to job state (pure, no I/O)
- `store/` the SQLite event log: one writer thread, group commit
- `runtime/` the step loop, supervisor, crash recovery, budgets, effect journal
- `env/` directory and container environments with snapshots
- `agents/`, `llm/` a fake agent and a Gemini agent with sandboxed tools
- `verify/` grader, tamper checks, the monitor
- `tasks/` the 20 tasks
- `chaos/` the crash harness

```sh
uv run pytest          # 843 tests
uv run python benchmarks/density.py --agents 1000,5000,10000
uv run python -m switchboard.chaos.cli exhaustive
```

## todos

- run many workers against one store, with heartbeat leases and failover (the fencing part is
  done: a stale owner can't write after a takeover)
- a Postgres store behind the same interface
- multi-container environments (app + database), snapshotted consistently
- more real rollouts, to find hacks the models come up with on their own

How this would scale to 100k+ agents is sketched in [docs/scaling.md](docs/scaling.md).
