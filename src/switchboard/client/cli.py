"""Entry point for the `sb` command: a thin, plain-text client of the daemon (no TUI).

The socket is `--socket`, else `$SB_SOCKET`, else `~/.switchboard/sb.sock`. `--socket` may be
given before or after the subcommand. Exit status: 0 success, 1 the daemon answered with an
error (or could not start), 2 the daemon is unreachable or usage is wrong, 130 `attach`
detached with Ctrl-C (the job is not affected: attach only subscribes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from switchboard import __version__
from switchboard.client.conn import Client, ClientError

DEFAULT_HOME = "~/.switchboard"
ATTACH_STOP_STATES = frozenset({"verifying", "done", "failed", "cancelled"})
EXIT_ERROR = 1
EXIT_UNREACHABLE = 2
EXIT_DETACHED = 130


def default_socket() -> str:
    return os.environ.get("SB_SOCKET") or str(Path(DEFAULT_HOME + "/sb.sock").expanduser())


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--socket",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="daemon socket (default: $SB_SOCKET or ~/.switchboard/sb.sock)",
    )
    parser = argparse.ArgumentParser(
        prog="sb",
        description="Switchboard: durable runtime for long-running agent rollouts.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"sb {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_, description=help_, parents=[common])

    d = add("daemon", "run the daemon in the foreground")
    d.add_argument("--db", default=str(Path(DEFAULT_HOME + "/sb.db").expanduser()))
    d.add_argument("--envs", default=str(Path(DEFAULT_HOME + "/envs").expanduser()))
    d.add_argument("--max-running", type=int, default=8, metavar="N")
    d.add_argument("--shutdown-timeout", type=float, default=10.0, metavar="SECONDS")
    d.add_argument(
        "--verify-projection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="on startup compare the jobs projection with a fold of the log "
        "(default: all jobs when there are at most 1000, else a sample of 100)",
    )
    d.add_argument("--sink-db", metavar="PATH", help="give jobs a persistent fake external sink")
    d.add_argument(
        "--sink-naive", action="store_true", help="make the sink non-idempotent (baseline runs)"
    )

    s = add("submit", "submit a fake job; prints its id")
    s.add_argument("--steps", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--task-ref", default="fake")
    s.add_argument("--budget-seconds", type=float)
    s.add_argument("--budget-tokens", type=int)
    s.add_argument("--budget-usd-micros", type=int)
    s.add_argument("--env-template", metavar="DIR")
    s.add_argument("--token-interval", type=float, help="seconds per fake token (default 0.01)")
    s.add_argument("--effect-every", type=int, help="external effect every N steps (needs a sink)")
    s.add_argument("--effect-policy", choices=["safe", "keyed", "unsafe"])

    ps = add("ps", "list jobs")
    ps.add_argument("--state", help="only jobs in this state")

    for name, help_ in (
        ("get", "show one job"),
        ("cancel", "cancel a job"),
        ("pause", "pause a job at its next step boundary"),
        ("resume", "resume a paused job"),
    ):
        add(name, help_).add_argument("job")
    a = add("attach", "stream a job's events; Ctrl-C detaches, the job keeps running")
    a.add_argument("job")
    a.add_argument("--live", action="store_true", help="only events from now on (no history)")
    e = add("extend", "raise a job's budget")
    e.add_argument("job")
    e.add_argument("--seconds", type=float, default=0.0)
    e.add_argument("--tokens", type=int, default=0)
    e.add_argument("--usd-micros", type=int, default=0)
    add("stats", "daemon statistics")
    add("shutdown", "ask the daemon to shut down gracefully")
    return parser


# ---- formatting ------------------------------------------------------------------------------


def _clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def format_jobs(jobs: Sequence[dict[str, Any]]) -> str:
    rows = [("JOB", "STATE", "STEP", "TOKENS", "USD", "ACTIVE_S", "UPDATED")]
    for j in jobs:
        step = 0 if j["last_step"] is None else j["last_step"] + 1  # completed steps
        u = j["usage"]
        rows.append(
            (
                j["job_id"],
                j["state"],
                str(step),
                str(u["tokens"]),
                f"{u['usd_micros'] / 1e6:.6f}",
                f"{u['active_seconds']:.1f}",
                _clock(j["updated_at"]),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )


def format_event(msg: dict[str, Any]) -> str:
    if "gap" in msg:
        g = msg["gap"]
        return f"-- gap: events {g['from_seq']}..{g['to_seq']} were dropped (re-attach to backfill)"
    e = msg["event"]
    p = e["payload"]
    match e["type"]:
        case "state_changed":
            detail = f"{p.get('from')} -> {p.get('to')} ({p.get('reason')})"
        case "step_completed" | "step_started" | "llm_responded":
            detail = f"step {e['step_id']}"
            if e["type"] == "step_completed" and p.get("done"):
                detail += " (agent done)"
        case "usage_recorded":
            detail = f"tokens={p.get('tokens_in', 0) + p.get('tokens_out', 0)}"
        case _:
            detail = json.dumps(p, separators=(",", ":"))[:80]
    return f"{e['seq']:>6}  {_clock(e['ts'])}  {e['job_id']}  {e['type']}  {detail}"


def format_job(j: dict[str, Any]) -> str:
    lines = [f"{k}: {v}" for k, v in j.items() if k not in ("budget", "usage")]
    lines.append(f"budget: {json.dumps(j['budget']) if j['budget'] else 'unlimited'}")
    lines.append(f"usage: {json.dumps(j['usage'])}")
    return "\n".join(lines)


def _flatten(d: dict[str, Any], prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend(_flatten(v, f"{prefix}{k}."))
        else:
            out.append(f"{prefix}{k}: {v}")
    return out


# ---- commands --------------------------------------------------------------------------------


def _say(text: str) -> None:
    print(text, flush=True)


async def _attach(client: Client, job: str, live: bool) -> None:
    stream = await client.subscribe(job, from_seq=None if live else 1)
    async for msg in stream:
        _say(format_event(msg))
        if "event" in msg:
            e = msg["event"]
            if e["type"] == "state_changed" and e["payload"].get("to") in ATTACH_STOP_STATES:
                return


async def _command(args: argparse.Namespace) -> int:
    async with await Client.connect(args.socket) as c:
        match args.command:
            case "submit":
                budget: dict[str, float | int] = {}
                if args.budget_seconds is not None:
                    budget["active_seconds"] = args.budget_seconds
                if args.budget_tokens is not None:
                    budget["tokens"] = args.budget_tokens
                if args.budget_usd_micros is not None:
                    budget["usd_micros"] = args.budget_usd_micros
                extra: dict[str, Any] = {}
                if args.env_template:
                    extra["env_template"] = args.env_template  # made absolute in main()
                if args.token_interval is not None:
                    extra["token_interval"] = args.token_interval
                if args.effect_every is not None:
                    extra["effect_every"] = args.effect_every
                if args.effect_policy is not None:
                    extra["effect_policy"] = args.effect_policy
                r = await c.request(
                    "submit",
                    kind="fake",
                    task_ref=args.task_ref,
                    budget=budget,
                    seed=args.seed,
                    steps=args.steps,
                    **extra,
                )
                _say(r["job_id"])
            case "ps":
                r = await c.request("list", **({"state": args.state} if args.state else {}))
                _say(format_jobs(r["jobs"]))
            case "get":
                _say(format_job((await c.request("get", job_id=args.job))["job"]))
            case "attach":
                await _attach(c, args.job, args.live)
            case "cancel" | "pause" | "resume":
                r = await c.request(args.command, job_id=args.job)
                _say(f"{args.job}: {r['job']['state']}")
            case "extend":
                r = await c.request(
                    "extend",
                    job_id=args.job,
                    active_seconds=args.seconds,
                    tokens=args.tokens,
                    usd_micros=args.usd_micros,
                )
                _say(f"{args.job}: {r['job']['state']}")
            case "stats":
                _say("\n".join(_flatten(await c.request("stats"))))
            case "shutdown":
                await c.request("shutdown")
                _say("daemon shutting down")
    return 0


def _run_daemon(args: argparse.Namespace) -> int:
    from switchboard.daemon.server import Daemon, DaemonError

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    daemon = Daemon(
        socket_path=args.socket,
        db_path=args.db,
        envs_dir=args.envs,
        max_running=args.max_running,
        shutdown_timeout=args.shutdown_timeout,
        sink_db=args.sink_db,
        sink_naive=args.sink_naive,
        verify_projection=args.verify_projection,
    )
    try:
        asyncio.run(daemon.run())
    except DaemonError as exc:
        print(f"sb daemon: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Not parser.set_defaults: the shared --socket action would then default in the
    # subparsers too and override a value given before the subcommand.
    if not hasattr(args, "socket"):
        args.socket = default_socket()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "daemon":
        return _run_daemon(args)
    if args.command == "submit" and args.env_template:
        args.env_template = os.path.abspath(args.env_template)  # the daemon's cwd differs
    try:
        return asyncio.run(_command(args))
    except KeyboardInterrupt:
        print("detached", file=sys.stderr)
        return EXIT_DETACHED
    except ClientError as exc:
        print(f"sb: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:  # refused, no such file, path too long, ...
        print(f"sb: cannot reach the daemon at {args.socket}: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE


if __name__ == "__main__":
    sys.exit(main())
