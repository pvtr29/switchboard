"""`python scripts/chaos.py exhaustive|random ...`: run a chaos mode, print and store results."""

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from switchboard.chaos.exhaustive import ExhaustiveConfig, format_exhaustive, run_exhaustive
from switchboard.chaos.random_mode import RandomConfig, format_random, run_random
from switchboard.chaos.results import write_result

DEFAULT_OUT = Path("benchmarks/results")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chaos", description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results directory")
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("exhaustive", help="crash at every commit window of every step")
    e.add_argument("--steps", type=int, default=3)
    e.add_argument("--seed", type=int, default=11)
    r = sub.add_parser("random", help="kill -9 a real daemon at random times")
    r.add_argument("--jobs", type=int, default=200)
    r.add_argument("--kills", type=int, default=20)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=10)
    r.add_argument("--timeout", type=float, default=180.0)
    r.add_argument("--keep", action="store_true", help="keep the work directory")
    args = p.parse_args(argv)

    if args.mode == "exhaustive":
        with tempfile.TemporaryDirectory(prefix="sbchaos") as tmp:
            result = asyncio.run(
                run_exhaustive(ExhaustiveConfig(steps=args.steps, seed=args.seed), Path(tmp))
            )
        print(format_exhaustive(result))
        ok = not result["failures"]
    else:
        result = asyncio.run(
            run_random(
                RandomConfig(
                    jobs=args.jobs,
                    kills=args.kills,
                    seed=args.seed,
                    steps=args.steps,
                    timeout=args.timeout,
                    keep=args.keep,
                )
            )
        )
        print(format_random(result))
        ok = bool(result["ok"])
    path = write_result(args.mode, result, args.out)
    print(f"\nresults: {path}")
    json.dumps(result)  # fail loudly if something non-serialisable slipped in
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
