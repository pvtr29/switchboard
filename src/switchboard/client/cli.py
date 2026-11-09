"""Entry point for the `sb` command."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from switchboard import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sb",
        description="Switchboard: durable runtime for long-running agent rollouts.",
    )
    parser.add_argument("--version", action="version", version=f"sb {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
