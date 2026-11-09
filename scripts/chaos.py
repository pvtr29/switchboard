"""Chaos harness entry point: `uv run python scripts/chaos.py exhaustive|random [options]`."""

import sys

from switchboard.chaos.cli import main

if __name__ == "__main__":
    sys.exit(main())
