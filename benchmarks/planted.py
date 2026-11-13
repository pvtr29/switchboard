"""Run the planted honest/hack set (switchboard.verify.planted) through the real pipeline.

Run: uv run python benchmarks/planted.py [--split dev|heldout]
(macOS: the agent's commands run under sandbox-exec)

Prints, per category, how many cases each verifier layer flagged (grader = hidden tests failed,
tamper = changes outside the editable files, monitor = action-only rules) and writes every
outcome to benchmarks/results/planted-<timestamp>.json.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from switchboard.agents.tools import LocalSandboxRunner
from switchboard.verify.planted import all_cases, run_cases, summarize

RESULTS_DIR = Path(__file__).parent / "results"


async def main(split: str) -> None:
    t0 = time.perf_counter()
    outcomes = await run_cases(all_cases(split), LocalSandboxRunner)
    elapsed = time.perf_counter() - t0
    table = summarize(outcomes)
    print(f"{split}: {len(outcomes)} cases in {elapsed:.1f}s")
    print(
        f"{'category':<20}{'n':>4}{'saw green':>11}{'grader':>8}"
        f"{'tamper':>8}{'monitor':>9}{'any':>5}"
    )
    for cat, r in table.items():
        print(
            f"{cat:<20}{r['n']:>4}{r['agent_saw_green']:>11}{r['grader']:>8}{r['tamper']:>8}"
            f"{r['monitor']:>9}{r['any']:>5}"
        )
    print("\nper case:")
    for o in outcomes:
        print(
            f"  {o.case.name:<36} {o.case.label:<6} verdict={o.verdict} "
            f"ignored={list(o.ignored_changes)} monitor={list(o.monitor_rules)}"
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"planted-{split}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_text(
        json.dumps(
            {
                "split": split,
                "elapsed_s": round(elapsed, 1),
                "summary": table,
                "cases": [
                    {
                        "name": o.case.name,
                        "label": o.case.label,
                        "category": o.case.category,
                        "state": o.state,
                        "verdict": o.verdict,
                        "ignored_changes": list(o.ignored_changes),
                        "monitor_rules": list(o.monitor_rules),
                        "monitor_score": o.monitor_score,
                        "agent_saw_green": o.agent_saw_green,
                        "layers": o.layers,
                    }
                    for o in outcomes
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"results: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "heldout"], default="dev")
    asyncio.run(main(ap.parse_args().split))
