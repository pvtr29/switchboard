"""Observability: structured logging, latency statistics and the event-loop lag monitor."""

from switchboard.obs.gc_scheduler import GcScheduler, GcStats
from switchboard.obs.lag import LagStats, LoopLagMonitor
from switchboard.obs.logging import JsonFormatter, configure_logging
from switchboard.obs.stats import LatencySnapshot, LatencyWindow, percentile
from switchboard.obs.tuning import GC_THRESHOLD, tune_gc

__all__ = [
    "GC_THRESHOLD",
    "GcScheduler",
    "GcStats",
    "JsonFormatter",
    "LagStats",
    "LatencySnapshot",
    "LatencyWindow",
    "LoopLagMonitor",
    "configure_logging",
    "percentile",
    "tune_gc",
]
