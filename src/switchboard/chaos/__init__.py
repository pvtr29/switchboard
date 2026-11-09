"""Chaos harness: crash the runtime at every commit window (exhaustive) and kill -9 the daemon
at random times (random), then check the recovery invariants.

Everything measures the effect journal and the recovery path against a FAKE external sink
(`SQLiteSink`), not against a real external system. There is no verifier yet, so "finished"
means the job reached `verifying`.
"""
