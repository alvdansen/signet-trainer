"""inference.stall_clock — pure decision for the parallel-inference watcher's stall-freshness clock.

issue #45 PR-1 must-fix #1: a PENDING render's ``pending_since`` clock must refresh on evidence of
life (a newly committed artifact), not only once at dispatch time — otherwise a genuinely healthy,
multi-hour H3 render gets declared STALLED at ``render_stall_minutes``, releases single-flight, and
lets a second metered A100 dispatch land OVER the still-running first (repeating until the session
cap trips).

The PR-2 verifier caught this refresh DECISION living only inline inside
``scripts/watch_parallel_inference.py::main()`` as a bare ``if progress != pending_progress:``
comparison — a semantic inversion of that one comparison (``!=`` -> ``==``) survived the entire new
watcher test suite, because every one of those tests scanned main()'s SOURCE (string presence /
ordering) rather than exercising the decision. Extracting it here as a pure, importable function
closes that hole: it is unit-testable directly, with zero Volume reads, zero subprocess, and zero
need to import the watcher script itself (see ``tests/test_watcher_pending_clock.py``).

Import tier: stdlib only, no package side effects — mirrors ``inference/samples_layout.py`` and
``inference/render_key.py`` for the same reason (the watcher must stay importable without dragging
``modal`` into ``sys.modules``).
"""

from __future__ import annotations

__all__ = ["next_pending_since"]


def next_pending_since(
    progress: frozenset[str],
    pending_progress: frozenset[str],
    pending_since: float,
    now: float,
) -> float:
    """The stall clock's refresh decision, and nothing else.

    Returns ``now`` when ``progress`` differs from the last-observed ``pending_progress`` snapshot —
    evidence of life, so the clock resets. Returns ``pending_since`` UNCHANGED when ``progress``
    equals ``pending_progress`` — no new evidence, so staleness keeps accumulating toward
    ``render_stall_minutes``.

    The caller owns updating its own ``pending_progress`` to ``progress``, and can tell whether a
    refresh happened by comparing the returned value against the ORIGINAL ``pending_since`` it passed
    in (``next_pending_since(...) != pending_since``) — this function makes no side effect and reads
    nothing itself, which is the whole point: the DECISION is testable in complete isolation from the
    Volume-polling loop it lives inside.
    """
    return now if progress != pending_progress else pending_since
