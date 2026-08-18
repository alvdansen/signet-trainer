"""issue #45 PR-1 must-fix #1 / #4 — the stall clock's refresh DECISION, tested behaviorally.

PR-2 verifier finding (MAJOR, test-theater): every test in ``tests/test_watcher_hardening.py`` scans
``main()``'s SOURCE (string presence / ordering) rather than exercising the refresh decision. A
semantic inversion of the one comparison that decides it — ``if progress != pending_progress:``
flipped to ``if progress == pending_progress:`` (the clock would reset ONLY when nothing new
committed, so a hung render would never stall and a live one would never refresh) — left the entire
suite green.

Two closures, deliberately layered:

  1. ``next_pending_since()`` (``inference/stall_clock.py``) is now the pure, importable function
     that OWNS the decision. Direct unit tests below pin its behavior in isolation — no Volume, no
     subprocess, no watcher import at all.
  2. The loop-level tests drive the REAL ``scripts/watch_parallel_inference.py::main()`` — imported
     via importlib (it is a standalone script, not a package: an `if __name__` guard would not save
     us, its module top-level itself gates on ``sys.argv`` and calls ``load_config``) — with every
     Volume/dispatch/session-cap dependency monkeypatched and a fully-controlled fake clock. This is
     the "loop-level test that a pending render which has committed clips but no terminal artifact
     does NOT trip the stall path" the PR-1 verifier's must-fix #4 explicitly demanded, and it catches
     the inversion whether it lives inside ``next_pending_since()`` OR at the main()-call-site that
     consumes its return value.

Zero GPU / zero Modal / zero spend: `dispatch_render`, `render_landed`, `render_progress_artifacts`,
`append_spend`, `session_cap_check`, `read_ledger`, `list_checkpoint_steps`,
`committed_render_stamps` and `refresh_grid` are all monkeypatched fakes; `time.time`/`time.sleep`
are replaced with a deterministic fake clock that raises once the test has seen enough polls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from signet_trainer.inference.stall_clock import next_pending_since

REPO = Path(__file__).resolve().parents[1]
WATCHER = REPO / "scripts" / "watch_parallel_inference.py"
SAMPLE_CONFIG = REPO / "configs" / "sample.yaml"


# ---- 1. the pure decision, tested directly (no watcher import needed) -------------------------


def test_next_pending_since_refreshes_when_progress_changes():
    assert next_pending_since(frozenset({"a"}), frozenset(), 100.0, 500.0) == 500.0


def test_next_pending_since_holds_when_progress_unchanged():
    assert next_pending_since(frozenset({"a"}), frozenset({"a"}), 100.0, 500.0) == 100.0


def test_next_pending_since_advances_iff_progress_changed():
    # Exhaustive over both directions — the exact property a `!=` <-> `==` inversion breaks.
    same = frozenset({"a", "b"})
    grown = frozenset({"a", "b", "c"})
    assert next_pending_since(same, same, 10.0, 20.0) == 10.0            # unchanged -> no refresh
    assert next_pending_since(grown, same, 10.0, 20.0) == 20.0           # changed -> refresh
    assert next_pending_since(frozenset(), frozenset(), 10.0, 20.0) == 10.0  # both empty -> no refresh


# ---- 2. loop-level: drive the REAL main() with everything Volume-shaped mocked ------------------


class _StopLoop(Exception):
    """Raised by the fake clock's sleep() once the test has observed enough polls — the clean,
    deterministic way to end main()'s `while time.time() < deadline` loop without waiting on the
    real 12h DEADLINE_HOURS or faking a Volume artifact landing."""


class _FakeTime:
    """A controllable stand-in for the stdlib `time` module, bound to the watcher module's own
    `time` name only (`monkeypatch.setattr(mod, "time", ...)`) — the REAL global `time` module is
    never touched, so nothing outside this one test is affected.

    `.time()` never advances on its own; only `.sleep()` advances the clock, mirroring how main()
    actually uses it (several `time.time()` reads happen almost instantaneously within one poll
    iteration; only the `time.sleep(POLL_SECONDS)` at the bottom represents real elapsed time).
    """

    def __init__(self, start: float = 1_700_000_000.0, max_sleeps: int | None = None):
        self.now = start
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds
        if self.max_sleeps is not None and self.sleeps >= self.max_sleeps:
            raise _StopLoop()


def _load_watcher():
    """Fresh module object for `scripts/watch_parallel_inference.py`, imported with a real minimal
    LTX config so its module-level `load_config` call succeeds. `sys.argv` is only touched for the
    duration of `exec_module` (restored immediately after) — the loaded module keeps its own
    `SAMPLE_CONFIG`/`_cfg` regardless of the real process's argv afterward.
    """
    old_argv = sys.argv
    sys.argv = ["watch_parallel_inference.py", str(SAMPLE_CONFIG)]
    try:
        spec = importlib.util.spec_from_file_location("watch_parallel_inference_under_test", WATCHER)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def _patch_common(monkeypatch, mod, tmp_path: Path, fake_time: _FakeTime, dispatches: list[int], *,
                   progress_fn):
    """Every dependency main() touches, neutered, except `render_landed` (always False — no render
    in this bundle ever lands; we are only exercising the PENDING/STALL machinery) and
    `render_progress_artifacts` (the one axis each test varies)."""
    monkeypatch.setattr(mod, "time", fake_time)
    monkeypatch.setattr(mod, "POLL_SECONDS", 60)
    monkeypatch.setattr(mod, "RENDER_STALL_MIN", 5.0)
    monkeypatch.setattr(mod, "HEARTBEAT_FILE", tmp_path / "heartbeat")
    monkeypatch.setattr(mod, "SENTINEL_FILE", tmp_path / "sentinel")
    monkeypatch.setattr(mod, "CAP_STOP_SENTINEL_FILE", tmp_path / "capstop")
    monkeypatch.setattr(mod, "list_checkpoint_steps", lambda: [50])
    monkeypatch.setattr(mod, "committed_render_stamps", lambda: [])
    monkeypatch.setattr(mod, "read_ledger", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(mod, "_resolve_cap", lambda *_a, **_k: 1000.0)
    monkeypatch.setattr(mod, "session_cap_check", lambda *_a, **_k: SimpleNamespace(allowed=True))
    monkeypatch.setattr(mod, "append_spend", lambda step: None)
    monkeypatch.setattr(mod, "dispatch_render", lambda step: (dispatches.append(step) or True))
    monkeypatch.setattr(mod, "render_landed", lambda step, ckpt: False)
    monkeypatch.setattr(mod, "render_progress_artifacts", lambda ckpt: progress_fn(ckpt))
    monkeypatch.setattr(mod, "refresh_grid", lambda: None)


_wpi = _load_watcher()


def test_stall_branch_fires_when_progress_never_changes(monkeypatch, tmp_path):
    """The genuinely-hung-render path (PR-1 must-fix #1's OTHER half, at loop level): with
    RENDER_STALL_MIN=5min and POLL_SECONDS=60s, flat (never-changing) progress for >5 real minutes
    must be declared STALLED and re-dispatched — driven through the REAL main() loop, so an inversion
    at EITHER the pure-function layer or the main()-call-site that consumes it is caught here too."""
    dispatches: list[int] = []
    fake_time = _FakeTime(max_sleeps=8)
    _patch_common(monkeypatch, _wpi, tmp_path, fake_time, dispatches,
                   progress_fn=lambda _ckpt: frozenset())  # never changes -> no evidence of life
    with pytest.raises(_StopLoop):
        _wpi.main()
    assert len(dispatches) >= 2, (
        f"a render with NO new committed artifact for longer than render_stall_minutes must be "
        f"declared STALLED and become eligible for re-dispatch; got {len(dispatches)} dispatch(es) "
        f"over 8 polls (60s each, stall gate 5min) — the stall path never fired"
    )


def test_growing_progress_never_trips_the_stall_path(monkeypatch, tmp_path):
    """issue #45 PR-1 must-fix #4 (the loop-level test explicitly demanded): a pending render that
    keeps committing NEW clips every poll (progress keeps changing) but has not yet produced its
    terminal identity-directory artifact (render_landed stays False the entire time) must NEVER be
    declared STALLED, no matter how long it runs — driven through the REAL main() loop over far more
    polls than render_stall_minutes would tolerate if progress were flat."""
    dispatches: list[int] = []
    fake_time = _FakeTime(max_sleeps=24)  # 24 * 60s = 24min >> the 5min stall gate, on purpose
    seen: dict[str, int] = {"n": 0}

    def growing_progress(_ckpt: str | None) -> frozenset[str]:
        seen["n"] += 1
        return frozenset({f"base/clip_{seen['n']:03d}.mp4"})

    _patch_common(monkeypatch, _wpi, tmp_path, fake_time, dispatches, progress_fn=growing_progress)
    with pytest.raises(_StopLoop):
        _wpi.main()
    assert seen["n"] >= 20, "the progress probe itself must have been polled repeatedly for this to be a real test"
    assert len(dispatches) == 1, (
        f"a render that keeps committing new clips every poll must never be declared STALLED even "
        f"though it hasn't landed yet; got {len(dispatches)} dispatches, expected exactly 1 (the "
        f"initial one) — a false STALL would double-book metered A100 spend for a render that is "
        f"plainly still alive"
    )
