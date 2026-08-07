"""Behavioral run-gate test (WR-05) — actually CALL main() and assert the approval RESULT is honored.

The structural gate tests (test_entrypoint_gate.py) are source-order text scans: they prove
``*.spawn(`` appears AFTER ``_require_approval`` in the source, but they pass even if main()
*ignores* the approval result (e.g. a refactor that changes ``if not _require_approval(approve):
raise SystemExit(1)`` to a bare ``_require_approval(approve)`` — the return value dropped). This test
closes that gap by invoking the real ``main`` body with:

  * ``run_dryrun`` monkeypatched to a no-op (keeps it CPU-only, zero GPU),
  * ``signet_trainer.modal.fns.train`` replaced by a recording stub (zero Modal spend — the stub's
    ``.spawn`` only appends to a list and hands back a fake ``FunctionCall``),
  * ``builtins.input`` raising EOFError to simulate non-interactive stdin.

``main`` is an ``@app.local_entrypoint()`` wrapper; its raw function body is ``main.info.raw_f``.
No metered dispatch ever runs — the stub records the call instead. Keep the existing structural
scans in test_entrypoint_gate.py; this is the behavioral complement.

D-10-DEF-17: the dispatch verb is now ``.spawn`` (ASYNC), and main() passes the returned
``FunctionCall`` to ``_watch_dispatch``, so the stub must model BOTH halves — the call and the
handle. A stub that only recorded the call would silently stop exercising the post-dispatch path.

IMPORT DISCIPLINE (Anti-Pattern 6): modal-touching imports (``signet_trainer.modal.entrypoint`` /
``.fns``) are done INSIDE the test bodies, never at module top — so pytest COLLECTION never pulls
``modal`` into ``sys.modules`` and the dry-run purity tests (test_dryrun_*.py) stay green.
"""

from __future__ import annotations

import builtins

import pytest

_MINIMAL_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
"""


class _RecordingCall:
    """Stand-in for a ``modal.FunctionCall`` — the handle ``.spawn()`` hands back (D-10-DEF-17).

    ``_watch_dispatch`` reads ``object_id`` (the re-attach handle it prints) and calls
    ``get(timeout=...)`` (the bounded synchronous watch). Returning instantly models the "job
    finished inside the watch window" branch; the expiry branch is a builtin ``TimeoutError`` from
    the real SDK and is not modelled here.
    """

    object_id = "fc-test-stub"

    def get(self, timeout: float | None = None) -> None:
        return None


class _RecordingFn:
    """Stand-in for a Modal Function — ``.spawn(...)`` records the call instead of dispatching."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _write_config(tmp_path) -> str:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(_MINIMAL_CONFIG, encoding="utf-8")
    return str(cfg_path)


def _raw_main():
    from signet_trainer.modal import entrypoint  # lazy: keep modal out of collection-time sys.modules

    return entrypoint, entrypoint.main.info.raw_f


def _stub_train(monkeypatch) -> _RecordingFn:
    """Replace fns.train with a recorder; main() does ``from ...fns import train`` at call time."""
    from signet_trainer.modal import fns  # lazy import (see module docstring)

    rec = _RecordingFn()
    monkeypatch.setattr(fns, "train", rec)
    return rec


def test_declined_approval_raises_and_dispatches_nothing(tmp_path, monkeypatch) -> None:
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 0)  # skip the heavy dry-run gate
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    rec = _stub_train(monkeypatch)

    with pytest.raises(SystemExit):
        raw_main(config=_write_config(tmp_path), approve=False, mode="train")

    assert rec.calls == [], "no metered dispatch may happen when approval is declined (MODL-02)"


def test_approved_run_dispatches_exactly_once(tmp_path, monkeypatch) -> None:
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 0)
    rec = _stub_train(monkeypatch)

    # --approve authorizes non-interactively; no input() should be consulted.
    raw_main(config=_write_config(tmp_path), approve=True, mode="train")

    assert len(rec.calls) == 1, "an approved run must dispatch train.spawn() exactly once"
