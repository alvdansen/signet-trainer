"""Phase 09.1 — RUNTIME session-cap ledger coverage (AUDIT-#4 / CR-01), CPU-only.

The Wave-1 suite (``test_watcher_hardening.py``) verifies the watchers' money-safety by SOURCE
SCAN only. That is how CR-01 slipped past: a source scan cannot see that the ledger never
actually advanced, because the watchers shelled out to ``sys.executable -c ...`` with ``env=None``
and a DISCARDED exit code — under the documented bare invocation (no ``PYTHONPATH=src``) the child
raised ``ModuleNotFoundError`` and exited 1 silently, so ``append_spend`` was a no-op and the
cumulative session cap could never trip.

This file adds the missing RUNTIME proof:
  1. ADVANCEMENT — ``append_spend`` twice against a real temp ledger; ``read_ledger`` reflects the
     advanced balance (the exact behaviour the broken subprocess never delivered).
  2. LOUD-ON-FAILURE — an un-landable write RAISES (OSError) and a value-guard violation RAISES
     (ValueError); a write failure is observable, never swallowed.
  3. ANTI-REGRESSION — a targeted structural guard that neither watcher shells out for
     ``append_spend`` again (locks the CR-01 subprocess pattern out for good), in ADDITION to the
     runtime assertions above.

Zero GPU / zero modal / zero spend. Pure stdlib + the pure-Python ``session_cap`` module.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from signet_trainer.modal.session_cap import append_spend, read_ledger

REPO = Path(__file__).resolve().parents[1]
FORK = REPO / "scripts" / "_watch_campaign_parallel_inference.py"
GENERALIZED = REPO / "scripts" / "watch_parallel_inference.py"
# issue #19 item 1 — FORK is not published in this repo; see test_watcher_hardening.py's identical
# filter for the full rationale. Filtering here keeps `test_watchers_do_not_subprocess_for_append_
# spend` from aborting on the fork before it ever scans the shipped watcher.
assert GENERALIZED.is_file(), "the shipped generalized watcher must always be present"
_WATCHERS = tuple(p for p in (FORK, GENERALIZED) if p.is_file())


def _strip_comments(src: str) -> str:
    # Line-wise ``#`` strip so a token that only appears in an explanatory comment (e.g. the comment
    # that DESCRIBES the removed subprocess) cannot false-pass or false-fail a structural scan.
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())


# ---- (1) ADVANCEMENT: the ledger balance actually advances (runtime round-trip) ----


def test_append_spend_advances_ledger_runtime(tmp_path):
    ledger = tmp_path / "SESSION-STATE.json"
    # A missing ledger is a fresh session -> 0.0 spent, never raises.
    assert read_ledger(ledger) == 0.0
    # Two dispatch-time appends of the campaign per-render estimate.
    append_spend(ledger, 2.46, run_ref="test render 1")
    append_spend(ledger, 2.46, run_ref="test render 2")
    # The balance advanced by the summed est_usd — the proof the broken subprocess never delivered.
    assert read_ledger(ledger) == pytest.approx(4.92, abs=1e-9)


def test_append_spend_creates_ledger_when_absent(tmp_path):
    # append_spend must materialise the ledger file (minimal skeleton) so the very FIRST render of a
    # session is counted — otherwise the cap would start from a phantom 0.0 forever.
    ledger = tmp_path / "nested" / "SESSION-STATE.json"
    assert not ledger.exists()
    append_spend(ledger, 1.0, run_ref="first render")
    assert ledger.exists()
    assert read_ledger(ledger) == pytest.approx(1.0, abs=1e-9)


# ---- (2) LOUD-ON-FAILURE: an un-landable / invalid write RAISES, never silently no-ops ----


def test_append_spend_raises_when_write_cannot_land(tmp_path):
    # Force an un-landable write: the ledger's PARENT is an existing regular file, so
    # _atomic_write_json's ``path.parent.mkdir(...)`` fails with an OSError. The in-process call
    # (unlike the discarded-exit-code subprocess) surfaces this LOUDLY and halts the watcher.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    unwritable_ledger = blocker / "SESSION-STATE.json"
    with pytest.raises(OSError):
        append_spend(unwritable_ledger, 1.0, run_ref="should raise")


def test_append_spend_raises_on_negative_est_usd(tmp_path):
    # The documented value-guard (WR-06): a negative est_usd would reduce the counted cumulative
    # spend (fail OPEN against the cap) — it must raise, not silently record.
    ledger = tmp_path / "SESSION-STATE.json"
    with pytest.raises(ValueError):
        append_spend(ledger, -1.0, run_ref="negative")
    # NaN is likewise rejected (would poison every later sum).
    with pytest.raises(ValueError):
        append_spend(ledger, float("nan"), run_ref="nan")


# ---- (3) ANTI-REGRESSION: neither watcher may reintroduce the fire-and-forget subprocess ----


def test_watchers_do_not_subprocess_for_append_spend():
    # Structural guard IN ADDITION to the runtime proof above: lock out the CR-01 pattern so a future
    # edit cannot silently reintroduce a discarded-exit-code subprocess for the ledger write.
    #
    # Issue #37 finding 1/2 (single-source accounting): GENERALIZED no longer owns a ledger write at
    # all — its own `def append_spend(step) -> None:` wrapper (the one this test used to isolate) was
    # deliberately REMOVED. Every dispatch this watcher triggers goes through the entrypoint-gate
    # subprocess in dispatch_render(), and THAT process now books the ledger in-process, once per
    # successful ``.spawn()`` (entrypoint.py's ``_watch_dispatch`` — covered by
    # test_entrypoint_session_cap.py, not this file). The CR-01 regression this test guards against —
    # a fire-and-forget ``sys.executable -c ...`` subprocess with a discarded exit code — is therefore
    # structurally impossible for a watcher that does not write the ledger itself; assert exactly
    # that absence, plus (belt-and-braces) that no CR-01-shaped subprocess call sneaks back in under
    # any other name.
    for path in _WATCHERS:
        src = _strip_comments(path.read_text(encoding="utf-8"))
        m = re.search(r"\ndef append_spend\(step: int\) -> None:\n(.*?)(?=\ndef )", src, re.DOTALL)
        if m is None:
            # No local ledger-write wrapper at all (the current GENERALIZED design) — the CR-01
            # subprocess pattern has nothing to hide inside. Still guard against it reappearing
            # ANYWHERE in the file under a different name.
            assert "append_spend as _append_spend" not in src, (
                f"{path.name}: imports the in-process append_spend alias but defines no wrapper that "
                "calls it — dead import, or the wrapper was renamed without updating this test"
            )
            assert not re.search(r'sys\.executable.*["\']-c["\']', src), (
                f"{path.name}: must not shell out to `sys.executable -c ...` for the ledger write "
                "(CR-01) — booking now happens in-process inside the entrypoint gate subprocess"
            )
            continue
        # A watcher that DOES still define its own wrapper (e.g. a reinstated campaign fork) must
        # keep the original in-process discipline: isolate the wrapper body (up to the next
        # top-level `def `) and assert it no longer builds a `-c` code string nor invokes the
        # interpreter.
        body = m.group(1)
        assert "sys.executable" not in body, f"{path.name}: append_spend must not shell out (CR-01)"
        assert '"-c"' not in body and "'-c'" not in body, (
            f"{path.name}: append_spend must not build a `-c` code string (CR-01)"
        )
        assert "sh(" not in body, f"{path.name}: append_spend must not call sh() (CR-01)"
        # It must instead import the pure function and call it in-process against the ledger path.
        assert "append_spend as _append_spend" in src, (
            f"{path.name}: must import `append_spend as _append_spend`"
        )
        assert "_append_spend(LEDGER_PATH" in body, (
            f"{path.name}: append_spend must call `_append_spend(LEDGER_PATH ...)` in-process"
        )


# ---- Documentation-only: WHY the in-process call is required (records the reproduced failure) ----


def test_bare_interpreter_import_reproduces_the_break(tmp_path):
    # Records the exact failure the verifier reproduced: under the documented bare invocation
    # (`python scripts/watch_parallel_inference.py <cfg>`, no PYTHONPATH=src, signet_trainer not
    # pip-installed) a `sys.executable -c "import signet_trainer"` child exits NON-ZERO — which the
    # old watcher swallowed. If signet_trainer IS importable without PYTHONPATH (pip-installed), the
    # documented failure does not reproduce and this record is skipped rather than false-failing.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(
        [sys.executable, "-c", "import signet_trainer"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )
    if r.returncode == 0:
        pytest.skip(
            "signet_trainer importable without PYTHONPATH (pip-installed) — the documented "
            "bare-invocation failure does not reproduce in this env"
        )
    assert r.returncode != 0
    assert "No module named" in r.stderr or "ModuleNotFoundError" in r.stderr
