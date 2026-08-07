"""Atomic-write regression tests for the session-spend ledger (P8-IN-04 / T-09-03-01 Tampering).

The ledger gates cumulative yolo spend; a crash mid-write must never leave a truncated,
unparseable SESSION-STATE.json (``read_ledger`` fails CLOSED on corrupt JSON and wedges the
harness). ``append_spend`` and ``consume_blanket`` therefore write through
``session_cap._atomic_write_json`` (temp file + ``os.replace`` atomic rename).

Pure stdlib-json — NO modal/CUDA dependency; runs on Windows/CI with zero spend (Anti-Pattern 6).
"""

from __future__ import annotations

import json

import pytest

from signet_trainer.modal import session_cap
from signet_trainer.modal.session_cap import append_spend, consume_blanket, read_ledger


def _blanket_ledger() -> dict:
    return {
        "blankets": [{"scope": ["train"], "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}],
        "spend": [],
    }


def test_append_spend_writes_via_temp_then_os_replace(tmp_path, monkeypatch) -> None:
    """append_spend must route through os.replace(tmp, path) — never a direct in-place write."""
    ledger = tmp_path / "SESSION-STATE.json"
    calls: list[tuple[str, str]] = []
    real_replace = session_cap.os.replace

    def _spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(session_cap.os, "replace", _spy_replace)
    append_spend(ledger, est_usd=1.25, run_ref="run-1")

    assert len(calls) == 1, "append_spend must perform exactly one atomic rename"
    src, dst = calls[0]
    assert src.endswith(".tmp"), "the write target must be a sibling .tmp file"
    assert dst == str(ledger), "os.replace must rename onto the real ledger path"
    # Final ledger is valid JSON with the appended spend.
    assert read_ledger(ledger) == pytest.approx(1.25)


def test_consume_blanket_writes_via_temp_then_os_replace(tmp_path, monkeypatch) -> None:
    """consume_blanket (the blanket-spend writer) must also be atomic."""
    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps(_blanket_ledger(), indent=2), encoding="utf-8")

    calls: list[tuple[str, str]] = []
    real_replace = session_cap.os.replace

    def _spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(session_cap.os, "replace", _spy_replace)
    consume_blanket(ledger, blanket_index=0, est_usd=2.0, run_ref="run-2")

    assert len(calls) == 1, "consume_blanket must perform exactly one atomic rename"
    assert calls[0][0].endswith(".tmp")
    assert calls[0][1] == str(ledger)
    assert read_ledger(ledger) == pytest.approx(2.0)


def test_crash_mid_write_leaves_old_ledger_intact(tmp_path, monkeypatch) -> None:
    """A crash during the rename must leave the OLD valid ledger — never a corrupt half-write.

    Simulate a mid-write failure by making os.replace raise; the original SESSION-STATE.json must
    be byte-for-byte unchanged and still parse (read_ledger returns the pre-crash total), so the
    harness never fails CLOSED against a truncated JSON.
    """
    ledger = tmp_path / "SESSION-STATE.json"
    pre_crash = {"spend": [{"ts": "2026-07-11T00:00:00+00:00", "est_usd": 3.0, "run_ref": "prior"}]}
    original_bytes = json.dumps(pre_crash, indent=2)
    ledger.write_text(original_bytes, encoding="utf-8")

    def _boom(src, dst):  # noqa: ANN001
        raise OSError("simulated crash during atomic rename")

    monkeypatch.setattr(session_cap.os, "replace", _boom)

    with pytest.raises(OSError):
        append_spend(ledger, est_usd=1.0, run_ref="doomed")

    # The real ledger is untouched: same bytes, still parseable, same cumulative total.
    assert ledger.read_text(encoding="utf-8") == original_bytes
    assert read_ledger(ledger) == pytest.approx(3.0)
