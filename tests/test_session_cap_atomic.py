"""Atomic-write regression tests for the session-spend ledger (P8-IN-04 / T-09-03-01 Tampering).

The ledger gates cumulative yolo spend; a crash mid-write must never leave a truncated,
unparseable SESSION-STATE.json (``read_ledger`` fails CLOSED on corrupt JSON and wedges the
harness). ``append_spend`` and ``consume_blanket`` therefore write through
``session_cap._atomic_write_json`` (temp file + ``os.replace`` atomic rename).

Pure stdlib-json — NO modal/CUDA dependency; runs on Windows/CI with zero spend (Anti-Pattern 6).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------------------------
# CR-02 — multi-writer safety. The ledger is multi-writer by design (N parallel watchers + the
# agent's own per-dispatch appends against the ONE project ledger); an unlocked read-modify-write
# silently LOSES concurrent entries (the cap then under-counts real spend — fail OPEN). These
# tests pin the lock + unique-staging fix; the concurrent one fails on the pre-fix code.
# ---------------------------------------------------------------------------------------------

_N_WRITERS = 4
_APPENDS_PER_WRITER = 60

# One worker process: N sequential $1.00 appends against the shared ledger (argv: ledger, n, tag).
_WORKER_SRC = """
import sys
from signet_trainer.modal.session_cap import append_spend
ledger, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for i in range(n):
    append_spend(ledger, est_usd=1.0, run_ref=f"{tag}-{i}")
"""


def test_concurrent_append_spend_loses_no_entries(tmp_path) -> None:
    """4 writer processes x 60 appends of $1.00 -> ALL 240 entries must survive (CR-02).

    Pre-fix this lost ~3/4 of the entries: every writer staged over the same fixed
    ``SESSION-STATE.json.tmp`` (PermissionError collisions on Windows) and the unsynchronised
    read-modify-write let a stale snapshot overwrite concurrent appends silently on POSIX.
    """
    ledger = tmp_path / "SESSION-STATE.json"
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER_SRC, str(ledger), str(_APPENDS_PER_WRITER), f"w{w}"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for w in range(_N_WRITERS)
    ]
    for proc in procs:
        _, stderr = proc.communicate(timeout=180)
        assert proc.returncode == 0, f"writer crashed:\n{stderr.decode('utf-8', 'replace')}"

    expected = float(_N_WRITERS * _APPENDS_PER_WRITER)
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(data["spend"]) == _N_WRITERS * _APPENDS_PER_WRITER, "lost-update: entries vanished"
    assert read_ledger(ledger) == pytest.approx(expected)
    # No writer debris left behind next to the live ledger.
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.lock"))


def test_staging_names_are_unique_per_writer(tmp_path, monkeypatch) -> None:
    """Two appends must stage through DISTINCT temp names (pre-fix both used ``<name>.tmp``)."""
    ledger = tmp_path / "SESSION-STATE.json"
    srcs: list[str] = []
    real_replace = session_cap.os.replace

    def _spy_replace(src, dst):
        srcs.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(session_cap.os, "replace", _spy_replace)
    append_spend(ledger, est_usd=1.0, run_ref="a")
    append_spend(ledger, est_usd=1.0, run_ref="b")

    assert len(srcs) == 2 and srcs[0] != srcs[1], "staging names must be unique per write"
    assert all(s.endswith(".tmp") for s in srcs)


def test_append_spend_fails_loudly_when_lock_is_held(tmp_path, monkeypatch) -> None:
    """A held (fresh, non-stale) lock must raise TimeoutError — a spend append is never silently
    skipped (CR-01), and the ledger must be left untouched."""
    ledger = tmp_path / "SESSION-STATE.json"
    lock = ledger.with_name(ledger.name + ".lock")
    lock.write_text("someone-else", encoding="utf-8")
    monkeypatch.setattr(session_cap, "DEFAULT_LOCK_TIMEOUT_S", 0.2)

    with pytest.raises(TimeoutError):
        append_spend(ledger, est_usd=1.0, run_ref="blocked")
    assert not ledger.exists(), "a blocked append must not touch the ledger"
    assert lock.exists(), "the contender must never steal a fresh lock"


def test_consume_blanket_fails_loudly_when_lock_is_held(tmp_path, monkeypatch) -> None:
    """consume_blanket (the other ledger writer) honors the same lock + TimeoutError posture."""
    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps(_blanket_ledger(), indent=2), encoding="utf-8")
    original_bytes = ledger.read_text(encoding="utf-8")
    lock = ledger.with_name(ledger.name + ".lock")
    lock.write_text("someone-else", encoding="utf-8")
    monkeypatch.setattr(session_cap, "DEFAULT_LOCK_TIMEOUT_S", 0.2)

    with pytest.raises(TimeoutError):
        consume_blanket(ledger, blanket_index=0, est_usd=1.0, run_ref="blocked")
    assert ledger.read_text(encoding="utf-8") == original_bytes


def test_stale_lock_is_broken_not_a_permanent_wedge(tmp_path) -> None:
    """A crashed writer's leftover lock (older than DEFAULT_LOCK_STALE_S) must be broken so the
    harness never wedges forever — the append then proceeds and releases cleanly."""
    ledger = tmp_path / "SESSION-STATE.json"
    lock = ledger.with_name(ledger.name + ".lock")
    lock.write_text("corpse", encoding="utf-8")
    ancient = time.time() - (session_cap.DEFAULT_LOCK_STALE_S + 60.0)
    os.utime(lock, (ancient, ancient))

    append_spend(ledger, est_usd=2.5, run_ref="after-crash")

    assert read_ledger(ledger) == pytest.approx(2.5)
    assert not lock.exists(), "the lock must be released after the append"
