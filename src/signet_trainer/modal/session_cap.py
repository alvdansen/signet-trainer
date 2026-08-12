"""Cumulative session-spend ledger + cap decision (D-8-YOLOCAP). PURE — Modal-agnostic.

Bounds yolo autonomy by ESTIMATED cumulative spend, not just the per-run ``$50`` guardrail (cost.py).
The ledger tracks the dispatch-time ``est_usd`` (what ``append_spend`` records); reconciliation
against actual Modal billing is out of scope.
A dispatch stays authorized only while ``projected + spent_so_far <= session_cap_usd``; once the
cumulative cap would be breached the harness drops to ask-first. The ledger is a plain
SESSION-STATE.json file written with stdlib ``json`` only (no pandas, no modal).

CRITICAL — Anti-Pattern 6: this module imports NOTHING from ``modal`` (it lives under modal/ for
cohesion but is pure-Python so it's unit-testable on Windows with no Modal install). It mirrors
``cost.py`` in shape: a frozen dataclass decision, pure functions, a documented default constant.

Cap + ledger-path source of truth (WR-02 — ONE authoritative chain, config-first). These pure
functions take the cap and ledger path as ARGUMENTS; the caller (the ``training-run`` skill) resolves
them from this chain and never hardcodes either:
  * house default = config: ``ModalConfig.session_spend_ledger_path`` (ledger path) and
    ``ModalConfig.session_cap_usd`` (cap), D-NOHARDCODE;
  * per-session override = the ``session_cap_usd`` the setup gate writes INTO that ledger file — when
    present it is the live cap, else fall back to the config default.
``DEFAULT_SESSION_CAP_USD`` below is only the proposed house figure the setup gate offers the operator.

Blanket-grant shape (D-8-BLANKET) — each entry in the SESSION-STATE.json ``blankets`` list is a dict::

    {
      "scope": ["train", "sample", "preprocess"],    # run types this grant pre-authorizes
      "cap_usd": 5.0,                                 # per-blanket cumulative ceiling (USD)
      "expires": "session" | "<ISO-8601 timestamp>",  # "session" never expires mid-session
      "spent_usd": 0.0                               # spend already attributed to this grant
    }

A blanket authorizes a dispatch iff ``run_type`` is in its ``scope`` AND
``projected + spent_usd <= cap_usd`` AND it is not expired. ``SESSION-STATE.template.json`` ships
with an empty ``blankets`` list; the entry schema lives here + in the 08-03 README. Cost prints
always still log regardless of blanket authorization.

CR-01 — airtight accounting: EVERY approved dispatch (strict, yolo, AND blanket) must be recorded
so ``read_ledger`` reflects estimated cumulative spend. ``append_spend`` is the single spend-append path
for strict/yolo; ``consume_blanket`` is the single blanket-spend path — it BOTH depletes the matched
blanket's ``spent_usd`` (so a ``cap_usd`` blanket can no longer authorize unbounded dispatches) AND
appends a ``spend`` entry (so blanket spend is visible to the cumulative session cap). Blanket
attribution is unambiguous: ``blanket_authorizes`` returns ``CapDecision.blanket_index`` naming WHICH
blanket authorized, and that index is fed straight to ``consume_blanket``.

CR-02 — multi-writer safety: the ledger has SEVERAL concurrent writers by design (N parallel
watchers, one per output_dir, plus the agent's own per-dispatch appends — all against the ONE
project ledger). Both writers therefore hold an exclusive sidecar lock (``_ledger_lock``) across
their whole read-modify-write and stage through a per-writer-unique temp file — without both, a
stale snapshot silently overwrites (LOSES) a concurrent writer's entry and the cumulative cap
under-counts real spend. Lock tuning lives in the ``DEFAULT_LOCK_*`` constants below.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# RESEARCH A3 — a proposed house default the operator confirms at setup. Config-driven live value lives on
# ModalConfig.session_cap_usd (never hardcoded in a skill, D-NOHARDCODE).
DEFAULT_SESSION_CAP_USD = 10.0

# CR-02 — ledger multi-writer safety tuning (documented defaults, never inline literals). The ledger
# is MULTI-WRITER by design: N parallel watchers (watch_parallel_inference.py, one per output_dir)
# plus the agent's own append_spend after each dispatch all append to the ONE project ledger
# (ModalConfig.session_spend_ledger_path). Each read-modify-write therefore holds an exclusive
# sidecar lockfile (``<ledger>.lock``, O_CREAT|O_EXCL — works on POSIX and Windows, stdlib-only per
# Anti-Pattern 6) for the WHOLE read→append→replace, or a concurrent writer's entry is silently
# lost and the cumulative cap under-counts real spend (fail OPEN).
DEFAULT_LOCK_TIMEOUT_S = 10.0  # a ledger write takes ms; 10s of contention means something is wrong
DEFAULT_LOCK_POLL_S = 0.02  # spin/backoff interval while another writer holds the lock
DEFAULT_LOCK_STALE_S = 30.0  # a lock older than this is a crashed writer's leftover — break it
DEFAULT_REPLACE_RETRIES = 25  # Windows: os.replace fails while a reader holds the file open; retry


@contextmanager
def _ledger_lock(
    path: Path,
    timeout_s: float | None = None,
    poll_s: float | None = None,
    stale_s: float | None = None,
) -> Iterator[None]:
    """Exclusive cross-process lock over one ledger's read-modify-write (CR-02).

    A sidecar ``<name>.lock`` created with ``os.open(O_CREAT | O_EXCL)`` — atomic on both POSIX and
    Windows, no new deps (Anti-Pattern 6: stdlib-only). Contending writers spin with backoff until
    the holder unlinks the lock; a lock older than ``stale_s`` is treated as a crashed writer's
    leftover and broken (a ledger write takes milliseconds, so a stale lock can only be a corpse).
    Acquisition past ``timeout_s`` raises ``TimeoutError`` LOUDLY — a spend append must never be
    silently skipped (CR-01 airtight accounting fails CLOSED, not open).

    ``None`` tuning args resolve to the module defaults at CALL time (not def time) so tests can
    monkeypatch the constants.
    """
    timeout_s = DEFAULT_LOCK_TIMEOUT_S if timeout_s is None else timeout_s
    poll_s = DEFAULT_LOCK_POLL_S if poll_s is None else poll_s
    stale_s = DEFAULT_LOCK_STALE_S if stale_s is None else stale_s
    lock = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # PermissionError: Windows raises EACCES (not EEXIST) while the holder's unlink of the
            # lock is still delete-pending — same meaning as contended, so re-poll, don't crash.
            try:
                if time.time() - lock.stat().st_mtime > stale_s:
                    lock.unlink()  # crashed writer's leftover — break it and re-contend
                    continue
            except OSError:
                pass  # lock vanished or is contended mid-check — just re-poll
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"could not acquire ledger lock {lock} within {timeout_s:.1f}s -- another "
                    "writer is wedged (or its stale lock is younger than "
                    f"{stale_s:.1f}s). The spend append was NOT recorded; fails CLOSED (CR-01)."
                )
            time.sleep(poll_s)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))  # holder pid, for post-mortem only
        yield
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass  # already broken as stale by a contender — nothing to release


@dataclass(frozen=True)
class CapDecision:
    """The result of a cumulative session-cap check — never auto-dispatches when ``allowed`` is False."""

    allowed: bool
    projected_usd: float
    spent_so_far: float
    session_cap_usd: float
    reason: str
    # CR-01: which blanket authorized this dispatch (index into the SESSION-STATE ``blankets`` list),
    # or None for session-cap decisions and for non-authorizing blanket results. The caller feeds this
    # straight to ``consume_blanket`` so the spend is attributed to the exact grant that authorized it.
    blanket_index: int | None = None


def session_cap_check(
    projected_usd: float,
    spent_so_far: float,
    session_cap_usd: float,
) -> CapDecision:
    """Decide whether a dispatch stays within the cumulative session cap (D-8-YOLOCAP).

    ``allowed = (projected_usd + spent_so_far) <= session_cap_usd`` — exactly-at-cap is allowed,
    mirroring ``guardrail_check``. The blocked ``reason`` names the cap and states the harness drops
    to ask-first. The caller MUST honor ``allowed=False`` and refuse the ``.remote()`` dispatch.

    WR-06: a negative or NaN ``projected_usd`` / ``spent_so_far`` fails CLOSED (``allowed=False``) — a
    negative projection would falsely shrink the cumulative and over-authorize, and ``NaN <= cap`` is
    always False anyway; both are rejected explicitly with a clear reason.
    """
    for name, val in (("projected_usd", projected_usd), ("spent_so_far", spent_so_far)):
        if val != val or val < 0:  # NaN (val != val) or negative
            return CapDecision(
                allowed=False,
                projected_usd=projected_usd,
                spent_so_far=spent_so_far,
                session_cap_usd=session_cap_usd,
                reason=(
                    f"invalid {name}={val!r} (negative or NaN) — fails CLOSED, drops to ask-first "
                    "(a negative projection would over-authorize the cumulative cap)"
                ),
            )
    cumulative = projected_usd + spent_so_far
    if cumulative <= session_cap_usd:
        return CapDecision(
            allowed=True,
            projected_usd=projected_usd,
            spent_so_far=spent_so_far,
            session_cap_usd=session_cap_usd,
            reason=(
                f"cumulative ${cumulative:.2f} (projected ${projected_usd:.2f} + spent "
                f"${spent_so_far:.2f}) within session cap ${session_cap_usd:.2f}"
            ),
        )
    return CapDecision(
        allowed=False,
        projected_usd=projected_usd,
        spent_so_far=spent_so_far,
        session_cap_usd=session_cap_usd,
        reason=(
            f"cumulative ${cumulative:.2f} (projected ${projected_usd:.2f} + spent "
            f"${spent_so_far:.2f}) exceeds session cap ${session_cap_usd:.2f} — drops to ask-first"
        ),
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write ``data`` as pretty JSON to ``path`` ATOMICALLY (P8-IN-04 / T-09-03-01 Tampering).

    Writes to a sibling ``<name>.<pid>.<uuid>.tmp`` then ``os.replace(tmp, path)`` — an atomic
    rename on both POSIX and Windows (``os.replace`` overwrites the destination). A crash mid-write
    therefore leaves the OLD, still-valid ledger intact rather than a truncated / half-written JSON
    that ``read_ledger`` would reject LOUDLY (fail CLOSED) and wedge the harness. The single
    whole-file writer for both ``append_spend`` and ``consume_blanket``.

    CR-02: the staging name is UNIQUE per writer (pid + uuid) — a fixed shared ``<name>.tmp`` let
    concurrent writers stage over each other (and collide with ``PermissionError`` on Windows).
    On Windows ``os.replace`` also fails with ``PermissionError`` while a READER (``read_ledger``
    in another process) transiently holds the destination open — that is retried with backoff
    (``DEFAULT_REPLACE_RETRIES``); POSIX never hits it. Any other ``OSError`` propagates unretried.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        for attempt in range(DEFAULT_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows sharing violation: a reader holds the destination
                if attempt == DEFAULT_REPLACE_RETRIES - 1:
                    raise
                time.sleep(DEFAULT_LOCK_POLL_S)
    except BaseException:
        try:
            tmp.unlink()  # never leave an orphaned staging file next to the ledger
        except OSError:
            pass
        raise


def read_ledger(ledger_path: str | Path) -> float:
    """Return the ESTIMATED cumulative spend: the summed ``est_usd`` of recorded dispatches.

    This is the dispatch-time estimate (``append_spend`` records ``est_usd``), NOT reconciled
    against actual Modal billing — reconciliation is out of scope for the yolo cap.
    A MISSING file is a FRESH session -> ``0.0`` spent (never raises). Reads ONLY the top-level
    ``spend`` list; tolerates the setup-gate keys (``triage_mode`` / ``session_cap_usd`` /
    ``blankets``) being present or absent.

    WR-06 fail-closed posture for a PRESENT-but-corrupt ledger (distinct from the missing-file case):
      * unparseable JSON -> raises ``ValueError`` LOUDLY. The cumulative cap cannot be computed, so
        the harness must NOT authorize spend against an unknown baseline — repair/delete the file.
      * a non-object ``spend`` entry -> raises ``ValueError`` (malformed ledger, fail closed).
      * a NEGATIVE ``est_usd`` in an entry -> clamped to ``0.0`` in the sum so it can never REDUCE the
        counted cumulative spend (a negative would fail open against the cap).
    """
    path = Path(ledger_path)
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"corrupted spend ledger {path}: not valid JSON ({exc}). The cumulative session cap "
            "CANNOT be computed, so the harness fails CLOSED — repair or delete the file (deleting "
            "resets to a fresh 0.0-spent session) before dispatching."
        ) from exc
    if not isinstance(data, dict):
        return 0.0
    spend = data.get("spend", [])
    if not isinstance(spend, list):
        return 0.0
    total = 0.0
    for i, entry in enumerate(spend):
        if not isinstance(entry, dict):
            raise ValueError(
                f"corrupted spend ledger {path}: spend[{i}] is not an object "
                f"({type(entry).__name__}) — fails CLOSED; repair the file before dispatching."
            )
        est = float(entry.get("est_usd", 0.0))
        total += max(0.0, est)  # a negative entry must never reduce cumulative spend (fail closed)
    return total


def append_spend(ledger_path: str | Path, est_usd: float, run_ref: str | None = None) -> None:
    """Append one dated spend entry (``{ts, est_usd, run_ref}``) to the ``spend`` list.

    Creates the file (with a minimal skeleton) when absent; preserves any existing setup-gate keys
    (``triage_mode`` / ``session_cap_usd`` / ``blankets``) — this operates ONLY on ``spend``.

    CR-02: the whole read-modify-write runs under ``_ledger_lock`` — the ledger is multi-writer
    (N parallel watchers + the agent's own dispatch appends), and an unlocked interleaving lets a
    stale snapshot overwrite (silently LOSE) a concurrent writer's entry, under-counting the
    cumulative cap. Raises ``TimeoutError`` if the lock cannot be acquired (never skip recording).

    WR-06: a negative or NaN ``est_usd`` is REJECTED — a negative append would silently reduce the
    counted cumulative spend (fail open against the cap), and NaN would poison every later sum.
    """
    if est_usd < 0 or est_usd != est_usd:  # negative OR NaN (NaN != NaN)
        raise ValueError(
            f"append_spend: est_usd must be a non-negative, finite number; got {est_usd!r} "
            "(a negative amount would reduce the counted cumulative spend — fail open)."
        )
    path = Path(ledger_path)
    with _ledger_lock(path):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
        data.setdefault("spend", [])
        data["spend"].append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "est_usd": float(est_usd),
                "run_ref": run_ref,
            }
        )
        _atomic_write_json(path, data)


def consume_blanket(
    ledger_path: str | Path,
    blanket_index: int,
    est_usd: float,
    run_ref: str | None = None,
) -> None:
    """Attribute one blanket-authorized dispatch's spend to the grant that authorized it (CR-01).

    ONE call does BOTH halves of blanket accounting so a blanket dispatch can never go un-counted:

      1. Adds ``est_usd`` to ``blankets[blanket_index]['spent_usd']`` so the blanket's own
         ``cap_usd`` ceiling DEPLETES. Without this a ``cap_usd: 5.0`` blanket would authorize an
         unbounded number of $3 dispatches (``spent_usd`` stays ``0.0`` forever — CR-01).
      2. Appends a ``{ts, est_usd, run_ref, blanket_index}`` entry to the top-level ``spend`` list so
         the dispatch is ALSO visible to ``read_ledger`` / the cumulative session cap (blanket spend
         was previously invisible to ``read_ledger``).

    ``blanket_index`` is ``CapDecision.blanket_index`` from ``blanket_authorizes`` — the exact grant
    that authorized this run. Raises ``ValueError`` on a negative ``est_usd`` (a negative amount would
    silently REDUCE both the blanket and the cumulative counters — fail open) or an out-of-range
    ``blanket_index``. This operates on the SAME SESSION-STATE.json ``append_spend`` uses; both are
    pure-stdlib (no modal), and both hold ``_ledger_lock`` across the whole read-modify-write
    (CR-02 — same lost-update exposure as ``append_spend``, same lock, same TimeoutError posture).
    """
    if est_usd < 0 or est_usd != est_usd:  # negative OR NaN (NaN != NaN) fails closed.
        raise ValueError(
            f"consume_blanket: est_usd must be a non-negative, finite number; got {est_usd!r} "
            "(a negative/NaN amount would corrupt the blanket + cumulative spend accounting)."
        )
    path = Path(ledger_path)
    with _ledger_lock(path):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        else:
            data = {}
        blankets = data.get("blankets", [])
        if not isinstance(blankets, list) or not (0 <= blanket_index < len(blankets)):
            n = len(blankets) if isinstance(blankets, list) else 0
            raise ValueError(
                f"consume_blanket: blanket_index {blanket_index} is out of range for {n} blanket(s) "
                f"in {path} — pass the CapDecision.blanket_index returned by blanket_authorizes."
            )
        blanket = blankets[blanket_index]
        if not isinstance(blanket, dict):
            raise ValueError(
                f"consume_blanket: blankets[{blanket_index}] in {path} is not a dict "
                f"({type(blanket).__name__}) — malformed SESSION-STATE, refusing to attribute spend."
            )
        blanket["spent_usd"] = float(blanket.get("spent_usd", 0.0)) + float(est_usd)
        data.setdefault("spend", [])
        data["spend"].append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "est_usd": float(est_usd),
                "run_ref": run_ref,
                "blanket_index": blanket_index,
            }
        )
        _atomic_write_json(path, data)


def _is_expired(expires: str, now: datetime) -> bool:
    """A blanket's ``expires`` gate — the literal ``"session"`` never expires mid-session; a past ISO
    timestamp does. An unparseable expiry fails CLOSED (treated as expired — never over-authorize)."""
    if expires == "session":
        return False
    try:
        exp_dt = datetime.fromisoformat(expires)
    except (TypeError, ValueError):
        return True
    if exp_dt.tzinfo is None:
        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return exp_dt < now


def blanket_authorizes(
    blankets: list[dict],
    run_type: str,
    projected_usd: float,
    now: datetime | None = None,
) -> CapDecision:
    """D-8-BLANKET consumption: does any blanket grant authorize this dispatch?

    Returns an ``allowed=True`` CapDecision for the FIRST blanket that authorizes ``run_type`` —
    i.e. ``run_type`` is in the blanket's ``scope`` AND ``projected_usd + spent_usd <= cap_usd``
    (exactly-at-cap allowed) AND it is not expired (``expires == "session"`` never expires
    mid-session; a past ISO timestamp is expired). If NO blanket matches, returns ``allowed=False``
    (out-of-scope / over-cap / expired all fall here) — the caller then drops to ask-first. This
    formalizes the D-6/D-7 blanket pattern (a LOCKED decision); it does NOT suppress cost prints.
    """
    now = now or datetime.now(timezone.utc)
    for index, blanket in enumerate(blankets):
        # WR-03: SESSION-STATE.json is hand/agent-authored — a malformed entry must fail CLOSED, never
        # authorize. A non-dict blanket is skipped; a ``scope`` written as a STRING (e.g.
        # "scope": "sample") must NOT fall through to Python substring membership ("train" in
        # "no-train-allowed" is True), which would authorize run types it was meant to exclude.
        if not isinstance(blanket, dict):
            continue
        scope = blanket.get("scope", [])
        if not isinstance(scope, (list, tuple)) or run_type not in scope:
            continue  # missing / non-list / out-of-scope all fail closed
        if _is_expired(blanket.get("expires", "session"), now):
            continue
        cap_usd = float(blanket.get("cap_usd", 0.0))
        spent_usd = float(blanket.get("spent_usd", 0.0))
        cumulative = projected_usd + spent_usd
        if cumulative <= cap_usd:
            return CapDecision(
                allowed=True,
                projected_usd=projected_usd,
                spent_so_far=spent_usd,
                session_cap_usd=cap_usd,
                reason=(
                    f"blanket grant authorizes {run_type!r}: cumulative ${cumulative:.2f} "
                    f"(projected ${projected_usd:.2f} + blanket spent ${spent_usd:.2f}) within "
                    f"blanket cap ${cap_usd:.2f}"
                ),
                # CR-01: name WHICH blanket authorized so the caller attributes the spend to the
                # exact grant (consume_blanket depletes this index's spent_usd + appends a spend entry).
                blanket_index=index,
            )
    return CapDecision(
        allowed=False,
        projected_usd=projected_usd,
        spent_so_far=0.0,
        session_cap_usd=0.0,
        reason=(
            f"no in-scope, under-cap, unexpired blanket authorizes {run_type!r} for projected "
            f"${projected_usd:.2f} — drops to ask-first"
        ),
    )
