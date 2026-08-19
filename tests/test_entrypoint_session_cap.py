"""Cumulative session-spend cap in the entrypoint gate (D-8-YOLOCAP; issue #37 findings 1 + 6).

The adversarial-audit finding: the single gate all metered runs pass through
(``entrypoint.main`` -> ``_require_approval``) never read or wrote the cumulative session-spend
ledger (``session_cap.py``) — only the per-run ``cost_guardrail_usd`` (a single-dispatch ceiling)
was enforced in code, so a fully exhausted session ledger authorized another guardrail-sized
dispatch, and again, and again. The fix threads ``session_cap`` into the same gate:

  * BEFORE ``_require_approval``: read the ledger, resolve the live cap (config default, overridden
    by SESSION-STATE.json's ``session_cap_usd`` when present — WR-02), and check
    ``projected + spent <= cap``.
  * The BINDING semantic (adversarial audit): going over cap is an ASK-FIRST DOWNGRADE, never a
    second hard refusal. It disables ``--approve`` for THIS dispatch (even when passed) and forces
    the interactive prompt; a PRESENT operator typing ``approved`` still proceeds over cap (the
    human is the fail-safe). A genuinely non-interactive over-cap request (``EOFError``, nobody to
    ask) refuses — that IS the yolo bound working, not a bug in it.
  * AFTER a successful ``.spawn()``: book the dispatch (``append_spend``) so the NEXT dispatch's
    cumulative check sees it — CR-01 airtight accounting, now enforced by code instead of resting on
    an agent remembering to run the equivalent skill-prose step.

IMPORT DISCIPLINE (Anti-Pattern 6): modal-touching imports are done INSIDE the test bodies, never at
module top, mirroring test_entrypoint_gate_behavioral.py — so pytest COLLECTION never pulls
``modal`` into ``sys.modules``.

FILESYSTEM DISCIPLINE: every config below sets ``modal.session_spend_ledger_path`` to a path INSIDE
``tmp_path`` — the schema default is project-relative (``.planning/harness/SESSION-STATE.json``),
and running these tests without the override would write that path onto the real repo tree as a
side effect of the test suite (caught by hand: an untracked ``.planning/`` directory appeared in
the worktree the first time this landed without the override).

``run_dryrun`` stubs below take ``mode=None`` (not a bare ``cfg``): issue #45's later merge threads
``run_dryrun(cfg, mode=mode)`` through ``main()`` for the mode-conditional dry-run refusals, and a
stub that only accepts ``cfg`` raises ``TypeError`` on that extra keyword.
"""

from __future__ import annotations

import builtins
import json

import pytest

# hourly_rate_usd(1.0) * est_hours(2.0) = $2.00 projected -- small and exact, easy to reason about
# against the cap figures each test below chooses.
_PROJECTED_USD = 2.0


class _RecordingCall:
    object_id = "fc-test-stub"

    def get(self, timeout: float | None = None) -> None:
        return None


class _RecordingFn:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _write_config(tmp_path, *, session_cap_usd: float, ledger_path) -> str:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        f"""
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
modal:
  hourly_rate_usd: 1.0
  est_hours: 2.0
  cost_guardrail_usd: 50.0
  session_cap_usd: {session_cap_usd}
  session_spend_ledger_path: "{str(ledger_path).replace(chr(92), '/')}"
""",
        encoding="utf-8",
    )
    return str(cfg_path)


def _raw_main():
    from signet_trainer.modal import entrypoint  # lazy: keep modal out of collection-time sys.modules

    return entrypoint, entrypoint.main.info.raw_f


def _stub_train(monkeypatch):
    from signet_trainer.modal import fns  # lazy import (see module docstring)

    rec = _RecordingFn()
    monkeypatch.setattr(fns, "train", rec)
    return rec


def test_within_cap_dispatches_with_raw_approve_and_books_the_ledger(tmp_path, monkeypatch) -> None:
    """Under cap: --approve authorizes as before, and the dispatch is booked to the ledger."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    ledger_path = tmp_path / "SESSION-STATE.json"

    raw_main(
        config=_write_config(tmp_path, session_cap_usd=10.0, ledger_path=ledger_path),
        approve=True,
        mode="train",
    )

    assert len(rec.calls) == 1, "an under-cap approved run must dispatch exactly once"
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(data["spend"]) == 1
    assert data["spend"][0]["est_usd"] == pytest.approx(_PROJECTED_USD)
    assert data["spend"][0]["run_ref"] == "fc-test-stub", (
        "the ledger entry must be reconcilable against the real dispatch (FunctionCall id)"
    )


def test_over_cap_disables_approve_but_a_present_operator_still_proceeds(tmp_path, monkeypatch) -> None:
    """ASK-FIRST DOWNGRADE: over cap, --approve is disabled but a typed 'approved' still dispatches."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    # cap 1.0 < projected 2.0 -> over cap even against a fresh (spend-0.0) ledger.
    ledger_path = tmp_path / "SESSION-STATE.json"
    prompts: list[str] = []

    def _typed_approval(prompt: str = "") -> str:
        prompts.append(prompt)
        return "approved"

    monkeypatch.setattr(builtins, "input", _typed_approval)

    # --approve IS passed, but the cap must still force the interactive prompt.
    raw_main(
        config=_write_config(tmp_path, session_cap_usd=1.0, ledger_path=ledger_path),
        approve=True,
        mode="train",
    )

    assert prompts, (
        "an over-cap dispatch must reach the interactive prompt even though --approve was passed "
        "(the cap disables --approve for this dispatch, it does not refuse outright)"
    )
    assert len(rec.calls) == 1, "a present operator typing 'approved' over cap must still dispatch"


def test_over_cap_non_interactive_refuses_with_no_dispatch(tmp_path, monkeypatch) -> None:
    """Over cap AND non-interactive (no operator to ask) refuses -- the yolo bound working."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    ledger_path = tmp_path / "SESSION-STATE.json"

    with pytest.raises(SystemExit):
        raw_main(
            config=_write_config(tmp_path, session_cap_usd=1.0, ledger_path=ledger_path),
            approve=True,
            mode="train",
        )

    assert rec.calls == [], "a non-interactive over-cap request must never dispatch"
    assert not ledger_path.exists() or json.loads(ledger_path.read_text(encoding="utf-8")).get(
        "spend", []
    ) == [], "a refused dispatch must never be booked to the ledger"


def test_session_cap_usd_override_in_ledger_file_takes_priority_over_config(tmp_path, monkeypatch) -> None:
    """WR-02: SESSION-STATE.json's session_cap_usd, when present, is the LIVE cap over the config's."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    ledger_path = tmp_path / "SESSION-STATE.json"
    # The setup gate writes session_cap_usd into the ledger; here it OVERRIDES the config's
    # generous 10.0 cap down to 0.5 -- projected $2.00 must now be refused non-interactively.
    ledger_path.write_text(json.dumps({"session_cap_usd": 0.5, "spend": []}), encoding="utf-8")

    with pytest.raises(SystemExit):
        raw_main(
            config=_write_config(tmp_path, session_cap_usd=10.0, ledger_path=ledger_path),
            approve=True,
            mode="train",
        )

    assert rec.calls == [], (
        "the ledger's session_cap_usd override (0.5) must be enforced instead of the config "
        "default (10.0) -- a stale, high config default must not silently widen the live cap"
    )


def test_cumulative_spend_across_two_dispatches_trips_the_cap(tmp_path, monkeypatch) -> None:
    """A SECOND dispatch that alone fits under cap is refused once prior spend is counted (CR-01)."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    ledger_path = tmp_path / "SESSION-STATE.json"
    # cap 3.0: the first $2.00 dispatch fits (cumulative 2.0 <= 3.0); a second $2.00 dispatch would
    # make cumulative 4.0 > 3.0 -- refused non-interactively.
    config = _write_config(tmp_path, session_cap_usd=3.0, ledger_path=ledger_path)

    raw_main(config=config, approve=True, mode="train")
    assert len(rec.calls) == 1, "the first, under-cap dispatch must succeed"

    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(SystemExit):
        raw_main(config=config, approve=True, mode="train")

    assert len(rec.calls) == 1, (
        "the second dispatch must be refused once the FIRST dispatch's spend is counted against "
        "the cap -- the ledger the first call wrote must be read by the second (CR-01)"
    )


def test_corrupt_ledger_refuses_with_an_actionable_message_not_a_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    """A present-but-unparseable ledger must refuse via SystemExit with a [signet-entrypoint]
    message on stderr -- never a bare ValueError traceback out of ``main()`` (medium finding,
    verify2_r.json #1)."""
    entrypoint, raw_main = _raw_main()
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg, mode=None: 0)
    rec = _stub_train(monkeypatch)
    ledger_path = tmp_path / "SESSION-STATE.json"
    ledger_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SystemExit):
        raw_main(
            config=_write_config(tmp_path, session_cap_usd=10.0, ledger_path=ledger_path),
            approve=True,
            mode="train",
        )

    assert rec.calls == [], "a corrupt ledger must fail closed -- never dispatch"
    err = capsys.readouterr().err
    assert "[signet-entrypoint]" in err, "the refusal must print the house actionable-message prefix"


def test_resolve_session_cap_usd_is_tolerant_of_a_missing_or_malformed_ledger(tmp_path) -> None:
    """The cap resolver never raises on a missing/malformed ledger -- falls back to the config default."""
    from signet_trainer.modal import entrypoint

    absent = tmp_path / "absent.json"
    assert entrypoint._resolve_session_cap_usd(absent, 10.0) == 10.0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not valid json", encoding="utf-8")
    assert entrypoint._resolve_session_cap_usd(malformed, 10.0) == 10.0

    not_a_dict = tmp_path / "list.json"
    not_a_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert entrypoint._resolve_session_cap_usd(not_a_dict, 10.0) == 10.0

    override = tmp_path / "override.json"
    override.write_text(json.dumps({"session_cap_usd": 42.0}), encoding="utf-8")
    assert entrypoint._resolve_session_cap_usd(override, 10.0) == 42.0
