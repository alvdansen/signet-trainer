"""Cumulative session-spend cap + ledger + blanket tests (D-8-YOLOCAP / D-8-BLANKET).

Pure arithmetic + stdlib-json — NO modal/CUDA dependency. ``signet_trainer.modal.session_cap`` must
be importable WITHOUT ``modal`` installed (Anti-Pattern 6, mirroring cost.py), so these tests run on
Windows/CI with zero Modal spend. Ledger IO is exercised against ``tmp_path`` only.
"""

from __future__ import annotations

import pytest

from signet_trainer.modal.session_cap import (
    CapDecision,
    DEFAULT_SESSION_CAP_USD,
    append_spend,
    blanket_authorizes,
    consume_blanket,
    read_ledger,
    session_cap_check,
)


# --------------------------------------------------------------------------------------------------
# session_cap_check — cumulative projected + spent vs the session cap (exactly-at-cap allowed)
# --------------------------------------------------------------------------------------------------


def test_session_cap_allows_under_cap() -> None:
    # 4.0 + 3.0 = 7.0 <= 10.0 -> allowed.
    decision = session_cap_check(projected_usd=4.0, spent_so_far=3.0, session_cap_usd=10.0)
    assert isinstance(decision, CapDecision)
    assert decision.allowed is True
    assert decision.projected_usd == pytest.approx(4.0)
    assert decision.spent_so_far == pytest.approx(3.0)
    assert decision.session_cap_usd == pytest.approx(10.0)


def test_session_cap_blocks_over_cap_and_names_the_cap() -> None:
    # 8.0 + 3.0 = 11.0 > 10.0 -> blocked; the reason names the cap and drops to ask-first.
    decision = session_cap_check(projected_usd=8.0, spent_so_far=3.0, session_cap_usd=10.0)
    assert decision.allowed is False
    assert "session cap" in decision.reason
    assert "ask-first" in decision.reason
    assert "10.00" in decision.reason


def test_session_cap_at_exactly_cap_is_allowed() -> None:
    # projected + spent == cap is on-budget (<=), mirroring test_guardrail_at_exactly_budget_is_allowed.
    decision = session_cap_check(projected_usd=7.0, spent_so_far=3.0, session_cap_usd=10.0)
    assert decision.allowed is True


def test_default_session_cap_constant_is_the_house_default() -> None:
    # RESEARCH A3 proposed house default (operator confirms at setup).
    assert DEFAULT_SESSION_CAP_USD == pytest.approx(10.0)


# --------------------------------------------------------------------------------------------------
# read_ledger / append_spend — plain SESSION-STATE.json spend ledger
# --------------------------------------------------------------------------------------------------


def test_read_ledger_missing_file_is_fresh_session() -> None:
    # A missing ledger is a fresh session -> 0.0 spent; never raises.
    missing = "definitely/does/not/exist/SESSION-STATE.json"
    assert read_ledger(missing) == 0.0


def test_append_spend_then_read_ledger_returns_cumulative_sum(tmp_path) -> None:
    ledger = tmp_path / "SESSION-STATE.json"
    append_spend(ledger, est_usd=3.28, run_ref="run-a")
    append_spend(ledger, est_usd=1.64, run_ref="run-b")
    assert read_ledger(ledger) == pytest.approx(4.92)


def test_append_spend_preserves_setup_gate_keys(tmp_path) -> None:
    # The ledger carries setup-gate fields (triage_mode/session_cap_usd/optimization_target/blankets);
    # append touches only the spend list and preserves the rest. The optimization_target assertion is
    # the money-path field-agnosticism proof (Phase 09.2 D-19): append_spend preserves the posture
    # WITHOUT knowing about it — cost/ledger arithmetic is orthogonal to the draft/pro axis.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(
        json.dumps(
            {
                "triage_mode": "strict",
                "session_cap_usd": 10.0,
                "optimization_target": "pro",
                "blankets": [],
                "spend": [],
            }
        ),
        encoding="utf-8",
    )
    append_spend(ledger, est_usd=2.0, run_ref="run-c")
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert data["triage_mode"] == "strict"
    assert data["session_cap_usd"] == 10.0
    assert data["optimization_target"] == "pro"  # D-19: the money path preserves it, field-agnostic
    assert data["blankets"] == []
    assert read_ledger(ledger) == pytest.approx(2.0)


# --------------------------------------------------------------------------------------------------
# blanket_authorizes — D-8-BLANKET consumption (in-scope / out-of-scope / over-cap / expired)
# --------------------------------------------------------------------------------------------------


def _sample_blanket() -> dict:
    return {"scope": ["sample"], "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}


def test_blanket_authorizes_in_scope_under_cap_unexpired() -> None:
    decision = blanket_authorizes([_sample_blanket()], run_type="sample", projected_usd=3.0)
    assert decision.allowed is True


def test_blanket_denies_out_of_scope_run_type() -> None:
    decision = blanket_authorizes([_sample_blanket()], run_type="train", projected_usd=3.0)
    assert decision.allowed is False


def test_blanket_denies_over_cap() -> None:
    # 6.0 > blanket cap 5.0 -> not authorized.
    decision = blanket_authorizes([_sample_blanket()], run_type="sample", projected_usd=6.0)
    assert decision.allowed is False


def test_blanket_denies_expired_past_timestamp() -> None:
    expired = {"scope": ["sample"], "cap_usd": 5.0, "expires": "2020-01-01T00:00:00+00:00", "spent_usd": 0.0}
    decision = blanket_authorizes([expired], run_type="sample", projected_usd=3.0)
    assert decision.allowed is False


def test_blanket_at_exactly_cap_is_authorized() -> None:
    # projected + blanket spent == cap is on-budget (<=), mirroring the session-cap boundary rule.
    blanket = {"scope": ["sample"], "cap_usd": 5.0, "expires": "session", "spent_usd": 2.0}
    decision = blanket_authorizes([blanket], run_type="sample", projected_usd=3.0)
    assert decision.allowed is True


# --------------------------------------------------------------------------------------------------
# CR-01 — airtight blanket accounting: blanket_index attribution + consume_blanket depletion
# --------------------------------------------------------------------------------------------------


def test_blanket_authorizes_names_the_matching_blanket_index() -> None:
    # CR-01: with multiple in-scope blankets, the decision must name WHICH one authorized so the
    # caller attributes the spend unambiguously. The first over-cap blanket is skipped; index 1 wins.
    blankets = [
        {"scope": ["sample"], "cap_usd": 1.0, "expires": "session", "spent_usd": 0.0},  # too small
        {"scope": ["sample"], "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0},  # authorizes
    ]
    decision = blanket_authorizes(blankets, run_type="sample", projected_usd=3.0)
    assert decision.allowed is True
    assert decision.blanket_index == 1


def test_non_authorizing_blanket_decision_has_no_index() -> None:
    decision = blanket_authorizes([_sample_blanket()], run_type="train", projected_usd=3.0)
    assert decision.allowed is False
    assert decision.blanket_index is None


def test_consume_blanket_depletes_matched_blanket_and_is_visible_to_read_ledger(tmp_path) -> None:
    # CR-01 depletion regression: a $5 blanket must NOT authorize unbounded $3 dispatches. After the
    # first consume_blanket the blanket's spent_usd depletes AND the spend is visible to read_ledger;
    # a second dispatch (3+3=6 > 5) is then correctly denied.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(
        json.dumps(
            {
                "triage_mode": "yolo",
                "session_cap_usd": 10.0,
                "optimization_target": "pro",
                "blankets": [{"scope": ["sample"], "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}],
                "spend": [],
            }
        ),
        encoding="utf-8",
    )

    blankets = json.loads(ledger.read_text(encoding="utf-8"))["blankets"]
    first = blanket_authorizes(blankets, run_type="sample", projected_usd=3.0)
    assert first.allowed is True
    consume_blanket(ledger, first.blanket_index, est_usd=3.0, run_ref="run-1")

    # The blanket's spent_usd depleted, and the spend is now visible to the cumulative ledger.
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert data["blankets"][0]["spent_usd"] == pytest.approx(3.0)
    assert read_ledger(ledger) == pytest.approx(3.0)
    # D-19: consume_blanket is field-agnostic too — it preserves the posture without reading it.
    assert data["optimization_target"] == "pro"

    # Re-evaluate with the depleted blanket: 3 (projected) + 3 (spent) = 6 > 5 cap -> DENIED now.
    second = blanket_authorizes(data["blankets"], run_type="sample", projected_usd=3.0)
    assert second.allowed is False


def test_consume_blanket_rejects_out_of_range_index(tmp_path) -> None:
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps({"blankets": [], "spend": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        consume_blanket(ledger, blanket_index=0, est_usd=3.0)


def test_blanket_string_scope_fails_closed() -> None:
    # WR-03: a hand-edited string scope must NOT authorize via Python substring membership. Without
    # the isinstance guard, "train" in "no-train-allowed" is True -> a string scope would authorize.
    string_scope = {"scope": "no-train-allowed", "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}
    decision = blanket_authorizes([string_scope], run_type="train", projected_usd=1.0)
    assert decision.allowed is False


def test_blanket_string_scope_exact_substring_still_fails_closed() -> None:
    # Even the run_type appearing as a literal substring ("sample" in "sample-only") must fail closed.
    string_scope = {"scope": "sample-only", "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}
    decision = blanket_authorizes([string_scope], run_type="sample", projected_usd=1.0)
    assert decision.allowed is False


def test_blanket_non_dict_entry_fails_closed() -> None:
    # A malformed non-dict blanket entry must be skipped, not crash and not authorize.
    decision = blanket_authorizes(["not-a-dict", _sample_blanket()], run_type="sample", projected_usd=1.0)
    assert decision.allowed is True  # the valid entry (index 1) still authorizes
    assert decision.blanket_index == 1


def test_consume_blanket_rejects_negative_amount(tmp_path) -> None:
    # A negative est_usd would silently REDUCE the blanket + cumulative counters (fail open).
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(
        json.dumps({"blankets": [{"scope": ["sample"], "cap_usd": 5.0, "expires": "session", "spent_usd": 0.0}], "spend": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        consume_blanket(ledger, blanket_index=0, est_usd=-3.0)


# --------------------------------------------------------------------------------------------------
# WR-06 — documented fail-closed invariants: bad expiry, corrupt ledger, negative/NaN amounts
# --------------------------------------------------------------------------------------------------


def test_blanket_unparseable_expiry_fails_closed() -> None:
    # 08-01-SUMMARY invariant: an unparseable expiry is treated as EXPIRED (never over-authorize).
    garbage = {"scope": ["sample"], "cap_usd": 5.0, "expires": "not-a-date", "spent_usd": 0.0}
    decision = blanket_authorizes([garbage], run_type="sample", projected_usd=1.0)
    assert decision.allowed is False


def test_blanket_none_expiry_fails_closed() -> None:
    none_exp = {"scope": ["sample"], "cap_usd": 5.0, "expires": None, "spent_usd": 0.0}
    decision = blanket_authorizes([none_exp], run_type="sample", projected_usd=1.0)
    assert decision.allowed is False


def test_session_cap_check_rejects_negative_projected() -> None:
    # A negative projection would falsely shrink cumulative and over-authorize -> fail CLOSED.
    decision = session_cap_check(projected_usd=-5.0, spent_so_far=3.0, session_cap_usd=10.0)
    assert decision.allowed is False


def test_session_cap_check_rejects_nan_projected() -> None:
    decision = session_cap_check(projected_usd=float("nan"), spent_so_far=3.0, session_cap_usd=10.0)
    assert decision.allowed is False


def test_append_spend_rejects_negative_amount(tmp_path) -> None:
    ledger = tmp_path / "SESSION-STATE.json"
    with pytest.raises(ValueError):
        append_spend(ledger, est_usd=-1.0, run_ref="bad")


def test_append_spend_rejects_nan_amount(tmp_path) -> None:
    ledger = tmp_path / "SESSION-STATE.json"
    with pytest.raises(ValueError):
        append_spend(ledger, est_usd=float("nan"), run_ref="bad")


def test_read_ledger_corrupt_json_raises_fail_closed(tmp_path) -> None:
    # A present-but-corrupt ledger must fail CLOSED loudly (distinct from the missing-file 0.0 case).
    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text("{ this is not valid json ]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_ledger(ledger)


def test_read_ledger_non_dict_spend_entry_raises(tmp_path) -> None:
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps({"spend": ["not-an-object"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_ledger(ledger)


def test_read_ledger_clamps_negative_entry_to_zero(tmp_path) -> None:
    # A sneaked-in negative entry must NOT reduce cumulative spend (fail closed for the cap).
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(
        json.dumps({"spend": [{"est_usd": 3.0}, {"est_usd": -2.0}]}), encoding="utf-8"
    )
    assert read_ledger(ledger) == pytest.approx(3.0)  # 3.0 + max(0, -2.0)


def test_read_ledger_non_dict_root_raises_fail_closed(tmp_path) -> None:
    # Issue #24 finding 1: a non-dict root used to return 0.0 (indistinguishable from a fresh,
    # missing-file session) while the strictly narrower malformed-entry case already raised. The
    # container being wrong must fail CLOSED exactly like the entry case, not fail open.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        read_ledger(ledger)


def test_read_ledger_non_list_spend_field_raises_fail_closed(tmp_path) -> None:
    # Issue #24 finding 1: "spend": {} is a plausible hand/agent-edit typo for "spend": [] — it must
    # raise, not silently read back as 0.0 spent.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps({"spend": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="spend is not a list"):
        read_ledger(ledger)


def test_read_ledger_valid_ledger_reads_unchanged(tmp_path) -> None:
    # The fail-closed shape checks must not disturb a well-formed ledger's normal read path.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(
        json.dumps(
            {
                "triage_mode": "yolo",
                "session_cap_usd": 10.0,
                "blankets": [],
                "spend": [{"ts": "2026-01-01T00:00:00+00:00", "est_usd": 1.5, "run_ref": "r1"}],
            }
        ),
        encoding="utf-8",
    )
    assert read_ledger(ledger) == pytest.approx(1.5)


def test_append_spend_refuses_non_dict_root(tmp_path) -> None:
    # Issue #24 finding 2: append_spend used to substitute {} for a non-dict root before overwriting
    # the file — the money-safe direction is to refuse and halt the harness, not reset the counter.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        append_spend(ledger, est_usd=1.0, run_ref="r1")
    # And the original corrupt content must survive untouched — no silent overwrite.
    assert json.loads(ledger.read_text(encoding="utf-8")) == ["not", "a", "dict"]


def test_consume_blanket_refuses_non_dict_root(tmp_path) -> None:
    # Issue #24 finding 2, blanket side: same refuse-not-reset posture as append_spend.
    import json

    ledger = tmp_path / "SESSION-STATE.json"
    ledger.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        consume_blanket(ledger, blanket_index=0, est_usd=1.0, run_ref="r1")
    assert json.loads(ledger.read_text(encoding="utf-8")) == ["not", "a", "dict"]


# --------------------------------------------------------------------------------------------------
# Anti-Pattern 6 invariant — session_cap.py must be Modal-agnostic
# --------------------------------------------------------------------------------------------------


def test_session_cap_module_does_not_import_modal() -> None:
    """session_cap.py must be Modal-agnostic (Anti-Pattern 6): importing it must not pull in ``modal``."""
    import signet_trainer.modal.session_cap as cap_mod

    src = open(cap_mod.__file__, encoding="utf-8").read()
    assert "import modal" not in src, "session_cap.py must not import modal (Modal-agnostic, D-8-YOLOCAP)"
    assert "from modal" not in src, "session_cap.py must not import from modal (Modal-agnostic)"
