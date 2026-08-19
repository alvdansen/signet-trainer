"""Cost-estimate + guardrail tests (MODL-03). Pure arithmetic — NO modal/CUDA dependency.

``signet_trainer.modal.cost`` must be importable WITHOUT ``modal`` installed (Anti-Pattern 6:
cost.py is Modal-agnostic), so these tests run on Windows/CI with zero Modal spend.
"""

from __future__ import annotations

import pytest

from signet_trainer.modal.cost import estimate_cost, guardrail_check, GuardrailDecision


def test_estimate_cost_basic_arithmetic() -> None:
    # 1.64 $/hr * 4 h = 6.56 (the plan's canonical example).
    assert estimate_cost(hourly_rate_usd=1.64, est_hours=4) == pytest.approx(6.56)


def test_estimate_cost_zero_hours_is_free() -> None:
    assert estimate_cost(hourly_rate_usd=1.64, est_hours=0) == 0.0


def test_estimate_cost_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        estimate_cost(hourly_rate_usd=-1.0, est_hours=4)
    with pytest.raises(ValueError):
        estimate_cost(hourly_rate_usd=1.64, est_hours=-2)


def test_guardrail_allows_under_budget() -> None:
    decision = guardrail_check(
        hourly_rate_usd=1.64, est_hours=2, cost_guardrail_usd=50.0
    )
    assert isinstance(decision, GuardrailDecision)
    assert decision.allowed is True
    assert decision.est_usd == pytest.approx(3.28)
    assert decision.guardrail_usd == pytest.approx(50.0)


def test_guardrail_blocks_over_budget() -> None:
    # 1.64 * 40 = 65.60 > 50.0 -> blocked, never silently launches.
    decision = guardrail_check(
        hourly_rate_usd=1.64, est_hours=40, cost_guardrail_usd=50.0
    )
    assert decision.allowed is False
    assert decision.est_usd == pytest.approx(65.60)
    assert "guardrail" in decision.reason.lower() or "budget" in decision.reason.lower()


def test_guardrail_at_exactly_budget_is_allowed() -> None:
    # est == guardrail is on-budget (<=), not blocked.
    decision = guardrail_check(
        hourly_rate_usd=10.0, est_hours=5, cost_guardrail_usd=50.0
    )
    assert decision.est_usd == pytest.approx(50.0)
    assert decision.allowed is True


def test_guardrail_prices_lives_as_a_multiplier() -> None:
    """issue #45 PR-2 — ``lives`` multiplies the estimate: 4 container lives at $1.64/hr x 2h each
    is $13.12, not $3.28. The single-life default (``lives=1``) stays byte-identical to every
    caller written before PR-2 (test_guardrail_allows_under_budget above still passes unmodified)."""
    decision = guardrail_check(hourly_rate_usd=1.64, est_hours=2, cost_guardrail_usd=50.0, lives=4)
    assert decision.est_usd == pytest.approx(1.64 * 2 * 4)
    assert decision.lives == 4
    assert decision.per_life_hours == pytest.approx(2.0)


def test_guardrail_prices_bounded_hours_not_est_hours_when_given() -> None:
    """``bounded_hours`` (the ``.with_options(timeout=...)`` product the arm actually dispatches
    with) overrides ``est_hours`` as the per-life basis — pricing ``est_hours`` alone would
    under-price a life that runs to the full config-derived timeout before Modal kills it."""
    decision = guardrail_check(
        hourly_rate_usd=1.64, est_hours=2.0, cost_guardrail_usd=50.0, lives=1, bounded_hours=3.0
    )
    assert decision.est_usd == pytest.approx(1.64 * 3.0)
    assert decision.per_life_hours == pytest.approx(3.0)


def test_guardrail_retry_worst_case_reproduces_the_audited_figure() -> None:
    """The exact regression the audit named: an LTX ``train`` dispatch (retries=10 -> 11 lives,
    est_hours=2.0, timeout_margin=1.5) authorizes $54.12 behind what used to be a $3.28 print."""
    decision = guardrail_check(
        hourly_rate_usd=1.64,
        est_hours=2.0,
        cost_guardrail_usd=50.0,
        lives=11,
        bounded_hours=2.0 * 1.5,
    )
    assert decision.est_usd == pytest.approx(54.12)
    assert decision.allowed is False, "the honest worst case exceeds the pre-PR-2 $50 default"
    assert "container life" in decision.reason, (
        "the worst-case basis must be NAMED in the reason — the operator authorizes on this line"
    )


def test_guardrail_lives_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        guardrail_check(hourly_rate_usd=1.64, est_hours=2.0, cost_guardrail_usd=50.0, lives=0)


def test_cost_module_does_not_import_modal() -> None:
    """cost.py must be Modal-agnostic (Anti-Pattern 6): importing it must not pull in ``modal``."""
    import sys

    # cost was imported at module top without modal having to be present; assert the module's
    # own source declares no modal dependency by checking it loaded and modal-free importability.
    import signet_trainer.modal.cost as cost_mod

    src = open(cost_mod.__file__, encoding="utf-8").read()
    assert "import modal" not in src, "cost.py must not import modal (Modal-agnostic, MODL-03)"
