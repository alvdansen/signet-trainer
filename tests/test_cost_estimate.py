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


def test_cost_module_does_not_import_modal() -> None:
    """cost.py must be Modal-agnostic (Anti-Pattern 6): importing it must not pull in ``modal``."""
    import sys

    # cost was imported at module top without modal having to be present; assert the module's
    # own source declares no modal dependency by checking it loaded and modal-free importability.
    import signet_trainer.modal.cost as cost_mod

    src = open(cost_mod.__file__, encoding="utf-8").read()
    assert "import modal" not in src, "cost.py must not import modal (Modal-agnostic, MODL-03)"
