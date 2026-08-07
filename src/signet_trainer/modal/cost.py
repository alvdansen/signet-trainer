"""Pre-launch cost estimate + guardrail (MODL-03). PURE arithmetic — Modal-agnostic.

There is no Modal SDK price API, so cost is estimated as ``hourly_rate_usd * est_hours``
(enochiatron precedent) and printed BEFORE any ``.remote()`` dispatch. A ``cost_guardrail_usd``
threshold blocks an over-budget launch so Phase 1 never auto-spends (T-01-MD2).

CRITICAL — Anti-Pattern 6: this module imports NOTHING from ``modal`` (it lives under modal/ for
cohesion but is pure-Python so it's unit-testable on Windows with no Modal install). Defaults are
the RESEARCH.md ``[ASSUMED 1.64]`` A100-80GB $/hr constant — confirm against live Modal pricing at
setup (A3) — and the enochiatron ``50.0`` guardrail.
"""

from __future__ import annotations

from dataclasses import dataclass

# RESEARCH.md A3 — confirm against live Modal pricing at setup (this is an assumed constant).
DEFAULT_HOURLY_RATE_USD = 1.64  # [ASSUMED] A100-80GB $/hr
DEFAULT_COST_GUARDRAIL_USD = 50.0  # enochiatron precedent


def estimate_cost(hourly_rate_usd: float, est_hours: float) -> float:
    """Return the estimated launch cost ``hourly_rate_usd * est_hours`` in USD.

    Pure arithmetic, no SDK call. Raises ``ValueError`` on negative inputs (a negative rate or
    duration is a config error, not a free run).
    """
    if hourly_rate_usd < 0:
        raise ValueError(f"hourly_rate_usd must be >= 0, got {hourly_rate_usd}")
    if est_hours < 0:
        raise ValueError(f"est_hours must be >= 0, got {est_hours}")
    return hourly_rate_usd * est_hours


@dataclass(frozen=True)
class GuardrailDecision:
    """The result of a guardrail check — never auto-launches when ``allowed`` is False."""

    allowed: bool
    est_usd: float
    guardrail_usd: float
    reason: str


def guardrail_check(
    hourly_rate_usd: float,
    est_hours: float,
    cost_guardrail_usd: float = DEFAULT_COST_GUARDRAIL_USD,
) -> GuardrailDecision:
    """Estimate cost and decide whether a launch is within budget (MODL-03).

    ``est_usd <= cost_guardrail_usd`` -> allowed; otherwise blocked. Exactly-at-budget is allowed.
    The caller MUST honor ``allowed=False`` and refuse the ``.remote()`` dispatch.
    """
    est_usd = estimate_cost(hourly_rate_usd, est_hours)
    if est_usd <= cost_guardrail_usd:
        return GuardrailDecision(
            allowed=True,
            est_usd=est_usd,
            guardrail_usd=cost_guardrail_usd,
            reason=f"estimate ${est_usd:.2f} within guardrail ${cost_guardrail_usd:.2f}",
        )
    return GuardrailDecision(
        allowed=False,
        est_usd=est_usd,
        guardrail_usd=cost_guardrail_usd,
        reason=(
            f"estimate ${est_usd:.2f} exceeds cost guardrail ${cost_guardrail_usd:.2f} — "
            "launch BLOCKED (over budget); raise cost_guardrail_usd or lower est_hours"
        ),
    )


def format_cost_line(decision: GuardrailDecision) -> str:
    """Human-readable one-line cost banner for the local_entrypoint to print before launch."""
    status = "ALLOWED" if decision.allowed else "BLOCKED"
    return (
        f"[signet-cost] est ${decision.est_usd:.2f} vs guardrail "
        f"${decision.guardrail_usd:.2f} -> {status}: {decision.reason}"
    )
