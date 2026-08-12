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


# --------------------------------------------------------------------------------------------------
# A RENDER BATCH's SIZE, so the banner above prints WORK next to its estimate.
#
# ``est_hours`` is a DECLARED number on every stage (there is no Modal price API, and no
# steps/second figure has been measured for Qwen-Image-Edit on any card in this program). For a
# training run that is genuinely all there is. For a RENDER it is not: the work is fully determined
# by the config before anything is dispatched — a grid is ``prompt modes x (base + band members) x
# held-out inputs`` images at a known denoise-step count each — so the counts are arithmetic rather
# than a guess, and the per-image budget the declared ``est_hours`` implies is a real quotient of
# the two.
#
# Printing that quotient is the point. "8.0 h" says nothing an operator can check; "8.0 h / 12
# images = 2400 s per image" is immediately either plausible or absurd — and it goes absurd in the
# direction that costs money, because a band that quietly became three times the work still prints
# the same ``est_hours``. This module stays PURE arithmetic: the caller reads the config and prints.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderBatchEstimate:
    """The SIZE of one render batch: how many images, how many denoise steps, at what budget.

    Every field is derived from declared config inputs. ``seconds_per_image_budget`` is the only one
    that mixes in ``est_hours``, and it is a BUDGET — what the declared estimate ALLOWS per image —
    never a prediction of what a render will take.
    """

    columns: int
    inputs: int
    images: int
    steps_per_image: int
    denoise_steps_total: int
    est_hours: float
    seconds_per_image_budget: float | None

    def describe(self) -> str:
        """One clause naming the multiplication, so the counts can be checked by eye."""
        return (
            f"{self.images} image(s) = {self.columns} column(s) x {self.inputs} held-out input(s), "
            f"{self.steps_per_image} denoise step(s) each = {self.denoise_steps_total} total"
        )


def render_batch_estimate(
    *,
    band_members: int,
    prompt_modes: int,
    held_out_inputs: int,
    steps_per_image: int,
    est_hours: float,
    include_base: bool = True,
) -> RenderBatchEstimate:
    """Size a base-vs-band render grid from its declared inputs. Pure arithmetic, no SDK call.

    ``include_base`` adds ONE un-adaptered column group, not one per band member: the base render is
    the convergence reference for the whole grid, so a three-member band is ``2 x (1 + 3) = 8``
    columns and never ``2 x 3 x 2 = 12``. Getting that wrong inflates the printed batch by exactly
    the number of redundant base renders a naive layout would perform — which is also the number of
    redundant renders such a layout would PAY for.

    Raises:
        ValueError: on a negative count, a non-positive step count, or a negative ``est_hours``. A
            batch that cannot be sized is a config error, not a free run.
    """
    for name, value in (
        ("band_members", band_members),
        ("prompt_modes", prompt_modes),
        ("held_out_inputs", held_out_inputs),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if steps_per_image <= 0:
        raise ValueError(f"steps_per_image must be >= 1, got {steps_per_image}")
    if est_hours < 0:
        raise ValueError(f"est_hours must be >= 0, got {est_hours}")

    columns = prompt_modes * (band_members + (1 if include_base else 0))
    images = columns * held_out_inputs
    return RenderBatchEstimate(
        columns=columns,
        inputs=held_out_inputs,
        images=images,
        steps_per_image=steps_per_image,
        denoise_steps_total=images * steps_per_image,
        est_hours=est_hours,
        # None rather than a division: a batch of zero images is a config that renders nothing, and
        # an "inf s/image" budget would read as a healthily generous one.
        seconds_per_image_budget=(est_hours * 3600.0 / images) if images else None,
    )


def format_render_batch_line(estimate: RenderBatchEstimate) -> str:
    """The render-batch banner the local_entrypoint prints beside ``format_cost_line``."""
    if estimate.seconds_per_image_budget is None:
        budget = "no images — nothing to budget"
    else:
        budget = (
            f"the declared est_hours={estimate.est_hours:g} allows "
            f"{estimate.seconds_per_image_budget:.0f} s per image"
        )
    return f"[signet-cost] render batch: {estimate.describe()}; {budget}"
