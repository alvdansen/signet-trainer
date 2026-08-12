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
from typing import Sequence

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


# --------------------------------------------------------------------------------------------------
# A WAN (musubi) RUN's SIZE — the same "print the WORK beside the estimate" move, for a run whose
# work is only PARTLY knowable, and which therefore has to say so.
#
# The qwen render batch above is fully determined before dispatch, so its line is pure arithmetic.
# A wan run is not, and pretending otherwise is the failure this whole layer exists to avoid. Three
# facts are knowable at $0 and one is not:
#
#   KNOWABLE  clips per MEDIA FILE per source (head/uniform/full/image are arithmetic on the
#             declared extraction; chunk/slide depend on video LENGTH, which lives on a Volume)
#   KNOWABLE  num_repeats, which multiplies them — and on musubi that means RUN LENGTH, not
#             mixture proportion, because the runner is epoch-driven
#   KNOWABLE  max_train_epochs, the recipe's actual stopping rule
#   NOT       how many media files are in each corpus directory. Pitfall 1: no filesystem touch
#             locally, and the directories are container paths on the dataset Volume.
#
# So the line prints PER-MEDIA clip instances and names the one multiplier it cannot supply, rather
# than inventing a corpus size to make a total. It also prints ``training.max_steps`` beside
# ``max_train_epochs`` for the reason SourceSpec.num_repeats records at length: musubi is
# EPOCH-driven and does not read max_steps at all, so an operator reading signet's step budget as
# the run length is wrong by whatever the corpus happens to be.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WanSourceView:
    """One source, reduced to the plain values the clip arithmetic needs.

    A flat record rather than a ``SourceSpec`` so this module keeps its pure-arithmetic character
    and its import closure (the caller does the extraction, exactly as it does for the render batch).
    """

    source_id: str
    kind: str
    extraction: str
    target_frames: tuple[int, ...]
    frame_sample: int
    num_repeats: int
    #: The source's media directory. Carried because clip instances are only summable WITHIN
    #: one corpus — two sources over different directories count different files.
    directory: str


@dataclass(frozen=True)
class WanSourceClips:
    """The per-media clip arithmetic for one source. ``None`` means corpus-dependent, not zero."""

    source_id: str
    kind: str
    extraction: str
    clips_per_media: int | None
    num_repeats: int
    instances_per_media: int | None
    directory: str = ""

    def describe(self) -> str:
        if self.clips_per_media is None:
            return (
                f"{self.source_id}={self.extraction}:? (corpus-dependent) x{self.num_repeats}"
            )
        return (
            f"{self.source_id}={self.extraction}:{self.clips_per_media}"
            f"x{self.num_repeats}={self.instances_per_media}"
        )


@dataclass(frozen=True)
class WanBatchEstimate:
    """What one musubi round will do per media file, and the multiplier nobody can supply locally."""

    per_source: tuple[WanSourceClips, ...]
    instances_per_media_total: int | None
    max_train_epochs: int
    declared_max_steps: int
    est_hours: float

    def describe(self) -> str:
        """Per-source breakdown, then a PER-CORPUS multiplier — never one cross-corpus total.

        ⛔ THE OLD TOTAL WAS IN INCOMMENSURABLE UNITS. It summed instances_per_media across every
        source and reported one number "per media file", but this feature's defining case is an
        IMAGE source and VIDEO sources over DIFFERENT directories — so the addends counted different
        files and the product was wrong in both directions. On the shipped example it printed
        "5 clip instance(s) per media file across 3 source(s)"; with 100 stills and 10 videos the
        true count is 100x1 + 10x2 + 10x2 = 140, while the printed recipe yields 5 x 110 = 550
        (~4x over) or 5 x 10 = 50 (3x under) depending on which count the operator substitutes.

        The banner's stated purpose is to be checkable BY EYE against the corpus the operator
        actually has. Grouping by directory is what makes that possible: one multiplier per corpus,
        each multiplied by the file count of that corpus, which is a number the operator can read
        off their own dataset.
        """
        breakdown = "; ".join(s.describe() for s in self.per_source) or "no sources"
        if not self.per_source:
            return f"{breakdown} -> no corpora"
        by_corpus: dict[str, int | None] = {}
        for source in self.per_source:
            corpus = source.directory or "<undeclared directory>"
            if corpus not in by_corpus:
                by_corpus[corpus] = 0
            if by_corpus[corpus] is None or source.instances_per_media is None:
                by_corpus[corpus] = None  # None propagates WITHIN its corpus, never across
            else:
                by_corpus[corpus] += source.instances_per_media
        parts = []
        for corpus, instances in by_corpus.items():
            if instances is None:
                parts.append(
                    f"{corpus} NOT SIZEABLE (chunk/slide clip count depends on video LENGTH, a "
                    f"Volume-side fact)"
                )
            else:
                parts.append(f"{corpus} x{instances}")
        per_corpus = "; ".join(parts)
        return (
            f"{breakdown} -> PER CORPUS: {per_corpus} — multiply each by the number of media files "
            f"in THAT directory. There is deliberately no single total: clip instances from "
            f"different directories count different files and cannot be added."
        )


def wan_batch_estimate(
    *,
    sources: Sequence[WanSourceView],
    max_train_epochs: int,
    declared_max_steps: int,
    est_hours: float,
) -> WanBatchEstimate:
    """Size a musubi round from its declared sources. Pure arithmetic, no SDK call, no filesystem.

    Raises:
        ValueError: on a non-positive epoch count or a negative ``est_hours`` — a round that runs
            zero epochs is a config error, not a free run.
    """
    from signet_trainer.runners.wan_musubi import wan_clips_per_media  # noqa: PLC0415

    if max_train_epochs <= 0:
        raise ValueError(f"max_train_epochs must be >= 1, got {max_train_epochs}")
    if est_hours < 0:
        raise ValueError(f"est_hours must be >= 0, got {est_hours}")

    per_source: list[WanSourceClips] = []
    for view in sources:
        clips = wan_clips_per_media(view.extraction, view.target_frames, view.frame_sample)
        per_source.append(
            WanSourceClips(
                source_id=view.source_id,
                kind=view.kind,
                extraction=view.extraction,
                clips_per_media=clips,
                num_repeats=view.num_repeats,
                instances_per_media=None if clips is None else clips * view.num_repeats,
                directory=view.directory,
            )
        )
    # None propagates: a total that silently drops an unknown source would UNDER-count, and this
    # number's only job is to be checkable by eye against the corpus the operator actually has.
    total: int | None = 0
    for source in per_source:
        if source.instances_per_media is None:
            total = None
            break
        total += source.instances_per_media
    return WanBatchEstimate(
        per_source=tuple(per_source),
        instances_per_media_total=total,
        max_train_epochs=max_train_epochs,
        declared_max_steps=declared_max_steps,
        est_hours=est_hours,
    )


def format_wan_batch_line(estimate: WanBatchEstimate) -> str:
    """The wan batch banner, printed BESIDE ``format_cost_line`` and never in place of it.

    It adds a line, never a decision: the guardrail arithmetic and its basis are byte-identical to
    every other mode, because turning per-media clip counts into hours would need a seconds-per-step
    figure nobody in this program has measured for Wan on any card.
    """
    return (
        f"[signet-cost] wan batch: {estimate.describe()}; musubi is EPOCH-driven "
        f"(--max_train_epochs {estimate.max_train_epochs}), so the run length is that x the clip "
        f"instances above x the media-file count on the dataset Volume — a number this gate cannot "
        f"read (Pitfall 1). training.max_steps={estimate.declared_max_steps} is signet's accounting "
        f"basis and is NOT passed to the runner. The guardrail above is priced on the DECLARED "
        f"cfg.modal.est_hours={estimate.est_hours:g}, which is not a measurement."
    )
