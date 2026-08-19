"""MiniMax-H3 arch math — the SINGLE home for H3 geometry (H3-04, Plan 10-03).

This is H3's analog of ``conditioning/strategy.py``'s "constants + one helper, imported never
duplicated" pattern. ``config/validators.py`` RE-EXPORTS from here; it never redefines the law.

CRITICAL — Anti-Pattern 6, and this module is the STRICTEST tier in the repo:
    stdlib + ``dataclasses`` ONLY. **No ``torch``. No ``modal``. No ``diffusers``. No ``peft``.**

    This is deliberately stricter than ``conditioning/strategy.py``, which DOES import ``torch``
    for tensor type hints. The reason is a closure, not a preference: ``config/validators.py``
    re-exports this module, and ``validators`` sits inside ``modal/app.py``'s ``download_image``
    import closure (``modal/app.py:176-208``). Adding ``torch`` here would widen that closure —
    exactly the BK-01 landmine, whose only symptom is a ``ModuleNotFoundError`` in a PAID container
    after every local gate has already passed. A future edit that "helpfully" adds a torch import
    to get a type hint will be caught by ``tests/test_h3_geometry.py`` — do not silence it.

Where the numbers come from
---------------------------
Every constant and formula below was TRANSCRIBED from diffusers at the pinned SHA
``9f169d98d0bce392a889c3b6524d0d97734dfc0e`` by ``scripts/_h3_probe_modal.py:172-231`` and then
CONFIRMED against live weights on a real A100 (``P10-1-MEASURED.md`` sections 4-5). Source-line
citations are kept inline so a future reader can re-verify without re-deriving:

    canvas          ``modular_pipeline.py::resolve_canvas_size`` — short edge 768, area cap 768*1344
    canvas multiple ``vae_spatial_compression_ratio(16) * patch_size[2](2) = 32``
    reference image ``before_encoder.py:490-492`` — short edge is a CONFIG value
                    (``before_encoder.py:217-221``), upscaling ON, NO area cap, each axis
                    independently ``round()``-ed to a multiple of 32, aspect clamped 1:4..4:1
    vision tokens   ``encoders.py:509`` — ``grid.prod() // merge_size**2`` with Qwen3-VL
                    ``patch_size 16`` / ``merge_size 2`` == ``H/32 * W/32``, i.e. EXACTLY the ref's
                    own latent row count. **Every image reference therefore bills TWICE.**
    frames          ``17n+5`` pixel -> ``5n+2`` latent (chunk-and-drop; NOT LTX's ``(F-1)//r+1``)
    audio           ``encoder_rates [2,4,4,5,5]`` => /800; ``32000/800`` = 40 latents/s/channel,
                    stereo, at 24 fps

Why this is NOT ``compute_seq_len``
-----------------------------------
``conditioning/strategy.py::compute_seq_len(width, height, frames)`` is a pure VIDEO token count
with a three-scalar signature. H3's sequence is a PACKED MULTI-MODAL layout::

    [ text (per-ref labels + vision blocks | prompt LAST) | ref blocks | target audio | target video ]

Two different functions, deliberately. Do NOT add H3 constants to ``strategy.py`` and do NOT try to
reuse ``compute_seq_len`` here — the signature is wrong and the modulus/offset are different.
"""

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "H3_AUDIO_CHANNELS",
    "H3_AUDIO_LATENTS_PER_SECOND",
    "H3_A100_80GB_USABLE_GIB",
    "H3_CAMPAIGN_ASPECT",
    "H3_CAMPAIGN_TARGET_FRAMES",
    "H3_CANVAS_MAX_PIXELS",
    "H3_CANVAS_MULTIPLE",
    "H3_CANVAS_SHORT_EDGE",
    "H3_DEFAULT_MODAL_GPU",
    "H3_FPS",
    "H3_FRAMES_PER_CHUNK",
    "H3_LATENTS_PER_CHUNK",
    "H3_MEASURED_PASSING_PACKED_ROWS",
    "H3_MEASURED_PASSING_PEAK_GIB",
    "H3_MIB_PER_PACKED_ROW",
    "H3_NOMINAL_PROMPT_TOKENS",
    "H3_PHASE10_REFERENCES_PER_SAMPLE",
    "H3_PHASE10_REFERENCE_SHORT_EDGE",
    "H3_PHASE10_TARGET_FRAMES",
    "H3_REFERENCE_ASPECT_MAX",
    "H3_REFERENCE_ASPECT_MIN",
    "H3_REFERENCE_IMAGE_SHORT_EDGE_SPEC",
    "H3_RESIDENT_GIB_RANK64",
    "H3_VALID_FRAME_COUNTS",
    "H3_VISION_TOKENS_EQUAL_LATENT_ROWS",
    "H3PackedLayout",
    "H3Reference",
    "h3_audio_rows",
    "h3_latent_frames",
    "h3_packed_seq_len",
    "h3_reference_pairing_domain",
    "h3_worst_case_packed_seq_len",
    "max_packed_rows_for_budget",
    "reference_image_size",
    "resolve_canvas_size",
    "rows_of",
]

# --------------------------------------------------------------------------------------------------
# Transcribed geometry constants (scripts/_h3_probe_modal.py:186-192; P10-0d section 5.5).
# --------------------------------------------------------------------------------------------------

#: ``vae_spatial_compression_ratio(16) * patch_size[2](2)``. Applied to BOTH the target canvas and
#: reference images, and the Qwen side independently agrees (``patch_size 16 * spatial_merge 2``).
H3_CANVAS_MULTIPLE: int = 32

#: Target-canvas short edge (``modular_pipeline.py::resolve_canvas_size``).
H3_CANVAS_SHORT_EDGE: int = 768

#: Target-canvas AREA CAP. References have no such cap — see ``reference_image_size``.
H3_CANVAS_MAX_PIXELS: int = 768 * 1344

#: ⚠ The TRUE spec short edge for reference IMAGES is 2048, but **Phase 10 runs at 896**, and this
#: is a CONFIG field (``before_encoder.py:217-221``) — this constant is the SPEC DEFAULT ONLY, never
#: a hardcoded runtime value. Callers pass ``ref_short_edge`` explicitly.
#:
#: Why 896 and not 1024/2048: the pairing domain forces it. Every sample carries exactly 2 reference
#: slots, and the worst of the 15 real pairs (``C+008``) is 12,394 rows at 896 — within 0.3% of the
#: 12,362-row configuration MEASURED passing on a real A100 at 76.36 GiB — but 14,026 rows at 1024,
#: where six of the twelve character-by-environment pairs exceed the ceiling. VAE latents cannot be
#: spatially downscaled after the fact, so a later higher-fidelity campaign needs a RE-ENCODE.
#: Budget for it. (``P10-1-MEASURED.md`` section 4 "Operator decision".)
H3_REFERENCE_IMAGE_SHORT_EDGE_SPEC: int = 2048

#: Reference-image aspect clamp, ``1:4 .. 4:1`` (hard ``ValueError`` outside; P10-0d section 5.1).
H3_REFERENCE_ASPECT_MIN: float = 0.25
H3_REFERENCE_ASPECT_MAX: float = 4.0

#: The frame law. ``17n+5`` pixel frames -> ``5n+2`` latent frames (chunk-and-drop).
H3_FRAMES_PER_CHUNK: int = 17
H3_LATENTS_PER_CHUNK: int = 5

#: The frame counts the released 5-15 s contract admits. Used ONLY to make an error message
#: actionable — ``h3_latent_frames`` accepts ANY conforming ``17n+5 >= 5``, it is not capped here.
H3_VALID_FRAME_COUNTS: tuple[int, ...] = (5, 22, 39, 56, 73, 90, 107, 124)

H3_FPS: float = 24.0
H3_AUDIO_LATENTS_PER_SECOND: float = 32000 / 800.0
H3_AUDIO_CHANNELS: int = 2

#: A NAMED FACT, not a toggle. ``encoders.py:509`` computes vision tokens as
#: ``grid.prod() // merge_size**2``; with Qwen3-VL ``patch_size 16`` and ``merge_size 2`` that is
#: exactly ``H/32 * W/32`` — identical to the reference's own VAE latent row count. So a reference
#: contributes its rows to the TEXT stream (as vision tokens) AND to the conditioning video stream.
#: **This double-billing is the single biggest driver of reference cost** and appears in no prior
#: artifact. Flipping this to False would be a lie about the architecture, not a configuration.
H3_VISION_TOKENS_EQUAL_LATENT_ROWS: bool = True

# --------------------------------------------------------------------------------------------------
# MEASURED VRAM facts (P10-1-MEASURED.md section 4). These are OBSERVATIONS on a real A100-80GB with
# gradient checkpointing ON and blocks_to_swap 0 — not derivations, and not tunables. They are the
# documented DEFAULTS a config supplies; every budget function still takes them as ARGUMENTS
# (D-NOHARDCODE), so a different GPU or rank simply passes different numbers.
# --------------------------------------------------------------------------------------------------

H3_A100_80GB_USABLE_GIB: float = 79.25
H3_RESIDENT_GIB_RANK64: float = 62.97  # weights 61.73 + rank-64 LoRA inject
H3_MIB_PER_PACKED_ROW: float = 1.21  # marginal activation cost, gradient checkpointing ON

#: The Modal GPU the three measured numbers above were taken ON — and the DEFAULT booking every
#: TRAIN-tier H3 stage (``h3_preprocess`` / ``h3_train``) dispatches onto (``h3.modal_gpu`` ->
#: ``.with_options(gpu=...)`` at the entrypoint, strictly after approval). Named here, next to the
#: triple, because the two are one fact: a budget that outgrows this card while the booking still
#: says this card is a config the coherence guard must refuse
#: (``config.validators.validate_h3_gpu_budget_coherence``), not a wider ceiling.
#:
#: RULING (bundle PR-5 rework, 2026-08-18): this field governs the TRAIN tier only. ``h3_sample``'s
#: GPU stays on its own pre-existing ``SIGNET_H3_SAMPLE_GPU`` env override (#55/PR#51 house audit,
#: an orthogonal fix for a Qwen3-VL text-encode OOM on a 3-reference render leg) — see
#: ``modal/entrypoint.py``'s h3_sample dispatch arm and ``modal/fns.py``'s ``H3_SAMPLE_GPU`` for the
#: full rationale. Threading this field into the sample dispatch too would make
#: ``.with_options(gpu=...)`` silently override an operator's exported env var with this field's
#: default on every run that has not also escalated ``modal_gpu`` — regressing #55 by composition.
H3_DEFAULT_MODAL_GPU: str = "A100-80GB"

#: The configuration MEASURED passing: 22f target + 2 refs @1024, 76.36 GiB peak, 16.6 s/it.
H3_MEASURED_PASSING_PACKED_ROWS: int = 12362
H3_MEASURED_PASSING_PEAK_GIB: float = 76.36

# --------------------------------------------------------------------------------------------------
# Phase-10 adopted geometry (operator decision 2026-08-05, LOCKED) and the campaign geometry it is
# measured against. Named here so validator MESSAGES can cite them without carrying integer literals.
# --------------------------------------------------------------------------------------------------

H3_PHASE10_TARGET_FRAMES: int = 22
H3_PHASE10_REFERENCE_SHORT_EDGE: int = 896
H3_PHASE10_REFERENCES_PER_SAMPLE: int = 2
H3_CAMPAIGN_TARGET_FRAMES: int = 124
H3_CAMPAIGN_ASPECT: tuple[int, int] = (16, 9)
H3_NOMINAL_PROMPT_TOKENS: int = 96


# --------------------------------------------------------------------------------------------------
# Canvas / reference / frame / audio geometry.
# --------------------------------------------------------------------------------------------------


def _round_to_multiple(value: float) -> int:
    """``max(32, round(value / 32) * 32)`` — the per-axis snap both sides of the arch agree on."""
    m = H3_CANVAS_MULTIPLE
    return max(m, round(value / m) * m)


def _check_aspect(width: float, height: float, what: str) -> None:
    """Enforce the ``1:4 .. 4:1`` clamp (``resolve_reference_image_size``, P10-0d section 5.1)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid {what} {width}x{height}: both axes must be positive.")
    ratio = width / height
    if ratio < H3_REFERENCE_ASPECT_MIN or ratio > H3_REFERENCE_ASPECT_MAX:
        raise ValueError(
            f"invalid {what} {width}x{height}: MiniMax-H3 clamps aspect to 1:4 .. 4:1 "
            f"(ratio must be in [{H3_REFERENCE_ASPECT_MIN}, {H3_REFERENCE_ASPECT_MAX}]); "
            f"got {ratio:.4f}."
        )


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """``(height, width)`` of the TARGET canvas — ``modular_pipeline.py::resolve_canvas_size``.

    Short edge 768, then an AREA CAP of ``768*1344``, then each axis independently snapped to a
    multiple of 32. ``16:9 -> (768, 1344)`` = 1,008 rows per latent frame (P10-1-MEASURED section 5).

    Note the asymmetry with ``reference_image_size``: the target IS area-capped, references are NOT.
    """
    _check_aspect(aspect_width, aspect_height, "target aspect")
    ratio = aspect_width / aspect_height
    if ratio >= 1.0:
        width, height = H3_CANVAS_SHORT_EDGE * ratio, float(H3_CANVAS_SHORT_EDGE)
    else:
        width, height = float(H3_CANVAS_SHORT_EDGE), H3_CANVAS_SHORT_EDGE / ratio
    area = width * height
    if area > H3_CANVAS_MAX_PIXELS:
        scale = (H3_CANVAS_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    return _round_to_multiple(height), _round_to_multiple(width)


def reference_image_size(
    width: int,
    height: int,
    short_edge: int = H3_REFERENCE_IMAGE_SHORT_EDGE_SPEC,
) -> tuple[int, int]:
    """``(height, width)`` a reference IMAGE is encoded at — ``before_encoder.py:490-492``.

    A reference NEVER binds the target geometry: it is prepared at its OWN resolution. Upscaling is
    intentional and there is **no area cap**, so a 4:1 reference at the 2048 spec short edge is
    encoded at ``8192x2048``. Each axis is independently ``round()``-ed to a multiple of 32.

    ``short_edge`` is a CONFIG value (``before_encoder.py:217-221``), NOT a constant — Phase 10 runs
    at 896, the released spec is 2048. See ``H3_REFERENCE_IMAGE_SHORT_EDGE_SPEC``.
    """
    _check_aspect(width, height, "reference image size")
    if short_edge <= 0:
        raise ValueError(f"invalid reference short edge {short_edge}: must be positive.")
    scale = short_edge / min(width, height)
    return _round_to_multiple(height * scale), _round_to_multiple(width * scale)


def rows_of(height: int, width: int) -> int:
    """Packed rows ONE latent frame of an ``(h, w)`` pixel canvas occupies: ``(h//32) * (w//32)``."""
    return (height // H3_CANVAS_MULTIPLE) * (width // H3_CANVAS_MULTIPLE)


def h3_latent_frames(pixel_frames: int) -> int:
    """``17n+5`` pixel frames -> ``5n+2`` latent frames. The frame LAW, single source of truth.

    ⚠ This is NOT LTX's ``(F-1)//8 + 1``: different modulus AND different offset. The H3 video VAE
    is not a clean temporal compressor — the pipeline contract is 17 pixel frames per chunk -> 5
    latent frames per chunk (~3.4x), and every piece of pipeline arithmetic uses the chunking
    contract, not the VAE's nominal ``temporal_downsample_factors`` (P10-0d section 5.4).

    ⚠ **The LTX campaign's ``{25, 49, 81}`` buckets are NOT valid H3 frame counts.** Multi-F
    bucketing must be re-derived, never carried (P10-1-MEASURED section 5).

    The floor is checked FIRST (the ``validate_frames`` CR-01 discipline): ``-12`` satisfies the
    modulo (``(-12 - 5) % 17 == 0``) and would otherwise yield a negative latent-frame count that
    only blows up much later as a degenerate tensor shape.
    """
    if pixel_frames < H3_LATENTS_PER_CHUNK:
        raise ValueError(
            f"invalid frame count {pixel_frames}: MiniMax-H3 frame counts follow the 17n+5 law, so "
            f"the floor is {H3_LATENTS_PER_CHUNK} (n = 0). Valid counts: "
            f"{', '.join(str(f) for f in H3_VALID_FRAME_COUNTS)}, ..."
        )
    remainder = (pixel_frames - H3_LATENTS_PER_CHUNK) % H3_FRAMES_PER_CHUNK
    if remainder != 0:
        n = (pixel_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK
        lower = H3_FRAMES_PER_CHUNK * n + H3_LATENTS_PER_CHUNK
        upper = lower + H3_FRAMES_PER_CHUNK
        raise ValueError(
            f"invalid frame count {pixel_frames}: MiniMax-H3 requires frames of the form 17n+5 "
            f"(i.e. (frames - {H3_LATENTS_PER_CHUNK}) % {H3_FRAMES_PER_CHUNK} == 0); got remainder "
            f"{remainder}. Nearest valid counts: {lower} or {upper}. Valid counts: "
            f"{', '.join(str(f) for f in H3_VALID_FRAME_COUNTS)}, ... "
            f"(NOTE: LTX's 25/49/81 buckets are NOT valid H3 counts — re-derive, do not carry.)"
        )
    return H3_LATENTS_PER_CHUNK * ((pixel_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK) + 2


def h3_audio_rows(pixel_frames: int) -> int:
    """Target audio rows for a clip of ``pixel_frames`` at 24 fps (stereo). A small but real term."""
    seconds = pixel_frames / H3_FPS
    return int(round(seconds * H3_AUDIO_LATENTS_PER_SECOND)) * H3_AUDIO_CHANNELS


# --------------------------------------------------------------------------------------------------
# The packed sequence.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class H3Reference:
    """One reference image slot: its SOURCE pixel size plus a human label for error messages.

    The label is what a budget refusal names (e.g. ``C+008``), so it must survive into the message —
    a refusal that says "some pair is over budget" is not actionable at 3am.
    """

    width: int
    height: int
    label: str


@dataclass(frozen=True)
class H3PackedLayout:
    """The packed-sequence breakdown, in the model's own order.

    Packed order (``P10-1-MEASURED.md`` section 7, ``prompt_position``)::

        [ text (per-ref labels + vision blocks | prompt LAST) | ref blocks | target audio | target video ]

    ``n_vision`` is INFORMATIONAL — it is already a component of ``n_text`` and must NOT be added to
    the total a second time. It is surfaced separately because the each-image-ref-bills-TWICE rule is
    the biggest single driver of reference cost and a reader needs to see it broken out.
    """

    n_text: int
    n_vision: int
    n_cond_video: int
    n_cond_audio: int
    n_target_audio: int
    n_target_video: int
    total: int

    def describe(self) -> str:
        """One-liner for the dry-run banner."""
        return (
            f"packed {self.total} rows = text {self.n_text} (incl {self.n_vision} vision) "
            f"+ ref video {self.n_cond_video} + ref audio {self.n_cond_audio} "
            f"+ target audio {self.n_target_audio} + target video {self.n_target_video}"
        )


def _default_label(index: int, kind: str) -> str:
    """``A``, ``B``, ``C``, ... for characters; ``E1``, ``E2``, ... for environments."""
    if kind == "character" and index < len(string.ascii_uppercase):
        return string.ascii_uppercase[index]
    prefix = "E" if kind == "environment" else "R"
    return f"{prefix}{index + 1}"


def _coerce_reference(item: object, index: int, kind: str) -> H3Reference:
    """Accept ``H3Reference``, ``(w, h)`` or ``(w, h, label)`` — labels default per ``kind``."""
    if isinstance(item, H3Reference):
        return item
    if isinstance(item, Sequence) and not isinstance(item, str | bytes):
        parts = tuple(item)
        if len(parts) == 2:
            return H3Reference(int(parts[0]), int(parts[1]), _default_label(index, kind))
        if len(parts) == 3:
            return H3Reference(int(parts[0]), int(parts[1]), str(parts[2]))
    raise TypeError(
        f"invalid reference {item!r}: expected H3Reference, (width, height) or "
        f"(width, height, label); got {type(item).__name__}."
    )


def _coerce_references(items: Sequence[object], kind: str) -> tuple[H3Reference, ...]:
    return tuple(_coerce_reference(item, i, kind) for i, item in enumerate(items))


def h3_packed_seq_len(
    target_frames: int,
    aspect: tuple[float, float],
    references: Sequence[object],
    prompt_tokens: int,
    ref_short_edge: int,
    n_cond_audio: int = 0,
) -> H3PackedLayout:
    """The packed sequence length for ONE sample — ``scripts/_h3_probe_modal.py:388-408``, exactly.

    ::

        n_text       = prompt_tokens + sum(vision_tokens_i + 2) + 6 * len(references)
        n_cond_video = sum(ref_rows_i)              and   vision_tokens_i == ref_rows_i
        total        = n_text + n_cond_video + n_cond_audio + n_target_audio + n_target_video

    The ``+ 2`` per reference is its ``<|vision_start|>`` / ``<|vision_end|>`` sentinels; the ``+ 6``
    is its text label block. ``vision_tokens_i == ref_rows_i`` is the each-image-ref-bills-TWICE
    rule (``encoders.py:509``) — the biggest single driver of reference cost.

    ``references`` are SOURCE pixel sizes, as ``H3Reference`` / ``(w, h)`` / ``(w, h, label)``;
    they are re-encoded at ``ref_short_edge`` here. ``n_cond_audio`` is 0 for a video-only corpus
    (measured, D-10-AUDIO): no reference soundtracks.
    """
    if prompt_tokens < 0:
        raise ValueError(f"invalid prompt_tokens {prompt_tokens}: must be >= 0.")
    if n_cond_audio < 0:
        raise ValueError(f"invalid n_cond_audio {n_cond_audio}: must be >= 0.")

    refs = _coerce_references(references, "reference")
    canvas_height, canvas_width = resolve_canvas_size(*aspect)
    latent_frames = h3_latent_frames(target_frames)

    n_target_video = latent_frames * rows_of(canvas_height, canvas_width)
    n_target_audio = h3_audio_rows(target_frames)

    ref_rows = [
        rows_of(*reference_image_size(ref.width, ref.height, short_edge=ref_short_edge))
        for ref in refs
    ]
    # encoders.py:509 — identical count, not an approximation. See H3_VISION_TOKENS_EQUAL_LATENT_ROWS.
    vision_tokens = list(ref_rows) if H3_VISION_TOKENS_EQUAL_LATENT_ROWS else [0] * len(ref_rows)

    n_vision = sum(vision_tokens)
    n_text = prompt_tokens + sum(v + 2 for v in vision_tokens) + 6 * len(refs)
    n_cond_video = sum(ref_rows)
    total = n_text + n_cond_video + n_cond_audio + n_target_audio + n_target_video

    return H3PackedLayout(
        n_text=n_text,
        n_vision=n_vision,
        n_cond_video=n_cond_video,
        n_cond_audio=n_cond_audio,
        n_target_audio=n_target_audio,
        n_target_video=n_target_video,
        total=total,
    )


def h3_reference_pairing_domain(
    character_references: Sequence[object],
    environment_references: Sequence[object],
    references_per_sample: int = H3_PHASE10_REFERENCES_PER_SAMPLE,
) -> tuple[tuple[str, tuple[H3Reference, ...]], ...]:
    """Every reference combination the corpus can actually produce, as ``(label, references)``.

    ⛔ **Every sample carries EXACTLY ``references_per_sample`` slots** (operator ruling). A
    non-environment segment gets ``references_per_sample`` rotating character refs; an
    environment-bearing segment SUBSTITUTES the environment ref for its LAST character slot — it is
    **never appended**. A combination with a different slot count reaching a caller is a bug.

    D-10-ASYM is still honored: the reference REGIME varies across the corpus (character+character
    vs character+environment) and that asymmetry is DESIRED — it stops the adapter binding to a
    fixed reference regime. Only the reference COUNT is fixed.

    For the Phase-10 corpus (3 character + 4 environment refs, 2 slots) this is
    ``C(3,2) = 3`` character pairs PLUS ``C(3,1) * 4 = 12`` character-by-environment pairs = **15**.
    """
    characters = _coerce_references(character_references, "character")
    environments = _coerce_references(environment_references, "environment")

    if references_per_sample < 1:
        raise ValueError(
            f"invalid references_per_sample {references_per_sample}: must be >= 1 "
            f"(Phase 10 fixes it at {H3_PHASE10_REFERENCES_PER_SAMPLE})."
        )
    if references_per_sample > len(characters):
        raise ValueError(
            f"invalid references_per_sample {references_per_sample}: only {len(characters)} "
            f"character reference(s) exist, so no all-character combination of that size can be "
            f"drawn. Add character references or lower the slot count."
        )

    domain: list[tuple[str, tuple[H3Reference, ...]]] = []
    for combo in itertools.combinations(characters, references_per_sample):
        domain.append(("+".join(r.label for r in combo), combo))
    # The environment ref SUBSTITUTES for the last character slot -- hence combinations of
    # (references_per_sample - 1) characters, never `references_per_sample` characters + 1 env.
    #
    # #39 finding 1 / step 2: skip this leg ENTIRELY below 2 slots rather than let it run. At
    # references_per_sample == 1, `itertools.combinations(characters, 0)` yields exactly the empty
    # combo, so `pair = (env,)` has length 1 -- which EQUALS references_per_sample and sails past
    # the length invariant below even though no such sample is resolvable: an environment ref needs
    # a character slot to substitute for (h3_ref.py's own rule), and at 1 total slot there is none
    # left. `resolve_reference_slots` / `H3RefStrategy._resolve_slots` refuse every environment-
    # bearing sample below 2 slots by construction, so a length-1 "pair" here would price a layout
    # the runtime can never produce -- the schema-level mirror guard (config/schema.py,
    # H3Config._check_no_reference_fields) refuses this at config load, but this function must stay
    # correct for any OTHER caller that reaches it directly (tests, future code) too.
    if references_per_sample >= 2:
        for combo in itertools.combinations(characters, references_per_sample - 1):
            for env in environments:
                pair = (*combo, env)
                domain.append(("+".join(r.label for r in pair), pair))

    for label, pair in domain:
        if len(pair) != references_per_sample:
            raise ValueError(
                f"internal invariant violated: enumerated pair {label!r} has {len(pair)} "
                f"references, expected exactly {references_per_sample}. The environment reference "
                f"SUBSTITUTES for the last character slot, it is never appended."
            )
    return tuple(domain)


def h3_worst_case_packed_seq_len(
    target_frames: int,
    aspect: tuple[float, float],
    character_references: Sequence[object],
    environment_references: Sequence[object],
    prompt_tokens: int,
    ref_short_edge: int,
    references_per_sample: int = H3_PHASE10_REFERENCES_PER_SAMPLE,
    n_cond_audio: int = 0,
) -> tuple[H3PackedLayout, str]:
    """The budget PRIMITIVE: the MOST EXPENSIVE pair in the real pairing domain, and its label.

    ⛔ **Price the worst case, never one nominal pair.** The 15 pairs differ in row cost by up to
    12%, so a validator fed only the nominal ``A+B`` pair passes at config load and then OOMs on the
    first costlier segment — precisely the failure H3-04 exists to prevent. Concretely, at reference
    short edge 1024 the nominal pair reports 12,362 rows and PASSES, while six of the twelve
    character-by-environment pairs are over the ceiling.

    Returns ``(layout, label)`` so a refusal can NAME the offending pair. Ties are broken toward the
    LAST enumerated pair, which is deterministic and matters here: character refs A (832x1248) and
    C (1024x1536) are both 2:3, so they encode identically at every short edge and every ``A+x``
    pair ties exactly with its ``C+x`` counterpart.
    """
    domain = h3_reference_pairing_domain(
        character_references, environment_references, references_per_sample
    )
    if not domain:
        raise ValueError(
            "empty reference pairing domain: at least one character-reference combination is "
            "required to price the worst case."
        )

    worst_layout: H3PackedLayout | None = None
    worst_label = ""
    for label, pair in domain:
        layout = h3_packed_seq_len(
            target_frames, aspect, pair, prompt_tokens, ref_short_edge, n_cond_audio=n_cond_audio
        )
        if worst_layout is None or layout.total >= worst_layout.total:
            worst_layout, worst_label = layout, label
    assert worst_layout is not None  # domain is non-empty, checked above
    return worst_layout, worst_label


def max_packed_rows_for_budget(
    gpu_usable_gib: float,
    resident_gib: float,
    mib_per_row: float,
) -> int:
    """Max packed rows that fit: ``int((gpu_usable - resident) * 1024 / mib_per_row)``.

    ``P10-1-MEASURED.md`` section 4 reports this as "~13,800 rows" in prose; from the measured
    ``79.25 / 62.97 / 1.21`` this helper computes **13,777**. The helper is the authority —
    ALWAYS compute it, never hardcode either figure, and never hardcode the inputs either (a
    different GPU or LoRA rank simply passes different numbers).
    """
    if mib_per_row <= 0:
        raise ValueError(
            f"invalid mib_per_row {mib_per_row}: the marginal per-row activation cost must be "
            f"positive (measured {H3_MIB_PER_PACKED_ROW} MiB/row with gradient checkpointing ON)."
        )
    activation_budget_gib = gpu_usable_gib - resident_gib
    if activation_budget_gib <= 0:
        raise ValueError(
            f"no activation budget: resident weights {resident_gib} GiB already meet or exceed the "
            f"{gpu_usable_gib} GiB usable on this GPU, so ZERO packed rows fit. This is a GPU-class "
            f"decision (H200/B200), not a geometry one."
        )
    return int(activation_budget_gib * 1024 / mib_per_row)
