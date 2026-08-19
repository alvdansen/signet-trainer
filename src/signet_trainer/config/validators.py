"""Fail-fast LTX-2.3 config validators (CONF-02, D-06).

The four verified constraints (RESEARCH.md Q3, from ``ltx_core`` +
``ValidationConfig.validate_video_dims`` + enochiatron's ``validate()``):

    (F - 1) % 8 == 0   (equivalently F % 8 == 1)   — latent temporal scale 8
    H % 32 == 0                                     — latent height scale 32
    W % 32 == 0                                     — latent width scale 32
    batch_size == 1                                 — single-bucket (multi-bucket has
                                                      per-sample shapes; enochiatron enforced this)

Each function raises a clear ``ValueError`` naming the offending value and the rule, and
returns the value unchanged on success so it can be used directly as a Pydantic validator.

NO filesystem touch, NO ``modal`` / ``ltx_core`` import here (Pitfall 1 / Anti-Pattern 6) —
this module must run free on Windows/CI, before any GPU is provisioned.

The ``seq_len`` helper is NOT redefined here: it lives in ``conditioning/strategy.py`` as
the single source of truth and is re-exported below for callers that want it from the
config namespace.

**The same rule now holds for the MiniMax-H3 geometry** (Phase 10, H3-04): the ``17n+5`` frame law,
the canvas/reference sizing, the packed-``seq_len`` arithmetic and the VRAM ceiling all live in
``conditioning/h3_geometry.py`` and are re-exported below — never duplicated here. A source-scan
test (``tests/test_h3_config_validators.py``) forbids inline ``% 17`` and any literal packed-row or
ceiling number in this module, because a duplicated copy drifts silently and its only symptom is an
OOM in a PAID container.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath, PureWindowsPath

# Single home for the seq-len math (Plan 01-01). Re-export, do not duplicate.
from signet_trainer.conditioning.strategy import (
    HEIGHT_SCALE,
    TIME_SCALE,
    WIDTH_SCALE,
    compute_seq_len,
)

# Single home for the MiniMax-H3 arch math (Plan 10-03). Re-export, do not duplicate.
# ``h3_geometry`` is the CONFINED tier: stdlib + dataclasses only, so this import adds ZERO new
# third-party roots to the ``config.validators`` closure (and therefore to ``download_image``'s).
from signet_trainer.conditioning.h3_geometry import (
    H3_A100_80GB_USABLE_GIB,
    H3_CAMPAIGN_ASPECT,
    H3_CAMPAIGN_TARGET_FRAMES,
    H3_CANVAS_MULTIPLE,
    H3_MEASURED_PASSING_PACKED_ROWS,
    H3_MEASURED_PASSING_PEAK_GIB,
    H3_MIB_PER_PACKED_ROW,
    H3_NOMINAL_PROMPT_TOKENS,
    H3_PHASE10_REFERENCE_SHORT_EDGE,
    H3_PHASE10_REFERENCES_PER_SAMPLE,
    H3_PHASE10_TARGET_FRAMES,
    H3_RESIDENT_GIB_RANK64,
    H3_VALID_FRAME_COUNTS,
    H3PackedLayout,
    h3_latent_frames,
    h3_packed_seq_len,
    h3_worst_case_packed_seq_len,
    max_packed_rows_for_budget,
    resolve_canvas_size,
)

# Single home for H3's RENDER frame-count law (D-10-DEF-15) — a DIFFERENT law from the 17n+5
# TRAINING bucket law above (satisfying one says nothing about the other). Re-export, do not
# duplicate (#20 / #4). ``h3_pipeline_source`` is ALSO stdlib-confined by its own module
# docstring ("tests/ must be able to drive this without importing modal.fns ... without
# diffusers or torch"), so this import adds ZERO new third-party roots to the
# ``config.validators`` closure either — same guarantee as the ``h3_geometry`` import above.
from signet_trainer.inference.h3_pipeline_source import (
    assert_h3_frame_count_is_renderable,
    h3_aligned_num_frames,
    h3_renderable_frame_bounds,
)

# Single home for musubi's LAWS (family #4). Re-export, do not duplicate — ``config/sources.py`` is
# pydantic + stdlib only, so this adds no third-party root to the closure.
from signet_trainer.config.sources import (
    MUSUBI_FRAME_MULTIPLE,
    MUSUBI_RESOLUTION_STEP,
    musubi_floor_frames,
)

# Single home for the Qwen-Image-Edit-2511 arch math (family #3). Re-export, do not duplicate.
# ``qwen_edit_geometry`` is the same CONFINED tier as ``h3_geometry`` — stdlib + dataclasses only —
# so this import adds ZERO new third-party roots to the ``config.validators`` closure (and therefore
# to ``modal/app.py::download_image``'s).
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_BLOCK_COUNT,
    QWEN_EDIT_CANVAS_MULTIPLE,
    QWEN_EDIT_CONDITION_IMAGE_SIZE,
    QWEN_EDIT_FRAMES,
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_PATCH_DIM,
    QWEN_EDIT_PIXELS_PER_ROW_EDGE,
    QWEN_EDIT_VAE_IMAGE_SIZE,
    QwenEditPackedLayout,
    qwen_edit_packed_layout,
    qwen_edit_rows_of,
    qwen_edit_vae_latent_shape,
)

__all__ = [
    "validate_frames",
    "validate_height",
    "validate_width",
    "validate_batch_size",
    "validate_training_dims",
    "validate_conditioning_mode",
    "validate_conditioning_items",
    "validate_volume_relative_path",
    "validate_resolution_bucket",
    "validate_resolution_buckets",
    "validate_inpaint_height",
    "validate_inpaint_width",
    "validate_inpaint_dims",
    "validate_inpaint_resolution_bucket",
    "validate_inpaint_resolution_buckets",
    "validate_a2v_lora_targets",
    "A2V_CROSS_MODAL_ATTN_TARGETS",
    "H3_LORA_TARGET_REGEX",
    "H3_VISUAL_CONDITION_PIN",
    "compute_seq_len",
    # --- MiniMax-H3 (Phase 10, H3-04) -------------------------------------------------------
    "validate_h3_frames",
    "validate_h3_resolution_bucket",
    "validate_h3_resolution_buckets",
    "validate_h3_seq_len_budget",
    "validate_h3_reference_budget",
    # Re-exported H3 geometry (defined in conditioning/h3_geometry.py, NOT here).
    "H3_A100_80GB_USABLE_GIB",
    "H3_CAMPAIGN_ASPECT",
    "H3_CAMPAIGN_TARGET_FRAMES",
    "H3_CANVAS_MULTIPLE",
    "H3_MEASURED_PASSING_PACKED_ROWS",
    "H3_MEASURED_PASSING_PEAK_GIB",
    "H3_MIB_PER_PACKED_ROW",
    "H3_NOMINAL_PROMPT_TOKENS",
    "H3_PHASE10_REFERENCE_SHORT_EDGE",
    "H3_PHASE10_REFERENCES_PER_SAMPLE",
    "H3_PHASE10_TARGET_FRAMES",
    "H3_RESIDENT_GIB_RANK64",
    "H3_VALID_FRAME_COUNTS",
    "H3PackedLayout",
    "h3_latent_frames",
    "h3_packed_seq_len",
    "h3_worst_case_packed_seq_len",
    "max_packed_rows_for_budget",
    # --- Wan 2.1 via the musubi-tuner RUNNER (family #4, slice A) ----------------------------
    "validate_wan_frames",
    "validate_wan_height",
    "validate_wan_width",
    "validate_wan_training_dims",
    # Re-exported musubi laws (defined in config/sources.py, NOT here).
    "MUSUBI_FRAME_MULTIPLE",
    "MUSUBI_RESOLUTION_STEP",
    # --- Qwen-Image-Edit-2511 (family #3) ----------------------------------------------------
    "QWEN_EDIT_LORA_LEAVES",
    "QWEN_EDIT_LORA_TARGET_REGEX",
    "validate_qwen_edit_frames",
    "validate_qwen_edit_resolution_bucket",
    "validate_qwen_edit_resolution_buckets",
    "validate_qwen_edit_rank_alpha_lock",
    "validate_qwen_edit_lora_coverage",
    "validate_qwen_edit_row_budget",
    # Re-exported qwen_edit geometry (defined in conditioning/qwen_edit_geometry.py, NOT here).
    "QWEN_EDIT_BLOCK_COUNT",
    "QWEN_EDIT_CANVAS_MULTIPLE",
    "QWEN_EDIT_CONDITION_IMAGE_SIZE",
    "QWEN_EDIT_FRAMES",
    "QWEN_EDIT_LATENT_CHANNELS",
    "QWEN_EDIT_MAX_CONTROL_SLOTS",
    "QWEN_EDIT_PATCH_DIM",
    "QWEN_EDIT_PIXELS_PER_ROW_EDGE",
    "QWEN_EDIT_VAE_IMAGE_SIZE",
    "QwenEditPackedLayout",
    "qwen_edit_packed_layout",
    "qwen_edit_rows_of",
    "qwen_edit_vae_latent_shape",
]

# Reference-control modes signet accepts TODAY (Phase 9). Phase 5 shipped {none, single_frame};
# Phase 6 (REF-02) added ``multi_frame``; Phase 7 (REF-03) added ``ic_lora`` (in-context
# video-to-video conditioning); Phase 9 (GATE-SPEC-inpaint-a2v rev 2) adds ``inpaint`` (masked-region
# per-token spatial denoise mask, prior-project flexible/mask semantics). Kept as a module constant so the
# error message and the check share one source of truth — ``validate_conditioning_mode`` reads it
# and needs no other change.
ALLOWED_CONDITIONING_MODES = (
    "none",
    "single_frame",
    "multi_frame",
    "ic_lora",
    "inpaint",
    "audio_to_video",
)

# Audio-to-video (a2v) cross-modal attention LoRA targets — the CANONICAL fail-fast list (GATE-SPEC
# rev 2 item 3: their ABSENCE is "the #1 silent a2v failure" — without them the adapter is
# structurally blind to the audio branch, so an a2v config that forgets them silently trains a
# video-only LoRA). Kept HERE (the import-light config layer) so ``schema.py`` can enforce their
# presence at config load WITHOUT importing the heavy ``lora/peft.py``. ``lora/peft.py`` keeps a
# byte-identical ``A2V_CROSS_MODAL_LORA_TARGETS`` (the two are asserted equal by
# ``tests/test_a2v_lora_targets.py`` — single source of truth enforced by test, not a cross-import).
# The names mirror the LTX-2.3 dual-stream transformer's cross-modal attention submodule
# (``audio_to_video_attn``); the check_lora_targets probe (lora/peft.py) reports what a real model
# actually exposes at the gated GPU exercise.
A2V_CROSS_MODAL_ATTN_TARGETS = (
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
)

# MiniMax-H3 LoRA target form (Phase 10, H3-02) — a PATH REGEX, not a suffix list. Kept HERE (the
# import-light config layer) for exactly the same reason as A2V_CROSS_MODAL_ATTN_TARGETS above: so
# ``schema.py`` can select it as the family default WITHOUT importing the heavy ``lora/peft.py``
# (which pulls in ``peft``/``torch`` and would widen ``modal/app.py::download_image``'s import
# closure — the BK-01 landmine). ``lora/peft.py`` DERIVES a byte-identical copy from its leaf list;
# ``tests/test_h3_config_schema.py`` asserts the two are equal, so the single source of truth is
# enforced by TEST rather than by a cross-import.
#
# ⚠ It must be a regex and not a suffix list: H3's ``ff.net.0.proj`` / ``ff.net.2`` leaf names are
# BYTE-IDENTICAL to LTX's, so a bare suffix list also matches ``token_refiner.refiner_blocks.*`` —
# 12 modules of text-stream refiner that are not part of the generative stack. PEFT matches a bare
# ``str`` target with ``re.fullmatch`` (``check_target_module_exists`` -> ``match_target_against_key``),
# so the prefix anchors the pattern to the main stack and yields exactly 300 modules / 0 collateral.
H3_LORA_TARGET_REGEX = (
    r"transformer_blocks\.\d+\."
    r"(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)"
)

# Qwen-Image-Edit-2511 LoRA leaves (family #3) — the FOURTEEN LoRA-targetable ``Linear`` leaves per
# transformer block, MEASURED on the live checkpoint (``qwen_image_edit_2511_bf16.safetensors``,
# 40,861,031,560 B, 1934 tensors, 60 blocks, hidden dim 3072). Kept HERE for exactly the reason
# ``H3_LORA_TARGET_REGEX`` and ``A2V_CROSS_MODAL_ATTN_TARGETS`` are: ``schema.py`` must select the
# family default WITHOUT importing the heavy ``lora/peft.py`` (which pulls ``peft``/``torch`` and
# would widen ``modal/app.py::download_image``'s closure — the BK-01 landmine). ``lora/peft.py``
# DERIVES a byte-identical copy from its own leaf lists and a test asserts the two are equal, so the
# single source of truth is enforced by TEST rather than by a cross-import.
#
# ⚠ FOURTEEN, not six. Qwen-Image-Edit is a DUAL-STREAM MMDiT: ``img_*`` and ``txt_*`` are separate
# parameter sets joined only inside attention, so every leaf exists twice. The house LTX invariant
# "attn1 + attn2 + ff.net" (TIER-TAXONOMY.yaml:48, HARD-FENCED D-07) transfers as an INTENT — never
# ship an attention-only adapter, the MLP is where identity lives — but NOT as a literal list: its
# six names match ZERO modules on this checkpoint (Qwen has no ``attn1.``/``attn2.``/``ff.`` path
# anywhere), and a 6-of-14 adapter is a DIFFERENT SHAPE, not a smaller one. See
# ``validate_qwen_edit_lora_coverage`` for what that costs a chained round.
#
# ⚠ A REGEX, not a suffix list. ai-toolkit's real inclusion rule is "prefix ``transformer_blocks``
# AND module type in {Linear, QLinear}" (``lora_special.py:342-434``); PEFT has no type filter, so
# the anchored PATH regex is the only faithful signet expression of it. On THIS checkpoint the 14
# suffixes happen to be collateral-free (the non-block ``Linear`` leaves are ``img_in``, ``txt_in``,
# ``norm_out.linear``, ``proj_out`` and the two ``time_text_embed`` projections — none suffix-match),
# but that is a coincidence of this release, and an anchor costs nothing.
QWEN_EDIT_LORA_LEAVES = (
    # image-stream attention
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    # text-stream attention (the dual-stream half that a ported LTX list silently omits)
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    # feed-forward, both streams — the "ff.net is ~2/3 of block params" fence, in Qwen's spelling
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    # AdaLN modulation projections, both streams
    "img_mod.1",
    "txt_mod.1",
)

#: The anchored path regex the 14 leaves compose to. Built from the tuple above rather than typed
#: out, so a leaf added to the list cannot fail to reach the pattern.
QWEN_EDIT_LORA_TARGET_REGEX = r"transformer_blocks\.\d+\.(" + "|".join(
    leaf.replace(".", r"\.") for leaf in QWEN_EDIT_LORA_LEAVES
) + r")"

# D-10-REFPIN — the level a VISUAL conditioning row is pinned at, as a CONFIG DEFAULT.
#
# ⚠ H3's time convention is INVERTED (``t = 1 - sigma``; ``t = 1`` is CLEAN,
# ``x_t = t*x0 + (1-t)*eps``), so 0.999 reads as "almost clean", not "almost noise". Source:
# ``modular_pipeline.py::keyframe_noise_aug``, applied at ``before_denoise.py:948-955``.
#
# Declared HERE rather than imported from ``train/h3_step.py`` for the same reason as
# ``H3_LORA_TARGET_REGEX`` above: ``config/validators`` sits inside ``modal/app.py``'s
# ``download_image`` import closure (the BK-01 landmine), and a cross-import would widen it for a
# single float. ``train/h3_step.H3_VISUAL_CONDITION_PIN`` is the source of truth and
# ``tests/test_h3_config_schema.py`` asserts the two are equal, so the duplication is enforced by
# TEST rather than by an import — the established precedent, not a new exception.
H3_VISUAL_CONDITION_PIN = 0.999

# Inpaint-mode spatial scales — STRICTER than the video-wide %32 rule, inpaint-mode ONLY.
# [precedent] prior-project HANDOFF-2026-06-30-NIGHT: "÷64 required for inpaint; validated by the smoke"
# (outranks the yaml's ÷32 comment; all precedent-validated inpaint dims are ÷64). The mechanism is
# unconfirmed — adopted defensively so the CPU gate catches it before any GPU dollar (GATE-SPEC
# rev 2 "Top risks"). Frames keep the SAME F % 8 == 1 rule as every other mode (validate_frames).
INPAINT_HEIGHT_SCALE = 64
INPAINT_WIDTH_SCALE = 64


def validate_frames(frames: int) -> int:
    """Reject frame counts where ``(F - 1) % TIME_SCALE != 0`` (LTX needs ``F % 8 == 1``).

    Positivity is checked FIRST (CR-01): ``(1 - 1) % 8 == 0`` and negatives such as ``-7``
    also satisfy the modulo, so without a ``< 1`` floor a zero/negative frame count would pass
    validation and only blow up deep inside ``torch.zeros`` with a negative/degenerate
    ``seq_len`` — defeating the "clear ValueError naming the field, before any GPU" contract.
    """
    if frames < 1:
        raise ValueError(
            f"invalid frame count {frames}: frames must be a positive integer >= 1 "
            f"(LTX-2.3 requires (frames - 1) % {TIME_SCALE} == 0, i.e. frames in 1, 9, 17, ...)."
        )
    if (frames - 1) % TIME_SCALE != 0:
        raise ValueError(
            f"invalid frame count {frames}: LTX-2.3 requires (frames - 1) % {TIME_SCALE} == 0 "
            f"(equivalently frames % {TIME_SCALE} == 1); got remainder {(frames - 1) % TIME_SCALE}. "
            f"Nearest valid counts: {frames - ((frames - 1) % TIME_SCALE)} or "
            f"{frames + (TIME_SCALE - (frames - 1) % TIME_SCALE)}."
        )
    return frames


def validate_height(height: int) -> int:
    """Reject heights where ``H % HEIGHT_SCALE != 0`` (latent height scale 32).

    Positivity is checked FIRST (CR-01): ``0 % 32 == 0`` and ``(-32) % 32 == 0`` both pass the
    divisibility test, so a zero/negative height would silently yield a degenerate (``seq_len==0``)
    or negative batch. The ``<= 0`` guard names the offending field before any tensor is built.
    """
    if height <= 0:
        raise ValueError(
            f"invalid height {height}: height must be a positive multiple of {HEIGHT_SCALE}."
        )
    if height % HEIGHT_SCALE != 0:
        raise ValueError(
            f"invalid height {height}: LTX-2.3 requires height % {HEIGHT_SCALE} == 0; "
            f"got remainder {height % HEIGHT_SCALE}."
        )
    return height


def validate_width(width: int) -> int:
    """Reject widths where ``W % WIDTH_SCALE != 0`` (latent width scale 32).

    Positivity is checked FIRST (CR-01): ``0 % 32 == 0`` and ``(-32) % 32 == 0`` both pass the
    divisibility test, so a zero/negative width would silently yield a degenerate (``seq_len==0``)
    or negative batch. The ``<= 0`` guard names the offending field before any tensor is built.
    """
    if width <= 0:
        raise ValueError(
            f"invalid width {width}: width must be a positive multiple of {WIDTH_SCALE}."
        )
    if width % WIDTH_SCALE != 0:
        raise ValueError(
            f"invalid width {width}: LTX-2.3 requires width % {WIDTH_SCALE} == 0; "
            f"got remainder {width % WIDTH_SCALE}."
        )
    return width


def validate_batch_size(batch_size: int) -> int:
    """Reject any ``batch_size != 1`` (single-bucket; multi-bucket has per-sample shapes)."""
    if batch_size != 1:
        raise ValueError(
            f"invalid batch_size {batch_size}: LTX-2.3 multi-bucket training is single-bucket — "
            f"batch_size must == 1 (each batch is one resolution/length bucket with its own shape)."
        )
    return batch_size


def validate_conditioning_mode(mode: str) -> str:
    """Reject any reference-control ``mode`` not in the Phase-5 allowlist (fail-fast, before GPU).

    Mirrors the positivity-first / clear-``ValueError`` shape of the dim validators: names the
    offending value AND lists the allowed set so a hand-edited YAML gets an actionable message,
    and returns the value unchanged on success so it plugs straight into a Pydantic field_validator.
    Phases 6/7 widen ``ALLOWED_CONDITIONING_MODES`` (multi-frame / IC-LoRA) — this function does not
    change, only the constant it reads.
    """
    if mode not in ALLOWED_CONDITIONING_MODES:
        allowed = ", ".join(repr(m) for m in ALLOWED_CONDITIONING_MODES)
        raise ValueError(
            f"invalid conditioning mode {mode!r}: allowed values are {{{allowed}}} "
            f"(Phase 7 (REF-03) added 'ic_lora' for in-context video-to-video conditioning; "
            f"Phase 9 added 'inpaint' for masked-region spatial inpainting and 'audio_to_video' "
            f"for frozen-audio-driven video generation)."
        )
    return mode


def validate_a2v_lora_targets(targets: Sequence[str]) -> Sequence[str]:
    """Fail-fast: an ``audio_to_video`` config MUST carry the cross-modal attention LoRA targets.

    GATE-SPEC rev 2 item 3 names their absence "the #1 silent a2v failure": PEFT injects a LoRA
    ONLY into modules whose names match ``lora.target_modules`` (suffix match), so an a2v config
    whose target list omits ``audio_to_video_attn.{to_q,to_k,to_v,to_out.0}`` trains a LoRA that is
    STRUCTURALLY BLIND to the audio→video cross-attention — the audio conditioning can never steer
    the video, and the run silently degrades to a video-only overfit. This guard turns that silent
    failure into a config-load ``ValueError`` naming the missing targets, before any GPU spend.

    Suffix-aware (PEFT semantics): a config MAY list the fuller path (e.g.
    ``audio_transformer_blocks.audio_to_video_attn.to_q``); acceptance is by suffix so both the short
    and the qualified forms satisfy the guard. Returns ``targets`` unchanged on success.
    NO filesystem / ltx_core import (Pitfall 1 / Anti-Pattern 6).
    """
    # A bare ``str`` is a PEFT target REGEX (the MiniMax-H3 target form). ``list()`` would explode it
    # into single CHARACTERS and report every a2v target as missing — the same char-explosion class
    # that was live in ``build_lora_config`` until Plan 10-02. Refuse honestly instead of guessing:
    # the a2v cross-modal guard is a suffix check and has no regex semantics to fall back on.
    if isinstance(targets, str):
        raise ValueError(
            f"audio_to_video mode was given a REGEX target form ({targets!r}): the cross-modal "
            f"attention guard is a suffix check over a target LIST and cannot verify a regex. a2v "
            f"is an LTX-family feature — list the targets explicitly, including "
            f"{list(A2V_CROSS_MODAL_ATTN_TARGETS)}."
        )
    have = list(targets)
    missing = [
        req
        for req in A2V_CROSS_MODAL_ATTN_TARGETS
        if not any(t == req or t.endswith("." + req) for t in have)
    ]
    if missing:
        raise ValueError(
            f"audio_to_video mode is missing the cross-modal attention LoRA target(s) {missing}: "
            f"without {list(A2V_CROSS_MODAL_ATTN_TARGETS)} in lora.target_modules the adapter is "
            f"structurally blind to the audio→video cross-attention (GATE-SPEC rev 2: 'the #1 silent "
            f"a2v failure'). Add them to lora.target_modules (short suffix or the fully-qualified "
            f"audio_transformer_blocks.* path both satisfy this)."
        )
    return have


def _item_field(item: object, name: str) -> object:
    """Read ``name`` from a conditioning item, supporting both dict and Pydantic-model items.

    The validator is called from two paths: directly from a test (dicts) and from the
    ``SignetConfig`` cross-field validator (``ConditioningItem`` models). Attribute access covers
    the model path; subscription covers the dict path — so the same fail-fast rule serves both.
    """
    try:
        return item[name]  # mapping / dict path
    except (TypeError, KeyError):
        return getattr(item, name)


def validate_conditioning_items(
    items: Sequence[object], training_dims: tuple[int, int, int]
) -> Sequence[object]:
    """Fail-fast validate a multi-frame ``conditioning_items`` list (REF-02, D-6-FAILFAST).

    Each item carries a ``frame_index`` (which pixel frame the keyframe anchors) and a ``strength``.
    Mirrors ``validate_frames``' error shape — names the offending item index, its value, and the
    nearest valid indices — so a hand-edited multi-frame YAML dies at config load, before any GPU.

    Rules per item (``F = training_dims[2]``):

        frame_index % TIME_SCALE == 0     — keyframe indexes an on-latent-grid pixel frame
        0 <= frame_index <= F - 1         — in range of the clip's pixel frames
        0.0 <= strength <= 1.0            — conditioning strength (denoise_mask = 1 - strength)
        frame_index unique across items   — duplicates would last-win in ``build_denoise_mask``'s
                                            in-order region writes (and stack two
                                            ``VideoConditionByLatentIndex`` at one latent index),
                                            then FAIL the dry-run per-region gate with an opaque
                                            message (WR-06)

    Returns ``items`` unchanged on success so it plugs straight into a Pydantic model_validator.
    Items may be ``ConditioningItem`` models OR dicts — fields are read via attribute-or-key access.
    NO filesystem, NO ltx_core import (keeps the module CPU/Windows-importable).
    """
    frames = training_dims[2]
    seen: dict[int, int] = {}
    for i, item in enumerate(items):
        frame_index = _item_field(item, "frame_index")
        strength = _item_field(item, "strength")
        remainder = frame_index % TIME_SCALE
        if remainder != 0:
            raise ValueError(
                f"invalid conditioning_items[{i}].frame_index {frame_index}: LTX-2.3 requires "
                f"frame_index % {TIME_SCALE} == 0; got remainder {remainder}. "
                f"Nearest valid indices: {frame_index - remainder} or "
                f"{frame_index + (TIME_SCALE - remainder)}."
            )
        if not (0 <= frame_index <= frames - 1):
            raise ValueError(
                f"invalid conditioning_items[{i}].frame_index {frame_index}: must be in range "
                f"0..{frames - 1} (F = {frames})."
            )
        if not (0.0 <= strength <= 1.0):
            raise ValueError(
                f"invalid conditioning_items[{i}].strength {strength}: conditioning strength must "
                f"be in [0.0, 1.0]."
            )
        if frame_index in seen:
            raise ValueError(
                f"invalid conditioning_items[{i}].frame_index {frame_index}: duplicates "
                f"conditioning_items[{seen[frame_index]}] (each keyframe must anchor a distinct "
                f"frame — build_denoise_mask writes regions in order, so with differing strengths "
                f"the last item would silently win)."
            )
        seen[frame_index] = i
    return items


def validate_volume_relative_path(value: str, field: str = "reference image path") -> str:
    """Reject absolute or ``..``-escaping reference-image paths (T-06-02 / WR-09, fail-fast).

    The schema documents these paths as "Volume-relative … resolved Modal-side", where they are
    joined as ``CHECKPOINTS_DIR / value``. ``pathlib`` join semantics make an ABSOLUTE ``value``
    REPLACE the Volume prefix entirely (``/etc/...``, ``C:/...``) and let ``..`` segments escape
    it — so a typo'd path would produce a confusing read from container paths instead of a clear
    config error. Enforce the documented contract at config load, before any container spend.

    Checks BOTH PurePosixPath and PureWindowsPath forms: configs are authored on Windows, so a
    ``C:/`` drive-absolute or a backslash ``..\\`` traversal must be caught even though the
    Modal-side join is POSIX. Returns ``value`` unchanged on success (Pydantic-validator shape).
    NO filesystem touch — pure string/path parsing (Pitfall 1).
    """
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError(
            f"invalid {field} {value!r}: must be Volume-relative (no absolute paths, no '..' "
            f"segments) — it is joined under the checkpoints Volume prefix Modal-side, and "
            f"pathlib join semantics let an absolute path REPLACE the prefix entirely and "
            f"'..' escape it (T-06-02)."
        )
    return value


def validate_resolution_bucket(bucket: str) -> str:
    """Parse + validate ONE ``WxHxF`` resolution-bucket string (WR-04, fail-fast at config load).

    ``cfg.data.resolution_buckets`` are ``WxHxF`` CLI strings the entrypoint's post-approval
    ``_parse_resolution_buckets`` splits into ``(F, H, W)`` tuples. Validating there is too late: a
    malformed string (wrong arity, non-int parts) raises ValueError AFTER the dry-run gate, the cost
    print, and the approval pause — a burned approval. Worse, a plausibly mis-ordered ``25x352x768``
    (FxHxW instead of WxHxF) parses "successfully" into garbage dims and dispatches a metered
    container with no ``%32`` / ``(F-1)%8`` check firing. Apply the SAME width/height/frame rules the
    ``training_dims`` validators enforce, at config load — before any GPU. Returns ``bucket`` unchanged
    on success so it plugs into a Pydantic field_validator. NO filesystem / ltx_core import (Pitfall 1).
    """
    parts = bucket.split("x")
    if len(parts) != 3:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: expected 'WxHxF' — exactly three lowercase-'x'-"
            f"separated integers (e.g. '768x352x25'); got {len(parts)} part(s)."
        )
    try:
        width, height, frames = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: each of W, H, F must be an integer "
            f"('WxHxF', e.g. '768x352x25')."
        ) from None
    # Reuse the exact fail-fast dim rules (W%32, H%32, (F-1)%8, positivity) — same messages, one home.
    validate_width(width)
    validate_height(height)
    validate_frames(frames)
    return bucket


def validate_resolution_buckets(buckets: Sequence[str]) -> Sequence[str]:
    """Validate every ``WxHxF`` bucket in a list (WR-04), naming the offending index. Fail-fast."""
    for i, bucket in enumerate(buckets):
        try:
            validate_resolution_bucket(bucket)
        except ValueError as exc:
            raise ValueError(f"resolution_buckets[{i}]: {exc}") from None
    return buckets


def validate_training_dims(dims: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a ``[W, H, F]`` triple against all three resolution/frame constraints.

    Returns the dims unchanged on success. The single source of truth for both the
    fail-fast schema validators and the dry-run synthetic batch (RESEARCH.md Q3/Open-Q2).
    """
    # NOTE (WR-05): this arity guard is UNREACHABLE via SignetConfig — the field is typed
    # ``tuple[int, int, int]`` so Pydantic rejects any non-3 length with its own ValidationError
    # before this custom validator runs. It is kept deliberately for the standalone-call contract
    # (this function is documented as independently callable, e.g. from tests / Phase-2 hydration).
    if len(dims) != 3:
        raise ValueError(f"training_dims must be exactly [width, height, frames]; got {dims!r}")
    width, height, frames = dims
    validate_width(width)
    validate_height(height)
    validate_frames(frames)
    return (width, height, frames)


# --------------------------------------------------------------------------------------------------
# Inpaint-mode dim validators (Phase 9, GATE-SPEC-inpaint-a2v rev 2) — the ÷64 rule.
# [precedent] prior-project HANDOFF-2026-06-30-NIGHT: "÷64 required for inpaint; validated by the smoke".
# These COMPOSE the base %32 / (F-1)%8 / positivity rules (inpaint is STRICTER, not different), so
# every error message a non-inpaint config would get still fires first, then the %64 layer on top.
# Applied cross-field by SignetConfig when conditioning.mode == "inpaint"; the video-wide %32 rule
# stays the law for every other mode.
# --------------------------------------------------------------------------------------------------


def validate_inpaint_height(height: int) -> int:
    """Reject inpaint-mode heights where ``H % INPAINT_HEIGHT_SCALE != 0`` (÷64 rule).

    Runs the base ``validate_height`` first (positivity + %32), then the stricter %64 layer.
    [precedent] prior-project HANDOFF-2026-06-30-NIGHT — ÷64 required for inpaint; validated by the smoke.
    """
    validate_height(height)
    if height % INPAINT_HEIGHT_SCALE != 0:
        raise ValueError(
            f"invalid height {height} for inpaint mode: precedent-validated inpaint requires "
            f"height % {INPAINT_HEIGHT_SCALE} == 0 (stricter than the video-wide %{HEIGHT_SCALE} "
            f"rule); got remainder {height % INPAINT_HEIGHT_SCALE}. "
            f"[precedent] prior-project HANDOFF-2026-06-30-NIGHT '÷64 required for inpaint'."
        )
    return height


def validate_inpaint_width(width: int) -> int:
    """Reject inpaint-mode widths where ``W % INPAINT_WIDTH_SCALE != 0`` (÷64 rule).

    Runs the base ``validate_width`` first (positivity + %32), then the stricter %64 layer.
    [precedent] prior-project HANDOFF-2026-06-30-NIGHT — ÷64 required for inpaint; validated by the smoke.
    """
    validate_width(width)
    if width % INPAINT_WIDTH_SCALE != 0:
        raise ValueError(
            f"invalid width {width} for inpaint mode: precedent-validated inpaint requires "
            f"width % {INPAINT_WIDTH_SCALE} == 0 (stricter than the video-wide %{WIDTH_SCALE} "
            f"rule); got remainder {width % INPAINT_WIDTH_SCALE}. "
            f"[precedent] prior-project HANDOFF-2026-06-30-NIGHT '÷64 required for inpaint'."
        )
    return width


def validate_inpaint_dims(dims: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a ``[W, H, F]`` triple against the inpaint-mode rules (÷64 H/W + F % 8 == 1).

    Composes ``validate_training_dims`` (the base %32 + frame + positivity rules) with the
    inpaint %64 layer. Frames need NO extra check: the base ``validate_frames`` already enforces
    ``(F - 1) % 8 == 0`` (equivalently ``F % 8 == 1``), which is the inpaint frame rule too.
    Returns the dims unchanged on success (same contract as ``validate_training_dims``).
    """
    validate_training_dims(dims)
    width, height, frames = dims
    validate_inpaint_width(width)
    validate_inpaint_height(height)
    return (width, height, frames)


def validate_inpaint_resolution_bucket(bucket: str) -> str:
    """Validate ONE ``WxHxF`` bucket string against the inpaint-mode rules (÷64 H/W + F % 8 == 1).

    Composes ``validate_resolution_bucket`` (arity/int parse + the base W%32 / H%32 / (F-1)%8
    rules) with the stricter inpaint %64 layer — so a malformed bucket dies with the SAME message
    it gets in every other mode, and a ÷32-clean-but-not-÷64 bucket (e.g. ``736x416x25``) dies
    with the inpaint-specific message. Fail-fast at config load, before any GPU (the GATE-SPEC
    rev-2 CPU gate). Returns ``bucket`` unchanged on success.
    """
    validate_resolution_bucket(bucket)
    width, height, _frames = (int(part) for part in bucket.split("x"))
    validate_inpaint_width(width)
    validate_inpaint_height(height)
    return bucket


def validate_inpaint_resolution_buckets(buckets: Sequence[str]) -> Sequence[str]:
    """Validate EVERY ``WxHxF`` bucket against the inpaint ÷64 rules, naming the offending index.

    The inpaint analogue of ``validate_resolution_buckets`` — applied by ``SignetConfig`` when
    ``conditioning.mode == "inpaint"`` (cross-field: mode lives on conditioning, buckets on data).
    """
    for i, bucket in enumerate(buckets):
        try:
            validate_inpaint_resolution_bucket(bucket)
        except ValueError as exc:
            raise ValueError(f"resolution_buckets[{i}]: {exc}") from None
    return buckets


# --------------------------------------------------------------------------------------------------
# MiniMax-H3 validators (Phase 10, H3-04). Two things break the LTX analogy outright:
#
#   1. The frame law is ``17n+5`` — a DIFFERENT modulus AND a DIFFERENT offset from LTX's
#      ``(F-1) % 8``. Sharing ``validate_frames``' body would be dishonest, so these are new
#      functions, not a parameterization.
#   2. The packed-``seq_len`` VRAM BUDGET refusal is a NEW CLASS of validator with no LTX analog.
#      ``P10-1-MEASURED.md`` section 8.5 on the OOMs it exists to prevent: "This run would have
#      been caught locally."
#
# Every H3 number below arrives from ``conditioning/h3_geometry.py`` (re-exported at the top) or
# from an argument. Nothing is re-derived here — see the module docstring.
# --------------------------------------------------------------------------------------------------


def validate_h3_frames(frames: int) -> int:
    """Reject frame counts that are not of the form ``17n+5`` (MiniMax-H3's frame law).

    A NEW function rather than a parameterization of ``validate_frames``: LTX's rule is
    ``(F - 1) % TIME_SCALE`` and H3's is ``(F - 5) % 17`` — different modulus AND different offset.

    Positivity is checked FIRST with its own message (the CR-01 discipline), because ``-12``
    satisfies H3's modulo outright (``(-12 - 5) % 17 == 0``) and would otherwise pass the law and
    only blow up deep inside a degenerate tensor shape. The LAW itself is then delegated to
    ``h3_latent_frames`` — the single source of truth — whose ``ValueError`` already names the
    offending value, the ``17n+5`` rule, the nearest valid counts, and the fact that LTX's
    ``{25, 49, 81}`` buckets are NOT valid H3 counts (the exact mistake a carried-over campaign
    config makes).

    Returns ``frames`` unchanged on success so it plugs straight into a Pydantic field_validator.
    """
    if frames < 1:
        raise ValueError(
            f"invalid frame count {frames}: frames must be a positive integer >= 1 "
            f"(MiniMax-H3 additionally requires the 17n+5 law, i.e. frames in "
            f"{', '.join(str(f) for f in H3_VALID_FRAME_COUNTS)}, ...)."
        )
    h3_latent_frames(frames)  # raises with the law's own actionable message
    return frames


def validate_h3_resolution_bucket(bucket: str) -> tuple[int, int, int]:
    """Parse + validate ONE ``WxHxF`` H3 resolution-bucket string, returning ``(W, H, F)``.

    Same fail-fast contract as ``validate_resolution_bucket`` (arity, int parse, then the dim
    rules), with ONE change: the frame rule is ``validate_h3_frames`` instead of ``validate_frames``.
    The spatial rules are REUSED verbatim — H3's ``H3_CANVAS_MULTIPLE`` is 32, identical to LTX's
    ``HEIGHT_SCALE`` / ``WIDTH_SCALE``, so re-implementing them would only create drift (a test
    asserts the three stay equal).

    ⚠ Deliberate divergence from the LTX singular: this returns the PARSED ``(W, H, F)`` triple
    rather than the bucket string, because every H3 caller (the budget check, the dry-run banner)
    needs the integers and would otherwise re-split the string. ``validate_h3_resolution_buckets``
    keeps the Pydantic-validator shape and returns its input unchanged.
    """
    parts = bucket.split("x")
    if len(parts) != 3:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: expected 'WxHxF' — exactly three lowercase-'x'-"
            f"separated integers (e.g. '1344x768x22'); got {len(parts)} part(s)."
        )
    try:
        width, height, frames = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: each of W, H, F must be an integer "
            f"('WxHxF', e.g. '1344x768x22')."
        ) from None
    validate_width(width)
    validate_height(height)
    validate_h3_frames(frames)
    return (width, height, frames)


def validate_h3_resolution_buckets(buckets: Sequence[str]) -> Sequence[str]:
    """Validate every ``WxHxF`` H3 bucket in a list, naming the offending index. Fail-fast.

    Returns ``buckets`` UNCHANGED (the Pydantic field_validator shape) — the parsed triples are
    available from the singular ``validate_h3_resolution_bucket``.
    """
    for i, bucket in enumerate(buckets):
        try:
            validate_h3_resolution_bucket(bucket)
        except ValueError as exc:
            raise ValueError(f"resolution_buckets[{i}]: {exc}") from None
    return buckets


def validate_h3_seq_len_budget(
    packed_seq_len: int, max_packed_rows: int, *, label: str = ""
) -> int:
    """Refuse a packed sequence that exceeds the declared GPU ceiling. NO LTX analog exists.

    This is the check that would have caught P10-1b's OOMs locally, for free, before any metered
    dispatch. CPU-pure: no torch, no GPU, no filesystem.

    ``label`` names the offending reference PAIR (e.g. ``C+008``) — callers get it from
    ``h3_worst_case_packed_seq_len``. A refusal that cannot say WHICH pair blew the budget is not
    actionable, because the operator's next move is to change that pair's fidelity.

    Returns ``packed_seq_len`` unchanged on success.
    """
    if packed_seq_len < 1:
        raise ValueError(
            f"invalid packed sequence length {packed_seq_len}: must be a positive integer "
            f"(it is computed by conditioning.h3_geometry.h3_packed_seq_len)."
        )
    if max_packed_rows < 1:
        raise ValueError(
            f"invalid max_packed_rows {max_packed_rows}: must be a positive integer "
            f"(compute it with conditioning.h3_geometry.max_packed_rows_for_budget — never "
            f"hardcode a ceiling)."
        )
    if packed_seq_len > max_packed_rows:
        # COMPUTED, never hardcoded: the no-reference t2v baseline at campaign length. It proves
        # that trimming references cannot rescue campaign length — the constraint is H3-at-80-GiB
        # generally, not the reference pathway.
        t2v_baseline = h3_packed_seq_len(
            H3_CAMPAIGN_TARGET_FRAMES,
            H3_CAMPAIGN_ASPECT,
            [],
            H3_NOMINAL_PROMPT_TOKENS,
            H3_PHASE10_REFERENCE_SHORT_EDGE,
        ).total
        pair = f" for reference pair {label}" if label else ""
        raise ValueError(
            f"packed sequence too long{pair}: {packed_seq_len} rows exceeds the declared ceiling "
            f"of {max_packed_rows} rows by {packed_seq_len - max_packed_rows}. This configuration "
            f"OOMs before the first optimizer step — refused locally, before any metered dispatch. "
            f"The only two levers that actually move are the target frame count and "
            f"reference_image_short_edge. Known-good Phase-10 geometry: "
            f"{H3_PHASE10_TARGET_FRAMES}f target + {H3_PHASE10_REFERENCES_PER_SAMPLE} references "
            f"at short edge {H3_PHASE10_REFERENCE_SHORT_EDGE} on one A100-80GB (the measured "
            f"anchor is {H3_MEASURED_PASSING_PACKED_ROWS} rows at "
            f"{H3_MEASURED_PASSING_PEAK_GIB} GiB peak). NOTE: trimming references does NOT rescue "
            f"campaign length — a {H3_CAMPAIGN_TARGET_FRAMES}f NO-REFERENCE t2v baseline is "
            f"{t2v_baseline} rows and also OOMs, so that is an H200-class GPU decision, deferred "
            f"to P10-4."
        )
    return packed_seq_len


def validate_h3_reference_budget(
    target_frames: int,
    aspect: tuple[float, float],
    character_references: Sequence[object],
    environment_references: Sequence[object],
    prompt_tokens: int,
    ref_short_edge: int,
    gpu_usable_gib: float = H3_A100_80GB_USABLE_GIB,
    resident_gib: float = H3_RESIDENT_GIB_RANK64,
    mib_per_packed_row: float = H3_MIB_PER_PACKED_ROW,
    references_per_sample: int = H3_PHASE10_REFERENCES_PER_SAMPLE,
    n_cond_audio: int = 0,
) -> H3PackedLayout:
    """The CALLER-FACING budget check — prices the WORST pair, then delegates the refusal.

    This is the function ``schema.py`` and ``dryrun/shapes.py`` call. **Neither of them may price a
    nominal pair itself.**

    ⛔ The blocker this exists to prevent: at ``ref_short_edge=1024`` the nominal ``A+B`` pair
    reports 12,362 rows and PASSES config load, while six of the twelve character-by-environment
    pairs are over the ceiling — and the first such segment OOMs a metered run. Every sample carries
    exactly ``references_per_sample`` slots, but the pairing REGIME varies by design (D-10-ASYM:
    roughly 12 of 88 segments swap their second character ref for an environment ref, and that
    asymmetry is DESIRED because it stops the adapter binding to a fixed reference regime). The 15
    pairs differ in row cost by up to 12%, so one nominal pair is not a budget.

    Returns the WORST ``H3PackedLayout`` on success, for the dry-run banner. Every budget number
    arrives as an argument (D-NOHARDCODE); the defaults are the DOCUMENTED measured values from
    ``h3_geometry``, so a different GPU or LoRA rank simply passes different numbers.
    """
    validate_h3_frames(target_frames)  # the law fails fast, before any pricing work
    max_packed_rows = max_packed_rows_for_budget(
        gpu_usable_gib, resident_gib, mib_per_packed_row
    )
    worst_layout, worst_label = h3_worst_case_packed_seq_len(
        target_frames,
        aspect,
        character_references,
        environment_references,
        prompt_tokens,
        ref_short_edge,
        references_per_sample=references_per_sample,
        n_cond_audio=n_cond_audio,
    )
    validate_h3_seq_len_budget(worst_layout.total, max_packed_rows, label=worst_label)
    return worst_layout


# --------------------------------------------------------------------------------------------------
# Qwen-Image-Edit-2511 (family #3). Same doctrine as the H3 block above: the geometry lives in
# ``conditioning/qwen_edit_geometry.py`` and is re-exported, never duplicated. Nothing below
# re-derives a row count, a patch dim or an area budget — each one arrives from that module or from
# an argument.
# --------------------------------------------------------------------------------------------------


def validate_qwen_edit_frames(frames: int) -> int:
    """Pin the frame count to EXACTLY 1 — Qwen-Image-Edit is an IMAGE model.

    A NEW function rather than a parameterization of ``validate_frames`` / ``validate_h3_frames``,
    and deliberately not a modulus at all: there is no law here, there is a single legal value.

    ⚠ This pin is doing REAL work, not decoration. LTX's law is ``(F - 1) % 8 == 0`` and
    ``(1 - 1) % 8 == 0``, so ``1`` is ALREADY legal under the both-families pre-screen in
    ``SignetConfig._check_training_dims`` — which means a ``9`` typo in a qwen_edit config sails
    through that pre-screen with LTX's blessing and would otherwise reach the packer as a
    nine-frame video request against an image model. This is the only place that stops it.

    Returns ``frames`` unchanged on success so it plugs straight into a Pydantic field_validator.
    """
    if frames != QWEN_EDIT_FRAMES:
        raise ValueError(
            f"invalid frame count {frames} for model.family 'qwen_edit': Qwen-Image-Edit-2511 is "
            f"an IMAGE model and F must be EXACTLY {QWEN_EDIT_FRAMES}. Note that LTX's frame law "
            f"ALSO admits 1 ((1 - 1) % {TIME_SCALE} == 0), so the shared pre-screen cannot catch "
            f"this — a video frame count reaching here means this family-exact pin is the only "
            f"guard standing. If you want video, you want a different family."
        )
    return frames


def validate_qwen_edit_resolution_bucket(bucket: str) -> tuple[int, int, int]:
    """Parse + validate ONE ``WxHxF`` qwen_edit bucket string, returning ``(W, H, F)``.

    Same fail-fast contract and same return-the-parsed-triple divergence as
    ``validate_h3_resolution_bucket`` (every qwen_edit caller needs the integers and would otherwise
    re-split the string). Two rule changes from the LTX singular:

      * the frame rule is ``validate_qwen_edit_frames`` — exactly 1, so ``1024x1024x9`` is refused
        here even though it is a perfectly legal LTX bucket;
      * the spatial rules are REUSED verbatim. ``QWEN_EDIT_CANVAS_MULTIPLE`` is 32, identical to
        LTX's ``HEIGHT_SCALE``/``WIDTH_SCALE``, so re-implementing them would only create drift.
        32 is twice the strict 16-pixel packing requirement (``QWEN_EDIT_PIXELS_PER_ROW_EDGE``),
        which is what guarantees an EVEN latent edge and therefore an exact 2x2 patch pack.
    """
    parts = bucket.split("x")
    if len(parts) != 3:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: expected 'WxHxF' — exactly three lowercase-'x'-"
            f"separated integers (e.g. '1024x1024x1'); got {len(parts)} part(s)."
        )
    try:
        width, height, frames = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"invalid resolution bucket {bucket!r}: each of W, H, F must be an integer "
            f"('WxHxF', e.g. '1024x1024x1')."
        ) from None
    validate_width(width)
    validate_height(height)
    validate_qwen_edit_frames(frames)
    return (width, height, frames)


def validate_qwen_edit_resolution_buckets(buckets: Sequence[str]) -> Sequence[str]:
    """Validate every ``WxHxF`` qwen_edit bucket in a list, naming the offending index. Fail-fast.

    Returns ``buckets`` UNCHANGED (the Pydantic field_validator shape) — the parsed triples are
    available from the singular ``validate_qwen_edit_resolution_bucket``.
    """
    for i, bucket in enumerate(buckets):
        try:
            validate_qwen_edit_resolution_bucket(bucket)
        except ValueError as exc:
            raise ValueError(f"resolution_buckets[{i}]: {exc}") from None
    return buckets


def _qwen_edit_target_would_match(resolved: object, key: str) -> bool:
    """Would ``resolved`` (a PEFT ``target_modules`` value) select the module named ``key``?

    Mirrors PEFT's own two-form resolution so a coverage check answers the question PEFT will
    actually be asked at injection time:

      * a bare ``str`` is a REGEX matched with ``re.fullmatch``
        (``check_target_module_exists`` -> ``match_target_against_key``) — NOT ``re.search``, which
        would over-report by accepting ``...attn.to_q.base_layer``;
      * a list/tuple is a SUFFIX list, matched on a DOT boundary, so ``to_q`` does not accidentally
        satisfy a required ``add_q_proj`` and ``mlp.net.2`` does not satisfy ``txt_mlp.net.2``.

    Kept private and local: ``lora/peft.py`` owns the injection-time probe against a real model, and
    this is only the config-load question "does this target form cover the required leaf names?".
    """
    if isinstance(resolved, str):
        return re.fullmatch(resolved, key) is not None
    for target in resolved:
        if key == target or key.endswith("." + target):
            return True
    return False


def validate_qwen_edit_lora_coverage(resolved: object) -> object:
    """Refuse a qwen_edit LoRA target set that does not cover all FOURTEEN required leaves.

    Runs on EVERY qwen_edit config — default or explicit override — because it guards two failure
    modes that point in opposite directions and neither one fails loud on its own:

      1. **The ported LTX list.** Pasting LTX's suffixes into a qwen_edit config matches ZERO
         modules on this checkpoint: Qwen has no ``attn1.`` / ``attn2.`` / ``ff.`` path anywhere.
         ``train/loop.py``'s "no trainable parameters" guard WOULD eventually fire — on a metered
         GPU, after the weights load, not at config load. (This is the mirror image of the H3
         hazard at ``schema.py:164-171``, where the LTX list matched 104 WRONG modules and the
         guard never fired at all.)
      2. **The plausible six-leaf adapter.** Someone reads the house invariant "attn1 + attn2 +
         ff.net" (``harness_data/TIER-TAXONOMY.yaml:48``, HARD-FENCED D-07) and writes the four
         attention leaves plus the two image MLP leaves, dropping the four ``txt_*`` leaves and the
         two ``*_mod.1`` projections. That is not a smaller adapter, it is a DIFFERENT SHAPE —
         dual-stream MMDiT keeps ``img_*`` and ``txt_*`` as separate parameter sets. The
         consequence is specific: such an adapter CANNOT warm-start from a published 14-leaf
         primer, because ``train/loop.py:177-189``'s ``should_warm_start`` is a weights-only load
         at step 0. Depending on ``strict``, that is either a hard key error deep inside a metered
         container or — worse — a silent partial load that leaves eight leaves per block at init
         while the optimizer trains all fourteen.

    The house fence's INTENT (never ship an attention-only adapter; the MLP is where identity is
    encoded) is what carries over to this family. Its LTX leaf LIST does not.

    Returns ``resolved`` unchanged on success so it can be used inline.
    """
    missing = [
        leaf
        for leaf in QWEN_EDIT_LORA_LEAVES
        if not _qwen_edit_target_would_match(resolved, f"transformer_blocks.0.{leaf}")
    ]
    if missing:
        raise ValueError(
            f"qwen_edit LoRA targets do not cover {len(missing)} of the "
            f"{len(QWEN_EDIT_LORA_LEAVES)} required leaves: {missing}. Qwen-Image-Edit is a "
            f"DUAL-STREAM MMDiT — img_* and txt_* are separate parameter sets joined only in "
            f"attention — so the LTX six-leaf house invariant (attn1 + attn2 + ff.net) is NOT the "
            f"same adapter here; it is 6 of 14, and on this checkpoint its literal names match "
            f"ZERO modules. An adapter missing any of these cannot warm-start from a 14-leaf "
            f"published primer (train/loop.py:177-189 is a weights-only load: the missing modules "
            f"stay at init while the optimizer trains them). Remove lora.target_modules to take "
            f"the family default, or cover all fourteen."
        )
    return resolved


def validate_qwen_edit_rank_alpha_lock(lora: object, qwen_edit: object, training: object) -> None:
    """Enforce the chained-edit rank/alpha lock at config load, on CPU, before any dispatch.

    ``QWEN-CHAINED-EDIT-METHOD.md`` locks ``rank == alpha == 42`` across EVERY round of a chain —
    only dataset, name and steps change between rounds. The lock is machine-checked here rather
    than remembered because ``lora_A`` / ``lora_B`` are RANK-SHAPED: change the rank mid-chain and
    no later round can warm-start from an earlier one, and the published primer becomes unloadable.

    ``rank_alpha_lock: null`` is the deliberate exit from the chain. It still requires
    ``rank == alpha`` (PEFT scale ``alpha / rank`` must be 1.0 — the recipe's scaling, independent
    of the chain), and it FORBIDS ``training.init_adapter_path``: warm-starting without declaring
    the chain's rank is precisely the silent partial load the lock exists to prevent.

    All three arguments are duck-typed for the same reason ``qwen_edit_packed_layout``'s are —
    ``config.validators`` must not import ``config.schema`` (the dependency runs the other way).
    """
    lock = getattr(qwen_edit, "rank_alpha_lock")
    rank = int(getattr(lora, "rank"))
    alpha = int(getattr(lora, "alpha"))
    init_adapter_path = getattr(training, "init_adapter_path", None)

    if lock is None:
        if rank != alpha:
            raise ValueError(
                f"qwen_edit with rank_alpha_lock disabled still requires rank == alpha (PEFT "
                f"scale alpha/rank == 1.0, the house recipe): got rank={rank}, alpha={alpha}."
            )
        if init_adapter_path is not None:
            raise ValueError(
                f"training.init_adapter_path is set ({init_adapter_path!r}) while "
                f"qwen_edit.rank_alpha_lock is null: a chained round MUST carry the chain's locked "
                f"rank/alpha, because lora_A/lora_B are rank-shaped and a warm start across a rank "
                f"change is a silent partial load, not an error. Set rank_alpha_lock to the "
                f"chain's value (the house chain: 42)."
            )
        return

    lock = int(lock)
    if rank != lock or alpha != lock:
        raise ValueError(
            f"qwen_edit chain lock violated: lora.rank={rank}, lora.alpha={alpha}, "
            f"qwen_edit.rank_alpha_lock={lock}. QWEN-CHAINED-EDIT-METHOD.md locks rank == alpha == "
            f"{lock} across EVERY round of a chain: only dataset / name / steps change between "
            f"rounds. Changing rank re-shapes lora_A/lora_B so no later round can warm-start from "
            f"an earlier one. Set rank and alpha to {lock}, or set rank_alpha_lock: null to leave "
            f"the chain deliberately (which then forbids init_adapter_path)."
        )


def validate_qwen_edit_row_budget(
    packed_rows: int, max_packed_rows: int, *, label: str = ""
) -> int:
    """Refuse a qwen_edit packed sequence that exceeds a DECLARED row ceiling.

    Structurally the H3 check (``validate_h3_seq_len_budget``) with one deliberate difference, and
    the difference is the honest part: **there is no measured MiB-per-row figure for
    Qwen-Image-Edit on any card in this program**, so there is no
    ``max_packed_rows_for_budget``-style derivation here and no default ceiling. The caller passes
    an integer an operator measured, or does not call this at all.

    ``qwen_edit.max_packed_rows == 0`` therefore means CEILING DISABLED, and the dry-run banner is
    required to print ``ceiling=DISABLED (unmeasured)`` in that state — an operator reading a
    headroom number must never be able to mistake "nobody measured this" for "this is safe".
    Synthesising a plausible ceiling from H3's A100 numbers would be exactly that mistake:
    different model, different row width, different resident weights.

    Returns ``packed_rows`` unchanged on success.
    """
    if packed_rows < 1:
        raise ValueError(
            f"invalid packed row count {packed_rows}: must be a positive integer (it is computed "
            f"by conditioning.qwen_edit_geometry.qwen_edit_packed_layout)."
        )
    if max_packed_rows < 1:
        raise ValueError(
            f"invalid max_packed_rows {max_packed_rows}: must be a positive integer, or 0 to "
            f"disable the ceiling entirely. There is no measured MiB-per-row figure for "
            f"Qwen-Image-Edit in this program, so a ceiling here is an operator's measurement — "
            f"never a derived default."
        )
    if packed_rows > max_packed_rows:
        where = f" for {label}" if label else ""
        raise ValueError(
            f"qwen_edit packed sequence too long{where}: {packed_rows} rows exceeds the declared "
            f"ceiling of {max_packed_rows} rows by {packed_rows - max_packed_rows}. Refused "
            f"locally, before any metered dispatch. The levers, in order of effect: "
            f"qwen_edit.control_slots (EVERY slot is priced at the full control_area_px budget, "
            f"filled or blank, so the control block is normally the majority of the sequence), "
            f"then qwen_edit.control_area_px, then training_dims. NOTE the ceiling itself is a "
            f"DECLARED number, not a measured one: if it was guessed, measure it before trusting "
            f"this refusal."
        )
    return packed_rows


# --------------------------------------------------------------------------------------------------
# Wan 2.1 via the musubi-tuner RUNNER (family #4, slice A) — the THIRD dims branch.
#
# WHY A THIRD BRANCH RATHER THAN A WIDENED LITERAL. The design's own worked geometry, 1280x720x21,
# is rejected by BOTH existing laws on BOTH axes: 720 % 32 == 16 (LTX/H3 spatial step is 32) and 21
# satisfies neither ``(F-1) % 8 == 0`` nor ``(F-5) % 17 == 0``. Adding ``"wan"`` to
# ``ModelConfig.family`` without these functions produces a family that cannot be configured at all
# — the point ``tests/test_multisource_verifier_gaps.py::
# test_the_fixtures_geometry_is_rejected_by_every_family_that_exists`` was written to make.
#
# ⚠ THE WAN LAW IS LOOSER THAN LTX'S ON BOTH AXES, so widening the SignetConfig pre-screen with it
# lets dims through that the LTX arm then rejects — by design, and provably hole-free: the
# family-exact ``else:`` arm re-asserts ``validate_training_dims`` verbatim, so an LTX config sees
# the identical verdict and the identical message it saw before. What changes for a rejected LTX
# config is only WHERE the error is raised (model validator rather than field validator); the four
# existing rejection cases in the suite ([768,512,32], [768,512,22], [770,...], [...,510,...], every
# zero/negative triple) all fail the WAN law too and therefore still die at the field level, exactly
# as they did. Measured, not assumed — see the regression comm on this slice.
# --------------------------------------------------------------------------------------------------


def validate_wan_frames(frames: int) -> int:
    """Reject frame counts that are not ``N*4+1`` (musubi's Wan latent temporal stride).

    A NEW function rather than a parameterization of ``validate_frames``: LTX's modulus is 8 and
    musubi's is 4, and the failure MODE differs too. LTX raises; musubi SILENTLY FLOORS
    ``(f-1)//4*4+1``, so an accepted 30 trains at 29 and the config then describes a run nobody
    performed. signet refuses and names the floored value, the same posture
    ``SourceSpec.target_frames`` takes on the per-source clip lengths.

    Positivity is checked FIRST (the CR-01 discipline): ``-3 % 4 == 1`` in Python, so a negative
    frame count satisfies the modulo outright and would otherwise pass the law.
    """
    if frames < 1:
        raise ValueError(
            f"invalid frame count {frames}: frames must be a positive integer >= 1 "
            f"(musubi/Wan additionally requires the N*{MUSUBI_FRAME_MULTIPLE}+1 law, i.e. frames "
            f"in 1, 5, 9, ..., 21, ..., 45, ...)."
        )
    if frames % MUSUBI_FRAME_MULTIPLE != 1:
        raise ValueError(
            f"invalid frame count {frames} for model.family 'wan': musubi requires "
            f"frames % {MUSUBI_FRAME_MULTIPLE} == 1 (the Wan latent temporal stride) and would "
            f"SILENTLY floor this to {musubi_floor_frames(frames)} rather than raise. Declare "
            f"{musubi_floor_frames(frames)}, or "
            f"{musubi_floor_frames(frames) + MUSUBI_FRAME_MULTIPLE}."
        )
    return frames


def validate_wan_height(height: int) -> int:
    """Reject heights that are not a multiple of musubi's 16-pixel bucketing step.

    ⚠ ASYMMETRY WITH ``musubi_resolution_warnings``, and it is deliberate. A SOURCE's
    ``resolution`` is only WARNED about when it is not a multiple of 16, because refusing it would
    refuse Timothy's own working ``[640, 360]``. ``training_dims`` is a different object: it is
    signet's own statement of the view it prices in the cost line and the dry-run banner, and a
    priced number that is off by a silent crop is a number the operator cannot check against the
    run. Declare the floored value there; the source may keep the un-floored one.
    """
    if height <= 0:
        raise ValueError(
            f"invalid height {height}: height must be a positive multiple of "
            f"{MUSUBI_RESOLUTION_STEP}."
        )
    if height % MUSUBI_RESOLUTION_STEP != 0:
        raise ValueError(
            f"invalid height {height} for model.family 'wan': musubi buckets to a multiple of "
            f"{MUSUBI_RESOLUTION_STEP} and would train this at "
            f"{height - height % MUSUBI_RESOLUTION_STEP}. training_dims is the number signet "
            f"PRICES, so declare {height - height % MUSUBI_RESOLUTION_STEP}."
        )
    return height


def validate_wan_width(width: int) -> int:
    """Reject widths that are not a multiple of musubi's 16-pixel bucketing step.

    Same asymmetry as :func:`validate_wan_height`; see its note.
    """
    if width <= 0:
        raise ValueError(
            f"invalid width {width}: width must be a positive multiple of "
            f"{MUSUBI_RESOLUTION_STEP}."
        )
    if width % MUSUBI_RESOLUTION_STEP != 0:
        raise ValueError(
            f"invalid width {width} for model.family 'wan': musubi buckets to a multiple of "
            f"{MUSUBI_RESOLUTION_STEP} and would train this at "
            f"{width - width % MUSUBI_RESOLUTION_STEP}. training_dims is the number signet "
            f"PRICES, so declare {width - width % MUSUBI_RESOLUTION_STEP}."
        )
    return width


def validate_wan_training_dims(dims: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a ``[W, H, F]`` triple against musubi's Wan geometry. Returns it unchanged.

    Deliberately NOT a coherence check against ``data.sources`` — that check needs the source list
    and therefore lives cross-field on ``SignetConfig``, where the family is also known. This is the
    LAW only, so it can run in the field-level pre-screen beside the LTX and H3 laws.
    """
    if len(dims) != 3:
        raise ValueError(f"training_dims must be exactly [width, height, frames]; got {dims!r}")
    width, height, frames = dims
    validate_wan_width(width)
    validate_wan_height(height)
    validate_wan_frames(frames)
    return (width, height, frames)
