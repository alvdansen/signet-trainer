"""prep.qwen_edit_encode — the signet-NATIVE Qwen-Image-Edit-2511 pre-encode (family #3, slice 2).

⛔ **There is no canonical Qwen-Image-Edit pre-encode either.** ai-toolkit's
``extensions_built_in/diffusion_models/qwen_image/qwen_image_edit_plus.py`` re-encodes its control
images **inside every training step** (``:280``, under the ``torch.no_grad`` opened at ``:212``) and
caches nothing; diffusers' ``pipeline_qwenimage_edit_plus.py`` is an inference pipeline with no
target image at all. So this module is signet-native by necessity, the same way ``prep/h3_encode``
was — the enochiatron "never write a custom encoder" landmine does not apply because there is
nothing canonical to prefer. Every constant below is TRANSCRIBED from one of those two files with
its citation kept inline.

Import tier — **Modal-side EXCEPTION** (10-PATTERNS "Anti-Pattern 6", tier 3)
---------------------------------------------------------------------------
This module MAY import the heavy backends (Pillow / numpy / diffusers / transformers), but **ONLY
function-local**. ``torch`` is the sole module-scope third-party import; the intra-repo imports are
all from the confined tier (``conditioning/qwen_edit_geometry`` is stdlib + ``dataclasses`` only,
``conditioning/qwen_edit`` is torch + stdlib, ``prep/h3_grad_contract`` is torch only,
``prep/h3_vae_contract`` is torch + intra-repo). The reason is concrete: reading the parity
CONSTANTS and running the CPU contract suite on Windows must never require a backend to be
installed. ``modal`` is never imported here at all — the thin ``@app.function`` wrapper belongs in
``modal/fns.py``, so this logic stays source-scannable on a CPU box with zero spend.

The traps this module exists to get right
-----------------------------------------
Each degrades **silently** — valid shapes, plausible tensors, off-distribution conditioning:

1. **A control image goes down BOTH channels, at TWO DIFFERENT sizes.**
   ``qwen_image_edit_plus.py:66`` sets ``encode_control_in_text_embeddings=True``, so the same
   source file is resized to ``CONDITION_IMAGE_SIZE`` (384², the Qwen2.5-VL channel,
   ``qwen_image_edit_plus.py:179-187``) **and** to ``VAE_IMAGE_SIZE`` (1024², the VAE channel,
   ``:217`` / ``:265-275``). Conflating the budgets mis-sizes one stream and nothing raises.
2. **The VAE takes 5-D ``[B, C, F, H, W]``.** ``AutoencoderKLQwenImage._encode`` unpacks exactly
   five (``autoencoder_kl_qwenimage.py:783``). The F axis is inserted at ``qwen_image.py:430``
   and squeezed again at ``:446``; this module keeps it at rest and adds/removes the BATCH axis at
   one chokepoint (the D-10-DEF-9 idiom).
3. **``[-1, 1]`` pixel normalization**, NOT H3's ImageNet-over-``[0,1]`` and NOT a bare ``[0,1]``.
   ``qwen_image_edit_plus.py:278`` (``control_img * 2 - 1``) and ``image_processor.py:225``
   (``2.0 * images - 1.0``) agree.
4. **Per-channel latent standardisation, and this VAE has NO ``scaling_factor``.**
   ``(x - latents_mean) / latents_std`` over 16-vectors read off ``vae.config``
   (``autoencoder_kl_qwenimage.py:687-688``; applied at ``qwen_image.py:434-444`` and
   ``pipeline_qwenimage_edit_plus.py:420-430``). A ``scaling_factor`` fallback would silently
   scale by 1.0.
5. **The text encoder's VISION half is load-bearing.** ``_get_qwen_prompt_embeds`` passes
   ``pixel_values`` and ``image_grid_thw`` (``pipeline_qwenimage_edit_plus.py:266-267``). A plain
   text LLM in that slot produces ``mat1 and mat2 shapes cannot be multiplied (5376x1280 and
   3840x1280)`` — a failure this house has already hit and fixed.
6. **The 64-token prefix drop** (``pipeline_qwenimage_edit_plus.py:217, 272``). Skipping it leaves
   the system block inside the conditioning; over-dropping empties a short caption.

⚠ **What is deliberately NOT copied from ``prep/h3_encode``**: the ``float16`` round-trip. That is a
MiniMax-H3 parity step (``encoders.py:102-136``); nothing in the Qwen path rounds, and adding it
here would be an invented ~11-bit precision loss.

Output contract (a divergence here is silent downstream corruption)
-------------------------------------------------------------------
::

    <root>/qwen_edit_latents/<rel>.pt         {"latents": [16, 1, H_lat, W_lat], "num_frames": 1,
                                               "height": H_lat, "width": W_lat, "source_wh",
                                               "encoded_wh", "latent_rows", "stem"}
    <root>/qwen_edit_control_latents/<rel>.pt {"controls": [ per-slot dicts ], "num_controls",
                                               "stem", + the slot-0 allowlist ALIAS}
    <root>/qwen_edit_conditions/<rel>.pt      {"prompt_embeds": [1, L, 3584],
                                               "prompt_embeds_mask": [1, L] int64, version,
                                               cache_key, ...}   — NOT a latent source

⚠ ``height`` / ``width`` are the **LATENT** grid dims, exactly as in ``h3_latents`` — that is what
``data/precomputed.py::_normalize_video_latents`` consumes for the legacy patchified
``(f h w) c -> c f h w`` fixup, and ``conditioning/qwen_edit_geometry.qwen_edit_vae_latent_shape``
(``:226-230``) names that path by hand. The PIXEL size travels separately as ``source_wh`` /
``encoded_wh``, the ``h3_reference_latents`` precedent (``prep/h3_encode.py:795-796``), because the
two differ by the VAE's 8x factor and pricing an image off a latent grid under-counts it ~64x.

⚠ ``qwen_edit_conditions`` carries Qwen2.5-VL text embeddings and NO ``"latents"`` key, so it must
never be added to ``data/precomputed.py::_VIDEO_LATENT_SOURCE_DIRS`` — the same trap as
``h3_conditions``.

⛔ **OPEN WIRING GAP, named here because this module is where it becomes load-bearing.** Neither
``qwen_edit_latents`` nor ``qwen_edit_control_latents`` is in ``_VIDEO_LATENT_SOURCE_DIRS`` today
(``data/precomputed.py:94-96`` lists only the LTX and H3 names). The payloads written here satisfy
that allowlist's contract by construction and are ASSERTED against it below
(``_assert_latent_payload``), so adding the two names is a one-line data-layer change — but it is
not this module's to make, and until it lands a cached ``[C, F, H, W]`` latent simply passes through
un-normalized (correct today, because nothing writes the legacy 2-D form). See
:func:`qwen_edit_allowlist_gap`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# The SLOT contract is owned by the module whose strategy consumes it, and imported rather than
# restated — the ``prep/h3_encode.py:91-95`` doctrine for ``H3_REFERENCE_KINDS``: the writer can
# never accept a slot shape the reader will reject. ``conditioning/qwen_edit`` is torch + stdlib +
# confined siblings, so this import pulls no backend and keeps the "constants readable on a CPU box"
# property the whole tier depends on.
from signet_trainer.conditioning.qwen_edit import (
    QWEN_EDIT_BLANK_FILLS,
    QWEN_EDIT_DATA_SOURCES,
    QwenControlSlot,
    resolve_control_slots,
)

# Single home for the qwen_edit arch math. RE-EXPORT, never duplicate (``qwen_edit_geometry``'s own
# header, and the reason ``h3_ref.py:64`` imports ``H3PackedLayout`` rather than restating it).
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_CONDITION_IMAGE_SIZE,
    QWEN_EDIT_FRAMES,
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_PATCH,
    QWEN_EDIT_VAE_IMAGE_SIZE,
    qwen_edit_area_budget_size,
    qwen_edit_rows_of,
)

# ⛔ THE RUNTIME IDIOM (D-10-DEF-10), imported UNDER ITS OWN NAME and deliberately NOT aliased.
#
# ``h3_grad_contract`` is family-named for historical reasons only — its mechanism is
# ``torch.no_grad`` plus a ``requires_grad_(False)`` that VERIFIES itself, which has nothing H3 about
# it. Re-implementing it here would recreate precisely the failure its own docstring names: *"That
# was paid for on run 7. It was REMEMBERED, not STRUCTURAL, so the H3 leg inherited nothing."*
# Family #3 inherits the structure instead.
#
# The name matters mechanically, not only stylistically: ``model_forward_sites`` matches on the
# decorator NAME (``H3_NO_GRAD_DECORATOR == "h3_no_grad"``, ``h3_grad_contract.py:90, 295-298``).
# A local alias — ``qwen_no_grad = h3_no_grad`` — would leave every forward site in this module
# reporting ``grad_free=False``, and the structural scan would silently stop covering family #3
# while looking clean. If the module is ever renamed to a family-neutral ``prep/grad_contract`` with
# a ``no_grad`` marker, this import follows it in the same commit.
from signet_trainer.prep.h3_grad_contract import h3_no_grad

# The device READ is a pure probe with no message of its own (parameters -> .device -> __dict__
# scan), so it is single-sourced. The device REFUSAL is written locally, below, because its message
# must name THIS site and THIS VAE class — an "[h3-vae-contract]" tag on a Qwen failure sends the
# reader to the wrong file.
from signet_trainer.prep.h3_vae_contract import h3_component_device

logger = logging.getLogger(__name__)

__all__ = [
    "QWEN_EDIT_BLANK_FILL_RGB",
    "QWEN_EDIT_CONDITIONS_DIR",
    "QWEN_EDIT_CONTROL_LATENTS_DIR",
    "QWEN_EDIT_CONTROL_POSTERIOR",
    "QWEN_EDIT_ENCODE_SEED",
    "QWEN_EDIT_IMG_PROMPT_TEMPLATE",
    "QWEN_EDIT_LATENTS_DIR",
    "QWEN_EDIT_PIXEL_RANGE",
    "QWEN_EDIT_PRECOMPUTED_DIRS",
    "QWEN_EDIT_PROMPT_TEMPLATE",
    "QWEN_EDIT_PROMPT_TEMPLATE_START_IDX",
    "QWEN_EDIT_TARGET_POSTERIOR",
    "QWEN_EDIT_TEXT_PAYLOAD_VERSION",
    "QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY",
    "QWEN_EDIT_TOKENIZER_MAX_LENGTH",
    "QWEN_EDIT_VAE_ENCODE_RANK",
    "QwenEditControlSource",
    "QwenEditPreparedImage",
    "assert_qwen_edit_cache_complete",
    "assert_qwen_edit_encode_device",
    "assert_qwen_edit_vae_encode_input",
    "build_qwen_edit_presentation",
    "encode_qwen_edit_control_latents",
    "encode_qwen_edit_latents",
    "encode_qwen_edit_target_latents",
    "encode_qwen_edit_text_conditions",
    "free_qwen_edit_text_encoder",
    "prepare_qwen_edit_image",
    "qwen_edit_allowlist_gap",
    "qwen_edit_blank_image",
    "qwen_edit_cache_census",
    "qwen_edit_control_identity",
    "qwen_edit_normalize_pixels",
    "qwen_edit_pixels_from_image",
    "qwen_edit_realized_image_rows",
    "qwen_edit_text_cache_is_current",
    "qwen_edit_text_cache_key",
    "qwen_edit_vae_latent_stats",
    "resolve_qwen_edit_control_sources",
    "write_qwen_edit_precomputed",
]


# --------------------------------------------------------------------------------------------------
# Parity constants. Each carries its line-verified citation. NONE is a tunable — they are checkpoint
# contracts, and a config knob that changes one is a bug, not a feature (D-NOHARDCODE).
# --------------------------------------------------------------------------------------------------

#: The three precomputed source dirs, declared HERE and cross-checked against the reader's own
#: declaration at import time (below). The ``qwen_edit_`` prefix is load-bearing for the reason H3's
#: ``h3_`` prefix is: a different VAE and a different text encoder produce tensors of the SAME RANK
#: as the LTX ones, so an unprefixed name could be paired into an LTX run and be silently wrong.
QWEN_EDIT_LATENTS_DIR: str = "qwen_edit_latents"
QWEN_EDIT_CONDITIONS_DIR: str = "qwen_edit_conditions"
QWEN_EDIT_CONTROL_LATENTS_DIR: str = "qwen_edit_control_latents"

#: Writer-side order: target, controls, text. ``PrecomputedDataset`` pairs by RELATIVE PATH and is
#: order-agnostic, so this tuple exists for the CENSUS (which needs a stable iteration order in its
#: messages), not for the wire.
QWEN_EDIT_PRECOMPUTED_DIRS: tuple[str, ...] = (
    QWEN_EDIT_LATENTS_DIR,
    QWEN_EDIT_CONTROL_LATENTS_DIR,
    QWEN_EDIT_CONDITIONS_DIR,
)

if set(QWEN_EDIT_PRECOMPUTED_DIRS) != set(QWEN_EDIT_DATA_SOURCES):
    # A cross-import guard rather than a hopeful duplicate. The WRITER and the READER
    # (``QwenEditStrategy.get_data_sources``) must name the same three dirs or every sample silently
    # drops out of the ``PrecomputedDataset`` index — the pairing is by directory name, and a
    # directory nobody writes is a source with zero files, which is a FileNotFoundError at best and
    # an empty dataset at worst. Import-time because that is the cheapest possible moment.
    raise RuntimeError(
        f"[qwen-edit-encode] the writer declares {sorted(QWEN_EDIT_PRECOMPUTED_DIRS)} but "
        f"QwenEditStrategy.get_data_sources() declares {sorted(QWEN_EDIT_DATA_SOURCES)}. "
        f"PrecomputedDataset pairs sources by directory NAME, so a disagreement here means the "
        f"trainer looks for a directory the pre-encode never created."
    )

#: ⚠⚠ **``[-1, 1]``, NOT ``[0, 1]`` and NOT H3's ImageNet-over-``[0,1]``.** Two independent paths
#: land here: ai-toolkit ``qwen_image_edit_plus.py:278`` (``control_img * 2 - 1``, over a ``[0,1]``
#: ``ToTensor`` base) and diffusers ``image_processor.py:225`` (``2.0 * images - 1.0``, reached from
#: ``pipeline_qwenimage_edit_plus.py:684``). A sibling family's convention pasted in here produces
#: perfectly-shaped latents that are silently off-distribution.
QWEN_EDIT_PIXEL_RANGE: tuple[float, float] = (-1.0, 1.0)

#: ``AutoencoderKLQwenImage._encode`` unpacks exactly five axes
#: (``autoencoder_kl_qwenimage.py:783``: ``_, _, num_frame, height, width = x.shape``) and prices its
#: temporal chunking off ``num_frame`` (``:789``, ``iter_ = 1 + (num_frame - 1) // 4``).
QWEN_EDIT_VAE_ENCODE_RANK: int = 5

#: Posterior policy — **the two halves differ, deliberately, and this is a DECISION rather than a
#: port.** ai-toolkit's shared ``encode_images`` draws ``latent_dist.sample()`` for everything
#: (``qwen_image.py:432``) and re-encodes the CONTROL images inside every training step
#: (``qwen_image_edit_plus.py:280``), so its control conditioning is re-sampled ~2000 times per run
#: and never cached. diffusers takes the ``argmax`` / ``.mode()`` for exactly those images
#: (``pipeline_qwenimage_edit_plus.py:414, 419`` via ``retrieve_latents:150-151``).
#:
#: signet caches once, so re-sampling per step is not available and one of the two must be chosen:
#:
#:   * TARGET  -> ``sample``, which is what the training path does, made REPRODUCIBLE by a fixed CPU
#:     generator (ai-toolkit passes no generator at all and therefore draws off global RNG — a cache
#:     that cannot be re-derived). The H3 precedent for a fixed, run-seed-INDEPENDENT encode seed is
#:     ``prep/h3_encode.py:171-175``.
#:   * CONTROL -> ``mode``, which is what the INFERENCE pipeline does for control images. Caching a
#:     single stochastic draw would freeze one sample of the posterior as "the" conditioning for the
#:     whole run; the mode is the distribution's own answer and matches what the sampler will see.
#:
#: Both are named constants so a future ruling flips one line rather than hunting call sites.
QWEN_EDIT_TARGET_POSTERIOR: str = "sample"
QWEN_EDIT_CONTROL_POSTERIOR: str = "mode"

#: The fixed posterior seed, deliberately INDEPENDENT of ``training.seed``. Same reasoning as
#: ``H3_KEYFRAME_ENCODE_SEED`` (``prep/h3_encode.py:171-175``): a cache seeded from the run seed is a
#: different cache per run, and nothing about the tensor shape or dtype reveals it.
QWEN_EDIT_ENCODE_SEED: int = 42

#: The edit-plus prompt template, verbatim from ``pipeline_qwenimage_edit_plus.py:216``. It is NOT
#: the plain QwenImage template — this one instructs the VL model to describe the INPUT image before
#: applying the instruction, which is the whole reason the vision half is in the loop.
QWEN_EDIT_PROMPT_TEMPLATE: str = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter or "
    "modify the image. Generate a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}"
    "<|im_end|>\n<|im_start|>assistant\n"
)

#: One per condition image, 1-INDEXED, prepended to the caption inside the template's ``{}``
#: (``pipeline_qwenimage_edit_plus.py:240-246, 253``). The index is what the caption's ``ctrl_img_N``
#: addressing refers to, which is why a slot may never slide (DIVERGE #4).
QWEN_EDIT_IMG_PROMPT_TEMPLATE: str = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

#: Tokens dropped off the FRONT of every encoded sequence — the system block through
#: ``<|im_start|>user\n`` (``pipeline_qwenimage_edit_plus.py:217``, applied at ``:272``).
#: ⚠ [INFERENCE, not verified on this box] that 64 is exactly that prefix's token length: the only
#: cached Qwen tokenizer here has an empty vocab and downloading is out of scope. The number is
#: TRANSCRIBED from the pipeline, not derived, and the image tokens sit AFTER it — so a control
#: image's visual tokens survive the drop and are part of ``prompt_embeds``.
QWEN_EDIT_PROMPT_TEMPLATE_START_IDX: int = 64

#: ``pipeline_qwenimage_edit_plus.py:214``; ``check_inputs`` refuses beyond it at ``:381-382``.
QWEN_EDIT_TOKENIZER_MAX_LENGTH: int = 1024

#: RGB for each declared blank fill. ``QWEN_EDIT_BLANK_FILLS`` (the NAMES) is owned by
#: ``conditioning/qwen_edit`` and imported; only the PIXELS are decided here, because only the
#: encoder needs them.
#: ⚠ ``gray`` is 128, a DECLARED choice: the midpoint of ``[0, 255]`` is 127.5 and is not
#: representable in uint8. 128 normalizes to ``+0.0039``, 127 to ``-0.0039``; neither is "neutral",
#: and pretending otherwise is how an arbitrary constant becomes folklore. A blank slot is a real
#: image the adapter sees on every padded row.
QWEN_EDIT_BLANK_FILL_RGB: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}

if set(QWEN_EDIT_BLANK_FILL_RGB) != set(QWEN_EDIT_BLANK_FILLS):
    raise RuntimeError(
        f"[qwen-edit-encode] blank fills {sorted(QWEN_EDIT_BLANK_FILL_RGB)} do not cover the "
        f"declared {sorted(QWEN_EDIT_BLANK_FILLS)}. A fill the config accepts but the encoder "
        f"cannot render fails only when a sample happens to need it."
    )

#: Bumped whenever the text payload gains or loses a field a reader depends on, or whenever the KEY
#: composition changes. The ``h3_text_payload`` precedent: a reader that finds a version it does not
#: know REFUSES rather than reading the fields it recognizes.
QWEN_EDIT_TEXT_PAYLOAD_VERSION: int = 1
QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: str = "qwen_edit_text_payload_version"

#: Where the cache key lives inside the text payload. It is stored rather than encoded in the
#: FILENAME because ``PrecomputedDataset`` pairs sources by relative path — a hash-named file would
#: never pair. See :func:`qwen_edit_text_cache_key`.
_QWEN_EDIT_CACHE_KEY_KEY: str = "cache_key"


def qwen_edit_allowlist_gap() -> str:
    """The one-line statement of the open ``_VIDEO_LATENT_SOURCE_DIRS`` gap. Not a stub — a NOTICE.

    ``data/precomputed.py:94-96`` allowlists ``{"latents", "reference_latents", "h3_latents",
    "h3_reference_latents"}``. The two qwen latent sources are absent, which is HARMLESS today (the
    allowlist only routes payloads through the legacy patchified ``[seq_len, C]`` einops fixup, and
    nothing writes that form) and becomes wrong the moment a legacy-form cache appears.

    Returned as a string rather than raised so the preprocess stage can PRINT it in its banner: a
    known gap that is announced on every run is institutional memory; a known gap in a docstring is
    a thing someone rediscovers.
    """
    return (
        f"[qwen-edit-encode] NOTICE: {QWEN_EDIT_LATENTS_DIR}/ and "
        f"{QWEN_EDIT_CONTROL_LATENTS_DIR}/ are NOT in "
        f"data/precomputed.py::_VIDEO_LATENT_SOURCE_DIRS. Their payloads satisfy that allowlist's "
        f"contract by construction (asserted at write time), so adding the two names is a one-line "
        f"change — until then a cached [C, F, H, W] latent passes through un-normalized, which is "
        f"correct only because nothing writes the legacy 2-D patchified form."
    )


# --------------------------------------------------------------------------------------------------
# Pixels: the ONE resize site, the ONE normalization, the ONE tensor builder.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenEditPreparedImage:
    """ONE image already resized to ONE channel's area budget, carrying the rows it will cost.

    ⛔ **This type exists so the two channels cannot disagree about what an image IS** — the
    ``H3PreparedReference`` pattern (``prep/h3_encode.py:551-568``) applied to the failure mode
    Qwen actually has. H3's was one image prepared by two phases at two geometries; Qwen's is one
    image that is SUPPOSED to be prepared at two geometries, which is worse, because the correct
    behaviour and the bug look identical from a call site. Carrying ``area_px`` on the value makes
    "which budget is this?" a property of the object rather than of the caller's memory.

    ``image`` is the RESIZED image and ``latent_rows`` is ITS packed row count, produced together by
    :func:`prepare_qwen_edit_image`. There is no call form that pairs one budget's pixels with
    another budget's row count.

    ``latent_rows`` is meaningful for the VAE channel (it is the block's contribution to
    ``hidden_states``) and is computed for the VL channel too — where it is NOT the vision-token
    count, because Qwen2.5-VL grids at ``patch 14 x merge 2 = 28`` and then merges 2x2
    (``processing_qwen2_5_vl.py:123-128``), not at this family's 16. The field is deliberately not
    reused across channels; ``blank`` / ``fill`` travel with it so a synthetic slot is never
    mistaken for a file.
    """

    image: Any
    source_wh: tuple[int, int]
    encoded_wh: tuple[int, int]
    area_px: int
    latent_rows: int
    blank: bool = False
    fill: str | None = None


def prepare_qwen_edit_image(
    image: Any, area_px: int, *, blank: bool = False, fill: str | None = None
) -> QwenEditPreparedImage:
    """THE single image-resize site — every consumer of a Qwen control/target image goes through it.

    The sizing arithmetic is NOT re-derived here: it is
    ``qwen_edit_geometry.qwen_edit_area_budget_size``, which owns the aspect-preserving area fit and
    the ``round(edge / 32) * 32`` snap (ai-toolkit ``qwen_image_edit_plus.py:270-271`` for the VAE
    channel and ``:181-182`` for the VL channel — the same four lines twice).

    ⚠ **ORIENTATION — the fork is already ruled, and this function inherits the ruling.**
    ai-toolkit computes ``ratio = shape[2] / shape[3]``, i.e. ``H / W``, then assigns
    ``width = sqrt(area * ratio)`` and calls ``F.interpolate(size=(height, width))`` — so the two
    locals hold each other's value and a 2:1 landscape control is squeezed into a 1:2 portrait
    tensor before BOTH encodes. diffusers passes ``ratio = image_width / image_height``
    (``pipeline_qwenimage_edit_plus.py:678-680``) and preserves orientation.
    ``qwen_edit_area_budget_size`` chose diffusers-correct and recorded the divergence in its own
    docstring (``qwen_edit_geometry.py:252-263``) during phase 1; this module does not re-open a
    settled decision. The two agree exactly for SQUARE sources, which is what the shipped example
    config uses — so a non-square corpus is the first place the choice is observable, and
    :func:`prepare_qwen_edit_image` logs it at WARNING for exactly that reason.

    ⛔ **Re-preparing at a DIFFERENT budget RAISES.** The same source legitimately goes down both
    channels at 384² and 1024², so the correct call is twice FROM THE SOURCE. Resizing an
    already-1024-budget copy down to 384 double-resamples and is not what either channel sees at
    inference; an idempotent same-budget call is returned as-is (the ``prepare_h3_reference_images``
    contract, ``prep/h3_encode.py:588-592``).

    Args:
        image: A Pillow-like image (``.size`` -> ``(width, height)``, ``.convert``, ``.resize``), or
            an already-prepared value.
        area_px: The channel's pixel-AREA budget — ``qwen_edit.control_area_px`` (VAE) or
            ``qwen_edit.condition_area_px`` (VL). NEVER a constant at the call site: both are config
            fields, and the whole trap is that they are two different numbers.
        blank: whether these pixels are a synthetic fill rather than a file.
        fill: which fill, when ``blank``. Both are carried onto the result so a synthetic slot is
            never mistaken for one that had an image.

    Returns:
        The prepared image, its source size, its encoded size and its packed row count.
    """
    from PIL import Image as PILImage  # noqa: PLC0415 — deliberate function-local (Modal-side tier)

    if isinstance(image, QwenEditPreparedImage):
        if int(image.area_px) != int(area_px):
            raise ValueError(
                f"[qwen-edit-encode] this image was prepared at an area budget of {image.area_px} "
                f"px and is being reused at {area_px} px. The SAME control image goes down BOTH "
                f"channels at DIFFERENT budgets (VL {QWEN_EDIT_CONDITION_IMAGE_SIZE}, VAE "
                f"{QWEN_EDIT_VAE_IMAGE_SIZE} — qwen_image_edit_plus.py:66-67), so the correct call "
                f"is twice FROM THE SOURCE image. Re-fitting an already-resized copy resamples it "
                f"twice and produces pixels neither channel sees at inference."
            )
        return image

    rgb = image.convert("RGB") if hasattr(image, "convert") else image
    source_width, source_height = (int(v) for v in rgb.size)
    target_width, target_height = qwen_edit_area_budget_size(
        source_width, source_height, int(area_px)
    )
    if (source_width, source_height) != (target_width, target_height):
        rgb = rgb.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        if source_width != source_height:
            # Not a refusal — the ruling is diffusers-correct and this function obeys it. It is a
            # WARNING because the two implementations agree only on SQUARE sources, so a non-square
            # image that is actually RESIZED here is the first geometry where a signet adapter and
            # an ai-toolkit adapter were conditioned on different pixels. An operator comparing the
            # two deserves to learn that from the log, not from a docstring.
            logger.warning(
                "[qwen-edit-encode] non-square %dx%d -> %dx%d at area budget %d. signet preserves "
                "orientation (diffusers pipeline_qwenimage_edit_plus.py:678-680); ai-toolkit "
                "computes ratio = H/W (qwen_image_edit_plus.py:178-187 and :266-275) and would "
                "produce %dx%d instead. Adapters trained under the two are NOT numerically "
                "interchangeable on non-square control images.",
                source_width,
                source_height,
                target_width,
                target_height,
                int(area_px),
                target_height,
                target_width,
            )

    return QwenEditPreparedImage(
        image=rgb,
        source_wh=(source_width, source_height),
        encoded_wh=(target_width, target_height),
        area_px=int(area_px),
        latent_rows=qwen_edit_rows_of(target_width, target_height),
        blank=bool(blank),
        fill=fill,
    )


def qwen_edit_blank_image(fill: str, width: int, height: int) -> Any:
    """A solid ``fill``-coloured RGB image — the real pixels behind a blank control slot.

    ⛔ **A blank slot is NOT a zero latent.** ``QwenEditStrategy._control_rows`` refuses to
    synthesize one and says why (``conditioning/qwen_edit.py:785-795``): *"a zero block packs to the
    correct shape and prices the correct number of rows"* and would teach the adapter that an empty
    slot looks like something the encoder never emits. Rendering the fill here and encoding it
    through the same VAE as every other slot is the alternative that docstring names first —
    *"Encode the blanks in prep (the payload then carries them, blank=True)"* — which is why this
    module does it rather than shipping a ``blank_latent_fn``.
    """
    from PIL import Image as PILImage  # noqa: PLC0415 — deliberate function-local

    rgb = QWEN_EDIT_BLANK_FILL_RGB.get(fill)
    if rgb is None:
        raise ValueError(
            f"[qwen-edit-encode] unknown blank fill {fill!r}: expected one of "
            f"{sorted(QWEN_EDIT_BLANK_FILL_RGB)}."
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"[qwen-edit-encode] blank slot size {width}x{height} is degenerate. A blank fill costs "
            f"real rows and must be sized exactly like a real control, or the packed sequence "
            f"length stops being a constant of the config."
        )
    return PILImage.new("RGB", (int(width), int(height)), rgb)


def qwen_edit_pixels_from_image(image: Any) -> torch.Tensor:
    """An ALREADY-RESIZED image -> ``[3, 1, H, W]`` uint8 pixels, the rank the VAE path carries.

    ⛔ Deliberately does NOT resize. The one resize lives in :func:`prepare_qwen_edit_image`; a
    second one here would be a second source of truth about encode geometry, which is the defect
    that factoring exists to make impossible (the ``_reference_pixels`` contract,
    ``prep/h3_encode.py:820-826``).

    The singleton axis at position 1 is the FRAME axis, not a batch axis: ``AutoencoderKLQwenImage``
    is a causal 3-D VAE and an image is one frame (``qwen_image.py:430`` inserts exactly this axis
    with ``images.unsqueeze(2)`` on a batched tensor). The BATCH axis is added and removed by
    :func:`encode_qwen_edit_latents`, at one chokepoint.
    """
    import numpy as np  # noqa: PLC0415 — deliberate function-local

    # np.array (a COPY), not np.asarray: Pillow hands back a READ-ONLY buffer, and torch.from_numpy
    # on a non-writable array warns and yields a tensor whose writes are undefined behaviour.
    array = np.array(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"[qwen-edit-encode] expected an HxWx3 RGB image, got array shape {array.shape}. "
            f"Convert to RGB before encoding — an RGBA or L image reaches the VAE at the wrong "
            f"channel count and the failure names a conv, not a colour space."
        )
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(1).contiguous()


def qwen_edit_normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """uint8 ``[0, 255]`` -> float32 ``[-1, 1]``. The Qwen VAE's pixel convention, and only it.

    ::

        pixels.to(float32).div(255.0)   # -> [0, 1]  (ToTensor / VaeImageProcessor's base)
        x * 2 - 1                       # -> [-1, 1]

    ⚠ Both halves are cited and both are load-bearing. ai-toolkit reaches ``[0,1]`` through
    ``ToTensor`` (``dataloader_mixins.py:1047-1053``) and then writes the second line literally
    (``qwen_image_edit_plus.py:278``); diffusers collapses the pair into
    ``2.0 * images - 1.0`` (``image_processor.py:225``, ``do_normalize`` defaulting True at
    ``:120``). Substituting H3's ImageNet-over-``[0,1]`` recipe — which is what a copy-paste from
    ``prep/h3_encode.imagenet_normalize`` would do — is a silent off-distribution encode at a
    perfectly valid shape.
    """
    return pixels.to(torch.float32).div(255.0).mul(2.0).sub(1.0)


# --------------------------------------------------------------------------------------------------
# The VAE contract: rank, device, latent statistics.
# --------------------------------------------------------------------------------------------------


def assert_qwen_edit_vae_encode_input(pixels: Any, *, what: str) -> None:
    """Refuse anything but 5-D ``[B, C, F, H, W]``, BY NAME, before it reaches ``vae.encode``.

    ⛔ The D-10-DEF-9 refusal, re-stated for family #3 because the CLASS transfers even though the
    symptom does not. ``AutoencoderKLQwenImage._encode`` opens with
    ``_, _, num_frame, height, width = x.shape`` (``autoencoder_kl_qwenimage.py:783``), so a 4-D
    input raises ``ValueError: not enough values to unpack (expected 5, got 4)`` — LOUD, unlike
    H3's, which read the canvas height as the frame count and could have returned a plausible
    tensor. The guard is still worth its lines: it names the SITE and the AXIS, where the library's
    message names neither, and it keeps the rank discipline symmetric across families so a reader
    who learned it on H3 finds it here.
    """
    rank = int(getattr(pixels, "dim", lambda: -1)())
    if rank == QWEN_EDIT_VAE_ENCODE_RANK:
        return
    shape = tuple(int(v) for v in getattr(pixels, "shape", ()))
    raise TypeError(
        f"[qwen-edit-encode] {what} would hand AutoencoderKLQwenImage a {rank}-D tensor {shape}; "
        f"_encode requires {QWEN_EDIT_VAE_ENCODE_RANK}-D [B, C, F, H, W] "
        f"(autoencoder_kl_qwenimage.py:783 unpacks exactly five axes and prices its temporal "
        f"chunking off axis 2). An image is F = {QWEN_EDIT_FRAMES}, never a missing axis. "
        f"encode_qwen_edit_latents adds and removes the BATCH axis itself; any other path to "
        f"vae.encode must do the same."
    )


def assert_qwen_edit_encode_device(tensor: Any, component: Any, *, what: str) -> None:
    """Refuse a tensor that is not on its component's device, BY NAME, before the forward.

    ⛔ **D-10-DEF-12, inherited whole.** The finding is H3's and is measured there
    (``prep/h3_vae_contract.py:241-286``): a CPU tensor reached a CUDA-resident VAE, cuDNN declined
    it, ``_select_conv_backend`` fell through to ``ConvBackend.Slow3d``, and Slow3d has no CUDA
    kernel — so PyTorch's own message is ``Could not run 'aten::slow_conv3d_forward' with arguments
    from the 'CUDA' backend``, which names a kernel and never mentions placement.

    It applies here unchanged and for the same mechanical reason: ``AutoencoderKLQwenImage`` is
    built out of ``QwenImageCausalConv3d`` (``autoencoder_kl_qwenimage.py:436, 590, 700-701``), so
    the failing dispatch is the same conv3d dispatch. The device READ is single-sourced
    (``h3_component_device``); only this message is local, because a reader chasing an
    ``[h3-vae-contract]`` tag out of a Qwen traceback opens the wrong file.

    Type equality, not index equality: a sharded component answers with one of its parameters'
    devices, and the dispatch failure being guarded is CPU-vs-CUDA.
    """
    device = h3_component_device(component)
    if device is None:
        return
    actual = getattr(tensor, "device", None)
    if actual is None or actual.type == device.type:
        return
    raise RuntimeError(
        f"[qwen-edit-encode] {what} would run a component whose weights are on {device} against a "
        f"tensor on {actual} (D-10-DEF-12, inherited from the H3 leg). PyTorch does NOT say that: "
        f"AutoencoderKLQwenImage is built from QwenImageCausalConv3d, cuDNN declines a non-CUDA "
        f"input, the dispatcher falls through to ConvBackend.Slow3d, and Slow3d has no CUDA kernel "
        f"— so the observable failure names 'aten::slow_conv3d_forward' and never mentions "
        f"placement. encode_qwen_edit_latents moves the pixels onto the component's own device at "
        f"ONE chokepoint; any other path to a component forward must do the same."
    )


def qwen_edit_vae_latent_stats(vae: Any) -> tuple[Any, Any]:
    """``(latents_mean, latents_std)`` read off the VAE's OWN config. No fallback, ever.

    ⛔ **This VAE has NO ``scaling_factor`` and NO ``shift_factor``.** Per-channel standardisation
    replaces them: ``latents_mean`` / ``latents_std`` are 16-vectors defaulted at
    ``autoencoder_kl_qwenimage.py:687-688`` and applied as ``(x - mean) / std`` at
    ``pipeline_qwenimage_edit_plus.py:420-430`` (ai-toolkit multiplies by the reciprocal at
    ``qwen_image.py:434-444`` — algebraically identical). A helper that fell back to
    ``getattr(config, "scaling_factor", 1.0)`` would silently scale by 1.0 and produce latents whose
    every channel is off by its own mean and its own variance, at exactly the right shape.

    Refused rather than defaulted for the same reason ``sigma_fn`` is refused rather than defaulted
    (``conditioning/qwen_edit.py:747-754``): the wrong answer here is indistinguishable from the
    right one until a sample renders.
    """
    config = getattr(vae, "config", None)
    stats: list[Any] = []
    for name in ("latents_mean", "latents_std"):
        value = getattr(config, name, None)
        if value is None and isinstance(config, dict):
            value = config.get(name)
        if value is None:
            raise RuntimeError(
                f"[qwen-edit-encode] the mounted VAE's config declares no {name!r}. "
                f"AutoencoderKLQwenImage standardises latents PER CHANNEL "
                f"({QWEN_EDIT_LATENT_CHANNELS}-vectors, autoencoder_kl_qwenimage.py:687-688) and "
                f"has no scaling_factor to fall back to — a default of 1.0 would leave every "
                f"channel off by its own mean and variance at a perfectly valid shape. Either the "
                f"wrong VAE is mounted or its config was not loaded."
            )
        length = len(value) if hasattr(value, "__len__") else 1
        if length != QWEN_EDIT_LATENT_CHANNELS:
            raise RuntimeError(
                f"[qwen-edit-encode] the mounted VAE declares {length} {name} element(s), expected "
                f"{QWEN_EDIT_LATENT_CHANNELS} (conditioning/qwen_edit_geometry."
                f"QWEN_EDIT_LATENT_CHANNELS). The wrong VAE is mounted — LTX's and H3's are "
                f"different widths."
            )
        stats.append(value)
    return stats[0], stats[1]


def _channel_axis(tensor: torch.Tensor) -> int:
    """Channel axis: 1 for ``[B, C, F, H, W]``, 0 for ``[C, F, H, W]``."""
    if tensor.dim() == QWEN_EDIT_VAE_ENCODE_RANK:
        return 1
    if tensor.dim() == QWEN_EDIT_VAE_ENCODE_RANK - 1:
        return 0
    raise ValueError(
        f"[qwen-edit-encode] expected a 4-D [C, F, H, W] or 5-D [B, C, F, H, W] tensor, got shape "
        f"{tuple(tensor.shape)}."
    )


def _broadcast_channel_vector(values: Any, reference: torch.Tensor, what: str) -> torch.Tensor:
    """Reshape a per-channel vector so it broadcasts against ``reference``'s channel axis."""
    vector = torch.as_tensor(values, dtype=reference.dtype, device=reference.device).flatten()
    axis = _channel_axis(reference)
    channels = int(reference.shape[axis])
    if vector.numel() not in (1, channels):
        raise ValueError(
            f"[qwen-edit-encode] {what} has {vector.numel()} element(s) but the tensor has "
            f"{channels} channel(s). AutoencoderKLQwenImage ships "
            f"{QWEN_EDIT_LATENT_CHANNELS}-element latents_mean / latents_std vectors."
        )
    shape = [1] * reference.dim()
    shape[axis] = vector.numel()
    return vector.reshape(shape)


def _latent_distribution(encoder_output: Any) -> Any:
    """The posterior out of whatever ``vae.encode`` returned. Both call shapes are real.

    ai-toolkit reads ``.latent_dist`` (``qwen_image.py:432``); diffusers reaches the same object via
    ``retrieve_latents`` (``pipeline_qwenimage_edit_plus.py:145-151``), which also accepts the
    ``return_dict=False`` tuple form. Accepting both is not permissiveness — it is refusing to have
    an opinion about which of the two the caller's diffusers version returns.
    """
    distribution = getattr(encoder_output, "latent_dist", None)
    if distribution is not None:
        return distribution
    if isinstance(encoder_output, (tuple, list)) and encoder_output:
        return encoder_output[0]
    raise TypeError(
        f"[qwen-edit-encode] vae.encode returned {type(encoder_output).__name__}, which carries "
        f"neither a .latent_dist (qwen_image.py:432) nor a leading posterior in the "
        f"return_dict=False tuple form (pipeline_qwenimage_edit_plus.py:145-151). Refusing to guess "
        f"which tensor is the posterior."
    )


@h3_no_grad
def encode_qwen_edit_latents(
    vae: Any,
    pixels: torch.Tensor,
    latents_mean: Any,
    latents_std: Any,
    *,
    posterior: str,
    seed: int = QWEN_EDIT_ENCODE_SEED,
) -> torch.Tensor:
    """The Qwen VAE encode recipe. THE chokepoint every image on this path routes through.

    ::

        pixels    = pixels.to(vae.device)                     # D-10-DEF-12
        pixels    = pixels.to(float32).div(255).mul(2).sub(1) # [-1, 1]
        posterior = vae.encode(pixels[None]).latent_dist
        latents   = posterior.sample(generator=Generator().manual_seed(42))  |  posterior.mode()
        return (latents - latents_mean) / latents_std

    ⛔ **``@h3_no_grad`` is load-bearing, not hygiene** (D-10-DEF-10). ``.eval()`` switches
    norm/dropout behaviour and touches autograd not at all, and a freshly-loaded diffusers VAE
    arrives with ``requires_grad=True`` on every parameter — so without the decorator ``vae.encode``
    builds and RETAINS a graph for a backward that never comes. On the H3 leg that exhausted an
    80 GiB A100 on a 544 MiB allocation. The decorator is the shared marker imported under its own
    name so ``h3_grad_contract.model_forward_sites`` still recognises it (see the import comment).

    ⛔ **The pixels are placed on the VAE's own device HERE**, once, off the component itself
    (D-10-DEF-12). The move happens BEFORE the normalization on purpose: uint8 crosses the bus
    instead of float32, and the arithmetic runs where the weights are. Precision is untouched —
    ``div``/``mul``/``sub`` are correctly-rounded elementwise float32 ops.

    ⛔ **The rank contract is satisfied from ONE place** (D-10-DEF-9): the caller hands
    ``[C, F, H, W]``, this function adds the batch axis, asserts, encodes and takes it back off, so
    "same rank as pixels" below is true rather than merely advertised.

    ⚠ **No ``float16`` round-trip.** ``prep/h3_encode.encode_video_latents`` rounds through fp16
    before normalizing because that is a MiniMax-H3 parity requirement; Qwen has no such step in
    either reference implementation, and adding one would invent an ~11-bit precision loss.

    Args:
        vae: ``AutoencoderKLQwenImage``. Passed in, never constructed here, so this function needs
            no backend import — and has no business having an opinion about where it lives.
        pixels: RGB in ``[0, 255]``, ``[C, F, H, W]`` or ``[B, C, F, H, W]``, uint8 expected.
        latents_mean: the 16-vector from :func:`qwen_edit_vae_latent_stats`.
        latents_std: its partner.
        posterior: ``"sample"`` or ``"mode"``. **REQUIRED, never defaulted** — the two halves of
            this pipeline deliberately differ (see ``QWEN_EDIT_TARGET_POSTERIOR`` /
            ``QWEN_EDIT_CONTROL_POSTERIOR``), and a default here would let one of them be chosen by
            omission.
        seed: the fixed CPU-generator seed for ``"sample"``. Independent of the run seed.

    Returns:
        Normalized latents, same rank as ``pixels``.
    """
    if posterior not in ("sample", "mode"):
        raise ValueError(
            f"[qwen-edit-encode] invalid posterior policy {posterior!r}: expected 'sample' (the "
            f"TARGET policy — ai-toolkit's training path, qwen_image.py:432) or 'mode' (the CONTROL "
            f"policy — the inference pipeline's argmax for exactly those images, "
            f"pipeline_qwenimage_edit_plus.py:414-419)."
        )

    # ⛔ D-10-DEF-12. Placement decided ONCE, here, off the component — not at the call sites, where
    # the H3 leg decided it once and forgot it once. A `None` device means the component holds no
    # tensors at all, and moving to a guessed device would be worse than leaving the pixels alone.
    device = h3_component_device(vae)
    if device is not None:
        pixels = pixels.to(device)

    normalized = qwen_edit_normalize_pixels(pixels)

    unbatched = normalized.dim() == QWEN_EDIT_VAE_ENCODE_RANK - 1
    if unbatched:
        normalized = normalized.unsqueeze(0)
    assert_qwen_edit_vae_encode_input(normalized, what="encode_qwen_edit_latents")
    # The move above cannot be dropped by a later refactor without this going red BY NAME: the
    # library's own failure names a conv kernel and never mentions device (D-10-DEF-12).
    assert_qwen_edit_encode_device(normalized, vae, what="encode_qwen_edit_latents")

    distribution = _latent_distribution(vae.encode(normalized))
    if posterior == "sample":
        # CPU generator, FIXED seed — reproducible across devices and independent of the run seed.
        # ai-toolkit passes NO generator here, so its draw comes off global RNG and its cache cannot
        # be re-derived; that is the one thing about this line that is not a transcription.
        latents = distribution.sample(generator=torch.Generator().manual_seed(int(seed)))
    else:
        latents = distribution.mode()

    mean = _broadcast_channel_vector(latents_mean, latents, "latents_mean")
    std = _broadcast_channel_vector(latents_std, latents, "latents_std")
    latents = (latents - mean) / std

    # Give back the rank we were given (D-10-DEF-9). The batch axis was OURS.
    return latents[0] if unbatched else latents


def _assert_image_latent(latents: torch.Tensor, what: str) -> tuple[int, int, int, int]:
    """``[C, F, H, W]`` with the right C and ``F == 1``, or a named refusal. Returns the four dims."""
    if latents.dim() == QWEN_EDIT_VAE_ENCODE_RANK:
        if int(latents.shape[0]) != 1:
            raise ValueError(
                f"[qwen-edit-encode] {what} encoded to batch {int(latents.shape[0])}; this path "
                f"encodes ONE image per call (batch_size = 1, the house recipe)."
            )
        latents = latents[0]
    channels, frames, height, width = (int(v) for v in latents.shape)
    if channels != QWEN_EDIT_LATENT_CHANNELS:
        raise RuntimeError(
            f"[qwen-edit-encode] {what} encoded to {channels} latent channel(s), expected "
            f"{QWEN_EDIT_LATENT_CHANNELS} (vae.config.z_dim, "
            f"pipeline_qwenimage_edit_plus.py:210). The wrong VAE is mounted — LTX's is 128 and "
            f"H3's is 24. Refusing to write a mis-shaped {QWEN_EDIT_LATENTS_DIR}/ payload."
        )
    if frames != QWEN_EDIT_FRAMES:
        raise RuntimeError(
            f"[qwen-edit-encode] {what} encoded to {frames} latent frame(s); qwen_edit is an IMAGE "
            f"family and every latent carries exactly {QWEN_EDIT_FRAMES}. img_shapes hardcodes it "
            f"(qwen_image_edit_plus.py:236 builds (1, h//2, w//2)), so a multi-frame latent packs "
            f"into rows the transformer's RoPE prices as a single frame."
        )
    if height % QWEN_EDIT_PATCH or width % QWEN_EDIT_PATCH:
        raise RuntimeError(
            f"[qwen-edit-encode] {what} encoded to a {height}x{width} latent grid, which is not "
            f"divisible by the {QWEN_EDIT_PATCH}x{QWEN_EDIT_PATCH} pack "
            f"(qwen_image_edit_plus.py:223-229). An odd latent edge makes the pack lossy, silently."
        )
    return channels, frames, height, width


def _latent_payload(
    latents: torch.Tensor, prepared: QwenEditPreparedImage, what: str
) -> dict[str, Any]:
    """The committed latent-source payload for one image. The dict shape is a HARD contract.

    ⚠ ``height`` / ``width`` are the **LATENT** grid dims — the ``h3_latents`` convention
    (``prep/h3_encode.py:454, 473-478``) and what ``_normalize_video_latents`` consumes. The PIXEL
    sizes travel as ``source_wh`` / ``encoded_wh``, the ``h3_reference_latents`` precedent
    (``prep/h3_encode.py:791-796``): the two differ by the VAE's 8x factor, and a consumer that
    priced an image off the latent grid would under-count it 64x.

    Both are DERIVED from the encoded tensor rather than taken on trust, and the declared
    ``latent_rows`` is cross-checked against the grid — a payload whose row count and grid disagree
    would describe two different images to the same RoPE clock.
    """
    _channels, frames, height, width = _assert_image_latent(latents, what)
    if latents.dim() == QWEN_EDIT_VAE_ENCODE_RANK:
        latents = latents[0]

    rows = (height // QWEN_EDIT_PATCH) * (width // QWEN_EDIT_PATCH)
    if rows != int(prepared.latent_rows):
        raise RuntimeError(
            f"[qwen-edit-encode] {what}: the prepared image was priced at "
            f"{int(prepared.latent_rows)} packed row(s) for {prepared.encoded_wh[0]}x"
            f"{prepared.encoded_wh[1]} px, but the encoded {height}x{width} latent grid prices "
            f"{rows}. The row count drives img_shapes, the loss split and the packed budget — a "
            f"disagreement here means the resize and the encode describe different images."
        )
    return {
        "latents": latents,
        "num_frames": frames,
        # LATENT grid dims. See the docstring — the pixel size is source_wh / encoded_wh.
        "height": height,
        "width": width,
        "source_wh": tuple(int(v) for v in prepared.source_wh),
        "encoded_wh": tuple(int(v) for v in prepared.encoded_wh),
        "latent_hw": (height, width),
        "latent_rows": rows,
    }


@h3_no_grad
def encode_qwen_edit_target_latents(
    vae: Any,
    prepared: Any,
    latents_mean: Any,
    latents_std: Any,
    *,
    stem: str,
    seed: int = QWEN_EDIT_ENCODE_SEED,
) -> dict[str, Any]:
    """Encode the TARGET image and build the committed ``qwen_edit_latents/`` payload.

    The target is the image the loss is computed against — the packed PREFIX
    (``conditioning/qwen_edit.py:17-31``, DIVERGE #1). It is encoded under
    ``QWEN_EDIT_TARGET_POSTERIOR``, which is ``sample`` with a fixed generator: that is what
    ai-toolkit's training path does (``qwen_image.py:432``), made re-derivable.

    ``stem`` travels INTO the payload because the control set is matched to the target BY STEM and
    ``QwenEditStrategy`` needs it at train time, when the directories no longer exist
    (``conditioning/qwen_edit.py:1069-1091``). It is carried, never inferred.

    Args:
        prepared: a :class:`QwenEditPreparedImage` at the TARGET canvas, or a raw image (prepared
            here at the training canvas's own area — see the caller). Passing an unprepared image is
            accepted only when its size already satisfies the canvas rule.
    """
    if not isinstance(prepared, QwenEditPreparedImage):
        raise TypeError(
            f"[qwen-edit-encode] the target must be a QwenEditPreparedImage, got "
            f"{type(prepared).__name__}. Every consumer of an image on this path goes through "
            f"prepare_qwen_edit_image, which resizes ONCE and carries the resulting row count "
            f"alongside the pixels — passing a raw image lets the encode see one geometry while "
            f"img_shapes and the packed budget are billed against another."
        )
    pixels = qwen_edit_pixels_from_image(prepared.image)
    latents = encode_qwen_edit_latents(
        vae,
        pixels,
        latents_mean,
        latents_std,
        posterior=QWEN_EDIT_TARGET_POSTERIOR,
        seed=seed,
    )
    payload = _latent_payload(latents, prepared, "target image")
    payload["stem"] = str(stem)
    return payload


# --------------------------------------------------------------------------------------------------
# Control slots: CONFIG ORDER, and a missing file is an ERROR.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenEditControlSource:
    """One resolved control SOURCE — which file (or which blank) occupies slot ``index``.

    The filesystem half of the slot contract. The SLOT half — index uniqueness, stem agreement,
    blank-fill validity, gap filling — is not re-implemented here: this value is handed to
    ``conditioning/qwen_edit.resolve_control_slots``, which owns it. One resolver for slot
    semantics; a second one would drift.
    """

    index: int
    stem: str
    path: str | None
    blank: bool
    fill: str | None = None

    def as_entry(self) -> dict[str, Any]:
        """The mapping shape ``resolve_control_slots`` reads (``slot`` / ``stem`` / ``path`` / ...)."""
        return {
            "slot": self.index,
            "stem": self.stem,
            "path": self.path,
            "blank": self.blank,
            "fill": self.fill,
        }


def resolve_qwen_edit_control_sources(
    target_stem: str,
    control_dirs: Any,
    *,
    control_slots: int,
    blank_slot_fill: str,
    extensions: Any = (".png", ".jpg", ".jpeg", ".webp"),
    blank_slots: Any = (),
    exists: Any = None,
) -> list[QwenControlSlot]:
    """Resolve slot -> file in **CONFIG ORDER**, refusing a missing file instead of dropping it.

    ⛔ **THE DELIBERATE DIVERGENCE FROM ai-toolkit** (``conditioning/qwen_edit.py:53-63``,
    DIVERGE #4). ``toolkit/dataloader_mixins.py:979-989`` walks the configured ``control_path`` list,
    builds ``file_name_no_ext + ext`` inside each directory, and appends to ``found_control_images``
    **only inside** ``if os.path.exists(...)`` (``:984-985``). With ``control_path = [dirA, dirB,
    dirC]`` and the stem missing from ``dirB``, the list does not merely get shorter — **dirC's
    image slides into slot 1, where dirB's belongs.** Slot identity is positional, the caption's
    ``ctrl_img_N`` addressing is positional, and the model was conditioned on that order, so the
    sample trains against a different request than the one its caption describes. Nothing raises,
    the shapes are right, and the loss curve is ordinary.

    This function refuses. A slot with no file is either an EXPLICIT blank (declared in
    ``blank_slots``, occupying its own index) or an error naming the directory, the stem and the
    index. The list is never shortened and an image is never moved.

    ``exists`` is injected (defaulting to ``Path.exists``) so the whole resolution — including the
    refusal — is provable on CPU with no filesystem, which is the only way a "we do not do what
    ai-toolkit does" claim gets a test.

    Args:
        target_stem: the target's file stem; the join key.
        control_dirs: the ORDERED config directory list. ``len`` must equal ``control_slots``.
        control_slots: how many slots every sample carries (1..3).
        blank_slot_fill: which synthetic image fills a declared blank.
        extensions: candidate suffixes, tried in order. The FIRST that exists wins, and a stem
            present under two suffixes in one directory is an error — two files claiming one slot is
            ambiguity, and resolving it by list order is the bug this function exists to refuse.
        blank_slots: indices declared blank up front (no file is looked for).
        exists: ``(Path) -> bool``. Injected for the CPU contract test.

    Returns:
        Exactly ``control_slots`` :class:`QwenControlSlot` values in index order, produced by
        ``conditioning/qwen_edit.resolve_control_slots`` so the writer and the reader cannot
        disagree about what a slot is.
    """
    checker = (lambda path: Path(path).exists()) if exists is None else exists
    directories = [Path(str(d)) for d in control_dirs]
    declared_blank = {int(i) for i in blank_slots}

    if len(directories) != int(control_slots):
        raise ValueError(
            f"[qwen-edit-encode] {len(directories)} control director(ies) configured for "
            f"{control_slots} slot(s). The mapping is POSITIONAL — directory i fills slot i, which "
            f"is what the caption's ctrl_img_{{i+1}} refers to — so an unequal count cannot be "
            f"resolved without guessing which directory was meant for which slot."
        )
    for index in sorted(declared_blank):
        if not 0 <= index < int(control_slots):
            raise ValueError(
                f"[qwen-edit-encode] blank slot {index} is outside 0..{int(control_slots) - 1}."
            )

    sources: list[QwenEditControlSource] = []
    for index, directory in enumerate(directories):
        if index in declared_blank:
            sources.append(
                QwenEditControlSource(
                    index=index, stem=target_stem, path=None, blank=True, fill=blank_slot_fill
                )
            )
            continue

        found = [directory / f"{target_stem}{ext}" for ext in extensions]
        present = [candidate for candidate in found if checker(candidate)]
        if len(present) > 1:
            raise ValueError(
                f"[qwen-edit-encode] slot {index}: stem {target_stem!r} matches "
                f"{[p.name for p in present]} in {directory}. Two files claim one slot; picking by "
                f"suffix order would silently decide which image the caption's ctrl_img_"
                f"{index + 1} refers to."
            )
        if not present:
            raise FileNotFoundError(
                f"[qwen-edit-encode] slot {index}: no control image for stem {target_stem!r} in "
                f"{directory} (tried {[Path(target_stem).name + e for e in extensions]}). This is "
                f"an ERROR, never a drop. ai-toolkit appends only inside "
                f"`if os.path.exists(...)` (dataloader_mixins.py:984-985), so a gap here would "
                f"slide every LATER control one slot to the left — slot {index + 1}'s image would "
                f"become slot {index}'s, the caption's ctrl_img_N addressing would re-point, and "
                f"the sample would train against a request nobody wrote. Supply the file, or "
                f"declare slot {index} blank explicitly (blank_slots) so the {blank_slot_fill!r} "
                f"fill occupies its own index."
            )
        sources.append(
            QwenEditControlSource(
                index=index, stem=target_stem, path=str(present[0]), blank=False, fill=None
            )
        )

    # The SLOT contract belongs to the reader's module. Round-tripping through it here means a
    # payload this writer produces is, by construction, one ``QwenEditStrategy`` can resolve.
    return resolve_control_slots(
        target_stem,
        [source.as_entry() for source in sources],
        control_slots=int(control_slots),
        blank_slot_fill=blank_slot_fill,
    )


@h3_no_grad
def encode_qwen_edit_control_latents(
    vae: Any,
    slots: Any,
    prepared_images: Any,
    latents_mean: Any,
    latents_std: Any,
    *,
    stem: str,
    control_slots: int,
    seed: int = QWEN_EDIT_ENCODE_SEED,
) -> dict[str, Any]:
    """Encode every control slot (blanks included) -> the ``qwen_edit_control_latents/`` payload.

    ⛔ **Every slot is encoded, including blanks.** A blank costs the same rows as a real control
    (the sequence length is a constant of the config, ``qwen_edit_geometry.py:307-311``) and its VAE
    latent is a specific tensor, not the zero one. Encoding it here is what lets the strategy run
    with ``blank_latent_fn=None`` — the alternative its own refusal names first
    (``conditioning/qwen_edit.py:793-794``).

    ⛔ **Slot order is INDEX order and the list is never shortened.** ``slots`` comes from
    :func:`resolve_qwen_edit_control_sources`, i.e. from the reader's own ``resolve_control_slots``,
    so the per-slot entries below are already index-complete and stem-checked.

    Args:
        slots: the resolved ``QwenControlSlot`` values, in index order.
        prepared_images: index-aligned :class:`QwenEditPreparedImage` values, one per slot, all at
            the VAE channel's ``control_area_px`` budget. Blanks are rendered by
            :func:`qwen_edit_blank_image` and prepared like any other image.

    Returns:
        ``{"controls": [ per-slot dict ], "num_controls", "stem", "control_slots"}`` plus the
        slot-0 ALIAS keys (``latents`` / ``num_frames`` / ``height`` / ``width``).

        ⚠ The top-level ``latents`` key ALIASES **slot 0 only**. It exists so the payload satisfies
        ``data/precomputed.py``'s video-latent allowlist contract, whose ``_normalize_video_latents``
        reads ``data["latents"]`` unconditionally — the ``h3_reference_latents`` precedent
        (``prep/h3_encode.py:811-816``). A consumer that reads the top-level tensor sees ONE control
        where the sample carries three; ``conditioning/qwen_edit._control_entries`` refuses exactly
        that mistake (``:1034-1060``).
    """
    resolved = list(slots)
    images = list(prepared_images)
    if len(resolved) != int(control_slots):
        raise ValueError(
            f"[qwen-edit-encode] got {len(resolved)} resolved slot(s) for {control_slots} declared "
            f"slot(s). The packed sequence length is priced off the DECLARED count at config load; "
            f"a different realized count is a sequence nobody budgeted."
        )
    if len(images) != len(resolved):
        raise ValueError(
            f"[qwen-edit-encode] got {len(images)} prepared image(s) for {len(resolved)} slot(s). "
            f"They are paired POSITIONALLY, so an unequal count would attach one slot's identity to "
            f"another slot's pixels — a mislabelled control that reorders silently."
        )

    encoded: list[dict[str, Any]] = []
    for position, slot in enumerate(resolved):
        if int(slot.index) != position:
            raise ValueError(
                f"[qwen-edit-encode] slot list is not in index order: position {position} carries "
                f"index {slot.index}. Slot identity is positional and the caption addresses it "
                f"positionally, so an out-of-order list is a re-addressed control set."
            )
        prepared = images[position]
        if not isinstance(prepared, QwenEditPreparedImage):
            raise TypeError(
                f"[qwen-edit-encode] slot {position} image is a {type(prepared).__name__}, not a "
                f"QwenEditPreparedImage. Every image on this path goes through "
                f"prepare_qwen_edit_image, which resizes ONCE at a DECLARED budget and carries the "
                f"resulting row count with the pixels."
            )
        if bool(prepared.blank) != bool(slot.blank):
            raise ValueError(
                f"[qwen-edit-encode] slot {position} is resolved blank={slot.blank} but its "
                f"prepared image is blank={prepared.blank}. A blank slot must be the RENDERED fill "
                f"and a real slot must be the file — swapping them trains the adapter on the wrong "
                f"pixels while every count still agrees."
            )

        pixels = qwen_edit_pixels_from_image(prepared.image)
        latents = encode_qwen_edit_latents(
            vae,
            pixels,
            latents_mean,
            latents_std,
            posterior=QWEN_EDIT_CONTROL_POSTERIOR,
            seed=seed,
        )
        entry = {
            # The SLOT DESCRIPTOR first. Without slot / stem / path / blank / fill the entry is a
            # sized tensor with no identity, and ``resolve_control_slots`` can neither order it nor
            # check it against the target — the D-10-DEF-2 gap, in this family's terms.
            "slot": int(slot.index),
            "stem": str(slot.stem),
            "path": slot.path,
            "blank": bool(slot.blank),
            "fill": slot.fill,
            **_latent_payload(latents, prepared, f"control slot {position}"),
        }
        encoded.append(entry)

    payload: dict[str, Any] = {
        "controls": encoded,
        "num_controls": len(encoded),
        "control_slots": int(control_slots),
        "stem": str(stem),
    }
    # Allowlist ALIAS — slot 0 only. See the Returns note.
    payload["latents"] = encoded[0]["latents"]
    payload["num_frames"] = encoded[0]["num_frames"]
    payload["height"] = encoded[0]["height"]
    payload["width"] = encoded[0]["width"]
    return payload


# --------------------------------------------------------------------------------------------------
# The Qwen2.5-VL channel: presentation, cache key, encode.
# --------------------------------------------------------------------------------------------------


def build_qwen_edit_presentation(caption: str, condition_images: int) -> str:
    """The exact string handed to the processor — ``template.format(picture_blocks + caption)``.

    Transcribed from ``pipeline_qwenimage_edit_plus.py:240-253``: one 1-indexed
    ``"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>"`` per condition image, concatenated
    with no separator, prepended to the caption, and the whole thing dropped into the edit-plus
    template's single ``{}``.

    ⚠ The count and the ORDER are load-bearing twice over. ``Picture 1`` is the caption's
    ``ctrl_img_1``, and the processor pairs the i-th vision block with the i-th image in the list it
    is handed — so a presentation built for 3 images and a list of 2 mis-pairs every block after the
    gap, which is the same class of silent re-addressing that :func:`resolve_qwen_edit_control_sources`
    refuses on the filesystem side.
    """
    if condition_images < 0:
        raise ValueError(f"[qwen-edit-encode] condition_images must be >= 0, got {condition_images}.")
    blocks = "".join(
        QWEN_EDIT_IMG_PROMPT_TEMPLATE.format(i + 1) for i in range(int(condition_images))
    )
    return QWEN_EDIT_PROMPT_TEMPLATE.format(blocks + caption)


def qwen_edit_control_identity(
    path: Any, *, mode: str, data: bytes | None = None, chunk: int = 1 << 20
) -> str:
    """The per-control identity that goes INTO the text cache key. ``'content'`` or ``'path'``.

    ⚠ **``'path'`` is bug-compatibility and nothing else**, exactly as
    ``QwenEditConfig.control_cache_key_mode`` says (``config/schema.py:650-658``): ai-toolkit hashes
    the control PATH STRING, so overwriting a control file IN PLACE reuses a STALE embedding — and
    overwriting control images in place is precisely the chained-edit workflow this family exists
    for. ``'content'`` hashes the bytes and is the default.

    ``data`` lets a caller that already holds the bytes hand them over, which is also what makes the
    content mode provable on CPU with no filesystem. It is deliberately **bytes rather than an
    injected reader callable**: ``h3_grad_contract.model_forward_sites`` classifies "a parameter you
    were handed is being invoked" as a model forward and flagged the reader version of this function
    — correctly, by its own rule. Laundering the call through a local would have silenced the scan
    for real forwards too, so the parameter stopped being callable instead.
    """
    if mode not in ("content", "path"):
        raise ValueError(
            f"[qwen-edit-encode] invalid control cache key mode {mode!r}: expected 'content' or "
            f"'path' (config/schema.py QwenEditConfig.control_cache_key_mode)."
        )
    text = str(path)
    if mode == "path":
        # Normalized separators so the SAME file does not get two identities on two platforms.
        return hashlib.sha256(text.replace("\\", "/").encode("utf-8")).hexdigest()

    digest = hashlib.sha256()
    if data is not None:
        digest.update(data)
        return digest.hexdigest()
    with Path(text).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def qwen_edit_text_cache_key(
    *,
    caption: str,
    controls: Any,
    condition_area_px: int,
    text_encoder_id: str,
    template: str = QWEN_EDIT_PROMPT_TEMPLATE,
    drop_idx: int = QWEN_EDIT_PROMPT_TEMPLATE_START_IDX,
) -> str:
    """The text-embedding cache key. **It MUST include the control images.** This is the whole point.

    ⛔ **A caption-only key is a documented footgun, and on this family it is worse than documented.**
    ``qwen_image_edit_plus.py:66`` sets ``encode_control_in_text_embeddings=True``: every control
    image is rendered through the Qwen2.5-VL tower at ``CONDITION_IMAGE_SIZE`` and its visual tokens
    land INSIDE ``prompt_embeds`` (``pipeline_qwenimage_edit_plus.py:240-268``; the tokens sit after
    the 64-token drop, so they survive it). The cached "text" embedding is therefore a function of
    the control PIXELS, not only of the caption.

    So the failure a caption-only key produces is not "a slightly stale prompt". Round 2 of a chain
    overwrites the control images and keeps the captions — the deliberate workflow this family is
    built for — and a caption-keyed cache would hand round 2 **round 1's control images**, encoded,
    inside the text stream, while the VAE channel correctly carries round 2's. The two channels
    would describe different images to the same forward pass, at a perfectly valid shape, for the
    whole run.

    The key also covers ``condition_area_px`` (the same file at a different budget is different
    pixels), the TEMPLATE and ``drop_idx`` (both change what the encoder is asked, and the template
    is edit-plus-specific), the text encoder identity, and the payload VERSION.

    ⚠ **The key is stored IN the payload, not in the filename.** ``PrecomputedDataset`` pairs
    sources by relative path (``data/precomputed.py:207-238``), so a hash-named file would never
    pair with its latents — the sample would silently vanish from the index. The key is therefore
    a value the incremental gate compares (:func:`qwen_edit_text_cache_is_current`), and the file
    name stays ``<rel>.pt``.

    Args:
        controls: one entry per slot, in slot order. Two forms are accepted:

            * a **mapping** carrying ``slot`` / ``blank`` / ``fill`` / ``identity`` / ``encoded_wh``
              — the rich form, which additionally keys on the resized geometry;
            * a **string** — an opaque per-slot identity at its list position, which is what
              ``modal/fns.py::qwen_edit_preprocess`` builds (``qwen_edit_control_identity(...)``
              for a real slot, ``"blank:<fill>"`` for a synthetic one).

            The string form is accepted rather than refused because the load-bearing property is
            the same in both: **the key changes when a control image changes.** It is lossy only in
            ``encoded_wh``, and the two forms produce different digests for the same corpus — a
            caller that switches forms pays exactly one re-encode, which is the correct outcome for
            a changed key composition and not a silent reuse.

            ``identity`` comes from :func:`qwen_edit_control_identity`; a blank slot has no file and
            contributes its FILL instead, which is what makes it distinguishable from a different
            blank.
    """
    entries = []
    for position, control in enumerate(controls):
        if isinstance(control, str):
            # The flattened form. ``blank:<fill>`` is the ONE structured prefix, so a blank slot
            # stays distinguishable from a real one whose content hash happens to be short.
            blank = control.startswith("blank:")
            control = {
                "slot": position,
                "blank": blank,
                "fill": control.split(":", 1)[1] if blank else None,
                "identity": None if blank else control,
            }
        index = control.get("slot", position)
        blank = bool(control.get("blank", False))
        identity = control.get("identity")
        if not blank and not identity:
            raise ValueError(
                f"[qwen-edit-encode] control slot {index} contributes no identity to the text cache "
                f"key. A non-blank slot MUST carry one (qwen_edit_control_identity): its pixels are "
                f"encoded into prompt_embeds, so a key that omits them cannot tell round 1's "
                f"controls from round 2's — and the chained-edit workflow overwrites controls in "
                f"place while keeping the caption."
            )
        size = control.get("encoded_wh")
        entries.append(
            {
                "slot": int(index),
                "blank": blank,
                "fill": control.get("fill") if blank else None,
                "identity": None if blank else str(identity),
                "wh": [int(size[0]), int(size[1])] if size is not None else None,
            }
        )
    entries.sort(key=lambda entry: entry["slot"])

    material = {
        "version": QWEN_EDIT_TEXT_PAYLOAD_VERSION,
        "caption": str(caption),
        "template": str(template),
        "drop_idx": int(drop_idx),
        "text_encoder_id": str(text_encoder_id),
        "condition_area_px": int(condition_area_px),
        "controls": entries,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def qwen_edit_text_cache_is_current(payload: Any, key: str) -> bool:
    """Is this cached text payload the one ``key`` describes? ``False`` means RE-ENCODE.

    The incremental gate. It answers ``False`` — never raises — for an absent, unversioned,
    wrong-version or wrong-key payload, because every one of those is a re-encode and not an
    operator error.

    ⚠ **An "exists on disk" check is NOT this function**, and substituting one is the footgun. The
    file name is ``<rel>.pt`` for every round of a chain (it has to be — sources pair by relative
    path), so existence is true from round 1 onward and would skip every subsequent re-encode. The
    key is what distinguishes rounds.
    """
    if not isinstance(payload, dict):
        return False
    if int(payload.get(QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY, -1)) != QWEN_EDIT_TEXT_PAYLOAD_VERSION:
        return False
    return str(payload.get(_QWEN_EDIT_CACHE_KEY_KEY, "")) == str(key)


@h3_no_grad
def encode_qwen_edit_text_conditions(
    text_encoder: Any,
    processor: Any,
    caption: str,
    condition_images: Any = (),
    *,
    cache_key: str,
    drop_idx: int = QWEN_EDIT_PROMPT_TEMPLATE_START_IDX,
    max_length: int = QWEN_EDIT_TOKENIZER_MAX_LENGTH,
) -> dict[str, Any]:
    """Run Qwen2.5-VL over the presentation + condition images -> the ``qwen_edit_conditions/`` payload.

    Transcribes ``_get_qwen_prompt_embeds`` (``pipeline_qwenimage_edit_plus.py:228-283``) for the
    ONE-sample case:

    ::

        txt    = template.format(picture_blocks + caption)
        inputs = processor(text=[txt], images=condition_images, padding=True, return_tensors="pt")
        out    = text_encoder(input_ids, attention_mask, pixel_values, image_grid_thw,
                              output_hidden_states=True)
        hidden = out.hidden_states[-1]              # the LAST hidden state (:270)
        hidden = hidden[mask.bool()][drop_idx:]     # masked, then the 64-token prefix drop (:271-272)

    ⛔ **The VISION half of the text encoder is REQUIRED.** ``pixel_values`` and ``image_grid_thw``
    are passed at ``:266-267``, and the measured checkpoint carries 714 vision tensors for exactly
    this reason. A plain text LLM in this slot fails with ``mat1 and mat2 shapes cannot be
    multiplied (5376x1280 and 3840x1280)`` — a real failure this house already hit and fixed. The
    assertion below turns that into a sentence naming the component.

    ⚠ **``hidden_states[-1]``, not a penultimate layer.** Unlike H3 — where reading the final,
    post-norm layer is silently wrong (``prep/h3_encode.py:1529-1534``) — Qwen's edit pipeline
    genuinely takes the last one. The two families' text paths differ HERE, which is precisely the
    kind of place a copy-paste between them does damage.

    ⚠ **Zero-padding, not garbage-padding.** The pipeline pads short sequences with
    ``new_zeros`` (``:276, 279``) and the transformer never turns ``encoder_hidden_states_mask``
    into an attention mask — it is threaded to the attention processor and dropped
    (``transformer_qwenimage.py:281, 338, 446``), so padded text positions ARE attended and are
    harmless only because they are zeros. At batch 1 there is nothing to pad and the mask is
    all-ones; the property is stated so a future batched writer cannot pad with anything else.

    Returns:
        The ``qwen_edit_conditions/`` payload. ``prompt_embeds`` / ``prompt_embeds_mask`` are the two
        keys ``conditioning/qwen_edit._text_payload`` reads first (``:561, 571``); the rest is
        provenance, and ``cache_key`` is what :func:`qwen_edit_text_cache_is_current` compares.
    """
    images = list(condition_images)
    presentation = build_qwen_edit_presentation(caption, len(images))

    inputs = processor(
        text=[presentation],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = _processor_inputs_to_dict(inputs)

    if images:
        missing = [key for key in ("pixel_values", "image_grid_thw") if inputs.get(key) is None]
        if missing:
            raise RuntimeError(
                f"[qwen-edit-encode] {len(images)} condition image(s) were supplied but the "
                f"processor returned no {missing}. Qwen-Image-Edit's text encoder is Qwen2.5-VL and "
                f"the VISION half is load-bearing — the pipeline passes pixel_values and "
                f"image_grid_thw (pipeline_qwenimage_edit_plus.py:266-267), and the measured "
                f"checkpoint carries 714 vision tensors for it. A text-only encoder or a text-only "
                f"processor in this slot fails downstream as 'mat1 and mat2 shapes cannot be "
                f"multiplied (5376x1280 and 3840x1280)', which names a matmul and never mentions "
                f"the vision tower."
            )

    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise RuntimeError(
            "[qwen-edit-encode] the processor returned no input_ids / attention_mask. The masked "
            "extraction (pipeline_qwenimage_edit_plus.py:220-225) selects valid positions with the "
            "mask; without it the prefix drop would slice into padding."
        )
    tokenized = int(input_ids.shape[-1])
    if tokenized > int(max_length):
        raise ValueError(
            f"[qwen-edit-encode] the presentation tokenizes to {tokenized} token(s), over the "
            f"tokenizer_max_length of {max_length} (pipeline_qwenimage_edit_plus.py:214; "
            f"check_inputs refuses beyond it at :381-382). Shorten the caption or reduce the "
            f"condition-image count — a truncated presentation drops vision blocks off the END, "
            f"which silently re-addresses ctrl_img_N."
        )

    inputs = {key: _to_device(value, text_encoder) for key, value in inputs.items()}
    outputs = text_encoder(**inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "[qwen-edit-encode] the text encoder returned no hidden_states — it must be called with "
            "output_hidden_states=True (pipeline_qwenimage_edit_plus.py:268). Refusing to guess at "
            "the conditioning."
        )
    # The LAST hidden state (:270). Stated as an index off the end so it cannot be mistaken for
    # H3's EXPECTED_H3_TEXT_ENCODER_LAYER, which is a specific interior layer out of 64.
    hidden = hidden_states[-1]
    if hidden.dim() == 3:
        hidden = hidden[0]
    mask = attention_mask.to(hidden.device)
    if mask.dim() == 2:
        mask = mask[0]

    selected = hidden[mask.bool()]
    if int(selected.shape[0]) <= int(drop_idx):
        raise RuntimeError(
            f"[qwen-edit-encode] the encoded presentation is {int(selected.shape[0])} valid "
            f"token(s), which does not survive the {drop_idx}-token prefix drop "
            f"(pipeline_qwenimage_edit_plus.py:217, applied at :272). The drop removes the system "
            f"block through '<|im_start|>user\\n'; a sequence this short means the template was not "
            f"applied, or a different tokenizer is mounted. Refusing to write an empty or negative "
            f"conditioning."
        )
    embeds = selected[int(drop_idx) :].detach().to("cpu")
    # ⚠ Built as ONES over the KEPT length, exactly as the pipeline does (:273): the mask marks
    # VALID positions, and at batch 1 every kept position is valid. It is not sliced from the
    # processor's mask, which covers the pre-drop, pre-selection stream — reusing that one would
    # mis-align by drop_idx and be undetectable, because the lengths only differ by a constant.
    attention = torch.ones(int(embeds.shape[0]), dtype=torch.int64)

    return {
        QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: QWEN_EDIT_TEXT_PAYLOAD_VERSION,
        _QWEN_EDIT_CACHE_KEY_KEY: str(cache_key),
        # The two keys conditioning/qwen_edit._text_payload looks for FIRST (:561, :571).
        "prompt_embeds": embeds.unsqueeze(0),
        "prompt_embeds_mask": attention.unsqueeze(0),
        "n_condition_images": len(images),
        "drop_idx": int(drop_idx),
        "tokenized_length": tokenized,
        "text_length": int(embeds.shape[0]),
    }


def _processor_inputs_to_dict(inputs: Any) -> dict[str, Any]:
    """A ``BatchFeature``/mapping -> a plain dict, without assuming which one arrived."""
    if isinstance(inputs, dict):
        return dict(inputs)
    data = getattr(inputs, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if hasattr(inputs, "keys"):
        return {key: inputs[key] for key in inputs.keys()}  # noqa: SIM118 — BatchFeature, not a dict
    raise TypeError(
        f"[qwen-edit-encode] the processor returned {type(inputs).__name__}, which is neither a "
        f"mapping nor a BatchFeature. Refusing to guess which attribute holds input_ids."
    )


def _to_device(value: Any, model: Any) -> Any:
    """Move a processor output onto the model's device when it is a tensor; pass anything else."""
    if not isinstance(value, torch.Tensor):
        return value
    device = h3_component_device(model)
    return value if device is None else value.to(device)


# --------------------------------------------------------------------------------------------------
# The precomputed writer + the partial-cache guard.
# --------------------------------------------------------------------------------------------------


def write_qwen_edit_precomputed(
    output_root: str | Path,
    rel_path: str | Path,
    *,
    target: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    text: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the provided payloads under the three ``qwen_edit_``-prefixed sources at the SAME rel path.

    ``PrecomputedDataset`` pairs sources by RELATIVE PATH (``data/precomputed.py:207-238``), so every
    source a run configures must receive a file at the identical ``<rel>.pt``. A missing one
    **silently drops the whole sample from the index rather than raising** — which is why a partial
    write is legal here (an unchanged text payload is legitimately skipped, see
    :func:`qwen_edit_text_cache_is_current`) and why :func:`assert_qwen_edit_cache_complete` exists
    to close the stage.

    The two LATENT sources are asserted against the video-latent dict contract before they are
    written, not after — see :func:`qwen_edit_allowlist_gap` for why that contract matters even
    though the allowlist entry is not yet in place.
    """
    root = Path(output_root)
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError(
            f"[qwen-edit-encode] rel_path must be relative to the precomputed root, got {rel}."
        )
    if rel.suffix != ".pt":
        rel = rel.with_suffix(".pt")

    written: dict[str, Path] = {}
    for dir_name, payload in (
        (QWEN_EDIT_LATENTS_DIR, target),
        (QWEN_EDIT_CONTROL_LATENTS_DIR, controls),
        (QWEN_EDIT_CONDITIONS_DIR, text),
    ):
        if payload is None:
            continue
        if dir_name in (QWEN_EDIT_LATENTS_DIR, QWEN_EDIT_CONTROL_LATENTS_DIR):
            _assert_latent_payload(dir_name, payload)
        else:
            _assert_text_payload(dir_name, payload)
        destination = root / dir_name / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        written[dir_name] = destination

    logger.info("[qwen-edit-encode] wrote %d source(s) for %s -> %s", len(written), rel, root)
    return written


def _assert_latent_payload(dir_name: str, payload: Any) -> None:
    """Refuse to write a latent source the video-latent allowlist contract would mis-route."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"[qwen-edit-encode] {dir_name}/ payloads must be dicts carrying a 'latents' tensor, "
            f"got {type(payload).__name__}."
        )
    missing = [key for key in ("latents", "num_frames", "height", "width") if key not in payload]
    if missing:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ payload is missing {missing}. The committed shape is "
            f"{{'latents': [C, F, H, W], 'num_frames', 'height', 'width'}} — the same one "
            f"data/precomputed.py::_normalize_video_latents reads unconditionally for every "
            f"allowlisted latent source."
        )
    latents = payload["latents"]
    if not isinstance(latents, torch.Tensor) or latents.dim() != QWEN_EDIT_VAE_ENCODE_RANK - 1:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ 'latents' must be a 4-D [C, F, H, W] tensor, got "
            f"{type(latents).__name__} "
            f"{tuple(latents.shape) if isinstance(latents, torch.Tensor) else ''}. F is carried "
            f"EXPLICITLY at rest and squeezed only at the model boundary "
            f"(qwen_edit_geometry.qwen_edit_vae_latent_shape)."
        )
    if int(latents.shape[0]) != QWEN_EDIT_LATENT_CHANNELS:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ 'latents' has {int(latents.shape[0])} channel(s), "
            f"expected {QWEN_EDIT_LATENT_CHANNELS}. Qwen's VAE is not LTX's (128) and not H3's "
            f"(24) — writing this would corrupt the run silently."
        )
    if int(payload["num_frames"]) != QWEN_EDIT_FRAMES:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ declares num_frames={payload['num_frames']}; "
            f"qwen_edit is an IMAGE family and F is EXACTLY {QWEN_EDIT_FRAMES}."
        )
    height, width = int(payload["height"]), int(payload["width"])
    if height < 1 or width < 1:
        raise ValueError(f"[qwen-edit-encode] {dir_name}/ latent grid {height}x{width} is degenerate.")
    if (height, width) != tuple(int(v) for v in latents.shape[2:]):
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ records a {height}x{width} LATENT grid but the tensor "
            f"is {tuple(int(v) for v in latents.shape[2:])}. These are the LATENT dims, not the "
            f"pixel size (which travels as source_wh / encoded_wh) — and the metadata drives the "
            f"legacy patchified fixup, so a disagreement reshapes latents wrongly, silently."
        )
    if dir_name == QWEN_EDIT_CONTROL_LATENTS_DIR:
        controls = payload.get("controls")
        if not isinstance(controls, list) or not controls:
            raise ValueError(
                f"[qwen-edit-encode] {dir_name}/ payload carries no non-empty 'controls' list. The "
                f"per-slot data is a LIST and the top-level 'latents' key aliases slot 0 ONLY — a "
                f"consumer reading the alias would see one control where the sample carries "
                f"several (conditioning/qwen_edit._control_entries refuses exactly that)."
            )
        if controls[0].get("latents") is not payload["latents"]:
            raise ValueError(
                f"[qwen-edit-encode] {dir_name}/ top-level 'latents' is not slot 0's tensor. The "
                f"alias exists to satisfy the allowlist contract; if it ever pointed somewhere else "
                f"the allowlist would normalize a tensor no consumer reads."
            )


def _assert_text_payload(dir_name: str, payload: Any) -> None:
    """Refuse to write a text payload the strategy could not read, or the gate could not key."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"[qwen-edit-encode] {dir_name}/ payloads must be dicts carrying the Qwen2.5-VL "
            f"embeddings and their mask, got {type(payload).__name__}. A bare tensor is refused by "
            f"conditioning/qwen_edit._text_payload (:576-581) because txt_seq_lens is derived from "
            f"the MASK (qwen_image_edit_plus.py:329)."
        )
    embeds = payload.get("prompt_embeds")
    mask = payload.get("prompt_embeds_mask")
    if not isinstance(embeds, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ payload needs both 'prompt_embeds' and "
            f"'prompt_embeds_mask' tensors (keys: {sorted(payload)})."
        )
    if embeds.dim() != 3 or mask.dim() != 2 or embeds.shape[1] != mask.shape[1]:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ expects embeddings [1, L, dim] and mask [1, L], got "
            f"{tuple(embeds.shape)} and {tuple(mask.shape)}. txt_seq_lens is the mask's SUM, so a "
            f"length mismatch prices a text stream of one length and attends over another."
        )
    if mask.dtype != torch.int64:
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ mask dtype is {mask.dtype}, expected int64. The "
            f"transformer casts it to int64 and SUMS it (qwen_image_edit_plus.py:326-329); an "
            f"additive -inf float mask — the LTX convention — sums to -inf and yields a garbage "
            f"sequence length instead of an error (DIVERGE #2)."
        )
    if not payload.get(_QWEN_EDIT_CACHE_KEY_KEY):
        raise ValueError(
            f"[qwen-edit-encode] {dir_name}/ payload carries no {_QWEN_EDIT_CACHE_KEY_KEY!r}. The "
            f"key is the ONLY thing that distinguishes round N's conditioning from round N-1's — "
            f"the file name is <rel>.pt in every round, so without it the incremental gate degrades "
            f"to an existence check and every later round reuses round 1's embeddings, control "
            f"images included."
        )


def qwen_edit_realized_image_rows(
    target: dict[str, Any], controls: dict[str, Any]
) -> dict[str, Any]:
    """The rows this sample ACTUALLY encoded to — the belt to the config-load braces.

    ⛔ **The config-load price is not always the realized one, and the divergence is measurable
    here.** ``qwen_edit_geometry.qwen_edit_packed_layout`` (``:313-315``) prices every control slot
    at one ``per_slot`` figure and says why: *"an arbitrarily-sized control image is fitted to the
    budget before the VAE ever sees it, so slot cost does not vary with source resolution."* The
    AREA is indeed fitted — but each edge is then snapped independently to a multiple of 32
    (``_snap``, ``qwen_edit_geometry.py:285-293``), and the snap does not preserve the product.
    Measured on this tree, at ``control_area_px = 1024*1024``::

        1024x1024 control -> 1024x1024 -> latent 128x128 -> 4096 rows   (== the priced figure)
        1024x 512 control -> 1440x 736 -> latent  92x180 -> 4140 rows   (+44)

    So a NON-SQUARE corpus realizes a longer packed sequence than the dry-run banner reports. It is
    inert today — ``max_packed_rows`` defaults to 0 (ceiling DISABLED, because no OOM boundary has
    been measured for this model on any card in this program) — and it stops being inert the moment
    somebody measures one. This function is what lets the stage compare the two at the CHEAP step,
    the ``h3_preprocess`` "realized packed rows" precedent (``modal/fns.py``), instead of discovering
    it as an OOM in a metered training container.

    Returns:
        ``{"n_target", "n_control", "per_slot", "n_image_stream"}``. ``n_image_stream`` is the
        ``hidden_states`` row count — target rows FIRST, controls appended — and deliberately
        EXCLUDES the text stream, which is a separate tensor (``QwenEditPackedLayout.n_image_stream``
        draws the same line for the same reason).
    """
    n_target = int(target["latent_rows"])
    per_slot = [int(entry["latent_rows"]) for entry in controls["controls"]]
    return {
        "n_target": n_target,
        "n_control": sum(per_slot),
        "per_slot": per_slot,
        "n_image_stream": n_target + sum(per_slot),
    }


def qwen_edit_cache_census(output_root: str | Path) -> dict[str, set[str]]:
    """``{source_dir: {rel_path_as_posix, ...}}`` for the three sources. Missing dir -> empty set."""
    root = Path(output_root)
    census: dict[str, set[str]] = {}
    for dir_name in QWEN_EDIT_PRECOMPUTED_DIRS:
        source = root / dir_name
        census[dir_name] = (
            {path.relative_to(source).as_posix() for path in source.rglob("*.pt")}
            if source.exists()
            else set()
        )
    return census


def assert_qwen_edit_cache_complete(
    output_root: str | Path, expected_rels: Any = None
) -> dict[str, int]:
    """THE PARTIAL-CACHE GUARD. Refuse a cache where the three sources do not cover the same samples.

    ⛔ **This is the guard the H3 leg paid for twice.** ``PrecomputedDataset`` pairs sources by
    relative path and keeps only the samples present in EVERY configured source
    (``data/precomputed.py:164-238``), so a sample missing one file is **silently dropped from the
    index** — not an error, not a warning, just a smaller dataset. The H3 leg hit both halves of
    this: a source dir that was never created at all killed the first training dispatch
    (``modal/fns.py``'s "container #4, bought for nothing" note), and a run that died mid-encode
    left **88 partial payloads** on the Volume that a later reader had to be taught to refuse
    (``prep/h3_text_payload.py:33-35``). Both are the same defect — a cache that is complete
    ENOUGH to load and wrong.

    It is also the reason a partial write is allowed per-sample: the incremental gate legitimately
    skips a text payload whose cache key is unchanged, so completeness cannot be checked at the
    write site. It is checked HERE, once, before the stage reports success.

    ⚠ Nothing is deleted. A stale or partial tree is REFUSED, never auto-removed — the
    ``h3_text_payload`` rule: *"The stale tree is NOT deleted, it is refused."*

    Args:
        expected_rels: optional iterable of the rel paths the stage believes it processed. When
            given, a source that is missing any of them fails even if all three sources agree with
            each other — which catches the case where the stage skipped a whole sample silently.

    Returns:
        ``{source_dir: file_count}`` on success.
    """
    census = qwen_edit_cache_census(output_root)
    counts = {name: len(paths) for name, paths in census.items()}

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(
            f"[qwen-edit-encode] source(s) {empty} hold ZERO .pt files under {output_root} "
            f"(counts: {counts}). All three are MANDATORY — QwenEditStrategy.get_data_sources() "
            f"declares every one, and data/precomputed.py raises FileNotFoundError on a configured "
            f"source whose directory does not exist. Refusing to report a successful pre-encode."
        )

    union: set[str] = set().union(*census.values())
    incomplete = {
        rel: sorted(name for name, paths in census.items() if rel not in paths)
        for rel in sorted(union)
        if any(rel not in paths for paths in census.values())
    }
    if incomplete:
        sample = dict(list(incomplete.items())[:10])
        raise RuntimeError(
            f"[qwen-edit-encode] PARTIAL CACHE under {output_root}: {len(incomplete)} sample(s) are "
            f"present in some sources and missing from others (counts: {counts}). First few, with "
            f"the sources each is MISSING from: {sample}. PrecomputedDataset pairs by relative path "
            f"and keeps only samples present in EVERY source, so this would not raise at train "
            f"time — it would silently train on a smaller corpus than the one that was encoded. "
            f"Re-run the pre-encode for the listed samples. Nothing here is deleted."
        )

    if expected_rels is not None:
        wanted = {Path(str(rel)).with_suffix(".pt").as_posix() for rel in expected_rels}
        for name, paths in census.items():
            absent = sorted(wanted - paths)
            if absent:
                raise RuntimeError(
                    f"[qwen-edit-encode] {name}/ is missing {len(absent)} sample(s) the stage "
                    f"believes it processed, e.g. {absent[:10]}. The three sources agree with each "
                    f"other, so the pairing check above passed — this catches the other half: a "
                    f"whole sample skipped by every branch, which leaves a self-consistent cache "
                    f"that is simply smaller than the corpus."
                )

    logger.info("[qwen-edit-encode] cache complete: %s", counts)
    return counts


# --------------------------------------------------------------------------------------------------
# Two-phase VRAM discipline.
# --------------------------------------------------------------------------------------------------


def free_qwen_edit_text_encoder(text_encoder: Any = None, processor: Any = None) -> None:
    """Drop Qwen2.5-VL before the transformer is loaded — the two-phase VRAM invariant.

    ⛔ The 9.38 GiB fp8 text encoder and the 40.86 GiB bf16 transformer are a 50 GiB pair before a
    single activation, and the recipe quantizes both to qfloat8 precisely because the margin is
    thin. The DISCIPLINE is H3's (``prep/h3_encode.free_text_encoder``, itself copied from the
    proven Gemma free): move to CPU, drop the reference, ``gc.collect()``, ``empty_cache()``.

    ⚠ **The CALLER must also drop ITS OWN references.** Loader-owned CUDA storage allocated with
    ``assign=True`` is freed only when the LAST reference goes away — ``Module.to("cpu")`` alone does
    not release it (the 06-09 run-5 finding). This helper can only reach its own locals.
    """
    import gc  # noqa: PLC0415 — deliberate function-local, mirroring the H3/Gemma free

    for obj in (text_encoder, processor):
        if obj is None:
            continue
        mover = getattr(obj, "to", None)
        if mover is None:
            continue
        try:
            mover("cpu")
        except Exception:  # noqa: BLE001 — a best-effort move must never abort the free
            logger.debug("[qwen-edit-encode] could not move %s to cpu", type(obj).__name__)

    del text_encoder, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            "[qwen-edit-encode] freed the Qwen2.5-VL text encoder (two-phase VRAM discipline) — "
            "cuda allocated=%.2f GiB. The transformer is loaded ONLY after this point.",
            torch.cuda.memory_allocated() / 2**30,
        )
