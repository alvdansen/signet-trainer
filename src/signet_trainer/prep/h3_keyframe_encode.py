"""prep.h3_keyframe_encode — encode fl2va KEYFRAMES. The canvas rule, enforced in one place.

The sibling of ``encode_h3_reference_latents``, and the difference is the whole point:

    a REFERENCE never binds the target geometry — it is prepared at its OWN resolution
      (``reference_image_short_edge``, no area cap, upscaling intentional)

    a KEYFRAME **IS** the target geometry — it is *"put onto the target canvas"*
      (``encoders.py:237``), because it claims to BE a frame of the video being generated

⛔ THE SILENT FAILURE THIS MODULE EXISTS TO PREVENT
---------------------------------------------------
MiniMax-H3's contract is **vision tokens == condition latent rows** — the Qwen3-VL vision block and
the VAE latent block are /32 grids of the SAME image (``encoders.py:285`` fl2va, ``:509`` ref2va).

If a keyframe is sized at ``reference_image_short_edge`` (896 -> 1600x896 -> 1400 rows) instead of
the target canvas (1344x768 -> 1008 rows), then:

* the packed sequence still assembles — ``n_text`` is simply bigger;
* every row-count guard in ``conditioning/h3_fl2va.make_h3_fl2va_position_ids_fn`` still PASSES,
  because it checks the *latent* rows and never sees the vision block;
* and every sample silently violates the contract the released weights were trained under.

No shape error. No crash. Quality damage only, discovered — if ever — from renders. That is the
"correct shape, silently wrong" class this campaign keeps paying for, and the reason both encoders
are driven from ONE size decision here rather than two config fields that can drift apart.

⛔ A KEYFRAME IS A SPATIAL ENCODE, NOT A SLICE OF THE VIDEO ENCODE
------------------------------------------------------------------
The tempting shortcut is to take latent frame 0 out of the already-encoded 22-frame target. It is
wrong: the video encode chunks 17 pixel frames -> 5 latent frames, so its frame 0 aggregates
several pixel frames under a different posterior draw. A keyframe is a single-frame ``[3, 1, H, W]``
spatial encode (``encoders.py:361-366``), exactly as a reference image is. This module never
touches the video payload.

CONFINED-ADJACENT: ``torch`` + stdlib + ``conditioning/h3_geometry`` + ``conditioning/h3_fl2va``
(for the shared anchor-set contract) + the shared encode primitives. The pure-geometry half is
unit-testable on Windows with no VAE and no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from signet_trainer.conditioning.h3_geometry import H3_CANVAS_MULTIPLE, rows_of
from signet_trainer.conditioning.h3_fl2va import validate_keyframe_anchors
from signet_trainer.prep.h3_encode import (
    H3_KEYFRAME_ENCODE_SEED,
    _grid_vision_tokens,
    _vision_merge_length,
    encode_video_latents,
    imagenet_normalize,
)

__all__ = [
    "H3PreparedKeyframe",
    "keyframe_latent_rows",
    "prepare_h3_keyframe_images",
    "validate_keyframe_vision_grid",
]


@dataclass(frozen=True)
class H3PreparedKeyframe:
    """One keyframe, already resized to the TARGET CANVAS, plus the anchor it binds to.

    ``image`` is the resized pixel source. It is handed to BOTH the VAE and the Qwen3-VL processor,
    which is what makes the vision-token/latent-row identity structural rather than hoped for.
    """

    image: Any
    anchor: str
    canvas_height: int
    canvas_width: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.anchor not in ("first", "last"):
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {self.anchor!r}.")
        for name, value in (("canvas_height", self.canvas_height), ("canvas_width", self.canvas_width)):
            if value <= 0 or value % H3_CANVAS_MULTIPLE:
                raise ValueError(
                    f"{name}={value} must be a positive multiple of {H3_CANVAS_MULTIPLE}: the VAE "
                    f"grid and Qwen3-VL's smart_resize both snap to it, and an off-multiple canvas "
                    f"makes the two disagree by a partial patch."
                )

    @property
    def latent_rows(self) -> int:
        return rows_of(self.canvas_height, self.canvas_width)


def keyframe_latent_rows(canvas_height: int, canvas_width: int) -> int:
    """Rows one keyframe occupies — identical to one target latent frame, by construction."""
    return rows_of(canvas_height, canvas_width)


def prepare_h3_keyframe_images(
    images: Sequence[Any],
    anchors: Sequence[str],
    canvas_height: int,
    canvas_width: int,
    *,
    labels: Sequence[str] | None = None,
) -> list[H3PreparedKeyframe]:
    """THE single keyframe-resize site. Every consumer goes through here — VAE and processor alike.

    Mirrors ``prepare_h3_reference_images``'s discipline with the one substantive change: the size
    is the TARGET CANVAS, supplied by the caller from ``resolve_canvas_size(*target_aspect)``, and
    there is no ``short_edge`` because a keyframe has no independent resolution to have.

    The resize is LANCZOS and **aspect-preserving only if the source already matches the canvas
    aspect**. It does not crop or pad: the staged corpus applies ONE crop box per shot to all three
    images before this function ever sees them, precisely so the keyframes and the target share a
    framing. A source whose aspect disagrees with the canvas is therefore a STAGING bug and is
    refused here rather than silently squashed — teaching "the answer is a squashed reference" is a
    documented failure of this project.

    Idempotent: an already-prepared keyframe passes through after its canvas AND its anchor are
    checked — the anchor is the field the whole fl2va contract rests on, and re-checking only the
    canvas would let a re-prepare call silently flip ``("first", "last")`` to ``("last", "first")``
    on an already-resized pair (order is meaning, exactly as ``validate_keyframe_anchors`` enforces
    below).

    The anchor SET (count, membership, order, no duplicates) is validated up front via
    ``conditioning.h3_fl2va.validate_keyframe_anchors`` — the single definition of that contract,
    shared with the packing layer — so an illegal request is refused HERE, before any resize is
    paid for, rather than surviving the entire pre-encode to be caught only at training time.
    """
    images = list(images)
    anchors = validate_keyframe_anchors(anchors)
    if len(images) != len(anchors):
        raise ValueError(
            f"got {len(images)} keyframe image(s) and {len(anchors)} anchor(s); they are paired "
            f"positionally and the anchor binds the `<Picture i>` block, so a mismatch packs a "
            f"request whose caption and tensor disagree."
        )
    labels = list(labels) if labels is not None else [f"keyframe_{i}" for i in range(len(images))]

    prepared: list[H3PreparedKeyframe] = []
    for index, (image, anchor) in enumerate(zip(images, anchors, strict=True)):
        if isinstance(image, H3PreparedKeyframe):
            if (image.canvas_height, image.canvas_width) != (canvas_height, canvas_width):
                raise ValueError(
                    f"keyframe {index} was prepared at "
                    f"{image.canvas_width}x{image.canvas_height} but this call declares "
                    f"{canvas_width}x{canvas_height}. The canvas is written into the cached payload; "
                    f"a payload whose declared canvas is not the one its pixels were resized at is "
                    f"a cache nobody can re-derive."
                )
            if image.anchor != anchor:
                raise ValueError(
                    f"keyframe {index} was prepared at anchor {image.anchor!r} but this call "
                    f"declares {anchor!r}. The anchor binds the `<Picture i>` block the caption "
                    f"describes; re-preparing an already-resized keyframe under a different anchor "
                    f"would silently flip which end of the clip it is pinned to while its pixels — "
                    f"and every downstream row count — stay unchanged. ORDER IS MEANING."
                )
            prepared.append(image)
            continue

        source = image.convert("RGB") if hasattr(image, "convert") else image
        width, height = source.size
        source_aspect = width / height
        canvas_aspect = canvas_width / canvas_height

        # ⛔ THIS TRIPWIRE IS DELIBERATELY LOOSE, AND A TIGHT ONE IS A BUG.
        #
        # `resolve_canvas_size(16, 9)` snaps each axis to a multiple of 32 and lands on 1344x768 =
        # **1.75**, not true 16:9 = 1.7778. So a genuine 1920x1080 keyframe is legitimately 1.6%
        # "off" the canvas. A 1%-tolerance check here rejects correctly-staged data.
        #
        # More importantly it would be INCONSISTENT with the target: `_h3_read_video_rgb` LANCZOS-
        # resizes every clip to the canvas with no aspect check whatsoever. Keyframe and target must
        # get the SAME treatment — that consistency, not aspect purity, is what stops the model
        # learning "the answer is a squashed version of the condition". The real protection is the
        # staging rule (ONE crop box per shot applied to all three images), which runs upstream.
        #
        # What survives is a tripwire for a GROSS mismatch — a portrait keyframe against a landscape
        # canvas, say — which means the wrong file was staged, not that a resize is needed.
        if max(source_aspect, canvas_aspect) / min(source_aspect, canvas_aspect) > 1.20:
            raise ValueError(
                f"keyframe {index} ({labels[index]!r}) is {width}x{height} "
                f"(aspect {source_aspect:.4f}) against a {canvas_width}x{canvas_height} canvas "
                f"(aspect {canvas_aspect:.4f}) — off by more than 20%. That is not a resize, it is "
                f"the wrong file: keyframes and their target share one crop box per shot at staging "
                f"time, so they cannot disagree this much. Re-stage the shot."
            )
        if (width, height) != (canvas_width, canvas_height):
            from PIL import Image as PILImage  # noqa: PLC0415

            source = source.resize((canvas_width, canvas_height), PILImage.Resampling.LANCZOS)
        prepared.append(
            H3PreparedKeyframe(
                image=source,
                anchor=anchor,
                canvas_height=canvas_height,
                canvas_width=canvas_width,
                label=labels[index],
            )
        )
    return prepared


def validate_keyframe_vision_grid(
    prepared: Sequence[H3PreparedKeyframe], vision_grid_thw: Any, processor: Any
) -> None:
    """Assert Qwen3-VL's vision grid equals the keyframe's latent-row count. THE F3 GUARD.

    ``image_grid_thw`` comes back from ``processor.image_processor`` as ``[n_images, 3]`` of
    ``(t, h, w)`` in UNMERGED patch units (``patch_size`` 16) — **not** the merged /32 grid
    ``rows_of`` computes. Qwen3-VL's own realized vision-token count is
    ``image_grid_thw.prod() // merge_size**2`` (``encoders.py:509``; ``Qwen3VLProcessor.__call__``
    computes it the identical way), so this guard reuses ``prep/h3_encode.py``'s existing
    primitives — ``_vision_merge_length`` (merge size READ off the mounted processor, never a
    literal) and ``_grid_vision_tokens`` (the divide) — instead of restating the arithmetic as a
    bare ``t * h * w``. Omitting the ``// merge_size**2`` divide is exactly the D-10-DEF-4 class of
    defect this guard exists to catch, only turned inward: a bare product is 4x (``merge_size**2``)
    too large at ``merge_size=2``, so it REJECTS every correctly-prepared keyframe and would ACCEPT
    one mistakenly sized at half the canvas.

    If the processor re-snapped the image — because it was handed a raw file, or one sized at
    ``reference_image_short_edge`` instead of the canvas — the counts diverge and the packed
    sequence silently violates the base model's contract.

    Cheap, exact, and it runs before any training step. Call it in PHASE A, once per sample.
    """
    if vision_grid_thw is None:
        raise ValueError(
            "no image_grid_thw supplied. The vision-token/latent-row identity is a contract of the "
            "released weights and it is unverifiable without the processor's own grid — refusing "
            "to encode a sample whose text and video blocks cannot be proven to agree."
        )
    grids = vision_grid_thw.tolist() if hasattr(vision_grid_thw, "tolist") else list(vision_grid_thw)
    if len(grids) != len(prepared):
        raise ValueError(
            f"processor returned {len(grids)} vision grid(s) for {len(prepared)} keyframe(s)."
        )
    merge_length = _vision_merge_length(processor)
    for index, (keyframe, grid) in enumerate(zip(prepared, grids, strict=True)):
        t, h, w = (int(v) for v in grid)
        vision_tokens = _grid_vision_tokens((t, h, w), merge_length)
        expected = keyframe.latent_rows
        if vision_tokens != expected:
            raise ValueError(
                f"keyframe {index} ({keyframe.label!r}) breaks the vision-token/latent-row "
                f"identity: Qwen3-VL produced {vision_tokens} vision token(s) "
                f"(unmerged grid t={t} h={h} w={w}, merge_length={merge_length}) but the VAE will "
                f"emit {expected} latent row(s) for a {keyframe.canvas_width}x{keyframe.canvas_height} "
                f"canvas. MiniMax-H3 was trained with these EQUAL (encoders.py:285). The usual cause "
                f"is the keyframe being sized at reference_image_short_edge instead of the target "
                f"canvas — which assembles cleanly, passes every row-count guard, and is wrong on "
                f"every sample."
            )


def encode_h3_keyframe_latents(
    vae: Any,
    prepared: Sequence[H3PreparedKeyframe],
    latents_mean: Any,
    latents_std: Any,
    *,
    seed: int = H3_KEYFRAME_ENCODE_SEED,
) -> dict[str, Any]:
    """Encode prepared keyframes to the committed payload. SINGLE-FRAME SPATIAL encodes.

    Each keyframe goes through the video VAE as ``[3, 1, H, W]`` — one pixel frame, so the temporal
    chunker never engages and the result is exactly one latent frame. This is the same path a
    reference image takes; what differs is only the size the pixels arrived at.

    Reuses ``encode_video_latents`` so the recipe (imagenet-normalize over [0,1], SEEDED posterior
    sample, the deliberate fp16 round BEFORE per-channel normalization) has exactly one definition.
    Restating any of it here is how the four traps get re-introduced.
    """
    if not prepared:
        raise ValueError("no keyframes to encode; an fl2va sample carries 1 or 2.")

    rows: list[torch.Tensor] = []
    for keyframe in prepared:
        pixels = _as_uint8_chw(keyframe.image)
        latents = encode_video_latents(
            vae, pixels.unsqueeze(1), latents_mean, latents_std, seed=seed
        )
        if int(latents.shape[1]) != 1:
            raise ValueError(
                f"keyframe {keyframe.label!r} encoded to {int(latents.shape[1])} latent frames; a "
                f"single-frame spatial encode must produce exactly 1. A value >1 means the temporal "
                f"chunker engaged, i.e. this was handed video rather than one frame."
            )
        rows.append(latents)

    return {
        "keyframe_latents": rows,
        "anchors": [k.anchor for k in prepared],
        "canvas_height": prepared[0].canvas_height,
        "canvas_width": prepared[0].canvas_width,
        "latent_rows_each": prepared[0].latent_rows,
        "encode_seed": seed,
    }


def _as_uint8_chw(image: Any) -> torch.Tensor:
    """Pillow-like -> uint8 ``[3, H, W]`` in ``[0, 255]``. `imagenet_normalize` divides by 255."""
    import numpy as np  # noqa: PLC0415

    if isinstance(image, torch.Tensor):
        return image
    arr = np.array(image.convert("RGB") if hasattr(image, "convert") else image, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
