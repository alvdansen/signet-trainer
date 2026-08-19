"""prep.h3_encode — the signet-NATIVE MiniMax-H3 pre-encode (H3-03, Plan 10-07).

⛔ **There is NO canonical H3 encoder anywhere** — not in this repo, not upstream. The LTX path
calls ltx-trainer's verified ``process_dataset.preprocess_dataset`` (``modal/fns.py::preprocess``);
MiniMax-H3 ships no training-side pre-encode at all, and neither community H3 trainer trains Ref2VA
(``P10-0d-REF2VA-PARTITION.md`` section 1.1). So this module is signet-native **by necessity**, not by
preference — the enochiatron "never write a custom encoder" landmine (RESEARCH Pitfall 1) does not
apply because there is no canonical encoder to prefer.

Import tier — **Modal-side EXCEPTION** (10-PATTERNS "Anti-Pattern 6", tier 3)
---------------------------------------------------------------------------
This module MAY import the heavy backends (``diffusers`` / ``transformers`` / ``torchvision`` /
``numpy`` / Pillow), but **ONLY function-local**. ``torch`` is the sole module-scope third-party
import. The reason is concrete, not stylistic: merely importing this module to read its parity
CONSTANTS — which ``tests/test_h3_preprocess_wiring.py`` and the CPU dry-run gate both do — must
never require a backend to be installed on Windows/CI. ``modal`` is never imported here at all: the
thin ``@app.function`` wrapper lives in ``modal/fns.py`` (plan 10-10) and calls into this module, so
the encode logic stays source-scannable on a CPU box with zero spend.

The four silent-corruption traps this module exists to get right
---------------------------------------------------------------
Each degrades **silently** — valid shapes, plausible tensors, off-distribution conditioning:

1. **The text-encoder layer.** H3 reads the Qwen3-VL hidden state at index
   ``EXPECTED_H3_TEXT_ENCODER_LAYER`` out of its 64 layers. The FINAL layer is post-norm and is NOT
   the conditioning the released weights were trained against; a final-layer encode is wrong at the
   correct ``(B, L, 5120)`` shape. The index is IMPORTED from ``models/h3_loader.py`` — there is
   deliberately no literal for it in this file.
2. **ImageNet pixel normalization over a ``[0, 1]`` base** (``div(255.0)`` FIRST, then mean/std) —
   **NOT** the ``[-1, 1]`` convention most video VAEs in this stack use.
3. **A FIXED-seed posterior sample** (``H3_KEYFRAME_ENCODE_SEED``), independent of the run seed.
4. **Two posterior policies in ONE pipeline** — image/video references and target video are
   ``.sample()``d; reference SOUNDTRACKS take the posterior ``.mode()`` and are never sampled.

Plus a fifth, easy to "clean up" by accident: the sampled latent is rounded through ``float16``
(~11 bits of mantissa) **before** the per-channel latent normalization. That is a deliberate parity
step — the released model's conditioning cannot be reproduced without it.

All five are pinned by a comment-stripped source scan in ``tests/test_h3_preprocess_wiring.py`` so a
later refactor cannot quietly drop one.

Provenance
----------
``P10-0d-REF2VA-PARTITION.md`` sections 3.4 / 5.1-5.5 (line-verified against the reference
implementation's ``encoders.py:102-136``, ``before_encoder.py:490-492``, ``packing_ref2va.py``) and
``P10-1-MEASURED.md`` sections 6-7 (measured / re-read at the pinned diffusers SHA). Geometry math
(the ``17n+5`` frame law, reference sizing, canvas) is NOT re-derived here — it is imported from
``conditioning/h3_geometry.py``. Architecture numbers are NOT re-derived here — they are imported
from ``models/h3_loader.py``.

Output contract (COMMITTED by plan 10-04 — a divergence here is silent downstream corruption)
---------------------------------------------------------------------------------------------
::

    <output_root>/h3_latents/<rel>.pt            {"latents": [C=24, F, H, W], "num_frames",
                                                  "height", "width"}   — ALLOWLISTED
    <output_root>/h3_conditions/<rel>.pt          Qwen3-VL hidden state, (L, 5120)
    <output_root>/h3_reference_latents/<rel>.pt   {"latents": [24, 1, H, W], ..., "references": [...]}
                                                  EXACTLY 2 refs/sample — ALLOWLISTED. Every entry
                                                  of "references" carries its DESCRIPTOR
                                                  (path / kind / subject_id) alongside its sizes:
                                                  conditioning/h3_ref requires all three and none
                                                  is recoverable from a latent (D-10-DEF-2).
    <output_root>/h3_audio_latents/<rel>.pt       audio-VAE latents (stereo) or an absent marker

``PrecomputedDataset`` pairs the four sources by RELATIVE PATH and is source-count-agnostic.
``h3_latents`` / ``h3_reference_latents`` are in ``data/precomputed.py::_VIDEO_LATENT_SOURCE_DIRS``,
which is only CORRECT because of the dict shape above; ``h3_conditions`` / ``h3_audio_latents``
deliberately are not.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch

from signet_trainer.conditioning.h3_geometry import (
    H3_CANVAS_MULTIPLE,
    H3_FRAMES_PER_CHUNK,
    H3_LATENTS_PER_CHUNK,
    H3_PHASE10_REFERENCES_PER_SAMPLE,
    h3_audio_rows,
    h3_latent_frames,
    reference_image_size,
    rows_of,
)

# The reference KINDS are owned by ``conditioning/h3_ref`` (the module whose ordering rule consumes
# them); this module imports rather than restating them, so the writer can never accept a kind the
# reader will reject. ``h3_ref`` is torch + stdlib + intra-repo at module scope — importing it here
# pulls no backend and keeps the "constants readable on a CPU box" property this module relies on.
from signet_trainer.conditioning.h3_ref import H3_REFERENCE_KINDS
from signet_trainer.models.h3_loader import (
    EXPECTED_H3_AUDIO_IN_CHANNELS,
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_TEXT_ENCODER_LAYER,
)

# The VAE's INPUT contract (D-10-DEF-9) lives in its own module because the test suite's stub has to
# obey exactly the same refusal — a stub that is more permissive than the component it stands in for
# is what let a 4-D tensor reach a metered container in the first place. torch + intra-repo only, so
# importing it pulls no backend.
from signet_trainer.prep.h3_vae_contract import (
    H3_VAE_FRAME_AXIS,
    H3_VAE_UNBATCHED_RANK,
    assert_h3_encode_device,
    assert_h3_vae_encode_input,
    h3_component_device,
)

# The RUNTIME IDIOM contract (D-10-DEF-10). ``.eval()`` is not ``no_grad``, and a diffusers component
# arrives with requires_grad=True on every parameter — so an encode outside a no-grad context builds
# and retains a graph for a backward that never comes (~78 GiB on a tiled 1344x768 clip). Every
# ``encode_*`` entry point below carries ``@h3_no_grad``, and the set of entry points is COMPUTED by
# ``h3_encode_entry_points`` rather than listed, so a helper added later cannot inherit the gap.
from signet_trainer.prep.h3_grad_contract import h3_no_grad

logger = logging.getLogger(__name__)

__all__ = [
    "H3_AUDIO_LATENTS_DIR",
    "H3_CONDITIONS_DIR",
    "H3_IMAGE_PAD_TOKEN",
    "H3_KEYFRAME_ENCODE_SEED",
    "H3_NOMINAL_LABEL_TOKENS",
    "H3_PIXEL_MEAN",
    "H3_PIXEL_STD",
    "H3_QWEN_TEMPORAL_PATCH",
    "H3_QWEN_VIDEO_SAMPLE_FPS",
    "H3_REFERENCE_LATENTS_DIR",
    "H3_VIDEO_LATENTS_DIR",
    "H3_VIDEO_PAD_TOKEN",
    "H3_VISION_END_TOKEN",
    "H3_VISION_START_TOKEN",
    "H3PreparedReference",
    "H3PresentationRef",
    "build_h3_presentation",
    "build_h3_processor_inputs",
    "encode_h3_audio_latents",
    "encode_h3_reference_latents",
    "encode_h3_text_conditions",
    "encode_h3_video_latents",
    "encode_video_latents",
    "free_text_encoder",
    "h3_absent_audio_payload",
    "h3_precomputed_complete",
    "imagenet_normalize",
    "prepare_h3_reference_images",
    "presentation_refs_from_prepared",
    "snap_reference_frames",
    "write_h3_precomputed",
]

# --------------------------------------------------------------------------------------------------
# Parity constants. Each carries its first-party / line-verified citation. NONE of these is a
# tunable: they are checkpoint contracts. A config knob that changes one is a bug, not a feature.
# --------------------------------------------------------------------------------------------------

#: ⚠⚠ **ImageNet normalization over a ``[0, 1]`` base — NOT ``[-1, 1]``.**
#: ``pixels.div(255.0)`` happens FIRST, then ``(x - mean) / std``. Most video VAEs in this stack
#: (LTX's included) are ``[-1, 1]``, so the wrong convention here produces perfectly-shaped latents
#: that are silently off-distribution — it DEGRADES rather than crashes, which is why it is a named
#: constant guarded by a source scan instead of an inline literal.
#: First-party: ``Ref2VA/video_vae/normalize.py`` + ``"pixel_norm_type": "imagenet"`` in the video
#: VAE config; transcribed at ``P10-0d-REF2VA-PARTITION.md`` sections 5.4 (trap 1) and 5.5.
H3_PIXEL_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
H3_PIXEL_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: ⚠ The conditioning VAE posterior is sampled under a **FIXED** seed, deliberately INDEPENDENT of
#: the run seed / request generator (``MINIMAX_H3_KEYFRAME_ENCODE_SEED``, ``encoders.py:102-136``;
#: P10-0d section 3.5 + 5.4 trap 2). A trainer that seeds this from the run seed is off-distribution
#: on every sample, and nothing about the tensor shape or dtype reveals it.
H3_KEYFRAME_ENCODE_SEED: int = 42

#: Qwen3-VL sees a reference VIDEO at 2 fps and merges frames in PAIRS, while the video VAE sees the
#: full 24 fps rate — two different sampling rates on the same reference, by design
#: (``MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS`` / ``MINIMAX_H3_QWEN_TEMPORAL_PATCH``, P10-0d section 5.2).
H3_QWEN_VIDEO_SAMPLE_FPS: float = 2.0
H3_QWEN_TEMPORAL_PATCH: int = 2

#: Qwen3-VL vision-block delimiters. A vision block is ``<|vision_start|> pad*N <|vision_end|>`` and
#: every row inside it is tagged 0 (VIDEO), even though it lives in the TEXT stream — the sentinels
#: are tagged 0 as well (P10-1-MEASURED section 7, ``adaln_tags``).
H3_VISION_START_TOKEN: str = "<|vision_start|>"
H3_VISION_END_TOKEN: str = "<|vision_end|>"
H3_IMAGE_PAD_TOKEN: str = "<|image_pad|>"
H3_VIDEO_PAD_TOKEN: str = "<|video_pad|>"

#: The M-RoPE modality mask ``ProcessorMixin.__call__`` returns alongside ``input_ids``, and the
#: three non-text values it uses (``create_mm_token_type_ids``: text 0, image 1, video 2, audio 3).
#: ``Qwen3VLModel.compute_3d_position_ids`` RAISES when multimodal grids arrive without this key —
#: which is how D-10-DEF-7 surfaced, on a metered container. The values are named rather than
#: written as literals because ``_assert_mm_token_type_ids`` reasons about all four.
H3_MM_TOKEN_TYPE_IDS_KEY: str = "mm_token_type_ids"
H3_MM_TYPE_TEXT: int = 0
H3_MM_TYPE_IMAGE: int = 1
H3_MM_TYPE_VIDEO: int = 2
H3_MM_TYPE_AUDIO: int = 3

#: NOMINAL token length billed for ONE emitted text segment (a ``<Picture i>: `` / ``<Audio j>: ``
#: label, or a ``<t seconds>`` timestamp) when no tokenizer is supplied to
#: ``build_h3_presentation``. It is the SAME figure ``h3_packed_seq_len`` bills per reference
#: (``n_text = prompt_tokens + sum(vision_tokens + 2) + 6 * len(references)``), so nominal spans and
#: the VRAM budget agree by construction. Pass ``tokenizer=`` for exact spans.
H3_NOMINAL_LABEL_TOKENS: int = 6

#: The four precomputed source directories, committed as convention by plan 10-04. The ``h3_``
#: prefix is deliberate: it keeps the LTX cache byte-identical and makes it impossible for an H3
#: cache to be paired into an LTX run.
H3_VIDEO_LATENTS_DIR: str = "h3_latents"
H3_CONDITIONS_DIR: str = "h3_conditions"
H3_REFERENCE_LATENTS_DIR: str = "h3_reference_latents"
H3_AUDIO_LATENTS_DIR: str = "h3_audio_latents"


# --------------------------------------------------------------------------------------------------
# Pixel + latent normalization.
# --------------------------------------------------------------------------------------------------


def _channel_dim(tensor: torch.Tensor) -> int:
    """Channel axis of a video tensor: 1 for ``[B, C, F, H, W]``, 0 for ``[C, F, H, W]``."""
    if tensor.dim() == 5:
        return 1
    if tensor.dim() == 4:
        return 0
    raise ValueError(
        f"expected a 4-D [C, F, H, W] or 5-D [B, C, F, H, W] video tensor, got shape "
        f"{tuple(tensor.shape)}."
    )


def _broadcast_channel_vector(values: Any, reference: torch.Tensor, what: str) -> torch.Tensor:
    """Reshape a per-channel vector so it broadcasts against ``reference``'s channel axis."""
    vector = torch.as_tensor(values, dtype=reference.dtype, device=reference.device).flatten()
    axis = _channel_dim(reference)
    channels = reference.shape[axis]
    if vector.numel() not in (1, channels):
        raise ValueError(
            f"{what} has {vector.numel()} element(s) but the tensor has {channels} channel(s). "
            f"MiniMax-H3's video VAE ships per-channel vectors of length "
            f"{EXPECTED_H3_IN_CHANNELS} in its config (latents_mean / latents_std)."
        )
    shape = [1] * reference.dim()
    shape[axis] = vector.numel()
    return vector.reshape(shape)


def imagenet_normalize(pixels: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize RGB pixels over a ``[0, 1]`` base — the H3 video VAE's pixel convention.

    The exact recipe, and the ORDER is the trap::

        pixels.to(torch.float32).div(255.0)      # <- FIRST: uint8 [0,255] -> float [0,1]
        (x - H3_PIXEL_MEAN) / H3_PIXEL_STD       # <- THEN: ImageNet mean/std

    ⚠ Reordering these (mean/std before the ``div``), or substituting the ``[-1, 1]`` convention
    that most video VAEs in this stack use, produces valid-shaped latents that are silently
    off-distribution. There is no shape error and no crash to catch it — this is trap 2 of the four
    (``encoders.py:102-136``; P10-0d section 5.4).

    Args:
        pixels: RGB video/image pixels in ``[0, 255]``, ``[C, F, H, W]`` or ``[B, C, F, H, W]``,
            any dtype (uint8 is the expected input).

    Returns:
        A float32 tensor of the same shape, ImageNet-normalized.
    """
    scaled = pixels.to(torch.float32).div(255.0)
    mean = _broadcast_channel_vector(H3_PIXEL_MEAN, scaled, "H3_PIXEL_MEAN")
    std = _broadcast_channel_vector(H3_PIXEL_STD, scaled, "H3_PIXEL_STD")
    return (scaled - mean) / std


@h3_no_grad
def encode_video_latents(
    vae: Any,
    pixels: torch.Tensor,
    latents_mean: Any,
    latents_std: Any,
    *,
    seed: int = H3_KEYFRAME_ENCODE_SEED,
    posterior: str = "sample",
) -> torch.Tensor:
    """The H3 video-VAE encode recipe, transcribed line-for-line from ``encoders.py:102-136``.

    ::

        pixels    = (pixels.to(float32).div(255.0) - pixel_mean) / pixel_std
        posterior = vae.encode(pixels, return_dict=False)[0]
        latents   = posterior.sample(generator=torch.Generator().manual_seed(42))
        latents   = latents.to(torch.float16).float()
        return (latents - latents_mean) / latents_std

    Every line is a trap, which is why this function is short and has no options beyond the two the
    architecture actually varies:

    * ``seed`` — the FIXED keyframe encode seed, NOT the run seed (trap 3). The generator is a CPU
      generator on purpose: it makes the draw reproducible regardless of which device the pixels
      live on (``randn_tensor`` generates on CPU and moves).
    * ``posterior`` — ``"sample"`` for images / video / target video, ``"mode"`` for reference
      SOUNDTRACKS only (trap 4). Two policies in one pipeline; see ``encode_h3_audio_latents``.

    ⚠ The ``float16`` round-trip is **deliberate**, not sloppiness: it rounds away ~11 bits of every
    conditioning latent's mantissa, and the released model's conditioning cannot be reproduced
    without it. It happens **BEFORE** the per-channel normalization — moving it after changes the
    result. Do not "clean this up".

    ⛔ **The VAE itself takes 5-D ``[B, C, F, H, W]`` and nothing else** (D-10-DEF-9). This function
    keeps the 4-D half of its documented contract by adding the batch axis before ``vae.encode`` and
    removing it again from the result — which is why "same rank as ``pixels``" below is true rather
    than merely advertised. Both producers on this path are 4-D (``_h3_read_video_rgb`` returns
    ``[3, F, H, W]``; ``_reference_pixels`` returns ``[3, 1, H, W]``), and handing either straight to
    the VAE made ``_encode`` read the canvas HEIGHT as the frame count. See
    ``prep/h3_vae_contract.py`` for why the resulting ``torch.cat`` error described the wrong thing.

    ⛔ **``@h3_no_grad`` is load-bearing, not hygiene** (D-10-DEF-10). The VAE's parameters arrive
    with ``requires_grad=True`` and ``.eval()`` does not change that, so without the decorator
    ``vae.encode`` builds and RETAINS an autograd graph for a backward that never comes. Tiling is on
    (256px tiles), so a 1344x768 canvas at 22 frames is ~36-42 encoder forwards per sample, each
    holding full float32 activations — the retained sum exhausted an 80 GiB A100 on a 544 MiB
    allocation. This is the chokepoint BOTH pixel producers route through, which is what makes one
    decorator sufficient here, exactly as it made one rank refusal sufficient above.

    ⛔ **The PIXELS are placed on the VAE's own device HERE** (D-10-DEF-12), for the third time at
    this same chokepoint and for the same reason: it is the one place both producers pass through.
    ``h3_preprocess`` used to write ``pixels.to("cuda")`` at the target-video call site and nothing
    at the reference one — ``_reference_pixels`` builds its tensor with ``torch.from_numpy``, on CPU
    — so a CPU tensor reached a CUDA-resident VAE, cuDNN declined it, and the dispatcher fell
    through to ``ConvBackend.Slow3d``, which has no CUDA kernel. The device comes off the component
    (``h3_component_device``), never from a literal ``"cuda"``: this function is handed a VAE it did
    not construct and has no business having an opinion about where it lives.

    ⚠ The move happens BEFORE ``imagenet_normalize``, which is not incidental. It is exactly what
    the target-video path already did and what attempt 5 proved on a real A100 — uint8 crosses the
    bus instead of float32 (4x less), and the normalization runs where the VAE is. Precision is
    untouched: ``imagenet_normalize`` builds its mean/std vectors on the reference tensor's own
    device, and float32 sub/div are correctly-rounded elementwise ops, so the values are the ones
    the video path has always produced.

    Args:
        vae: The H3 video VAE (a diffusers ``AutoencoderKL``-shaped module). Passed in, never
            constructed here, so this function needs no backend import.
        pixels: RGB pixels in ``[0, 255]``, ``[C, F, H, W]`` or ``[B, C, F, H, W]``.
        latents_mean: The 24-element per-channel ``latents_mean`` from ``video_vae/config.json``.
        latents_std: The matching per-channel ``latents_std``.
        seed: Fixed posterior seed. Defaults to ``H3_KEYFRAME_ENCODE_SEED``.
        posterior: ``"sample"`` (default) or ``"mode"``.

    Returns:
        Normalized latents, same rank as ``pixels``.
    """
    # ⛔ D-10-DEF-12. Placement is decided ONCE, here, off the component itself — not at the two call
    # sites, where it was decided once and forgotten once. A `None` device means the component holds
    # no tensors at all (nothing to disagree with), and moving to a guessed device would be worse
    # than leaving the pixels where they are.
    device = h3_component_device(vae)
    if device is not None:
        pixels = pixels.to(device)

    normalized = imagenet_normalize(pixels)

    # ⛔ D-10-DEF-9. A reference arrives as [3, 1, H, W] and a target clip as [3, F, H, W]; the VAE
    # requires [B, C, F, H, W]. Batching HERE (rather than at each call site) is what makes the
    # 5-D-only contract satisfiable from one place, and squeezing back preserves the rank promise
    # the docstring above has always made. A batched 5-D reference [1, 3, 1, H, W] then reaches
    # `_encode`'s `num_frames == 1` short-circuit — the single-frame spatial path a reference image
    # is supposed to take.
    unbatched = normalized.dim() == H3_VAE_UNBATCHED_RANK
    if unbatched:
        normalized = normalized.unsqueeze(0)
    assert_h3_vae_encode_input(normalized, what="encode_video_latents")
    # The move above cannot be dropped by a later refactor without this going red BY NAME: the
    # library's own failure names a conv kernel and never mentions device (D-10-DEF-12).
    assert_h3_encode_device(normalized, vae, what="encode_video_latents")

    distribution = vae.encode(normalized, return_dict=False)[0]

    if posterior == "sample":
        # CPU generator, FIXED seed — reproducible across devices and independent of the run seed.
        latents = distribution.sample(generator=torch.Generator().manual_seed(seed))
    elif posterior == "mode":
        latents = distribution.mode()
    else:
        raise ValueError(
            f"invalid posterior policy {posterior!r}: MiniMax-H3 runs exactly two — 'sample' "
            f"(images, video references, target video; fixed seed {H3_KEYFRAME_ENCODE_SEED}) and "
            f"'mode' (reference SOUNDTRACKS, which are never sampled). See P10-0d section 5.4."
        )

    # DELIBERATE ~11-bit parity rounding, and it happens BEFORE the per-channel normalization.
    latents = latents.to(torch.float16).float()

    mean = _broadcast_channel_vector(latents_mean, latents, "latents_mean")
    std = _broadcast_channel_vector(latents_std, latents, "latents_std")
    latents = (latents - mean) / std

    # Give back the rank we were given (D-10-DEF-9). The batch axis was OURS, added so the VAE's
    # 5-D-only contract could be met from one place; leaving it on would push a silent rank change
    # onto every downstream consumer of this function's documented "same rank as pixels".
    return latents[0] if unbatched else latents


# --------------------------------------------------------------------------------------------------
# Target video.
# --------------------------------------------------------------------------------------------------


@h3_no_grad
def encode_h3_video_latents(
    vae: Any,
    pixels: torch.Tensor,
    latents_mean: Any,
    latents_std: Any,
    *,
    pixel_frames: int | None = None,
    seed: int = H3_KEYFRAME_ENCODE_SEED,
) -> dict[str, Any]:
    """Encode the TARGET video and build the committed ``h3_latents/`` payload.

    The payload shape is a HARD contract from plan 10-04 — ``h3_latents`` is in
    ``data/precomputed.py::_VIDEO_LATENT_SOURCE_DIRS``, and that allowlist entry is only correct
    because the payload is ``{"latents": [C, F, H, W], "num_frames", "height", "width"}`` with the
    channel count H3's video VAE actually emits. Diverging here corrupts training silently.

    ``num_frames`` / ``height`` / ``width`` are the **LATENT** grid dims (that is what the legacy
    patchified ``(f h w) c -> c f h w`` fixup in ``_normalize_video_latents`` consumes), and they are
    DERIVED from the encoded tensor rather than taken on trust from the caller.

    Args:
        vae: The H3 video VAE.
        pixels: RGB pixels in ``[0, 255]``, ``[C, F, H, W]`` or ``[1, C, F, H, W]``.
        latents_mean: Per-channel ``latents_mean`` from the video VAE config.
        latents_std: Per-channel ``latents_std``.
        pixel_frames: Optional source frame count. When given, the encoded latent-frame count is
            cross-checked against the ``17n+5 -> 5n+2`` frame law (``h3_latent_frames``) so a VAE
            that padded or dropped a chunk is caught HERE, at the cheap step, not in the loop.
        seed: Fixed posterior seed.

    Returns:
        The ``h3_latents/`` payload dict.
    """
    latents = encode_video_latents(vae, pixels, latents_mean, latents_std, seed=seed)
    if latents.dim() == 5:
        if latents.shape[0] != 1:
            raise ValueError(
                f"encode_h3_video_latents expects ONE sample per call (batch_size=1, the "
                f"multi-bucket rule), got batch {latents.shape[0]}."
            )
        latents = latents[0]

    channels, num_frames, height, width = (int(v) for v in latents.shape)
    if channels != EXPECTED_H3_IN_CHANNELS:
        raise RuntimeError(
            f"[h3-encode] the video VAE emitted {channels} latent channel(s), expected "
            f"{EXPECTED_H3_IN_CHANNELS} (models/h3_loader.EXPECTED_H3_IN_CHANNELS). The wrong VAE "
            f"is mounted — LTX's is a different width. Refusing to write a mis-shaped "
            f"{H3_VIDEO_LATENTS_DIR}/ payload (plan 10-04 committed this dict shape)."
        )

    if pixel_frames is not None:
        expected_frames = h3_latent_frames(int(pixel_frames))
        if num_frames != expected_frames:
            raise RuntimeError(
                f"[h3-encode] {pixel_frames} pixel frames should encode to {expected_frames} latent "
                f"frame(s) under the {H3_FRAMES_PER_CHUNK}n+{H3_LATENTS_PER_CHUNK} law, but the VAE "
                f"produced {num_frames}. The clip was padded, truncated or chunked differently than "
                f"the pipeline contract — aborting before the cache is written."
            )

    return {
        "latents": latents,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


# --------------------------------------------------------------------------------------------------
# References.
# --------------------------------------------------------------------------------------------------


def snap_reference_frames(num_frames: int) -> int:
    """Largest conforming frame count ``<= num_frames`` — video references snap **DOWN**.

    A reference video is truncated to a conforming length so the VAE encodes it without padding
    (``encoders.py:729-736``). The law itself is NOT re-derived here: the modulus and offset come
    from ``conditioning/h3_geometry.py``, which owns them.
    """
    if num_frames < H3_LATENTS_PER_CHUNK:
        raise ValueError(
            f"cannot snap {num_frames} frame(s) down: the floor of the "
            f"{H3_FRAMES_PER_CHUNK}n+{H3_LATENTS_PER_CHUNK} law is {H3_LATENTS_PER_CHUNK} (n = 0)."
        )
    chunks = (num_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK
    return H3_FRAMES_PER_CHUNK * chunks + H3_LATENTS_PER_CHUNK


def _reference_descriptor(entry: Any, index: int) -> dict[str, str]:
    """Validate ONE caller-supplied slot descriptor -> the three fields the pool parser requires.

    ``conditioning/h3_ref._parse_reference_pool`` needs ``path`` / ``kind`` / ``subject_id`` on
    every cached slot, and **none of the three is recoverable from a latent tensor.** They are
    supplied by the caller (which read them out of the manifest) and merely checked here, because
    the encoder is the last place that can refuse before a full metered pre-encode writes a cache
    the trainer cannot read (D-10-DEF-2).

    ⛔ Nothing here defaults or infers. A guessed ``kind`` puts the environment reference in the
    wrong slot, and the shared rotary clock makes a reordered reference set a DIFFERENT request —
    it would train against conditioning nobody asked for, at a perfectly valid shape.
    """
    if not isinstance(entry, dict):
        raise TypeError(
            f"[h3-encode] reference descriptor {index} is a {type(entry).__name__}; expected a "
            'mapping like {"path": ..., "kind": ..., "subject_id": ...}.'
        )
    missing = [field for field in ("path", "kind", "subject_id") if not entry.get(field)]
    if missing:
        raise ValueError(
            f"[h3-encode] reference descriptor {index} is missing {missing}. Every cached slot must "
            f"carry path / kind / subject_id — selection needs 'kind' (D-10-REFORDER puts the "
            f"environment LAST) and 'subject_id' (images of one subject are grouped, and it is the "
            f"label a budget refusal names). Writing a descriptor-less payload caches a pre-encode "
            f"the trainer cannot read."
        )
    kind = str(entry["kind"])
    if kind not in H3_REFERENCE_KINDS:
        raise ValueError(
            f"[h3-encode] reference descriptor {index} declares kind {kind!r}; expected one of "
            f"{H3_REFERENCE_KINDS}. An unrecognized kind orders as a character, silently moving the "
            f"environment reference out of the last slot."
        )
    path = str(entry["path"])
    # Checked against BOTH flavours on purpose: manifests are authored POSIX-relative, but this
    # module is also exercised on Windows, where ``Path("/dataset/x").is_absolute()`` is False and a
    # container-absolute path would sail straight into the cache.
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise ValueError(
            f"[h3-encode] reference descriptor {index} carries an ABSOLUTE path. The cached "
            f"descriptor must stay relative: it is written into a payload that outlives this "
            f"container, so an absolute mount prefix is both non-portable and a client-property "
            f"leak into a committed artifact."
        )
    return {"path": path, "kind": kind, "subject_id": str(entry["subject_id"])}


@dataclass(frozen=True)
class H3PreparedReference:
    """ONE reference, already resized to its encode geometry, carrying the count both phases bill.

    ⛔ **This type exists so PHASE A and PHASE B cannot disagree about what a reference IS**
    (D-10-DEF-4). The pre-encode has two consumers of a reference image and they used to prepare it
    independently: PHASE A handed Qwen3-VL the **raw** file while counting vision tokens off the
    RESIZED geometry, and PHASE B resized before the VAE. Reference ``A`` (832x1248) therefore
    gridded at 1,014 vision tokens against the 1,176 the spans, the modality tags and the packed
    budget all assumed — a 162-row silent disagreement per reference.

    The fix is structural rather than arithmetic: ``image`` **is** the resized image and
    ``vision_tokens`` **is** its count, produced together by ``prepare_h3_reference_images``. There
    is no call form that pairs one reference's pixels with another reference's row count, and no
    call form that hands a consumer the raw file while counting the resized one.

    ``vision_tokens`` is billed TWICE by construction (``H3_VISION_TOKENS_EQUAL_LATENT_ROWS``):
    once as Qwen vision tokens in the TEXT stream and once as conditioning video rows.
    """

    image: Any
    source_wh: tuple[int, int]
    encoded_wh: tuple[int, int]
    short_edge: int
    vision_tokens: int


def prepare_h3_reference_images(images: Any, short_edge: int) -> list[H3PreparedReference]:
    """THE single reference-resize site — every consumer of a reference image goes through here.

    Both axes come back forced to a multiple of ``H3_CANVAS_MULTIPLE`` by
    ``h3_geometry.reference_image_size``, and that is what makes the agreement PROVABLE rather than
    asserted: Qwen3-VL's ``smart_resize`` snaps to ``patch_size * merge_size`` (= the same 32) and
    then only rescales when the total pixel count leaves its configured band, so an image that is
    already a multiple of 32 passes through it unchanged and its grid is exactly
    ``rows_of(height, width)``. Feeding the processor the RAW file — which is what PHASE A used to
    do — re-runs that snap against a different geometry and silently produces a different count.

    Idempotent: an already-prepared reference is returned as-is after its ``short_edge`` is checked,
    so PHASE B can pass PHASE A's list straight through without a second LANCZOS resample. A
    disagreeing ``short_edge`` RAISES — the same value is written into the cached payload as
    ``reference_short_edge``, and a payload whose declared short edge is not the one its pixels were
    resized at is a cache nobody can re-derive.

    Args:
        images: Pillow-like reference images (``.size`` -> ``(width, height)``, ``.convert``,
            ``.resize``), or already-prepared ``H3PreparedReference`` values.
        short_edge: The config-supplied reference short edge (896 for Phase 10). Never a constant.

    Returns:
        One ``H3PreparedReference`` per input, in the SAME order (D-10-REFORDER is load-bearing).
    """
    from PIL import Image as PILImage  # noqa: PLC0415 — deliberate function-local (Modal-side tier)

    prepared: list[H3PreparedReference] = []
    for index, entry in enumerate(images):
        if isinstance(entry, H3PreparedReference):
            if int(entry.short_edge) != int(short_edge):
                raise ValueError(
                    f"[h3-encode] reference {index} was prepared at short edge {entry.short_edge} "
                    f"but is being used at {short_edge}. The short edge is baked into the cached "
                    f"payload (reference_short_edge) and into every row count derived from it; two "
                    f"values in one sample means the spans, the modality tags and the packed budget "
                    f"describe different images. A different short edge needs a full RE-ENCODE — "
                    f"VAE latents cannot be spatially downscaled after the fact."
                )
            prepared.append(entry)
            continue

        rgb = entry.convert("RGB") if hasattr(entry, "convert") else entry
        source_width, source_height = (int(v) for v in rgb.size)
        # The aspect clamp and the round-to-32 sizing both live in h3_geometry — never re-derived.
        target_height, target_width = reference_image_size(
            source_width, source_height, short_edge=short_edge
        )
        if tuple(rgb.size) != (target_width, target_height):
            rgb = rgb.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        prepared.append(
            H3PreparedReference(
                image=rgb,
                source_wh=(source_width, source_height),
                encoded_wh=(target_width, target_height),
                short_edge=int(short_edge),
                vision_tokens=rows_of(target_height, target_width),
            )
        )
    return prepared


def _as_prepared_references(references: Any, what: str) -> list[H3PreparedReference]:
    """Refuse anything that is not an ``H3PreparedReference``, naming the function that makes one.

    This refusal IS the D-10-DEF-4 guard in its cheapest form. A raw Pillow image reaching a
    consumer is exactly how PHASE A ended up presenting one geometry while counting another, and
    a raw image satisfies every duck-typed thing a consumer asks of it — so nothing downstream
    would have complained.
    """
    prepared = list(references)
    bad = [i for i, ref in enumerate(prepared) if not isinstance(ref, H3PreparedReference)]
    if bad:
        raise TypeError(
            f"[h3-encode] {what} reference(s) {bad} are not H3PreparedReference values. Every "
            f"consumer of a reference image must take the output of prepare_h3_reference_images, "
            f"which resizes ONCE and carries the resulting vision-token count alongside the pixels. "
            f"Passing a raw image here is D-10-DEF-4: the consumer sees one geometry while the "
            f"spans, the modality tags and the packed budget are billed against another."
        )
    return prepared


@h3_no_grad
def encode_h3_reference_latents(
    vae: Any,
    images: Any,
    short_edge: int,
    latents_mean: Any,
    latents_std: Any,
    *,
    descriptors: Any,
    references_per_sample: int = H3_PHASE10_REFERENCES_PER_SAMPLE,
    seed: int = H3_KEYFRAME_ENCODE_SEED,
    explicit_manifest: bool = False,
) -> dict[str, Any]:
    """Encode the reference IMAGES and build the committed ``h3_reference_latents/`` payload.

    ⛔ **EXACTLY ``references_per_sample`` references per sample** (operator ruling, and the
    invariant ``h3_reference_pairing_domain`` already enforces on the pairing side). An
    ENVIRONMENT reference **SUBSTITUTES** for the last character slot — it is **never appended**.
    A caller handing this function a different count is a bug, so it raises rather than encoding
    a sample the budget was never priced for.

    Reference geometry (``before_encoder.py:490-492``; P10-0d section 5.1) — three things that
    surprise everyone:

    * **A reference NEVER binds the target geometry.** It is prepared at its OWN resolution.
    * **There is NO area cap** (the target canvas has one; references do not) and **upscaling is
      intentional**, so a 4:1 reference at the 2048 spec short edge encodes at 8192x2048.
    * Each axis is independently ``round()``-ed to a multiple of 32, aspect is hard-clamped to
      1:4 .. 4:1, and the resize is **LANCZOS**, aspect-preserving, **no crop and no pad**. The
      D-10-CROP crops are applied to the SOURCE files before this function ever sees them.

    ``short_edge`` is a REQUIRED argument, never a constant: it is a config field
    (``before_encoder.py:217-221``) — the released spec is 2048 and Phase 10 runs at 896 because
    the 2-slot pairing domain is what fits a single A100 (``P10-1-MEASURED.md`` section 4).
    A later higher-fidelity campaign needs a full RE-ENCODE; VAE latents cannot be downscaled after
    the fact.

    Each reference is encoded as ONE latent frame: a single frame goes through the spatial encoder
    alone, no temporal chunking (``encoders.py:110-112``).

    Args:
        vae: The H3 video VAE.
        images: Sequence of Pillow-like reference images (``.size`` -> ``(width, height)``,
            ``.resize(...)``, ``.convert("RGB")``).
        short_edge: Config-supplied reference short edge (896 for Phase 10).
        latents_mean: Per-channel ``latents_mean``.
        latents_std: Per-channel ``latents_std``.
        descriptors: One mapping per image, POSITIONALLY paired with ``images``, carrying
            ``path`` / ``kind`` / ``subject_id``. REQUIRED and never defaulted — a payload without
            them is a cache ``conditioning/h3_ref._parse_reference_pool`` refuses to read, and the
            refusal lands only after a whole metered pre-encode (D-10-DEF-2). They are propagated
            from the manifest by ``modal/fns.py::_h3_resolve_references``, never inferred here.
        references_per_sample: Required slot count. Defaults to the Phase-10 value.
        seed: Fixed posterior seed.
        explicit_manifest: issue #52 provenance — ``True`` when this row's references came from an
            explicit ``reference_paths`` list (the exact set, order load-bearing), ``False`` when
            they came from the rotating ``character_references`` pool (deliberately allowed to
            outnumber ``references_per_sample`` — the Embe corpus regime). Recorded verbatim as
            ``payload["explicit_manifest"]`` so ``conditioning/h3_ref.H3RefStrategy`` can refuse a
            cached-pool-size mismatch ONLY on a row positively marked explicit; a payload with no
            such key at all (every cache written before this field existed) reads as ``False`` —
            pool-derived, byte-identical to pre-fix behaviour. Defaults to ``False`` so every OTHER
            caller of this function (the both-modalities smoke, hand-built test fixtures) keeps
            writing the pool-derived shape it always has.

    Returns:
        The ``h3_reference_latents/`` payload. ``payload["references"]`` is the per-slot list, each
        entry ``{"latents": [24, 1, H, W], "num_frames", "height", "width", "source_wh",
        "encoded_wh", "latent_rows", "path", "kind", "subject_id"}``. ``payload["explicit_manifest"]``
        is the issue #52 provenance flag described above.

        ⚠ ``height`` / ``width`` are the LATENT grid dims; ``source_wh`` is the SOURCE pixel size,
        and that is what an ``H3Reference`` means by width/height. The parser reads ``source_wh``
        for exactly that reason — the two differ by the VAE's spatial factor, and pricing a
        reference off a latent grid would under-count it ~1000x.

        The TOP-LEVEL ``"latents"`` key is required by the plan-10-04 allowlist contract
        (``_normalize_video_latents`` reads it unconditionally) and ALIASES
        ``references[0]["latents"]`` — consumers must read ``"references"``, never assume the
        top-level tensor is the whole reference set.
    """
    items = list(images)
    if len(items) != references_per_sample:
        raise ValueError(
            f"[h3-encode] got {len(items)} reference(s), expected exactly {references_per_sample}. "
            f"Every H3 sample carries a FIXED number of reference slots; an environment reference "
            f"SUBSTITUTES for the last character slot, it is never appended. A different count was "
            f"never priced by the packed-sequence budget."
        )

    slots = [_reference_descriptor(entry, i) for i, entry in enumerate(descriptors)]
    if len(slots) != len(items):
        raise ValueError(
            f"[h3-encode] got {len(items)} image(s) but {len(slots)} descriptor(s). They are paired "
            f"POSITIONALLY, so an unequal count would attach one reference's identity to another "
            f"reference's latents — a mislabelled slot that reorders silently."
        )
    duplicates = sorted({s["path"] for s in slots if [d["path"] for d in slots].count(s["path"]) > 1})
    if duplicates:
        raise ValueError(
            f"[h3-encode] reference slots share path(s) {duplicates}. ``path`` is the JOIN KEY "
            f"H3RefStrategy gathers rows by (``by_path``), so two slots with one path collapse into "
            f"a single reference repeated twice — the same conditioning, silently, at the right "
            f"shape and the right row count."
        )

    # ⛔ D-10-DEF-4: the resize happens in ONE place, shared with PHASE A. This function does not own
    # a second copy of the geometry — if it did, the two phases could drift apart again exactly the
    # way they did (PHASE A presenting the raw image while counting the resized one).
    prepared = prepare_h3_reference_images(items, short_edge)

    encoded: list[dict[str, Any]] = []
    for index, reference in enumerate(prepared):
        source_width, source_height = reference.source_wh
        target_width, target_height = reference.encoded_wh
        pixels = _reference_pixels(reference.image)
        latents = encode_video_latents(
            vae, pixels, latents_mean, latents_std, seed=seed, posterior="sample"
        )
        if latents.dim() == 5:
            latents = latents[0]

        channels, num_frames, height, width = (int(v) for v in latents.shape)
        if channels != EXPECTED_H3_IN_CHANNELS:
            raise RuntimeError(
                f"[h3-encode] reference slot {index} encoded to {channels} latent channel(s), "
                f"expected {EXPECTED_H3_IN_CHANNELS}. The wrong VAE is mounted."
            )
        if num_frames != 1:
            raise RuntimeError(
                f"[h3-encode] reference slot {index} encoded to {num_frames} latent frames; a "
                f"single-frame reference must encode to exactly 1 (spatial encoder only, no "
                f"temporal chunking — encoders.py:110-112)."
            )

        encoded.append(
            {
                # The DESCRIPTOR first: without path / kind / subject_id the slot is a sized tensor
                # with no identity, and H3RefStrategy can neither order it (D-10-REFORDER) nor
                # gather it (``by_path``). D-10-DEF-2 was exactly this gap.
                **slots[index],
                "latents": latents,
                "num_frames": num_frames,
                # ⚠ LATENT grid dims. The SOURCE pixel size — what H3Reference means by
                # width/height — is ``source_wh``, and the pool parser reads it from there.
                "height": height,
                "width": width,
                "source_wh": (source_width, source_height),
                "encoded_wh": (target_width, target_height),
                # Packed rows this reference contributes to the CONDITIONING video stream. It bills
                # the SAME count again as Qwen vision tokens in the TEXT stream — the
                # each-image-ref-bills-TWICE rule (h3_geometry.H3_VISION_TOKENS_EQUAL_LATENT_ROWS).
                # Read off the PREPARED reference, not recomputed: it is the same integer PHASE A
                # presented to the processor, so the cache and the conditioning cannot disagree.
                "latent_rows": int(reference.vision_tokens),
            }
        )

    payload: dict[str, Any] = {
        "references": encoded,
        "num_references": len(encoded),
        "reference_short_edge": int(short_edge),
        # issue #52 provenance — see the Args docstring. A payload with no such key (every cache
        # written before this field existed) is read by H3RefStrategy as False, never as a
        # missing-therefore-refuse state.
        "explicit_manifest": bool(explicit_manifest),
    }
    # Plan-10-04 allowlist contract: h3_reference_latents routes through _normalize_video_latents,
    # which reads data["latents"] unconditionally. Slot 0 is the alias that satisfies it.
    payload["latents"] = encoded[0]["latents"]
    payload["num_frames"] = encoded[0]["num_frames"]
    payload["height"] = encoded[0]["height"]
    payload["width"] = encoded[0]["width"]
    return payload


def _reference_pixels(image: Any) -> torch.Tensor:
    """An ALREADY-RESIZED reference image -> ``[3, 1, H, W]`` uint8 pixels.

    ⛔ This function deliberately does NOT resize. The LANCZOS resample (no crop, no pad — aspect is
    preserved by the round-to-32 sizing) lives in ``prepare_h3_reference_images``, which is the ONE
    place a reference's encode geometry is decided (D-10-DEF-4). A second resize here would be a
    second source of truth, which is the defect this factoring exists to make impossible.
    """
    import numpy as np  # noqa: PLC0415 — deliberate function-local

    # np.array (a COPY), not np.asarray: Pillow hands back a READ-ONLY buffer, and
    # torch.from_numpy on a non-writable array warns and yields a tensor whose writes are
    # undefined behavior.
    array = np.array(image, dtype=np.uint8)
    # HWC -> CHW, then a singleton temporal axis: a reference image IS one latent frame.
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(1).contiguous()


# --------------------------------------------------------------------------------------------------
# The BOTH-MODALITIES smoke — the family closer for "a path only the reference case takes".
# --------------------------------------------------------------------------------------------------


@h3_no_grad
def h3_vae_smoke_encode(
    vae: Any,
    *,
    clip_pixels: torch.Tensor,
    reference_image: Any,
    reference_short_edge: int,
    reference_descriptor: Any,
    latents_mean: Any,
    latents_std: Any,
    clip_pixel_frames: int,
) -> str:
    """Encode ONE clip and ONE reference through the REAL VAE. Seconds. The whole family, at once.

    ⛔ **The class this closes: a code path that ONLY the single-frame / reference case takes.**
    ``AutoencoderKLMiniMaxH3._encode`` short-circuits ``num_frames == 1`` straight to
    ``_encode_clip``, so the reference branch is structurally unreachable from any video-side
    success. **D-10-DEF-9** (4-D pixels) and **D-10-DEF-12** (CPU pixels into a CUDA VAE) both live
    down there, and both were bought at the top of an 88-sample encode — five containers have now
    died there, after PHASE A had already pushed the whole corpus through Qwen3-VL-32B.

    ⛔ **No CPU test can reach either.** D-10-DEF-12 is a CUDA dispatch decision
    (``cudnn_is_acceptable`` -> ``ConvBackend.Slow3d`` -> no CUDA kernel) that does not exist on a
    CPU box, and every CPU stub in the suite is by definition device-agnostic. That is not a gap in
    the stubs — it is the boundary of what a stub can prove, and it is why this runs in-container.

    ⚠ **It calls the PRODUCTION entry points, not a parallel copy.** ``encode_h3_video_latents`` and
    ``encode_h3_reference_latents`` are the exact functions the 88-sample loop calls, so their
    channel-count checks, their frame-law cross-check and the chokepoint's rank + device refusals
    are all exercised for real. A smoke that re-implemented the encode would be the D-10-DEF-9
    hand-rolled-stub mistake with a bigger blast radius: it would go green on its own logic.

    ⚠ **Placement matters more than the check.** The caller must run this BEFORE PHASE A, not at the
    top of PHASE B — PHASE A is 88 samples through a 32B model and is most of what every dead
    container paid for. Run early it costs one small VAE load and two encodes; run late it saves
    nothing.

    Args:
        vae: The H3 video VAE, already loaded and on its device.
        clip_pixels: One real decoded clip, ``[C, F, H, W]`` uint8 — the target-video modality.
        reference_image: One real reference image (Pillow-like) — the single-frame modality.
            ``None`` on a NO-REFERENCE (ALPHA) encode: the single-frame branch is then unreachable
            by design (no reference is ever encoded on that run), so the smoke covers the clip
            modality only and says so in its report.
        reference_short_edge: The config's ``reference_image_short_edge``.
        reference_descriptor: That reference's manifest descriptor (``path`` / ``kind`` /
            ``subject_id``), passed through so the payload builder's real refusals run too.
            ``None`` iff ``reference_image`` is ``None``.
        latents_mean: Per-channel ``latents_mean`` from the video VAE config.
        latents_std: Per-channel ``latents_std``.
        clip_pixel_frames: The clip's source frame count, so the ``17n+5 -> 5n+2`` law is checked.

    Returns:
        A one-line report naming both realized latent shapes — printed by the caller so a passing
        smoke leaves evidence in the container log, not just an absence of failure.
    """
    video = encode_h3_video_latents(
        vae,
        clip_pixels,
        latents_mean,
        latents_std,
        pixel_frames=int(clip_pixel_frames),
    )
    if reference_image is None:
        # NO-REFERENCE (ALPHA): a run with zero reference slots never reaches the num_frames == 1
        # branch, so there is nothing for the reference leg to prove — and there is no reference to
        # encode it with. Skipped LOUDLY, not silently: the report names the reduced coverage.
        video_latents = video["latents"]
        return (
            f"[h3-vae-smoke] CLIP modality encoded on the real VAE — clip "
            f"{clip_pixel_frames}f -> latents {list(video_latents.shape)} "
            f"({video['num_frames']}f x {video['height']}x{video['width']}). NO-REFERENCE (ALPHA) "
            f"run: the single-frame reference branch (D-10-DEF-9 / D-10-DEF-12) is unreachable by "
            f"design on this encode and was NOT smoked."
        )
    # ⛔ references_per_sample=1 is not a shortcut, it is the point: ONE reference is enough to reach
    # the num_frames == 1 branch, and the campaign's real slot count is checked by the loop itself.
    # Encoding all of them here would multiply the cost of a preflight whose value is that it is
    # cheap.
    reference = encode_h3_reference_latents(
        vae,
        [reference_image],
        reference_short_edge,
        latents_mean,
        latents_std,
        descriptors=[reference_descriptor],
        references_per_sample=1,
    )
    slot = reference["references"][0]
    video_latents = video["latents"]
    reference_latents = slot["latents"]
    if int(reference_latents.shape[H3_VAE_FRAME_AXIS - 1]) != 1:
        raise RuntimeError(
            f"[h3-vae-smoke] the reference encoded to "
            f"{int(reference_latents.shape[H3_VAE_FRAME_AXIS - 1])} latent frames, expected 1. A "
            f"single-frame reference must take _encode's num_frames == 1 short-circuit (the spatial "
            f"encoder alone, no temporal chunking); anything else means the frame axis was misread."
        )
    return (
        f"[h3-vae-smoke] BOTH modalities encoded on the real VAE — clip "
        f"{clip_pixel_frames}f -> latents {list(video_latents.shape)} "
        f"({video['num_frames']}f x {video['height']}x{video['width']}); reference "
        f"{slot['encoded_wh'][0]}x{slot['encoded_wh'][1]} -> latents "
        f"{list(reference_latents.shape)} ({slot['latent_rows']} rows). The single-frame branch "
        f"(D-10-DEF-9 / D-10-DEF-12) is reachable ONLY from the reference producer and is now proven "
        f"before the 88-sample loop rather than at the top of it."
    )


# --------------------------------------------------------------------------------------------------
# Audio.
# --------------------------------------------------------------------------------------------------


@h3_no_grad
def encode_h3_audio_latents(
    audio_vae: Any,
    waveform: torch.Tensor | None,
    *,
    is_reference: bool,
    seed: int = H3_KEYFRAME_ENCODE_SEED,
    pixel_frames: int | None = None,
) -> torch.Tensor | None:
    """Encode a stereo waveform through the H3 audio VAE — with the OTHER posterior policy.

    ⚠ **Two different posterior policies live in one pipeline** (P10-0d section 5.4, trap 4):

    * ``is_reference=True``  -> the posterior ``.mode()``. Reference SOUNDTRACKS take the MEAN and
      are **never sampled**. (They are also never noised: reference audio pins at ``t = 1.0``,
      fully clean, while visual conditioning pins at ``max(t, 0.999)``.)
    * ``is_reference=False`` -> the posterior ``.sample()`` at the FIXED keyframe seed, matching
      every other visual encode.

    An audio reference never reaches the conditioner at all — it is encoded by the audio VAE alone
    (``MiniMaxH3Reference.audio``), which is why there is no Qwen leg in this function.

    ⚠ **This path is measured UNEXERCISED for Phase 10**: 0 of the 44 corpus clips carry an audio
    stream (D-10-AUDIO). When ``waveform`` is ``None`` or empty this returns ``None`` and the CALLER
    records an explicit absent marker (``h3_absent_audio_payload``) — fabricating silence would
    train the model on synthetic zeros it never saw, which is worse than an honest gap.

    ⛔ **``@h3_no_grad`` is here for the same reason it is on the video path, and it is here BECAUSE
    the path is unexercised** (D-10-DEF-10). The one-occurrence count of ``torch.no_grad`` in this
    module proved this helper was equally uncovered; it simply never ran, so it would have shipped
    unnoticed and cost its own metered container on the first campaign that targets audio. A guard
    added only where a failure has already been observed is how the same defect gets bought twice.

    ⚠ **#21 finding 2 — the return shape is CHANNEL-MAJOR ROWS, not the raw posterior.** The raw
    posterior out of ``audio_vae.encode`` carries the audio-channel axis at dim 0 (batch — stereo
    is ``batch_size=2``, never two channels on axis 1; ``h3_vae_contract.assert_h3_audio_vae_encode_input``
    is the refusal that names this) and the VAE's own ``EXPECTED_H3_AUDIO_IN_CHANNELS``-wide latent
    channel axis at dim 1, time last. Every consumer instead wants ONE ROW per
    ``(audio_channel, time)`` pair, audio-channel **slowest** —
    ``conditioning/h3_packing.py::_fill_h3_audio_positions``'s ``time.repeat(audio_channels)`` RoPE
    row order, and the row count ``conditioning/h3_geometry.h3_audio_rows`` prices, are both built
    on that layout. Nothing performed this transpose before writing the cache, so the whole audio
    arm was built on a wire format this encoder never produced — until now.

    Args:
        audio_vae: The H3 audio VAE.
        waveform: Stereo waveform at the VAE's rate, or ``None`` when the clip has no audio stream.
        is_reference: Selects the posterior policy. Required keyword — there is no safe default.
        seed: Fixed posterior seed (target audio only).
        pixel_frames: Optional target-video pixel-frame count. When given, the reshaped row count
            is cross-checked against ``h3_audio_rows(pixel_frames)`` (mirrors
            ``encode_h3_video_latents``'s ``pixel_frames`` cross-check) so a VAE time resolution
            that disagrees with the packed-sequence budget is caught HERE, before the cache is
            written, rather than surfacing as a downstream shape mismatch in the CPU preflight or
            (worse) a silently wrong row count that still happens to pack.

    Returns:
        The encoded audio latents as channel-major rows, ``[audio_channels * num_audio_latents,
        EXPECTED_H3_AUDIO_IN_CHANNELS]``, or ``None`` when there is no audio stream.
    """
    if waveform is None or waveform.numel() == 0:
        logger.info(
            "[h3-encode] no audio stream on this clip -> returning None; the caller records an "
            "explicit absent marker rather than fabricating silence."
        )
        return None

    # ⛔ D-10-DEF-12, the audio half — and it is here for the SAME reason `@h3_no_grad` is: this path
    # is measured unexercised (0 of 44 corpus clips carry a stream), so its `waveform.to("cuda")`
    # lives at the one call site and has never been executed. A guard added only where a failure has
    # already been observed is how the same defect gets bought twice. Placement is read off the
    # component, never written as a literal here.
    device = h3_component_device(audio_vae)
    if device is not None:
        waveform = waveform.to(device)
    assert_h3_encode_device(waveform, audio_vae, what="encode_h3_audio_latents")

    distribution = audio_vae.encode(waveform, return_dict=False)[0]
    if is_reference:
        latents = distribution.mode()
    else:
        latents = distribution.sample(generator=torch.Generator().manual_seed(seed))

    channels = int(latents.shape[1] if latents.dim() >= 3 else latents.shape[0])
    if channels != EXPECTED_H3_AUDIO_IN_CHANNELS:
        raise RuntimeError(
            f"[h3-encode] the audio VAE emitted {channels} latent channel(s), expected "
            f"{EXPECTED_H3_AUDIO_IN_CHANNELS} (models/h3_loader.EXPECTED_H3_AUDIO_IN_CHANNELS). "
            f"The wrong audio VAE is mounted."
        )

    # ⛔ #21 finding 2 (the boundary fix): raw posterior -> CHANNEL-MAJOR ROWS. ``squeeze(0)`` is a
    # no-op on the ordinary stereo-as-batch shape (batch=2) and only fires if a caller ever hands a
    # batch=1 waveform; ``permute(0, 2, 1)`` swaps the (channel, time) axes to (time, channel) per
    # audio-channel block, and ``reshape(-1, EXPECTED_H3_AUDIO_IN_CHANNELS)`` flattens
    # (audio_channel, time) into rows with audio_channel SLOWEST — exactly the row order
    # ``_fill_h3_audio_positions`` assumes (``time.repeat(audio_channels)``).
    latents = latents.squeeze(0).permute(0, 2, 1).reshape(-1, EXPECTED_H3_AUDIO_IN_CHANNELS)

    if pixel_frames is not None:
        expected_rows = h3_audio_rows(int(pixel_frames))
        if latents.shape[0] != expected_rows:
            raise RuntimeError(
                f"[h3-encode] {pixel_frames} pixel frame(s) should produce {expected_rows} "
                "channel-major audio row(s) (conditioning/h3_geometry.h3_audio_rows), but the "
                f"reshaped audio latents carry {latents.shape[0]}. The audio VAE's time resolution "
                "disagrees with the packed-sequence budget — refusing to write a mis-shaped "
                "h3_audio_latents/ payload."
            )
    return latents


def h3_absent_audio_payload(reason: str = "clip carries no audio stream") -> dict[str, Any]:
    """The explicit "no audio" marker written to ``h3_audio_latents/``.

    An EXPLICIT absence, not a zero tensor: a consumer can tell "this clip has no soundtrack" apart
    from "this clip is silent", and an encode that silently produced nothing cannot masquerade as a
    successful one. ``h3_audio_latents`` is deliberately NOT in the video-latent allowlist, so this
    marker is never routed through ``_normalize_video_latents``.
    """
    return {"latents": None, "present": False, "reason": reason}


# --------------------------------------------------------------------------------------------------
# The Qwen3-VL presentation + text conditioning.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class H3PresentationRef:
    """One reference as the Qwen3-VL PRESENTATION sees it (not as the budget sees it).

    Deliberately distinct from ``h3_geometry.H3Reference`` (which carries source pixel size + label
    for VRAM pricing): this one carries what the token stream needs — the kind, the per-vision-block
    pad counts, whether the reference has a soundtrack, and a video's merged-pair timestamps.
    """

    kind: str
    vision_token_counts: tuple[int, ...]
    has_audio: bool = False
    block_timestamps: tuple[float, ...] = field(default_factory=tuple)


def presentation_refs_from_prepared(prepared: Any) -> list[H3PresentationRef]:
    """``H3PreparedReference`` -> the ``H3PresentationRef`` the token stream needs, order preserved.

    The adapter exists so no caller ever writes ``vision_token_counts=(some_locally_computed_n,)``
    again: the count comes off the same object that carries the pixels the processor will see, which
    is what makes PHASE A's presentation and PHASE B's encode structurally the same reference
    (D-10-DEF-4).
    """
    return [
        H3PresentationRef(kind="image", vision_token_counts=(int(ref.vision_tokens),))
        for ref in _as_prepared_references(prepared, "presentation")
    ]


def _presentation_segments(
    prompt: str, references: Any
) -> list[tuple[str, bool]]:
    """Emit the presentation as ``(text, is_vision_block)`` segments, in the verified order.

    Transcribed from ``build_ref2va_presentation`` (P10-0d section 3.4). Numbering is PER-MODALITY
    and ONE-based; the prompt is emitted LAST and VERBATIM.
    """
    counts = {"image": 0, "video": 0, "audio": 0}
    segments: list[tuple[str, bool]] = []

    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            segments.append((f"<Audio {counts['audio']}>: ", False))

        if reference.kind == "image":
            counts["image"] += 1
            segments.append((f"<Picture {counts['image']}>: ", False))
            if len(reference.vision_token_counts) != 1:
                raise ValueError(
                    f"an image reference has exactly ONE vision block, got "
                    f"{len(reference.vision_token_counts)}."
                )
            segments.append((_vision_block(H3_IMAGE_PAD_TOKEN, reference.vision_token_counts[0]), True))
        elif reference.kind == "video":
            counts["video"] += 1
            segments.append((f"<Video {counts['video']}>: ", False))
            if len(reference.block_timestamps) != len(reference.vision_token_counts):
                raise ValueError(
                    f"a video reference needs one timestamp per vision block: got "
                    f"{len(reference.block_timestamps)} timestamp(s) and "
                    f"{len(reference.vision_token_counts)} block(s). Qwen3-VL samples the reference "
                    f"at {H3_QWEN_VIDEO_SAMPLE_FPS} fps and merges frames in groups of "
                    f"{H3_QWEN_TEMPORAL_PATCH}."
                )
            for timestamp, pad_count in zip(
                reference.block_timestamps, reference.vision_token_counts, strict=True
            ):
                segments.append((f"<{timestamp:.1f} seconds>", False))
                segments.append((_vision_block(H3_VIDEO_PAD_TOKEN, pad_count), True))
        else:
            raise ValueError(
                f"unknown reference kind {reference.kind!r}: MiniMax-H3 presents 'image' and "
                f"'video' references (an audio-only reference never reaches the conditioner)."
            )

    # The prompt goes LAST, VERBATIM: no chat template, no special tokens.
    segments.append((prompt, False))
    return segments


def _vision_block(pad_token: str, pad_count: int) -> str:
    """``<|vision_start|> pad*N <|vision_end|>`` — the wrapper every vision block wears."""
    if pad_count < 1:
        raise ValueError(f"a vision block needs at least one pad token, got {pad_count}.")
    return H3_VISION_START_TOKEN + pad_token * pad_count + H3_VISION_END_TOKEN


def build_h3_presentation(
    prompt: str,
    references: Any,
    *,
    tokenizer: Any = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Assemble the Qwen3-VL input string in the VERIFIED order, plus each vision block's span.

    The order (``build_ref2va_presentation``, P10-0d section 3.4) is::

        [ per-reference labels + vision blocks | prompt VERBATIM ]

    The labels and vision blocks come **FIRST**; the prompt comes **LAST**, verbatim, with **no chat
    template and no special tokens**. Numbering is per-modality and one-based
    (``<Audio j>: `` / ``<Picture i>: `` / ``<Video k>: ``), and a video reference emits one
    ``<t seconds>`` marker per merged frame pair before each of its vision blocks.

    Returns the string plus the HALF-OPEN ``(start, stop)`` token span of every vision block, in
    emission order, so ``train/h3_step.h3_token_tags`` can tag those rows 0 (VIDEO) — Qwen vision
    rows live in the TEXT stream but modulate as video, sentinels included.

    Span exactness:

    * ``tokenizer`` supplied -> spans are EXACT (each segment is tokenized in order), and each
      vision block is verified to tokenize to exactly ``pad_count + 2`` tokens. A tokenizer that
      does not know the sentinels would otherwise shred them into sub-word pieces and shift every
      downstream tag — so that is a hard error, not a warning.
    * ``tokenizer`` omitted -> spans are NOMINAL, billing ``H3_NOMINAL_LABEL_TOKENS`` per emitted
      TEXT segment, which is the same accounting ``h3_packed_seq_len`` uses. This is sound for
      planning because **the prompt is emitted LAST** — its length can never shift a vision span.

    Args:
        prompt: The caption, used VERBATIM.
        references: Sequence of ``H3PresentationRef``, in request order (order IS load-bearing).
        tokenizer: Optional Qwen3-VL tokenizer for exact spans.

    Returns:
        ``(presentation, vision_spans)``.
    """
    segments = _presentation_segments(prompt, references)

    presentation_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for text, is_vision in segments:
        presentation_parts.append(text)
        if tokenizer is None:
            length = _vision_block_nominal_length(text) if is_vision else H3_NOMINAL_LABEL_TOKENS
        else:
            length = len(tokenizer.encode(text, add_special_tokens=False))
            if is_vision:
                _assert_vision_block_tokenization(text, length)
        if is_vision:
            spans.append((cursor, cursor + length))
        cursor += length

    return "".join(presentation_parts), spans


def _vision_block_nominal_length(block: str) -> int:
    """Token length of a vision block: ``pad_count`` pads plus the two sentinels."""
    pad_token = H3_IMAGE_PAD_TOKEN if H3_IMAGE_PAD_TOKEN in block else H3_VIDEO_PAD_TOKEN
    return block.count(pad_token) + 2


def _assert_vision_block_tokenization(block: str, tokenized_length: int) -> None:
    """A vision block MUST tokenize to ``pads + 2``; anything else means the sentinels were split."""
    expected = _vision_block_nominal_length(block)
    if tokenized_length != expected:
        raise RuntimeError(
            f"[h3-encode] a vision block tokenized to {tokenized_length} token(s), expected "
            f"{expected} (pads + the two sentinels). The tokenizer does not recognize "
            f"{H3_VISION_START_TOKEN} / {H3_VISION_END_TOKEN} as single tokens, so every vision "
            f"span — and therefore every modality tag downstream — would be shifted. Load the "
            f"Qwen3-VL tokenizer that ships with the H3 weights."
        )


def _vision_merge_length(processor: Any) -> int:
    """``merge_size ** 2`` — READ off the mounted processor, never restated as a literal.

    Qwen3-VL computes an image's vision-token count as ``image_grid_thw.prod() // merge_size**2``.
    Hardcoding the divisor would make our count agree with the processor only by coincidence, on
    exactly the class of disagreement D-10-DEF-4 was.
    """
    image_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if merge_size is None:
        raise RuntimeError(
            "[h3-encode] the processor exposes no image_processor.merge_size, so the realized "
            "vision-token count cannot be derived from image_grid_thw and the PHASE A / PHASE B "
            "agreement cannot be checked. That check is the whole D-10-DEF-4 guard; refusing to "
            "encode without it rather than trusting a hardcoded merge size."
        )
    return int(merge_size) ** 2


def _grid_vision_tokens(grid_row: Any, merge_length: int) -> int:
    """One ``image_grid_thw`` row -> the pads the processor would have expanded it to."""
    prod = getattr(grid_row, "prod", None)
    if prod is not None:
        total = int(prod())
    else:
        total = 1
        for value in grid_row:
            total *= int(value)
    return total // merge_length


def _special_token_id(processor: Any, attribute: str, token: str) -> int:
    """A sentinel's id, preferring the processor's own attribute and falling back to the tokenizer."""
    token_id = getattr(processor, attribute, None)
    if token_id is None:
        tokenizer = getattr(processor, "tokenizer", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        token_id = convert(token) if convert is not None else None
    if token_id is None:
        raise RuntimeError(
            f"[h3-encode] cannot resolve the id of {token} (tried processor.{attribute} then "
            f"tokenizer.convert_tokens_to_ids). Without it the realized token stream cannot be "
            f"checked against the vision spans, and a shifted span mis-tags every conditioning row."
        )
    return int(token_id)


def build_h3_processor_inputs(
    processor: Any,
    presentation: str,
    references: Any = (),
    *,
    vision_spans: Any = (),
) -> dict[str, Any]:
    """Qwen3-VL inputs built from the presentation VERBATIM, with the expansion checked both ways.

    ⛔ **The processor's text path is deliberately not used** (D-10-DEF-4 defect 1).
    ``Qwen3VLProcessor.__call__`` expects **exactly ONE** ``<|image_pad|>`` per image and performs
    the expansion itself::

        merge_length = self.image_processor.merge_size**2
        while self.image_token in text[i]:
            num_image_tokens = image_grid_thw[index].prod() // merge_length
            text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)

    ``build_h3_presentation`` emits vision blocks **already expanded**, because its half-open vision
    spans — which ``train/h3_step.h3_token_tags`` consumes to tag conditioning rows — are only
    meaningful over the expanded stream. Handing that string to ``__call__`` exhausts the
    per-image replacement after 2 of 2,576 pads.

    ⚠ **That exhaustion happening to RAISE was luck, not a design working.** The installed
    ``transformers`` raises (``IndexError`` on ``image_grid_thw[index]`` at 5.2, ``StopIteration``
    from ``replacements_iters`` at 5.14). A version whose loop simply stops would leave the
    remaining pads un-expanded and condition the model at a **plausible but wrong shape**, with
    nothing to catch it. So the fix does not lean on the exception: it reproduces what ``__call__``
    does — ``image_processor(images=...)`` then ``tokenizer(text=...)``, merged, which is literally
    the ``BatchFeature(data={**text_inputs, **image_inputs})`` that function returns — and then
    asserts the agreement itself, in both directions:

      1. every reference's REALIZED grid (``image_grid_thw[i].prod() // merge_size**2``) equals the
         ``vision_tokens`` the prepared reference declared and the packed budget was priced on;
      2. the number of ``<|image_pad|>`` ids actually in ``input_ids`` equals the total the grids
         demand — the double-expansion defect itself, caught by arithmetic instead of by an
         exception a library version happens to raise;
      3. every vision span lands exactly on a ``<|vision_start|>`` / ``<|vision_end|>`` pair in the
         realized stream, so a shifted or shredded span cannot reach ``h3_token_tags``;
      4. the ``mm_token_type_ids`` mask marks image runs that equal the grids' counts **run by run,
         in order** — the M-RoPE half of the same class (``_assert_mm_token_type_ids``).

    Key-set parity with ``__call__`` is guarded separately and structurally by
    ``prep/h3_parity.py``, which runs the REAL ``__call__`` against this function on a synthetic
    input and diffs the output keys. That guard exists because this bypass inherits ALL of
    ``__call__``'s outputs and the gaps surface one at a time, each costing a metered container.

    ``add_special_tokens=False`` is explicit and load-bearing: the presentation is documented as
    emitted verbatim with no chat template and no special tokens, and ``build_h3_presentation``
    measures its spans the same way. Any injected BOS would shift every span by one.

    ⛔ **``mm_token_type_ids`` is emitted UNCONDITIONALLY** (D-10-DEF-7). ``ProcessorMixin.__call__``
    adds it whenever TEXT is given — not when images are — so a text-only sample gets an all-zero
    mask from it too. Gating ours on "has references" would make our output key set differ from
    ``__call__``'s on the text-only branch, which is precisely the divergence
    ``prep/h3_parity.py`` exists to forbid.

    ⚠ **The one thing NO assertion here can catch**: two references of the SAME encoded shape
    swapped with each other. ``get_rope_index`` pairs grids to runs purely by ORDER, and so does the
    embedding ``masked_scatter`` — both check only totals, which are identical under a swap. The
    only guard is structural and lives one level up: **ONE ``prepared`` list drives the
    presentation, the spans and the ``image_processor`` call** (``modal/fns.py`` PHASE A builds it
    once; this function consumes that same list for both). Splitting that list — preparing images
    for the processor separately from the ones the presentation was built from — silently reopens
    D-10-DEF-4 in a form nothing downstream can detect.

    Args:
        processor: The mounted Qwen3-VL processor.
        presentation: The string from ``build_h3_presentation``, pads already expanded.
        references: ``H3PreparedReference`` values — never raw images (see
            ``_as_prepared_references``).
        vision_spans: The half-open spans ``build_h3_presentation`` returned alongside the string.

    Returns:
        The model kwargs dict: tokenizer outputs merged with the image-processor outputs.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "[h3-encode] the processor exposes no .tokenizer, so the presentation cannot be "
            "tokenized without going through the processor's own text path — which re-expands the "
            "vision blocks this presentation has already expanded (D-10-DEF-4)."
        )

    prepared = _as_prepared_references(references, "text-conditioning")
    text_inputs = dict(tokenizer([presentation], add_special_tokens=False, return_tensors="pt"))
    input_ids = text_inputs.get("input_ids")
    if input_ids is None:
        raise RuntimeError("[h3-encode] the tokenizer returned no input_ids.")
    text_inputs[H3_MM_TOKEN_TYPE_IDS_KEY] = _mm_token_type_ids(processor, input_ids)
    if not prepared:
        # A text-only sample passes no grids, so `realized` is empty and the mask must be all text.
        _assert_mm_token_type_ids(text_inputs[H3_MM_TOKEN_TYPE_IDS_KEY], [])
        return text_inputs

    image_inputs = dict(
        processor.image_processor(images=[ref.image for ref in prepared], return_tensors="pt")
    )
    grid = image_inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError(
            "[h3-encode] the image processor returned no image_grid_thw, so the realized vision "
            "expansion is unknowable. Refusing to encode conditioning whose token count cannot be "
            "checked against the count the spans and the packed budget assume."
        )

    merge_length = _vision_merge_length(processor)
    realized = [_grid_vision_tokens(row, merge_length) for row in grid]
    expected = [int(ref.vision_tokens) for ref in prepared]
    if realized != expected:
        detail = ", ".join(
            f"slot {i}: {ref.source_wh[0]}x{ref.source_wh[1]} -> "
            f"{ref.encoded_wh[0]}x{ref.encoded_wh[1]} presents {want} but grids {got}"
            for i, (ref, want, got) in enumerate(zip(prepared, expected, realized, strict=True))
        )
        raise RuntimeError(
            f"[h3-encode] the processor's realized vision grid disagrees with the prepared "
            f"references ({detail}). The two must be equal BY CONSTRUCTION — every prepared "
            f"reference is a multiple of {H3_CANVAS_MULTIPLE} on both axes, which makes Qwen3-VL's "
            f"smart_resize a no-op and its grid exactly rows_of(h, w). A disagreement means the "
            f"image handed to the processor is not the image that was prepared, or the processor's "
            f"pixel band rescaled it — either way the vision spans, the modality tags and the "
            f"packed budget would all describe a different reference (D-10-DEF-4)."
        )

    pad_id = _special_token_id(processor, "image_token_id", H3_IMAGE_PAD_TOKEN)
    realized_pads = int((input_ids == pad_id).sum())
    if realized_pads != sum(realized):
        raise RuntimeError(
            f"[h3-encode] the presentation carries {realized_pads} {H3_IMAGE_PAD_TOKEN} token(s) "
            f"but the {len(prepared)} reference grid(s) demand {sum(realized)} ({realized}). The "
            f"pads in the token stream and the pads the vision grids expand to must be equal: too "
            f"few leaves vision rows unbacked, too many (the D-10-DEF-4 double-expansion) presents "
            f"the model padding it has no pixels for. This is checked by arithmetic on purpose — "
            f"the transformers version that RAISED on this is a coincidence, not the guard."
        )

    _assert_mm_token_type_ids(text_inputs[H3_MM_TOKEN_TYPE_IDS_KEY], realized)
    _assert_vision_spans_land_on_sentinels(processor, input_ids, vision_spans)
    return {**text_inputs, **image_inputs}


def _mm_token_type_ids(processor: Any, input_ids: Any) -> torch.Tensor:
    """The M-RoPE modality mask, built by the processor's own PUBLIC method — never reimplemented.

    ``ProcessorMixin.create_mm_token_type_ids`` reads ``self.image_token_ids`` /
    ``video_token_ids`` / ``audio_token_ids`` off the MOUNTED processor, so the pad-id vocabulary is
    never re-derived here — the same reason every other constant in this module is read rather than
    restated. Values are text 0, image 1, video 2, audio 3.

    It returns a LIST OF LISTS (it is documented to accept unpadded, ragged batches), so it needs a
    tensor conversion before joining a ``return_tensors="pt"`` dict and flowing through
    ``_to_device``.
    """
    if not hasattr(processor, "create_mm_token_type_ids"):
        raise RuntimeError(
            f"[h3-encode] the mounted processor ({type(processor).__name__}) has no "
            f"create_mm_token_type_ids. That method is a ProcessorMixin public API at the PINNED "
            f"transformers version (modal/app.py::TRANSFORMERS_VERSION); its absence means a "
            f"different version is installed. Refusing to hand-build the M-RoPE modality mask: it "
            f"encodes the processor's own pad-id vocabulary, and a re-derived one that disagrees is "
            f"silently wrong at a perfectly valid shape (D-10-DEF-7)."
        )
    return torch.tensor(processor.create_mm_token_type_ids(input_ids), dtype=torch.int)


def _assert_mm_token_type_ids(mm_token_type_ids: Any, realized: Any) -> None:
    """The modality mask must describe EXACTLY the vision blocks the grids describe, RUN BY RUN.

    ⛔ This is the silent half of D-10-DEF-7, and it is worse than the crash that surfaced it.
    ``Qwen3VLModel.compute_3d_position_ids`` raises only when ``mm_token_type_ids`` is **missing**;
    a mask that is present and WRONG raises nowhere, and the guard evaporates entirely on the
    ``inputs_embeds`` path.

    Two failure modes, both traced by hand through ``get_rope_index``:

    1. **All zeros.** ``create_mm_token_type_ids`` falls back to ``[self.image_token_id]`` and, if
       that is ``None``, ``np.isin(ids, [None])`` marks NOTHING. The mask is then one all-text run:
       ``get_rope_index`` emits ``arange(seq_len)`` positions, **never consumes the grid iterators**,
       and the concatenated length still equals ``seq_len``, so the assignment succeeds. Every
       vision row is positioned as text and nothing anywhere raises.
    2. **Per-run drift.** ``get_rope_index`` pulls ``next(grid_iters[type])`` once per CONTIGUOUS
       run and builds that run's positions from the GRID, not from the run's measured length; the
       final concat is checked only against the TOTAL sequence length. So a swapped or merged run
       cancels in the total (1,176 + 1,372 == 1,372 + 1,176) and reference A silently receives the
       other reference's H x W lattice. Hence the check is **per run, in order** — a count-only
       check would pass on exactly that swap.

    Args:
        mm_token_type_ids: The mask ``_mm_token_type_ids`` built, ``[1, L]`` or ``[L]``.
        realized: The per-reference vision-token counts the grids demand, IN PRESENTATION ORDER.
            Empty for a text-only sample.
    """
    expected_runs = [int(n) for n in realized]
    batched = getattr(mm_token_type_ids, "dim", lambda: 1)() == 2
    types = mm_token_type_ids[0] if batched else mm_token_type_ids
    values = [int(v) for v in types.tolist()]

    runs: list[list[int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])

    image_runs = [length for value, length in runs if value == H3_MM_TYPE_IMAGE]
    if expected_runs and not image_runs:
        raise RuntimeError(
            f"[h3-encode] mm_token_type_ids marks NO image tokens, but the reference grids demand "
            f"{expected_runs} ({sum(expected_runs)} vision token(s)). This is the all-zeros mask: "
            f"create_mm_token_type_ids falls back to [processor.image_token_id] and marks nothing "
            f"when that is None. It does NOT raise downstream — get_rope_index sees one all-text "
            f"run, never touches the grid iterators, and produces a correctly-shaped position_ids "
            f"in which every vision row is positioned as text (D-10-DEF-7). Check that the mounted "
            f"processor resolves image_token_id / image_token_ids."
        )
    if image_runs != expected_runs:
        raise RuntimeError(
            f"[h3-encode] mm_token_type_ids has image runs {image_runs} but the reference grids "
            f"demand {expected_runs}, in that order. get_rope_index consumes ONE grid per "
            f"CONTIGUOUS run of the type value and builds that run's positions from the GRID rather "
            f"than from the run's length, checking only the TOTAL at the end — so a swapped or "
            f"merged run cancels in the total and gives a reference the WRONG H x W lattice with "
            f"nothing raising. Our sentinels (<|vision_start|> / <|vision_end|>, type 0) are what "
            f"separate the blocks into one run each."
        )
    for marker, name in ((H3_MM_TYPE_VIDEO, "video"), (H3_MM_TYPE_AUDIO, "audio")):
        count = sum(length for value, length in runs if value == marker)
        if count:
            raise RuntimeError(
                f"[h3-encode] mm_token_type_ids marks {count} {name} token(s) (type {marker}), but "
                f"build_h3_processor_inputs only ever produces image_grid_thw. get_rope_index would "
                f"call next() on a grid iterator that is None for that modality. A VIDEO reference "
                f"is not a widening of this path — it needs pixel_values_videos, video_grid_thw, a "
                f"different sub-processor and per-merged-frame-group grid splitting, i.e. a "
                f"D-10-DEF-4-sized re-derivation."
            )


def _assert_vision_spans_land_on_sentinels(
    processor: Any, input_ids: Any, vision_spans: Any
) -> None:
    """Every ``build_h3_presentation`` span must bracket a real ``<|vision_start|>``/``<|vision_end|>``.

    The spans are what ``train/h3_step.h3_token_tags`` uses to tag Qwen vision rows as VIDEO. A span
    that is off by even one token mis-tags a conditioning row at a perfectly valid shape, so it is
    checked against the REALIZED stream rather than trusted from the string it was measured on.
    """
    spans = [tuple(int(v) for v in span) for span in vision_spans]
    if not spans:
        return
    start_id = _special_token_id(processor, "vision_start_token_id", H3_VISION_START_TOKEN)
    end_id = _special_token_id(processor, "vision_end_token_id", H3_VISION_END_TOKEN)
    ids = input_ids[0] if getattr(input_ids, "dim", lambda: 1)() == 2 else input_ids
    length = int(ids.shape[-1])
    for index, (start, stop) in enumerate(spans):
        if not 0 <= start < stop <= length:
            raise RuntimeError(
                f"[h3-encode] vision span {index} is {(start, stop)}, outside the realized "
                f"{length}-token stream. The presentation and its spans came from the same call, so "
                f"this means the tokenizer produced a different stream than the spans were measured "
                f"on — every downstream modality tag would be shifted."
            )
        if int(ids[start]) != start_id or int(ids[stop - 1]) != end_id:
            raise RuntimeError(
                f"[h3-encode] vision span {index} {(start, stop)} does not bracket "
                f"{H3_VISION_START_TOKEN} .. {H3_VISION_END_TOKEN} in the realized token stream "
                f"(found ids {int(ids[start])} .. {int(ids[stop - 1])}, expected {start_id} .. "
                f"{end_id}). A shifted span mis-tags conditioning rows as text at a perfectly valid "
                f"shape — nothing downstream raises, the adapter simply learns the wrong "
                f"modulation."
            )


@h3_no_grad
def encode_h3_text_conditions(
    text_encoder: Any,
    processor: Any,
    presentation: str,
    references: Any = (),
    *,
    vision_spans: Any,
) -> torch.Tensor:
    """Run Qwen3-VL over the presentation and take the layer H3 was actually trained against.

    ⚠⚠ **WARNING — the layer index is the single most dangerous number in this pipeline.** H3 reads
    the hidden state at ``EXPECTED_H3_TEXT_ENCODER_LAYER`` out of Qwen3-VL's 64 layers
    (``modular_pipeline.py::text_encoder_layer``). The FINAL layer is post-norm and is **NOT** the
    conditioning the released weights were trained against, so a final-layer encode is **silently
    wrong at the correct ``(B, L, 5120)`` shape** — no crash, no shape error, just off-distribution
    text features for every sample of the run (``P10-1-MEASURED.md`` section 6).

    The index is IMPORTED from ``models/h3_loader.py``, the single source of every H3 architecture
    number. It is deliberately never written as a literal in this module, and
    ``tests/test_h3_preprocess_wiring.py`` fails if one appears.

    ⚠ Its numeric equality with the DiT depth is a coincidence; the two must never be conflated.

    ⛔ The inputs are built by ``build_h3_processor_inputs``, NOT by ``processor(text=..., ...)``.
    That call double-expands the vision blocks and grids the RAW image geometry (D-10-DEF-4); the
    replacement checks the realized expansion against the prepared references in both directions.

    Args:
        text_encoder: The loaded Qwen3-VL model.
        processor: Its matching processor (builds pixel inputs for the vision blocks).
        presentation: The string from ``build_h3_presentation``.
        references: ``H3PreparedReference`` values the vision blocks stand in for — the RESIZED
            images and their vision-token counts travelling together, so the processor can never be
            handed one geometry while the spans bill another. Empty for a text-only encode.
        vision_spans: REQUIRED. The spans ``build_h3_presentation`` returned with the string; they
            are verified against the realized token stream before the encoder runs. Required rather
            than optional because an unchecked span silently mis-tags conditioning rows, and pass
            ``()`` deliberately for a text-only encode.

    Returns:
        The selected hidden state, squeezed to ``(L, text_dim)`` for a single sample.
    """
    inputs = build_h3_processor_inputs(
        processor, presentation, references, vision_spans=vision_spans
    )
    inputs = {key: _to_device(value, text_encoder) for key, value in inputs.items()}

    # ⚠ Redundant under ``@h3_no_grad`` and KEPT. It is the only occurrence of this idiom the module
    # had, and its being the only one is what D-10-DEF-10 was: the lesson was written down at the one
    # place it had already been paid for and nowhere else. Deleting it as "now redundant" would erase
    # the marker at the exact site a reader looks for it. Re-entering no_grad costs nothing.
    with torch.no_grad():
        outputs = text_encoder(**inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "[h3-encode] the text encoder returned no hidden_states — it must be called with "
            "output_hidden_states=True. Refusing to guess at the conditioning."
        )
    if len(hidden_states) <= EXPECTED_H3_TEXT_ENCODER_LAYER:
        raise RuntimeError(
            f"[h3-encode] the text encoder returned {len(hidden_states)} hidden state(s), which "
            f"does not reach EXPECTED_H3_TEXT_ENCODER_LAYER "
            f"({EXPECTED_H3_TEXT_ENCODER_LAYER}). MiniMax-H3 conditions on THAT layer, not the "
            f"final one — the final layer is post-norm and is not what the released weights were "
            f"trained against. A shorter tuple means the wrong text encoder is mounted; falling "
            f"back to the last available layer would be silently wrong at the correct shape."
        )

    hidden = hidden_states[EXPECTED_H3_TEXT_ENCODER_LAYER]
    if hidden.dim() == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    return hidden.detach().to("cpu")


def _to_device(value: Any, model: Any) -> Any:
    """Move a processor output onto the model's device when it is a tensor; pass anything else."""
    if not isinstance(value, torch.Tensor):
        return value
    device = getattr(model, "device", None)
    return value if device is None else value.to(device)


# --------------------------------------------------------------------------------------------------
# The precomputed writer.
# --------------------------------------------------------------------------------------------------

#: The four source dirs one sample's cache spans, in write order. ``h3_precomputed_complete`` and
#: ``write_h3_precomputed`` both read THIS tuple, so "is a sample complete" can never drift from
#: "what does a full encode write" — a fifth source added to one and not the other is the exact
#: split-brain a resume guard must not have.
H3_PRECOMPUTED_SOURCE_DIRS: tuple[str, ...] = (
    H3_VIDEO_LATENTS_DIR,
    H3_CONDITIONS_DIR,
    H3_REFERENCE_LATENTS_DIR,
    H3_AUDIO_LATENTS_DIR,
)


def _normalize_precomputed_rel(rel_path: str | Path) -> Path:
    """Sample rel path -> the ``<rel>.pt`` every source file is named by. Shared by write + probe."""
    rel = Path(rel_path)
    if rel.is_absolute():
        raise ValueError(f"rel_path must be relative to the precomputed root, got {rel}.")
    return rel if rel.suffix == ".pt" else rel.with_suffix(".pt")


def _atomic_save(payload: Any, destination: Path) -> None:
    """Atomic ``torch.save`` via per-PID staging + ``replace`` (``data/mask_encode.py::_atomic_save``).

    A container killed inside a bare ``torch.save`` publishes a TRUNCATED ``.pt`` at the canonical
    name — which the completeness probes then count as a finished sample and ``PrecomputedDataset``
    pairs into a training batch. Staging + rename makes the canonical name appear only whole. Not
    imported from ``data/mask_encode``: that module pulls numpy at module scope, and this one is
    pinned to torch-only imports (the CPU-gate closure test).
    """
    staging = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    torch.save(payload, staging)
    staging.replace(destination)


def h3_precomputed_complete(output_root: str | Path, rel_path: str | Path) -> bool:
    """True iff ALL FOUR sources already hold a file for ``rel_path`` under ``output_root``.

    The resume predicate for ``h3_preprocess``: a sample is only skippable when the whole quartet is
    on disk, because ``PrecomputedDataset`` pairs sources by rel path and a partial write (e.g. a
    PHASE-A-only ``h3_conditions/`` file from a container that died in PHASE B) must be re-encoded,
    never trusted. Keyed on the same dir tuple the writer uses (mirroring
    ``encode_mask_dataset(overwrite=False)``'s skip-if-present).
    """
    root = Path(output_root)
    rel = _normalize_precomputed_rel(rel_path)
    return all((root / dir_name / rel).is_file() for dir_name in H3_PRECOMPUTED_SOURCE_DIRS)


def write_h3_precomputed(
    output_root: str | Path,
    rel_path: str | Path,
    *,
    video: dict[str, Any] | None = None,
    text: torch.Tensor | None = None,
    references: dict[str, Any] | None = None,
    audio: Any = None,
) -> dict[str, Path]:
    """Write the provided payloads under the four ``h3_``-prefixed sources at the SAME rel path.

    ``PrecomputedDataset`` pairs sources by RELATIVE PATH, so every source a run configures must
    receive a file at the identical ``<rel>.pt`` — a missing one silently drops the whole sample
    from the index rather than raising.

    The two VIDEO-latent sources are written as ``{"latents": [C, F, H, W], "num_frames", "height",
    "width"}`` dicts. That is the shape plan 10-04 committed when it allowlisted them in
    ``_VIDEO_LATENT_SOURCE_DIRS``; the allowlist entry is only correct because of it, so the shape
    is asserted here rather than trusted.

    Args:
        output_root: The ``.precomputed`` root on the dataset Volume.
        rel_path: Sample-relative path (nested paths are fine; parents are created).
        video: ``encode_h3_video_latents`` payload -> ``h3_latents/``.
        text: The selected Qwen3-VL hidden state -> ``h3_conditions/``.
        references: ``encode_h3_reference_latents`` payload -> ``h3_reference_latents/``.
        audio: Audio latents, or ``h3_absent_audio_payload()`` -> ``h3_audio_latents/``.

    Returns:
        ``{source_dir_name: written_path}`` for every payload actually written.
    """
    root = Path(output_root)
    rel = _normalize_precomputed_rel(rel_path)

    written: dict[str, Path] = {}
    for dir_name, payload in zip(H3_PRECOMPUTED_SOURCE_DIRS, (video, text, references, audio)):
        if payload is None:
            continue
        if dir_name in (H3_VIDEO_LATENTS_DIR, H3_REFERENCE_LATENTS_DIR):
            _assert_video_latent_payload(dir_name, payload)
        destination = root / dir_name / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Staged, never bare: a kill inside torch.save must not publish a truncated canonical file.
        _atomic_save(payload, destination)
        written[dir_name] = destination

    logger.info("[h3-encode] wrote %d source(s) for %s -> %s", len(written), rel, root)
    return written


def _assert_video_latent_payload(dir_name: str, payload: Any) -> None:
    """Refuse to write a video-latent source that the 10-04 allowlist would then mis-route."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"[h3-encode] {dir_name}/ payloads must be dicts carrying a 'latents' tensor "
            f"(plan 10-04 allowlist contract), got {type(payload).__name__}."
        )
    missing = [key for key in ("latents", "num_frames", "height", "width") if key not in payload]
    if missing:
        raise ValueError(
            f"[h3-encode] {dir_name}/ payload is missing {missing}. The committed shape is "
            f"{{'latents': [C, F, H, W], 'num_frames', 'height', 'width'}} — this source is in "
            f"data/precomputed.py::_VIDEO_LATENT_SOURCE_DIRS and _normalize_video_latents reads "
            f"those keys unconditionally."
        )
    latents = payload["latents"]
    if not isinstance(latents, torch.Tensor) or latents.dim() != 4:
        raise ValueError(
            f"[h3-encode] {dir_name}/ 'latents' must be a 4-D [C, F, H, W] tensor, got "
            f"{type(latents).__name__} "
            f"{tuple(latents.shape) if isinstance(latents, torch.Tensor) else ''}."
        )
    if int(latents.shape[0]) != EXPECTED_H3_IN_CHANNELS:
        raise ValueError(
            f"[h3-encode] {dir_name}/ 'latents' has {int(latents.shape[0])} channel(s), expected "
            f"{EXPECTED_H3_IN_CHANNELS}. H3's video VAE is not LTX's — writing this would corrupt "
            f"the run silently."
        )
    height, width = int(payload["height"]), int(payload["width"])
    if height < 1 or width < 1:
        raise ValueError(f"[h3-encode] {dir_name}/ latent grid {height}x{width} is degenerate.")
    if (height, width) != tuple(int(v) for v in latents.shape[2:]):
        raise ValueError(
            f"[h3-encode] {dir_name}/ records a {height}x{width} latent grid but the tensor is "
            f"{tuple(int(v) for v in latents.shape[2:])}. The metadata drives the legacy "
            f"patchified fixup — a disagreement here reshapes latents wrongly, silently."
        )


# --------------------------------------------------------------------------------------------------
# The two-phase VRAM discipline.
# --------------------------------------------------------------------------------------------------


def free_text_encoder(text_encoder: Any = None, processor: Any = None) -> None:
    """Drop Qwen3-VL before anything large is loaded — the two-phase VRAM invariant.

    ⛔ **INVARIANT: Qwen3-VL-32B (~27 GiB int8 / ~62 GiB bf16) and the 61.7 GiB H3 transformer must
    NEVER coexist in VRAM.** Not "should not" — a single A100-80GB physically cannot hold both, and
    the failure mode is an OOM many minutes into a metered container.

    This discipline is MORE mandatory on H3 than it was on LTX, where Gemma-12B (~24 GiB) alongside
    the 22B was merely tight (06-09 runs 3-5 proved they cannot coexist either). The pattern is
    copied from the proven Gemma free at ``modal/fns.py:445-464``: move to CPU, drop the reference,
    ``gc.collect()``, ``torch.cuda.empty_cache()``.

    ⚠ **The CALLER must also drop ITS OWN references** (``text_encoder = None`` after this call, and
    null any attribute holding it). Loader-owned CUDA storage allocated with ``assign=True`` is
    freed only when the LAST reference goes away — ``Module.to("cpu")`` alone does not release it
    (06-09 run-5 finding). This helper can only reach its own locals.
    """
    import gc  # noqa: PLC0415 — deliberate function-local, mirroring fns.py's Gemma free

    for obj in (text_encoder, processor):
        if obj is None:
            continue
        mover = getattr(obj, "to", None)
        if mover is None:
            continue
        try:
            mover("cpu")
        except Exception:  # noqa: BLE001 — a best-effort move must never abort the free
            logger.debug("[h3-encode] could not move %s to cpu before free", type(obj).__name__)

    del text_encoder, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(
            "[h3-encode] freed the Qwen3-VL text encoder (two-phase VRAM discipline) — cuda "
            "allocated=%.2f GiB. The transformer is loaded ONLY after this point.",
            torch.cuda.memory_allocated() / 2**30,
        )
