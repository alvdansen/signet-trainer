"""conditioning.h3_packing — the two MiniMax-H3 WIRE-FORMAT contracts, transcribed from diffusers.

This module closes the two seams plan 10-08 deliberately left failing loud rather than guessing:

  1. the ``[C, F, H, W] -> rows`` **patchify**, whose intra-patch ELEMENT ORDER is a checkpoint
     contract (``prep/h3_encode.py`` writes ``[24, F, H, W]``; ``train/h3_step.py`` consumes
     ``[1, rows, patch_dim]``);
  2. the H3 **RoPE ``position_ids``** construction — 3-axis float64, ``t`` absolute from
     ``text_len``, h/w area-normalized to a fixed 32-unit extent.

⛔ NEITHER OF THESE MAY EVER BE GUESSED
---------------------------------------
Both fail SILENTLY at the correct shape. A wrong intra-patch order produces correctly-shaped,
entirely plausible rows trained against scrambled features: nothing raises, the loss curve looks
normal, and the adapter is garbage. A monotone ``position_ids`` stand-in trains the adapter against
the wrong geometry with, again, nothing raising. Everything below was therefore READ OUT OF THE
SOURCE, not inferred.

PROVENANCE — read, not derived
------------------------------
``diffusers`` at the SHA pinned as ``modal/app.py::DIFFUSERS_SHA``
(``9f169d98d0bce392a889c3b6524d0d97734dfc0e``), file
``src/diffusers/modular_pipelines/minimax_h3/before_denoise.py``:

    ``patchify_video_latents``                                   L44-73
    ``_ROPE_FRAME_RESCALE`` / ``_ROPE_FRAMES_PER_LATENT`` / ``_ROPE_SPATIAL_SCALE``   L36-41
    ``_spatial_position_grid``                                   L76-86
    ``_temporal_position_grid``                                  L89-98
    ``_frame_position_grid``                                     L101-109
    ``_fill_audio_positions``                                    L112-133
    ``MiniMaxH3Ref2VAPackedSequenceStep.build_ref2va_packed_sequence``   L559-722

The ``P10-1`` probe (``scripts/_h3_probe_modal.py:521-522``) is NOT an authority here: it ran a real
A100 forward/backward on a SYNTHETIC monotone ``position_ids`` and never patchified a real latent,
so it proves the forward SIGNATURE and nothing about either contract in this file.

Why the layouts line up
-----------------------
diffusers' ref2va layout is ``[ text | reference blocks | target audio | target video ]``
(``before_denoise.py`` L575). ``train/h3_step.h3_segment_offsets`` packs
``[ text | ref video | audio | target video ]`` where the audio block is reference soundtracks THEN
target audio. With **zero reference soundtracks** — which is the measured Phase-10 corpus, D-10-AUDIO,
0 of 44 clips carry an audio stream — the two orders are IDENTICAL. This module therefore refuses to
build a layout with reference audio rows rather than emitting one whose ordering it cannot vouch for.

CRITICAL — Anti-Pattern 6, the CONFINED tier
--------------------------------------------
Module scope is ``numpy`` + ``torch`` + stdlib + ``conditioning/h3_geometry`` (itself stdlib-only).
No ``modal``, no ``diffusers``, no ``peft``. ``numpy`` is deliberate and load-bearing, not
convenience: ``np.linspace(..., endpoint=False)`` is ``start + arange(n) * (stop - start) / n``,
which ``torch.linspace`` does NOT compute, and the float64 grid has to be reproduced exactly (the
reference implementation carries that warning in a comment at ``before_denoise.py`` L83-85). numpy
already ships with torch and is already in this package's measured import closure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from signet_trainer.conditioning.h3_geometry import (
    H3_AUDIO_CHANNELS,
    H3_CANVAS_MULTIPLE,
    h3_audio_rows,
    h3_latent_frames,
    reference_image_size,
    resolve_canvas_size,
    rows_of,
)

__all__ = [
    "H3_ROPE_FRAMES_PER_LATENT",
    "H3_ROPE_FRAME_RESCALE",
    "H3_ROPE_IMAGE_REFERENCE_SPAN",
    "H3_ROPE_SPATIAL_SCALE",
    "H3PositionIdsBuilder",
    "build_h3_ref2va_position_ids",
    "h3_frame_position_grid",
    "h3_latent_grid_of_reference",
    "h3_spatial_position_grid",
    "h3_temporal_position_grid",
    "h3_video_rows",
    "make_h3_position_ids_fn",
    "patchify_h3_video_latents",
]


# --------------------------------------------------------------------------------------------------
# Rotary constants — before_denoise.py L36-41, verbatim.
# --------------------------------------------------------------------------------------------------

#: One latent frame spans ``5/3 * frames_per_latent`` rotary units. The ``(1, 4, 4, 4, 4)`` pattern
#: mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping — the SAME chunk contract
#: ``h3_geometry.h3_latent_frames`` encodes, seen from the rotary side.
H3_ROPE_FRAME_RESCALE: float = 5.0 / 3.0
H3_ROPE_FRAMES_PER_LATENT: tuple[int, ...] = (1, 4, 4, 4, 4)

#: The spatial axes are normalized by the square root of the latent area and scaled by 32, so a
#: SQUARE canvas spans ``[0, 32)`` regardless of resolution. That normalization is why a reference
#: encoded at its own resolution still lands in the same rotary space as the target.
H3_ROPE_SPATIAL_SCALE: int = 32

#: ⚠ An IMAGE reference takes a single INTEGER rotary slot — **not** a latent frame's ``5/3`` units.
#: ``before_denoise.py`` L646-647 states this explicitly, and it is exactly the sort of asymmetry a
#: reimplementation gets wrong: an image is one frame, so the "obvious" move is to bill it like one.
H3_ROPE_IMAGE_REFERENCE_SPAN: float = 1.0

#: The reference kinds this module builds coordinates for. ``"audio"`` is deliberately absent — see
#: the module docstring: a reference soundtrack changes the packed ORDER in a way this module cannot
#: vouch for against ``train/h3_step.h3_segment_offsets``.
H3_ROPE_REFERENCE_KINDS: tuple[str, ...] = ("image", "video")


# --------------------------------------------------------------------------------------------------
# 1. Patchify — the intra-patch element order (before_denoise.py L44-73).
# --------------------------------------------------------------------------------------------------


def patchify_h3_video_latents(
    latents: torch.Tensor, patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """Pack ``(B, C, F, H, W)`` video latents into transformer rows. **Transcribed, not derived.**

    Row order is frame-block major then row-major; inside a row the elements run
    ``(channel, patch_t, patch_h, patch_w)`` with CHANNEL SLOWEST. That ordering is the whole reason
    this function exists as a transcription: ``permute(0, 2, 4, 6, 1, 3, 5, 7)`` puts the three
    block axes first and the four intra-patch axes after them in ``(c, t, h, w)`` order, and any
    other arrangement produces a tensor of exactly the right shape carrying scrambled features.

    Source: ``before_denoise.py::patchify_video_latents`` L44-73 at the pinned ``DIFFUSERS_SHA``.

    Args:
        latents: ``(batch_size, channels, num_frames, height, width)`` LATENT tensor.
        patch_size: the transformer's ``(t, h, w)`` patch — H3 ships ``(1, 2, 2)``. Read it off the
            live ``model.config.patch_size``; never restate it (``models/h3_loader`` owns the
            measured value).

    Returns:
        ``(batch_size * num_patches, channels * prod(patch_size))``.
    """
    patch_t, patch_h, patch_w = (int(p) for p in patch_size)
    if latents.ndim != 5:
        raise ValueError(
            f"invalid latents: patchify consumes (batch, channels, frames, height, width), got "
            f"{latents.ndim}-D {tuple(latents.shape)}. Use h3_video_rows for the "
            f"[C, F, H, W] PrecomputedDataset payload."
        )
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by the patch "
            f"{(patch_t, patch_h, patch_w)}."
        )

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def h3_video_rows(value: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Coerce a latent payload to the ``[1, rows, patch_dim]`` form the packed batch consumes.

    Accepts, in the three forms that actually reach the training path:

      * ``[C, F, H, W]`` — what ``PrecomputedDataset`` returns for an ``h3_``-prefixed video-latent
        source (``_normalize_video_latents`` output);
      * ``[1, C, F, H, W]`` — the same payload after a ``DataLoader`` collate;
      * ``[1, rows, patch_dim]`` — already row-major; passed through UNCHANGED so a caller that
        patchified upstream (or a CPU test driving synthetic rows) is never double-patchified.

    A leading batch dimension other than 1 is refused: H3 packs ONE sample per sequence, and the
    project's ``batch_size`` is LOCKED at 1 for exactly this reason (every sample carries its own
    geometry).
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"invalid latents: expected a torch.Tensor, got {type(value).__name__}.")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(
                f"invalid latents: H3 packs one sample per sequence, so the leading batch "
                f"dimension must be 1; got {tuple(value.shape)}."
            )
        return value
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5:
        raise ValueError(
            f"invalid latents: expected [C, F, H, W], [1, C, F, H, W] or [1, rows, patch_dim], got "
            f"{value.ndim}-D {tuple(value.shape)}."
        )
    if value.shape[0] != 1:
        raise ValueError(
            f"invalid latents: H3 packs one sample per sequence, so the leading batch dimension "
            f"must be 1; got {tuple(value.shape)}."
        )
    return patchify_h3_video_latents(value, patch_size).unsqueeze(0)


# --------------------------------------------------------------------------------------------------
# 2. The rotary grids (before_denoise.py L76-133).
# --------------------------------------------------------------------------------------------------


def h3_spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis — ``before_denoise.py::_spatial_position_grid``.

    ``dim // patch`` coordinates centred on the unit interval and scaled by 32; the right endpoint
    is EXCLUDED, so a square canvas spans ``[0, 32)``.

    ⚠ Built with numpy on purpose. ``np.linspace(..., endpoint=False)`` is
    ``start + arange(num) * (stop - start) / num``; ``torch.linspace`` computes something else, and
    the float64 grid has to be reproduced exactly (the reference implementation carries this same
    warning inline). Do not "simplify" this to torch.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * H3_ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def h3_temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame from ``origin`` — spacing ``5/3 * (1, 4, 4, 4, 4)``.

    Source: ``before_denoise.py::_temporal_position_grid`` L89-98. The spacing is NON-UNIFORM and
    the first latent frame is the short one — a uniform ``arange`` here would drift the whole clip's
    temporal coordinates and, again, raise nothing.
    """
    spans = torch.tensor(
        [
            H3_ROPE_FRAME_RESCALE * H3_ROPE_FRAMES_PER_LATENT[index % len(H3_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def h3_frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, w)`` rotary coordinates of ONE latent frame, plus the width axis they were built from.

    Source: ``before_denoise.py::_frame_position_grid`` L101-109. ``indexing="ij"`` then
    ``reshape(-1)`` means HEIGHT is the slow axis and WIDTH the fast one — the same row-major order
    ``patchify_h3_video_latents`` emits, which is what keeps coordinates and rows aligned.

    The width axis is returned because the audio rows are pinned to ITS two extremes.
    """
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = h3_spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = h3_spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_h3_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    """Place one CHANNEL-MAJOR audio block — ``before_denoise.py::_fill_audio_positions`` L112-133.

    Audio rows carry NO height coordinate (axis 1 stays 0.0) and are pinned to the two extremes of
    the width grid of *their own* block: the target grid for the generated soundtrack.
    """
    if num_audio_latents <= 0:
        return
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def _video_reference_span(num_latent_frames: int) -> float:
    """Rotary time a VIDEO reference advances the shared clock by — ``before_denoise.py`` L682-685.

    Summed sequentially in float64, deliberately: the reference implementation keeps this separate
    from its pairwise-summing sibling because the two orders differ in the last ulp from 16 latent
    frames onwards, and it keeps both, one per call site.
    """
    return sum(
        H3_ROPE_FRAME_RESCALE * H3_ROPE_FRAMES_PER_LATENT[index % len(H3_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


# --------------------------------------------------------------------------------------------------
# 3. The ref2va packed layout (before_denoise.py::build_ref2va_packed_sequence L559-722).
# --------------------------------------------------------------------------------------------------


def build_h3_ref2va_position_ids(
    n_text: int,
    reference_grids: Sequence[tuple[int, int, int]],
    target_grid: tuple[int, int, int],
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    *,
    audio_channels: int = H3_AUDIO_CHANNELS,
    reference_kinds: Sequence[str] | None = None,
) -> torch.Tensor:
    """Build the ``[seq, 3]`` float64 ``(t, h, w)`` RoPE coordinates of a ref2va packed sequence.

    Layout, verbatim from the reference implementation::

        [ text | reference blocks | target audio | target video ]

    with the shared rotary clock starting where the TEXT rows end and every reference block pushing
    it forward by the time that block occupies — 1.0 for an image, its ``5/3``-summed frame span for
    a video. That is why D-10-REFORDER is load-bearing: reordering the same references is a
    different request, and the coordinates prove it.

    Args:
        n_text: number of text rows (Qwen3-VL presentation: labels + vision blocks, prompt LAST).
        reference_grids: one ``(latent_frames, latent_height, latent_width)`` per reference, in
            PACKED ORDER. An image reference is exactly one latent frame.
        target_grid: the target's ``(latent_frames, latent_height, latent_width)``.
        num_audio_latents: target audio latents PER CHANNEL (the row count is this times
            ``audio_channels``).
        patch_size: the transformer's ``(t, h, w)`` patch, off the live model config.
        audio_channels: the channel count the soundtrack is packed channel-major over.
        reference_kinds: per-reference ``"image"`` / ``"video"``; defaults to all ``"image"``, which
            is the Phase-10 corpus (hand-drawn stills). ``"audio"`` is REFUSED — see the module
            docstring.

    Returns:
        ``[seq, 3]`` float64. float64 is not incidental: H3's RoPE is float64 throughout and the
        area-normalized grids carry meaningful low bits.
    """
    _, patch_h, patch_w = (int(p) for p in patch_size)
    grids = [tuple(int(v) for v in g) for g in reference_grids]
    kinds = list(reference_kinds) if reference_kinds is not None else ["image"] * len(grids)
    if len(kinds) != len(grids):
        raise ValueError(
            f"reference_kinds has {len(kinds)} entries but {len(grids)} reference grid(s) were "
            f"given — they are parallel sequences in packed order."
        )
    for kind in kinds:
        if kind not in H3_ROPE_REFERENCE_KINDS:
            raise ValueError(
                f"unsupported reference kind {kind!r}: this builder emits coordinates for "
                f"{H3_ROPE_REFERENCE_KINDS} only. A reference SOUNDTRACK (an 'audio' reference, or "
                f"a video reference carrying one) inserts reference-audio rows into the packed "
                f"order, and train/h3_step.h3_segment_offsets places the whole audio block AFTER "
                f"every reference video block — an ordering this module cannot vouch for against "
                f"the reference implementation. Measured: 0 of 44 Phase-10 corpus clips carry an "
                f"audio stream (D-10-AUDIO), so this is refused rather than guessed."
            )

    target_frames, target_h, target_w = (int(v) for v in target_grid)
    rows_per_target_frame = (target_h // patch_h) * (target_w // patch_w)
    n_target_video = target_frames * rows_per_target_frame
    n_target_audio = int(num_audio_latents) * int(audio_channels)
    n_reference_video = sum(f * (h // patch_h) * (w // patch_w) for f, h, w in grids)

    sequence_length = n_text + n_reference_video + n_target_audio + n_target_video
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    target_frame_grid, target_width_grid = h3_frame_position_grid(
        target_h, target_w, patch_h, patch_w
    )

    # Reference blocks, in request order. ``rotary_time`` is the shared audio/video clock: it starts
    # where the text rows end and every block pushes it forward by the time that block occupies.
    cursor = n_text
    rotary_time = float(n_text)
    for kind, (frames, height, width) in zip(kinds, grids, strict=True):
        n_video_rows = frames * (height // patch_h) * (width // patch_w)
        rows = slice(cursor, cursor + n_video_rows)
        cursor = rows.stop
        frame_grid, _ = h3_frame_position_grid(height, width, patch_h, patch_w)
        if kind == "image":
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image is a single frame and takes a single INTEGER rotary slot, not 5/3 units.
            rotary_time += H3_ROPE_IMAGE_REFERENCE_SPAN
        else:  # "video"
            frame_time = h3_temporal_position_grid(frames, rotary_time)
            position_ids[rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[rows, 1:] = frame_grid.repeat(frames, 1)
            rotary_time += _video_reference_span(frames)

    # The generated rows. Target audio and target video share the origin the references left behind.
    audio_start = cursor
    video_start = audio_start + n_target_audio
    _fill_h3_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        int(num_audio_latents),
        rotary_time,
        target_width_grid,
        int(audio_channels),
    )
    frame_time = h3_temporal_position_grid(target_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(rows_per_target_frame)
    position_ids[video_start:, 1:] = target_frame_grid.repeat(target_frames, 1)

    return position_ids


# --------------------------------------------------------------------------------------------------
# 4. The strategy-facing factory — every DERIVED count is cross-checked against a MEASURED one.
# --------------------------------------------------------------------------------------------------


def h3_latent_grid_of_reference(
    width: int,
    height: int,
    short_edge: int,
    patch_size: tuple[int, int, int],
) -> tuple[int, int, int]:
    """``(latent_frames, latent_height, latent_width)`` a reference IMAGE encodes to.

    The pixel size comes from ``h3_geometry.reference_image_size`` — the SAME function
    ``prep/h3_encode.encode_h3_reference_latents`` calls — so the derivation cannot disagree with
    the encode unless ``short_edge`` differs, which the caller's cross-check then catches.

    The VAE spatial compression ratio is DERIVED (``H3_CANVAS_MULTIPLE // patch_w``), never a
    literal: ``H3_CANVAS_MULTIPLE`` is itself ``vae_spatial_compression_ratio * patch_size[2]``.
    An image reference is exactly ONE latent frame (spatial encoder only, no temporal chunking —
    ``prep/h3_encode.py`` asserts this on the encoded tensor).
    """
    _, _, patch_w = (int(p) for p in patch_size)
    ratio = H3_CANVAS_MULTIPLE // patch_w
    pixel_height, pixel_width = reference_image_size(width, height, short_edge=short_edge)
    return 1, pixel_height // ratio, pixel_width // ratio


class H3PositionIdsBuilder:
    """The ``position_ids_fn`` seam ``conditioning/h3_ref.H3RefStrategy`` accepts, made real.

    The strategy's callback signature hands over ROW COUNTS plus the resolved ``H3Reference``
    objects; H3's RoPE needs GEOMETRY. This builder bridges the two by re-deriving each reference's
    latent grid through ``h3_geometry`` — and then **proves the derivation** against the row counts
    the strategy measured off the real tensors:

      * derived reference rows must equal ``n_cond_video``;
      * derived target rows must equal ``n_target_video``;
      * derived target audio rows must equal ``n_target_audio``;
      * the finished tensor must be exactly ``seq_len`` rows.

    That is what makes deriving geometry from a config safe rather than a second guess: if the cache
    on the Volume was encoded at a different short edge, a different canvas or a different frame
    count than this config declares, the FIRST batch raises with both numbers named — instead of
    training against coordinates for a geometry the latents do not have.

    ``n_cond_audio`` must be 0. A reference soundtrack changes the packed order (module docstring),
    and the Phase-10 corpus has none.
    """

    def __init__(
        self,
        *,
        target_frames: int | Sequence[int],
        target_aspect: tuple[int, int],
        reference_short_edge: int,
        patch_size: tuple[int, int, int],
        audio_channels: int = H3_AUDIO_CHANNELS,
    ) -> None:
        """``target_frames`` is a single count OR a sequence of declared FRAME BUCKETS.

        FRAME-COUNT BUCKETING. A run may mix target lengths — e.g. F=5 (one still, staged as a
        5-frame freeze) alongside F=22 (11 drawings held on twos) — so a corpus of uneven shot
        lengths can contribute stills from short shots and sequences from long ones in the SAME
        run. Each declared bucket is priced here once and the SAMPLE's own bucket is resolved
        per call from the row counts the strategy measured off the real tensors.

        ⛔ Buckets are keyed by LATENT-FRAME COUNT, never by target ROW count. Row count is
        injective in F only while the canvas is fixed; across canvases it collides (F=56 on
        768x768 gives 17*576 = 9,792 rows and F=39 on 768x1088 gives 12*816 = 9,792 — both legal
        under ``resolve_canvas_size``). Aspect bucketing is a separate change and keying by rows
        would hand it a silent wrong-geometry selection. The constructor also ASSERTS injectivity
        so that change trips an error instead of inheriting the bug.
        """
        if isinstance(target_frames, int):
            buckets = (int(target_frames),)
        else:
            buckets = tuple(sorted({int(f) for f in target_frames}))
        if not buckets:
            raise ValueError(
                "target_frames is empty: a builder must know at least one declared frame count. "
                "Pass an int for a single-bucket run or the declared bucket set for a "
                "frame-bucketed one."
            )
        self.target_frames_buckets: tuple[int, ...] = buckets
        # The scalar spelling survives ONLY in single-bucket mode. In a bucketed run there is no
        # one campaign frame count, and exposing a stale scalar is exactly how a consumer that
        # falls back to it (H3RefStrategy._absent_target_audio_rows reads
        # `position_ids_fn.n_target_audio_rows` via getattr) would silently price one bucket's
        # audio rows for another sample. Absent is better than wrong.
        self.target_frames: int | None = buckets[0] if len(buckets) == 1 else None
        self.target_aspect = (int(target_aspect[0]), int(target_aspect[1]))
        self.reference_short_edge = int(reference_short_edge)
        self.patch_size = tuple(int(p) for p in patch_size)  # type: ignore[assignment]
        self.audio_channels = int(audio_channels)

        # Target geometry is OWNED by conditioning/h3_geometry — the canvas resolution, the 17n+5
        # frame law and the audio-row count are all read from there, never re-derived here.
        canvas_height, canvas_width = resolve_canvas_size(*self.target_aspect)
        ratio = H3_CANVAS_MULTIPLE // self.patch_size[2]
        rows_per_latent_frame = rows_of(canvas_height, canvas_width)

        self._by_latent_frames: dict[int, dict[str, Any]] = {}
        for frames in buckets:
            latent_frames = h3_latent_frames(frames)   # re-validates the 17n+5 law per bucket
            if latent_frames in self._by_latent_frames:
                raise ValueError(
                    f"frame buckets {buckets} are not injective in latent-frame count: "
                    f"{frames} collides with "
                    f"{self._by_latent_frames[latent_frames]['target_frames']}. Two buckets that "
                    f"resolve to the same latent grid cannot be told apart from a batch."
                )
            self._by_latent_frames[latent_frames] = {
                "target_frames": frames,
                "target_grid": (latent_frames, canvas_height // ratio, canvas_width // ratio),
                "n_target_video_rows": latent_frames * rows_per_latent_frame,
                "n_target_audio_rows": h3_audio_rows(frames),
            }
        self._rows_per_latent_frame = rows_per_latent_frame
        self._by_target_video_rows: dict[int, dict[str, Any]] = {
            int(spec["n_target_video_rows"]): spec for spec in self._by_latent_frames.values()
        }

        single = self._by_latent_frames[h3_latent_frames(buckets[0])]
        self.target_grid: tuple[int, int, int] = single["target_grid"]
        self.n_target_video_rows: int = int(single["n_target_video_rows"])
        if len(buckets) == 1:
            self.n_target_audio_rows: int = int(single["n_target_audio_rows"])

    def resolve_bucket(self, n_target_video: int) -> dict[str, Any]:
        """The sample's own bucket, from the target-video rows the strategy measured.

        Loud on an undeclared count, and the message names every declared bucket — a sample whose
        cache was encoded at a frame count this run does not declare is a staging error, and
        picking the nearest bucket would train against coordinates for a geometry the latents do
        not have.
        """
        spec = self._by_target_video_rows.get(int(n_target_video))
        if spec is None:
            declared = ", ".join(
                f"F={s['target_frames']} -> {s['n_target_video_rows']} rows"
                for s in self._by_latent_frames.values()
            )
            raise ValueError(
                f"H3 RoPE geometry disagreement on the TARGET video rows: the batch carries "
                f"{n_target_video} row(s), which matches NO declared frame bucket at "
                f"h3.target_aspect={self.target_aspect} ({declared}). The cached target latents "
                f"were encoded at a different canvas or frame count than this config declares."
            )
        return spec

    def audio_rows_for(self, n_target_video: int) -> int:
        """Target-audio rows for the bucket this sample belongs to.

        ``H3RefStrategy`` calls this when a sample carries no audio stream and the rows have to be
        synthesized as noise. It is a METHOD rather than the ``n_target_audio_rows`` scalar
        precisely because a bucketed run has no single answer.
        """
        return int(self.resolve_bucket(n_target_video)["n_target_audio_rows"])

    def __call__(
        self,
        *,
        n_text: int,
        references: Sequence[Any] = (),
        n_cond_video: int = 0,
        n_cond_audio: int = 0,
        n_target_audio: int = 0,
        n_target_video: int = 0,
        seq_len: int = 0,
        **_ignored: Any,
    ) -> torch.Tensor:
        if int(n_cond_audio) != 0:
            raise ValueError(
                f"n_cond_audio is {n_cond_audio}, but this builder emits no reference-audio rows. "
                f"A reference soundtrack inserts audio rows into the reference region, which "
                f"changes the packed ORDER relative to train/h3_step.h3_segment_offsets. Measured: "
                f"0 of 44 Phase-10 corpus clips carry an audio stream (D-10-AUDIO)."
            )

        grids = [
            h3_latent_grid_of_reference(
                int(reference.width),
                int(reference.height),
                self.reference_short_edge,
                self.patch_size,
            )
            for reference in references
        ]
        _, patch_h, patch_w = self.patch_size
        derived_reference_rows = sum(
            f * (h // patch_h) * (w // patch_w) for f, h, w in grids
        )
        if derived_reference_rows != int(n_cond_video):
            raise ValueError(
                f"H3 RoPE geometry disagreement on the REFERENCE rows: the {len(grids)} resolved "
                f"reference(s) re-derive to {derived_reference_rows} row(s) at "
                f"reference_image_short_edge={self.reference_short_edge}, but the batch carries "
                f"{n_cond_video}. The cached reference latents were encoded at a different short "
                f"edge than this config declares (VAE latents cannot be spatially rescaled after "
                f"the fact — a change needs a full RE-ENCODE), or the descriptor sizes in the "
                f"payload are not the sizes that were encoded. Refusing to build coordinates for a "
                f"geometry the latents do not have."
            )
        # Resolve THIS sample's frame bucket from the rows the strategy measured off the real
        # tensors. In single-bucket mode this is the same assertion as before, phrased against a
        # set of one; the derivation is still PROVEN against the batch rather than assumed.
        bucket = self.resolve_bucket(int(n_target_video))
        if int(n_target_audio) != int(bucket["n_target_audio_rows"]):
            raise ValueError(
                f"H3 RoPE geometry disagreement on the TARGET audio rows: this sample's "
                f"{bucket['target_frames']} frames re-derive to "
                f"{bucket['n_target_audio_rows']} row(s) "
                f"(conditioning/h3_geometry.h3_audio_rows), but the batch carries {n_target_audio}."
            )
        if int(n_target_audio) % self.audio_channels:
            raise ValueError(
                f"invalid n_target_audio {n_target_audio}: audio rows are packed CHANNEL-MAJOR "
                f"over {self.audio_channels} channel(s), so the count must be divisible by it."
            )

        position_ids = build_h3_ref2va_position_ids(
            int(n_text),
            grids,
            # THIS SAMPLE's grid, not the builder's first bucket. Using `self.target_grid` here
            # would emit coordinates for the wrong number of latent frames on every sample outside
            # the first declared bucket — at a row count that still matches, so the seq_len check
            # below would pass and the run would train against silently wrong RoPE.
            bucket["target_grid"],
            int(n_target_audio) // self.audio_channels,
            self.patch_size,
            audio_channels=self.audio_channels,
        )
        if seq_len and int(position_ids.shape[0]) != int(seq_len):
            raise ValueError(
                f"built {position_ids.shape[0]} RoPE coordinate row(s) for a packed sequence of "
                f"{seq_len}. build_h3_packed_batch would reject this, but the mismatch means the "
                f"layout this builder assumes and the one the strategy assembled have diverged."
            )
        return position_ids


def make_h3_position_ids_fn(
    *,
    target_frames: int | Sequence[int],
    target_aspect: tuple[int, int],
    reference_short_edge: int,
    patch_size: tuple[int, int, int],
    audio_channels: int = H3_AUDIO_CHANNELS,
) -> Callable[..., torch.Tensor]:
    """Build the ``position_ids_fn`` for a run, from CONFIG values (D-NOHARDCODE).

    Every argument is a config field or a live-model fact: ``target_frames`` /
    ``h3.target_aspect`` / ``h3.reference_image_short_edge`` from the YAML, ``patch_size`` off
    ``model.config``. Nothing here is a literal, so a higher-fidelity campaign is a YAML edit plus
    the re-encode its short edge implies.
    """
    return H3PositionIdsBuilder(
        target_frames=target_frames,
        target_aspect=target_aspect,
        reference_short_edge=reference_short_edge,
        patch_size=patch_size,
        audio_channels=audio_channels,
    )
