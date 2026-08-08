"""conditioning.h3_fl2va — MiniMax-H3 ``fl2va`` conditioning: keyframe anchors + packed layout.

The sibling of ``conditioning/h3_ref.py``. Where a ``ref2va`` REFERENCE is a picture shown to the
model *before* the video clock starts, an ``fl2va`` KEYFRAME is a destination *on* it:

    ref2va   [ text | reference blocks | target audio | target video ]
                     ^ own spatial grid, own rotary slot, pushes the shared clock forward

    fl2va    [ text | keyframe blocks  | target audio | target video ]
                     ^ the TARGET's spatial grid, pinned to the rotary time of the
                       FIRST or LAST generated latent frame; the clock is NOT advanced

That difference is the entire reason this module exists. A reference says "here is what things look
like"; a keyframe says "the video is *at* this, *then*". Everything else about the two layouts —
the `<Picture i>` presentation, the bills-twice rule, the audio pinning — is identical, which is why
this module reuses ``h3_packing``'s helpers rather than restating any of them.

SOURCE. Upstream diffusers at the pinned ``DIFFUSERS_SHA``
(``9f169d98d0bce392a889c3b6524d0d97734dfc0e``),
``src/diffusers/modular_pipelines/minimax_h3/before_denoise.py::build_packed_sequence`` L267-360.
Read at that SHA, not recalled.

CRITICAL — the CONFINED tier
----------------------------
Module scope is ``numpy`` + ``torch`` + stdlib + ``conditioning/h3_packing`` (itself confined). No
``modal``, no ``diffusers``, no ``peft``. This file is unit-testable on Windows with zero GPU and
zero Modal spend, which is the point: the packing arithmetic is where an fl2va port goes silently
wrong, and silence is what this project keeps paying for.

⚠ THE TWO SUMMATION ORDERS ARE NOT INTERCHANGEABLE, AND THIS IS THE TRAP
------------------------------------------------------------------------
The ``"last"`` anchor time is computed with **numpy's PAIRWISE summation** over a strided array —
that is literally how the reference computes it. ``h3_packing._video_reference_span`` sums the same
series **sequentially**, for the ref2va call site. The two differ in the last ulp from 16 latent
frames onwards, and ``h3_packing`` already documents that it deliberately keeps both, one per call
site. Substituting the sequential sum here would produce rotary coordinates that are correct to
~1e-16, train without error, and be wrong in a way no test that rounds would ever catch.

⚠ A "LAST" KEYFRAME DOES **NOT** LAND ON THE LAST VIDEO FRAME'S ROTARY TIME
---------------------------------------------------------------------------
Upstream computes ``anchor_time = n_text + spans.sum() - RESCALE``, i.e. ``n_text + sum(spans[1:])``.
The last generated latent frame sits at ``n_text + sum(spans[:-1])`` (``h3_temporal_position_grid``).
Those agree only when ``spans[0] == spans[-1]``, and the ``(1, 4, 4, 4, 4)`` cycle makes that false
for most frame counts — at F=22 (7 latent frames) the cycle is ``(1,4,4,4,4,1,4)``, so the anchor
lands 5.0 rotary units PAST the last frame. That is the reference's behaviour, it is deliberate
here, and ``tests/test_h3_fl2va.py`` pins the exact numbers so a future "fix" has to argue with a
test instead of silently shifting every keyframe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from signet_trainer.conditioning.h3_packing import (
    H3_AUDIO_CHANNELS,
    H3_ROPE_FRAME_RESCALE,
    H3_ROPE_FRAMES_PER_LATENT,
    h3_frame_position_grid,
    h3_temporal_position_grid,
)

__all__ = [
    "VALID_KEYFRAME_ANCHORS",
    "build_h3_fl2va_position_ids",
    "h3_keyframe_anchor_time",
    "h3_keyframe_rows",
    "make_h3_fl2va_position_ids_fn",
    "validate_keyframe_anchors",
]

VALID_KEYFRAME_ANCHORS: tuple[str, ...] = ("first", "last")


def validate_keyframe_anchors(anchors: Sequence[str]) -> tuple[str, ...]:
    """Check anchors are ``"first"``/``"last"`` and that the order is one a caller meant.

    ORDER IS MEANING, exactly as D-10-REFORDER is for references: the anchors are consumed
    positionally and the i-th anchor binds the i-th ``<Picture i>`` block. ``("last", "first")`` is
    a legal but almost certainly unintended request — the caption would say Picture 1 aligns with
    the 0.00-second mark while the tensor pins it to the end — so it is refused rather than packed.
    """
    anchors = tuple(str(a) for a in anchors)
    if not anchors:
        raise ValueError(
            "fl2va needs at least one keyframe. A zero-keyframe request is `t2va`, a different "
            "task with a different text presentation — say so explicitly rather than packing an "
            "empty condition block."
        )
    if len(anchors) > 2:
        raise ValueError(
            f"fl2va takes at most 2 keyframes (first and last), got {len(anchors)}: {anchors}. "
            "More would need a rotary anchor the reference implementation does not define."
        )
    for index, anchor in enumerate(anchors):
        if anchor not in VALID_KEYFRAME_ANCHORS:
            raise ValueError(
                f"keyframe anchor {index} is {anchor!r}; must be one of {VALID_KEYFRAME_ANCHORS}."
            )
    if len(anchors) == 2 and anchors != ("first", "last"):
        raise ValueError(
            f"keyframe anchors {anchors} are out of order or duplicated. Two keyframes must be "
            '("first", "last") — the anchor binds the `<Picture i>` block positionally, so any '
            "other arrangement packs a request whose caption and tensor disagree."
        )
    return anchors


def h3_keyframe_rows(latent_height: int, latent_width: int, patch_h: int, patch_w: int) -> int:
    """Packed rows ONE keyframe occupies: exactly one TARGET latent frame's worth.

    ``before_denoise.py`` L307-310: ``rows_per_frame = (latent_height // patch_h) * (latent_width //
    patch_w)`` and ``num_condition_rows = len(keyframe_anchors) * rows_per_frame``. The keyframe is
    put onto the target canvas (``encoders.py:237``), so it does NOT get a reference's independent
    ``reference_image_short_edge`` sizing — which is why a keyframe is CHEAPER than a reference at
    this project's geometry (1008 rows vs 1400 at 16:9 / short edge 896).
    """
    return (latent_height // patch_h) * (latent_width // patch_w)


def h3_keyframe_anchor_time(anchor: str, n_text: int, num_latent_frames: int) -> float:
    """Rotary time a keyframe block sits at — ``before_denoise.py`` L325-337.

    ``"first"`` is the origin of the media clock, which is where the text rows end.

    ``"last"`` is ``n_text + spans.sum() - RESCALE``, where ``spans`` is the per-latent-frame rotary
    span built by STRIDED numpy multiplication and summed by numpy's PAIRWISE algorithm. Both the
    construction and the summation order are copied from the reference deliberately — see this
    module's docstring for why the sequential sibling in ``h3_packing`` is not a substitute.
    """
    if anchor == "first":
        return float(n_text)
    if anchor != "last":
        raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")

    # Verbatim construction: fill with the rescale, then multiply each residue class by its
    # frames-per-latent. Equivalent to a comprehension, but the STRIDED form is what the reference
    # runs and it is what makes the pairwise summation tree — and therefore the last ulp — match.
    spans = np.ones(num_latent_frames, dtype=np.float64) * H3_ROPE_FRAME_RESCALE
    for offset in range(len(H3_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(H3_ROPE_FRAMES_PER_LATENT)] *= H3_ROPE_FRAMES_PER_LATENT[offset]
    return float(n_text) + float(spans.sum()) - H3_ROPE_FRAME_RESCALE


def build_h3_fl2va_position_ids(
    n_text: int,
    keyframe_anchors: Sequence[str],
    target_grid: tuple[int, int, int],
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    *,
    audio_channels: int = H3_AUDIO_CHANNELS,
) -> torch.Tensor:
    """The ``[seq, 3]`` float64 ``(t, h, w)`` RoPE coordinates of an fl2va packed sequence.

    Layout::

        [ text | keyframe blocks | target audio | target video ]

    Args:
        n_text: number of text rows. Qwen3-VL presentation: a ``"<Picture i>: "`` label and a vision
            block per keyframe, then the prompt LAST and verbatim.
        keyframe_anchors: ``"first"`` / ``"last"`` per keyframe, in PACKED ORDER.
        target_grid: the target's ``(latent_frames, latent_height, latent_width)``.
        num_audio_latents: target audio latents PER CHANNEL.
        patch_size: the transformer's ``(t, h, w)`` patch, off the live model config.
        audio_channels: channels the soundtrack is packed channel-major over.

    Returns:
        ``[sequence_length, 3]`` float64.
    """
    anchors = validate_keyframe_anchors(keyframe_anchors)
    if n_text < 0:
        raise ValueError(f"n_text must be >= 0, got {n_text}.")
    if num_audio_latents < 0:
        raise ValueError(f"num_audio_latents must be >= 0, got {num_audio_latents}.")

    num_latent_frames, latent_height, latent_width = (int(v) for v in target_grid)
    _, patch_h, patch_w = (int(p) for p in patch_size)

    rows_per_frame = h3_keyframe_rows(latent_height, latent_width, patch_h, patch_w)
    num_condition_rows = len(anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = n_text + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = n_text
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)

    frame_grid, width_grid = h3_frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # The keyframes. Each takes the TARGET's spatial grid — the same `frame_grid` the generated
    # video rows use — and a single time coordinate. That shared spatial grid is the structural
    # claim of fl2va: the keyframe is the video at that instant, not a picture of it.
    for index, anchor in enumerate(anchors):
        anchor_time = h3_keyframe_anchor_time(anchor, n_text, num_latent_frames)
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio: channel-major, sharing the video clock, no height coordinate, pinned to the two
    # extremes of the width axis. Identical to ref2va (`before_denoise.py` L343-353).
    if num_audio_rows:
        audio_time = float(n_text) + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
        position_ids[audio_start:video_start, 2] = torch.cat(
            [
                torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
                torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
            ]
        )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = h3_temporal_position_grid(num_latent_frames, float(n_text))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    return position_ids


def make_h3_fl2va_position_ids_fn(
    *,
    target_grid: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    keyframe_anchors: Sequence[str],
    audio_channels: int = H3_AUDIO_CHANNELS,
) -> "Callable[..., torch.Tensor]":
    """The ``position_ids_fn`` seam an fl2va strategy accepts, with the geometry CROSS-CHECKED.

    ``H3RefStrategy`` calls its injected ``position_ids_fn`` with the REALIZED row counts it just
    assembled (``n_text`` / ``n_cond_video`` / ``n_cond_audio`` / ``n_target_audio`` /
    ``n_target_video`` / ``seq_len``). This builder computes the same counts INDEPENDENTLY from the
    declared geometry and refuses when they disagree.

    ⛔ WHY THE CHECK IS HERE AND NOT LATER. The realized-vs-declared mismatch is this campaign's
    most expensive recurring failure: the cache is encoded at one frame count while the config
    declares another (``training_dims[2]`` vs ``validation.frame_count`` are DIFFERENT NUMBERS), and
    the contradiction surfaces as *"RoPE geometry disagreement ... re-derives to 7056 row(s), but
    the batch carries 2016"* — raised INSIDE the model step, i.e. after both 61.7 GiB partition
    loads have been paid for. Every input this function needs is an integer available before any
    weight is touched, so the same contradiction is caught for free.

    The message names BOTH numbers and which side is which, because the failure is symmetric and
    "shapes disagree" sends the reader to the wrong file half the time.
    """
    anchors = validate_keyframe_anchors(keyframe_anchors)
    num_latent_frames, latent_height, latent_width = (int(v) for v in target_grid)
    _, patch_h, patch_w = (int(p) for p in patch_size)
    rows_per_frame = h3_keyframe_rows(latent_height, latent_width, patch_h, patch_w)

    expected_cond_video = len(anchors) * rows_per_frame
    expected_target_video = num_latent_frames * rows_per_frame

    def position_ids_fn(
        *,
        n_text: int,
        n_cond_video: int,
        n_cond_audio: int = 0,
        n_target_audio: int = 0,
        n_target_video: int,
        seq_len: int | None = None,
        **_ignored: object,
    ) -> torch.Tensor:
        if n_cond_video != expected_cond_video:
            raise ValueError(
                f"fl2va keyframe rows disagree: the batch carries {n_cond_video} conditioning "
                f"row(s), but {len(anchors)} keyframe(s) on a "
                f"{latent_height}x{latent_width} latent grid re-derive to {expected_cond_video} "
                f"({len(anchors)} x {rows_per_frame}). Either the staged keyframes were encoded at "
                f"a different canvas than `target_aspect` declares, or the anchor count and the "
                f"manifest's keyframe count differ. Fix the DATA or the CONFIG — do not relax this."
            )
        if n_target_video != expected_target_video:
            raise ValueError(
                f"fl2va target video rows disagree: the batch carries {n_target_video} row(s), but "
                f"{num_latent_frames} latent frame(s) on a {latent_height}x{latent_width} grid "
                f"re-derive to {expected_target_video} ({num_latent_frames} x {rows_per_frame}). "
                f"This is the `training_dims[2]` vs `validation.frame_count` confusion: the CACHE "
                f"was encoded at one frame count and this config declares another. The data is the "
                f"authority — re-read the frame count off the staged clips."
            )
        if n_cond_audio:
            raise ValueError(
                f"fl2va got {n_cond_audio} conditioning-audio row(s). The keyframe pathway has no "
                f"audio conditioning; a non-zero count means a ref2va payload reached an fl2va "
                f"strategy."
            )

        num_audio_latents = int(n_target_audio) // int(audio_channels) if audio_channels else 0
        if num_audio_latents * audio_channels != int(n_target_audio):
            raise ValueError(
                f"target audio rows {n_target_audio} are not divisible by audio_channels "
                f"{audio_channels}; the block is packed channel-major and a partial channel would "
                f"mis-pin every audio row's width coordinate."
            )

        ids = build_h3_fl2va_position_ids(
            n_text=int(n_text),
            keyframe_anchors=anchors,
            target_grid=(num_latent_frames, latent_height, latent_width),
            num_audio_latents=num_audio_latents,
            patch_size=(1, patch_h, patch_w),
            audio_channels=int(audio_channels),
        )
        if seq_len is not None and int(seq_len) != int(ids.shape[0]):
            raise ValueError(
                f"fl2va packed length disagrees: the strategy assembled {int(seq_len)} row(s), the "
                f"RoPE builder produced {int(ids.shape[0])}. These are built from the same "
                f"declared geometry, so a mismatch means one of the four blocks was assembled from "
                f"a different source than it was priced from."
            )
        return ids

    return position_ids_fn
