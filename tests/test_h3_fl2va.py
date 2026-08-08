"""fl2va packing tests — CPU only, no GPU, no Modal, no network.

These pin the arithmetic that goes silently wrong. Every expected value is derived from the
reference implementation at the pinned DIFFUSERS_SHA, not from running our own code and blessing
whatever it printed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from signet_trainer.conditioning.h3_fl2va import (
    build_h3_fl2va_position_ids,
    h3_keyframe_anchor_time,
    h3_keyframe_rows,
    validate_keyframe_anchors,
)
from signet_trainer.conditioning.h3_packing import (
    H3_ROPE_FRAME_RESCALE,
    H3_ROPE_FRAMES_PER_LATENT,
    h3_temporal_position_grid,
)

# The live keyframe_test geometry: 16:9 canvas 1344x768, F=22 -> 7 latent frames.
PATCH = (1, 1, 1)
LATENT_H, LATENT_W = 24, 42          # 768/32, 1344/32
TARGET_7 = (7, LATENT_H, LATENT_W)
ROWS_PER_FRAME = LATENT_H * LATENT_W  # 1008


def test_keyframe_costs_exactly_one_target_latent_frame():
    """The measured budget rests on this: a keyframe is one target frame's rows, nothing more."""
    assert h3_keyframe_rows(LATENT_H, LATENT_W, 1, 1) == ROWS_PER_FRAME == 1008


def test_first_anchor_is_the_media_clock_origin():
    assert h3_keyframe_anchor_time("first", n_text=200, num_latent_frames=7) == 200.0


def test_last_anchor_matches_the_reference_arithmetic_exactly():
    """Recompute the reference's expression independently and require bit equality."""
    n_text, frames = 200, 7
    spans = np.ones(frames, dtype=np.float64) * H3_ROPE_FRAME_RESCALE
    for offset in range(len(H3_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(H3_ROPE_FRAMES_PER_LATENT)] *= H3_ROPE_FRAMES_PER_LATENT[offset]
    expected = float(n_text) + float(spans.sum()) - H3_ROPE_FRAME_RESCALE
    assert h3_keyframe_anchor_time("last", n_text, frames) == expected


def test_last_anchor_does_NOT_sit_on_the_last_video_frame():
    """Documented, deliberate divergence — see the module docstring.

    If a future change makes these equal, it has changed every keyframe's rotary time. That must be
    an argued decision, not a silent one, so the gap is pinned here with its exact value.
    """
    n_text, frames = 200, 7
    anchor = h3_keyframe_anchor_time("last", n_text, frames)
    last_video_frame = float(h3_temporal_position_grid(frames, float(n_text))[-1])
    # (1,4,4,4,4,1,4) -> spans[0] = 5/3, spans[-1] = 20/3; the anchor lands 5.0 units past the frame.
    assert anchor - last_video_frame == pytest.approx(5.0, abs=1e-12)


def test_pairwise_and_sequential_summation_are_not_interchangeable():
    """The reason this module does not reuse `_video_reference_span`.

    They agree at small frame counts and diverge in the last ulp as the series grows. The test
    asserts the ORDER we use, not that a difference exists at F=7 — at 7 frames they may still be
    bit-identical, and that is fine; what matters is that we run the reference's order.
    """
    frames = 64
    spans = np.ones(frames, dtype=np.float64) * H3_ROPE_FRAME_RESCALE
    for offset in range(len(H3_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(H3_ROPE_FRAMES_PER_LATENT)] *= H3_ROPE_FRAMES_PER_LATENT[offset]
    pairwise = float(spans.sum())
    sequential = sum(
        H3_ROPE_FRAME_RESCALE * H3_ROPE_FRAMES_PER_LATENT[i % len(H3_ROPE_FRAMES_PER_LATENT)]
        for i in range(frames)
    )
    got = h3_keyframe_anchor_time("last", 0, frames) + H3_ROPE_FRAME_RESCALE
    assert got == pairwise
    assert abs(pairwise - sequential) < 1e-9  # same value to any sane tolerance...
    # ...which is exactly why substituting one for the other would never be caught by eye.


@pytest.mark.parametrize("anchors,expected_rows", [(("first",), 1), (("first", "last"), 2)])
def test_sequence_length_and_block_boundaries(anchors, expected_rows):
    n_text, n_audio, channels = 200, 37, 2
    ids = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=anchors, target_grid=TARGET_7,
        num_audio_latents=n_audio, patch_size=PATCH, audio_channels=channels,
    )
    expected = (
        n_text
        + expected_rows * ROWS_PER_FRAME
        + n_audio * channels
        + TARGET_7[0] * ROWS_PER_FRAME
    )
    assert ids.shape == (expected, 3)
    assert ids.dtype == torch.float64


def test_keyframe_rows_share_the_targets_spatial_grid():
    """The structural claim of fl2va, asserted rather than described.

    A keyframe's (h, w) coordinates must be IDENTICAL to a generated video frame's. If they ever
    differ, the keyframe has become a reference — a picture at its own grid — and the whole reason
    for this arm is gone.
    """
    n_text = 200
    ids = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=("first", "last"), target_grid=TARGET_7,
        num_audio_latents=0, patch_size=PATCH,
    )
    kf_first = ids[n_text : n_text + ROWS_PER_FRAME]
    video_start = n_text + 2 * ROWS_PER_FRAME
    first_video_frame = ids[video_start : video_start + ROWS_PER_FRAME]
    assert torch.equal(kf_first[:, 1:], first_video_frame[:, 1:])


def test_first_keyframe_shares_the_first_video_frames_time():
    """`"first"` is the one anchor that DOES coincide: both sit at the media clock origin."""
    n_text = 200
    ids = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=("first",), target_grid=TARGET_7,
        num_audio_latents=0, patch_size=PATCH,
    )
    kf = ids[n_text : n_text + ROWS_PER_FRAME]
    video_start = n_text + ROWS_PER_FRAME
    first_video_frame = ids[video_start : video_start + ROWS_PER_FRAME]
    assert torch.equal(kf[:, 0], first_video_frame[:, 0])


def test_keyframe_block_does_not_advance_the_video_clock():
    """A reference pushes the shared clock forward; a keyframe must not.

    The generated video's temporal coordinates have to be identical whether one keyframe is packed
    or two. This is the single cheapest guard against accidentally porting ref2va semantics.
    """
    n_text = 200
    one = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=("first",), target_grid=TARGET_7,
        num_audio_latents=0, patch_size=PATCH,
    )
    two = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=("first", "last"), target_grid=TARGET_7,
        num_audio_latents=0, patch_size=PATCH,
    )
    assert torch.equal(one[n_text + ROWS_PER_FRAME :, 0], two[n_text + 2 * ROWS_PER_FRAME :, 0])


def test_text_rows_are_their_own_index_on_the_time_axis():
    ids = build_h3_fl2va_position_ids(
        n_text=8, keyframe_anchors=("first",), target_grid=(7, 2, 2),
        num_audio_latents=0, patch_size=PATCH,
    )
    assert torch.equal(ids[:8, 0], torch.arange(8, dtype=torch.float64))


@pytest.mark.parametrize(
    "bad", [(), ("middle",), ("last", "first"), ("first", "first"), ("first", "last", "first")]
)
def test_invalid_anchor_sets_are_refused(bad):
    with pytest.raises(ValueError):
        validate_keyframe_anchors(bad)


def test_audio_rows_are_pinned_to_the_width_extremes():
    n_text, n_audio, channels = 4, 3, 2
    ids = build_h3_fl2va_position_ids(
        n_text=n_text, keyframe_anchors=("first",), target_grid=(7, 2, 2),
        num_audio_latents=n_audio, patch_size=PATCH, audio_channels=channels,
    )
    rows_per_frame = 2 * 2
    audio_start = n_text + rows_per_frame
    audio = ids[audio_start : audio_start + n_audio * channels]
    assert torch.all(audio[:, 1] == 0)                      # no height coordinate
    assert len(torch.unique(audio[:, 2])) == 2              # the two width extremes only


# ── the position_ids_fn seam: catch the geometry contradiction BEFORE the 61.7 GiB loads ─────────

from signet_trainer.conditioning.h3_fl2va import make_h3_fl2va_position_ids_fn  # noqa: E402


def _fn(anchors=("first", "last"), frames=7):
    return make_h3_fl2va_position_ids_fn(
        target_grid=(frames, LATENT_H, LATENT_W), patch_size=PATCH, keyframe_anchors=anchors
    )


def test_seam_builds_when_the_realized_counts_agree():
    ids = _fn()(
        n_text=200, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=0,
        n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME,
    )
    assert ids.shape == (200 + 2 * ROWS_PER_FRAME + 74 + 7 * ROWS_PER_FRAME, 3)


def test_seam_refuses_a_keyframe_count_that_disagrees_with_the_anchors():
    """One keyframe staged, two anchors declared — the manifest/config split."""
    with pytest.raises(ValueError, match="keyframe rows disagree"):
        _fn()(
            n_text=200, n_cond_video=1 * ROWS_PER_FRAME, n_cond_audio=0,
            n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME,
        )


def test_seam_refuses_the_frame_count_confusion_that_costs_two_partition_loads():
    """The cache was encoded at 5 frames (2 latents); the config declares 22 (7 latents).

    This is trap 3 — `training_dims[2]` vs `validation.frame_count`. Live, it raises INSIDE the
    model step after both 61.7 GiB loads. Here it costs nothing.
    """
    with pytest.raises(ValueError, match="target video rows disagree"):
        _fn()(
            n_text=200, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=0,
            n_target_audio=74, n_target_video=2 * ROWS_PER_FRAME,
        )


def test_seam_refuses_a_ref2va_payload():
    with pytest.raises(ValueError, match="conditioning-audio row"):
        _fn()(
            n_text=200, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=8,
            n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME,
        )


def test_seam_refuses_a_partial_audio_channel():
    with pytest.raises(ValueError, match="not divisible by audio_channels"):
        _fn()(
            n_text=200, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=0,
            n_target_audio=75, n_target_video=7 * ROWS_PER_FRAME,
        )


def test_seam_refuses_a_seq_len_that_disagrees_with_its_own_blocks():
    with pytest.raises(ValueError, match="packed length disagrees"):
        _fn()(
            n_text=200, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=0,
            n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME, seq_len=999,
        )


def test_seam_matches_the_measured_budget_for_the_live_geometry():
    """End-to-end tie-back: the seam's row count IS the 11,378 the budget script measured.

    n_text 2232 = prompt 200 + (1008 + 2) * 2 vision + 6 * 2 label.
    """
    ids = _fn()(
        n_text=2232, n_cond_video=2 * ROWS_PER_FRAME, n_cond_audio=0,
        n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME,
    )
    assert int(ids.shape[0]) == 11_378


def test_seam_matches_the_measured_budget_for_hero_single_keyframe():
    """hero is first-frame-only: n_text 1216 = 200 + (1008 + 2) + 6."""
    ids = _fn(anchors=("first",))(
        n_text=1216, n_cond_video=1 * ROWS_PER_FRAME, n_cond_audio=0,
        n_target_audio=74, n_target_video=7 * ROWS_PER_FRAME,
    )
    assert int(ids.shape[0]) == 9_354
