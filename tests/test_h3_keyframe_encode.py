"""Keyframe pre-encode tests — CPU only, no VAE, no GPU, no Modal.

The geometry half of the keyframe path is pure arithmetic, so the guard that matters most (the
vision-token/latent-row identity) is fully testable without touching a model.
"""

from __future__ import annotations

import pytest
import torch

from signet_trainer.conditioning.h3_geometry import resolve_canvas_size, rows_of
from signet_trainer.prep.h3_keyframe_encode import (
    H3PreparedKeyframe,
    keyframe_latent_rows,
    prepare_h3_keyframe_images,
    validate_keyframe_vision_grid,
)

CANVAS_H, CANVAS_W = resolve_canvas_size(16.0, 9.0)   # 768 x 1344
ROWS = rows_of(CANVAS_H, CANVAS_W)                    # 1008


class FakeImage:
    """Minimal Pillow stand-in: .size, .convert, .resize."""

    def __init__(self, width: int, height: int):
        self.size = (width, height)

    def convert(self, _mode):
        return self

    def resize(self, size, _resample=None):
        return FakeImage(*size)


def test_canvas_is_the_live_geometry():
    assert (CANVAS_H, CANVAS_W) == (768, 1344)
    assert ROWS == keyframe_latent_rows(CANVAS_H, CANVAS_W) == 1008


def test_a_keyframe_costs_the_same_as_a_target_latent_frame():
    """The budget rests on this. A reference at short edge 896 costs 1400 — a keyframe costs 1008."""
    from signet_trainer.conditioning.h3_geometry import reference_image_size

    ref_h, ref_w = reference_image_size(1600, 896, short_edge=896)
    assert rows_of(ref_h, ref_w) == 1400
    assert keyframe_latent_rows(CANVAS_H, CANVAS_W) == 1008


def test_prepare_resizes_a_matching_aspect_to_the_canvas():
    prepared = prepare_h3_keyframe_images(
        [FakeImage(1920, 1080), FakeImage(1344, 768)], ["first", "last"], CANVAS_H, CANVAS_W
    )
    assert [k.anchor for k in prepared] == ["first", "last"]
    for k in prepared:
        assert (k.canvas_height, k.canvas_width) == (CANVAS_H, CANVAS_W)
        assert k.latent_rows == ROWS


def test_a_true_16_9_source_is_ACCEPTED_even_though_the_canvas_is_1_75():
    """The canvas snaps to 1344x768 = 1.75; a real 1920x1080 frame is 1.6% off and is CORRECT.

    A tight aspect check here rejects correctly-staged data and, worse, disagrees with the target
    path — `_h3_read_video_rgb` resizes clips to the canvas with no aspect check at all. Keyframe
    and target must get the same treatment.
    """
    prepared = prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["first"], CANVAS_H, CANVAS_W)
    assert prepared[0].latent_rows == ROWS


def test_prepare_REFUSES_a_GROSS_aspect_mismatch():
    """Portrait against a landscape canvas is the wrong file, not a resize."""
    with pytest.raises(ValueError, match="off by more than 20%"):
        prepare_h3_keyframe_images([FakeImage(768, 1344)], ["first"], CANVAS_H, CANVAS_W)


def test_prepare_is_idempotent_but_refuses_a_canvas_change():
    once = prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["first"], CANVAS_H, CANVAS_W)
    twice = prepare_h3_keyframe_images(once, ["first"], CANVAS_H, CANVAS_W)
    assert twice[0] is once[0]
    with pytest.raises(ValueError, match="prepared at"):
        prepare_h3_keyframe_images(once, ["first"], 512, 896)


def test_image_and_anchor_counts_must_match():
    with pytest.raises(ValueError, match="paired"):
        prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["first", "last"], CANVAS_H, CANVAS_W)


@pytest.mark.parametrize("bad_anchor", ["middle", "FIRST", ""])
def test_bad_anchor_is_refused(bad_anchor):
    with pytest.raises(ValueError, match="anchor"):
        H3PreparedKeyframe(FakeImage(1344, 768), bad_anchor, CANVAS_H, CANVAS_W)


def test_off_multiple_canvas_is_refused():
    with pytest.raises(ValueError, match="multiple of"):
        H3PreparedKeyframe(FakeImage(1344, 768), "first", 770, 1344)


# ── the F3 guard: vision tokens == latent rows ───────────────────────────────────────────────────


def _prepared(n=2):
    return prepare_h3_keyframe_images(
        [FakeImage(1920, 1080)] * n, ["first", "last"][:n], CANVAS_H, CANVAS_W
    )


def test_vision_grid_matching_the_latent_rows_passes():
    # 1008 = 1 * 24 * 42, the /32 grid of 1344x768 in merged patch units.
    grid = torch.tensor([[1, 24, 42], [1, 24, 42]])
    validate_keyframe_vision_grid(_prepared(2), grid)


def test_vision_grid_sized_at_reference_short_edge_is_CAUGHT():
    """THE failure this guard exists for.

    A keyframe sized at reference_image_short_edge 896 gives 1600x896 -> 28x50 = 1400 vision
    tokens against 1008 latent rows. Everything downstream assembles cleanly and every row-count
    guard passes, because n_text simply grows. Only this check sees it.
    """
    grid = torch.tensor([[1, 28, 50], [1, 28, 50]])
    with pytest.raises(ValueError, match="vision-token/latent-row identity"):
        validate_keyframe_vision_grid(_prepared(2), grid)


def test_missing_vision_grid_refuses_rather_than_assuming():
    with pytest.raises(ValueError, match="unverifiable"):
        validate_keyframe_vision_grid(_prepared(1), None)


def test_vision_grid_count_must_match_the_keyframe_count():
    with pytest.raises(ValueError, match="vision grid"):
        validate_keyframe_vision_grid(_prepared(2), torch.tensor([[1, 24, 42]]))
