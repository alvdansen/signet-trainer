"""Keyframe pre-encode tests — CPU only, no VAE, no GPU, no Modal.

The geometry half of the keyframe path is pure arithmetic, so the guard that matters most (the
vision-token/latent-row identity) is fully testable without touching a model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from signet_trainer.conditioning.h3_geometry import resolve_canvas_size, rows_of
from signet_trainer.prep.h3_encode import _grid_vision_tokens, _vision_merge_length
from signet_trainer.prep.h3_keyframe_encode import (
    H3PreparedKeyframe,
    keyframe_latent_rows,
    prepare_h3_keyframe_images,
    validate_keyframe_vision_grid,
)

CANVAS_H, CANVAS_W = resolve_canvas_size(16.0, 9.0)   # 768 x 1344
ROWS = rows_of(CANVAS_H, CANVAS_W)                    # 1008


class FakeProcessor:
    """Minimal Qwen3-VL processor stand-in: only `.image_processor.merge_size` is read."""

    def __init__(self, merge_size: int = 2):
        self.image_processor = SimpleNamespace(merge_size=merge_size)


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


def test_prepare_idempotent_branch_REFUSES_a_reversed_anchor():
    """MAJOR audit finding: the idempotent branch used to check only the canvas.

    A single already-prepared "first" keyframe re-driven as "last" is a legal anchor SET on its
    own (one anchor, valid membership) so `validate_keyframe_anchors` alone cannot catch it — the
    per-item anchor comparison in the idempotent branch is what has to.
    """
    once = prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["first"], CANVAS_H, CANVAS_W)
    with pytest.raises(ValueError, match="anchor"):
        prepare_h3_keyframe_images(once, ["last"], CANVAS_H, CANVAS_W)


def test_prepare_idempotent_branch_REFUSES_a_reversed_pair():
    """PHASE A prepares [first, last]; a PHASE B re-drive from a last-then-first manifest read
    must not silently pass the already-resized pair through with the anchors swapped."""
    phase_a = prepare_h3_keyframe_images(
        [FakeImage(1920, 1080), FakeImage(1920, 1080)], ["first", "last"], CANVAS_H, CANVAS_W
    )
    # A reversed 2-anchor request is itself illegal (`validate_keyframe_anchors` order rule), so
    # exercise the idempotent-branch check directly with a 1-at-a-time re-drive of each slot.
    with pytest.raises(ValueError, match="anchor"):
        prepare_h3_keyframe_images([phase_a[0]], ["last"], CANVAS_H, CANVAS_W)
    with pytest.raises(ValueError, match="anchor"):
        prepare_h3_keyframe_images([phase_a[1]], ["first"], CANVAS_H, CANVAS_W)


def test_image_and_anchor_counts_must_match():
    with pytest.raises(ValueError, match="paired"):
        prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["first", "last"], CANVAS_H, CANVAS_W)


# ── validate_keyframe_anchors is now wired into the prep path ───────────────────────────────────


def test_prepare_REFUSES_duplicated_anchors():
    with pytest.raises(ValueError, match="out of order or duplicated"):
        prepare_h3_keyframe_images(
            [FakeImage(1920, 1080), FakeImage(1920, 1080)], ["first", "first"], CANVAS_H, CANVAS_W
        )


def test_prepare_REFUSES_a_reversed_pair_up_front():
    with pytest.raises(ValueError, match="out of order or duplicated"):
        prepare_h3_keyframe_images(
            [FakeImage(1920, 1080), FakeImage(1920, 1080)], ["last", "first"], CANVAS_H, CANVAS_W
        )


def test_prepare_REFUSES_more_than_two_keyframes():
    with pytest.raises(ValueError, match="at most 2"):
        prepare_h3_keyframe_images(
            [FakeImage(1920, 1080)] * 3, ["first", "last", "first"], CANVAS_H, CANVAS_W
        )


def test_prepare_REFUSES_an_empty_request():
    with pytest.raises(ValueError, match="at least one keyframe"):
        prepare_h3_keyframe_images([], [], CANVAS_H, CANVAS_W)


def test_prepare_REFUSES_an_illegal_anchor_before_any_resize_is_paid_for():
    """Fails at $0 — the caller never reaches the LANCZOS resize for an illegal anchor set."""
    with pytest.raises(ValueError, match="anchor"):
        prepare_h3_keyframe_images([FakeImage(1920, 1080)], ["middle"], CANVAS_H, CANVAS_W)


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
    # image_grid_thw is UNMERGED patch units (patch_size 16): 768/16 x 1344/16 = 48 x 84.
    # 1 * 48 * 84 = 4032 unmerged patches // merge_size**2 (4) = 1008 == the latent-row count.
    grid = torch.tensor([[1, 48, 84], [1, 48, 84]])
    validate_keyframe_vision_grid(_prepared(2), grid, FakeProcessor())


def test_omitting_the_merge_divide_would_reject_every_correctly_sized_keyframe():
    """BLOCKER regression pin: `t * h * w` with no `// merge_size**2` is the OPPOSITE bug.

    The unmerged grid for a correctly-prepared 1344x768 keyframe is [1, 48, 84]. A guard that
    forgets the divide computes 4032 against the 1008 latent rows and raises on exactly the data
    it exists to accept — while a keyframe mistakenly sized at half the canvas ([1, 24, 42], the
    merged-unit literal the old test bug hand-fed) would produce 1008 raw and wrongly PASS.
    """
    correctly_sized = torch.tensor([[1, 48, 84], [1, 48, 84]])
    validate_keyframe_vision_grid(_prepared(2), correctly_sized, FakeProcessor())  # must not raise

    wrongly_sized_but_bare_product_matches = torch.tensor([[1, 24, 42], [1, 24, 42]])
    with pytest.raises(ValueError, match="vision-token/latent-row identity"):
        validate_keyframe_vision_grid(_prepared(2), wrongly_sized_but_bare_product_matches, FakeProcessor())


def test_vision_grid_sized_at_reference_short_edge_is_CAUGHT():
    """THE failure this guard exists for.

    A keyframe sized at reference_image_short_edge 896 gives 1600x896 -> unmerged grid 56x100 ->
    5600 // 4 = 1400 vision tokens against 1008 latent rows. Everything downstream assembles
    cleanly and every row-count guard passes, because n_text simply grows. Only this check sees it.
    """
    grid = torch.tensor([[1, 56, 100], [1, 56, 100]])
    with pytest.raises(ValueError, match="vision-token/latent-row identity"):
        validate_keyframe_vision_grid(_prepared(2), grid, FakeProcessor())


def test_missing_vision_grid_refuses_rather_than_assuming():
    with pytest.raises(ValueError, match="unverifiable"):
        validate_keyframe_vision_grid(_prepared(1), None, FakeProcessor())


def test_vision_grid_count_must_match_the_keyframe_count():
    with pytest.raises(ValueError, match="vision grid"):
        validate_keyframe_vision_grid(_prepared(2), torch.tensor([[1, 48, 84]]), FakeProcessor())


def test_vision_grid_guard_reads_merge_size_off_the_processor_never_a_literal():
    """A different `merge_size` changes the divisor; the guard must track it, not hardcode 4."""
    # merge_size=1 -> divisor 1, so the UNMERGED grid itself must equal the latent rows.
    grid = torch.tensor([[1, 24, 42], [1, 24, 42]])  # 1*24*42 = 1008, unmerged == latent rows here
    validate_keyframe_vision_grid(_prepared(2), grid, FakeProcessor(merge_size=1))
    with pytest.raises(ValueError, match="vision-token/latent-row identity"):
        validate_keyframe_vision_grid(_prepared(2), grid, FakeProcessor(merge_size=2))


# ── parity: the guard's arithmetic agrees with h3_geometry's, at every merge size ────────────────


@pytest.mark.parametrize("merge_size", [1, 2, 4])
def test_guard_arithmetic_agrees_with_h3_geometry_rows_of(merge_size):
    """The audited defect was a divisor disagreement between this guard and every other site in
    the repo (h3_geometry.py, h3_encode.py). Pin the agreement structurally, not just at
    merge_size=2: for ANY merge size, an unmerged grid built as `(rows_of(h, w) * merge_size**2)`
    patches (i.e. exactly what a real /16 processor grid would report) must validate as equal to
    `rows_of(h, w)` through both `_grid_vision_tokens` directly and the guard end-to-end.
    """
    merge_length = merge_size**2
    latent_rows = rows_of(CANVAS_H, CANVAS_W)  # 1008
    # Distribute the unmerged patch count across h/w so t*h*w // merge_length == latent_rows,
    # mirroring how a real processor reports (t, h, w) rather than a single flattened count.
    unmerged_h = (CANVAS_H // 32) * merge_size
    unmerged_w = (CANVAS_W // 32) * merge_size
    assert unmerged_h * unmerged_w // merge_length == latent_rows
    assert _grid_vision_tokens((1, unmerged_h, unmerged_w), merge_length) == latent_rows

    grid = torch.tensor([[1, unmerged_h, unmerged_w]])
    validate_keyframe_vision_grid(_prepared(1), grid, FakeProcessor(merge_size=merge_size))


def test_vision_merge_length_helper_is_reused_not_restated():
    """`_vision_merge_length` is the single source for `merge_size ** 2` (D-10-DEF-4)."""
    assert _vision_merge_length(FakeProcessor(merge_size=2)) == 4
    assert _vision_merge_length(FakeProcessor(merge_size=1)) == 1
