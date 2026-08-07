"""INFR-01 Wave-0 gate — LTX frame ``8k+1`` + dim ``%32`` alignment.

This gate proves the sampling surface can only ever request an LTX-legal geometry. It does
NOT re-implement the modulo math — it drives the EXISTING shared validators
(``validators.validate_frames`` / ``validate_training_dims``) so the sampler and the
training path enforce one single source of truth (validators.py).

House frame buckets are the 8k+1 values 49 (= 8*6+1) and 81 (= 8*10+1); the canonical
ValidationConfig geometry is 768x352 (both %32). 33 is technically 8k+1-valid but is not a
house bucket; the real failure mode a gate must catch is an OFF-grid count (e.g. 50) or a
non-%32 width/height — asserted below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from signet_trainer.config.schema import ValidationConfig
from signet_trainer.config.validators import (
    validate_conditioning_items,
    validate_frames,
    validate_training_dims,
)

# The Phase-6 multi-frame sample config — the SAME source of truth that
# ``scripts/_stage_multi_frame_refs.py`` reads its keyframe (image, frame_index) targets from.
_MULTI_FRAME_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "configs" / "ltx23_multi_frame.example.yaml"
)

# The house frame buckets (8k+1). Both must pass the shared validator unchanged.
HOUSE_FRAME_BUCKETS = (49, 81)


@pytest.mark.parametrize("frames", HOUSE_FRAME_BUCKETS)
def test_house_frame_buckets_pass_validate_frames(frames: int) -> None:
    assert validate_frames(frames) == frames  # (F - 1) % 8 == 0


@pytest.mark.parametrize("bad_frames", [50, 48, 16, 0])
def test_off_grid_frame_counts_raise(bad_frames: int) -> None:
    # 50 -> (50-1)%8 == 1; 48 -> 7; 16 -> 7; 0 -> < 1 floor. All must fail fast.
    with pytest.raises(ValueError, match="frame"):
        validate_frames(bad_frames)


def test_canonical_validation_geometry_passes_training_dims() -> None:
    """The default ValidationConfig geometry (768x352, frame_count 49) is LTX-legal."""
    v = ValidationConfig()
    assert validate_training_dims((v.width, v.height, v.frame_count)) == (768, 352, 49)


@pytest.mark.parametrize(
    "dims",
    [
        (770, 352, 49),  # width 770 % 32 == 2  -> not %32
        (768, 510, 49),  # height 510 % 32 == 30 -> not %32
        (768, 352, 50),  # frames 50 -> (50-1) % 8 == 1 -> off the 8k+1 grid
    ],
)
def test_illegal_training_dims_raise(dims: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        validate_training_dims(dims)


def test_frame_gate_uses_the_shared_validator_not_inline_math() -> None:
    """Guard against a future rewrite that re-derives % 8 inline instead of calling validators."""
    # A 33-frame count IS 8k+1-valid ((33-1) % 8 == 0), so the shared validator accepts it;
    # this documents that the gate defers the rule to validators.validate_frames rather than
    # hard-coding a "must be 49/81" check.
    assert validate_frames(33) == 33


# --------------------------------------------------------------------------------------------------
# Phase 6 (REF-02) — multi-frame conditioning item frame_index alignment. Keyframe indices are on
# the latent temporal grid (``% 8 == 0``), a DIFFERENT rule than the pixel-frame ``8k+1`` count
# (validate_frames): a conditioning frame_index of 0, 8, 16, 24, ... indexes an existing pixel
# frame, so it must be a multiple of TIME_SCALE and within [0, F-1].
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("frame_index", [0, 8, 16, 24, 48])
def test_conditioning_frame_index_on_temporal_grid_passes(frame_index: int) -> None:
    items = [{"frame_index": frame_index, "strength": 1.0}]
    assert validate_conditioning_items(items, training_dims=(768, 512, 49)) == items


@pytest.mark.parametrize("bad_index", [25, 7, 33, 50])
def test_conditioning_frame_index_off_temporal_grid_raises(bad_index: int) -> None:
    # 33 IS 8k+1-valid as a pixel-frame COUNT, but as a keyframe INDEX (% 8) it is off-grid (33%8==1)
    # -> the item validator rejects it. Documents the two distinct rules.
    items = [{"frame_index": bad_index, "strength": 1.0}]
    with pytest.raises(ValueError):
        validate_conditioning_items(items, training_dims=(768, 512, 49))


# --------------------------------------------------------------------------------------------------
# Phase 6 (06-07 Task 3) — the keyframe indices that scripts/_stage_multi_frame_refs.py extracts come
# straight from configs/ltx23_multi_frame.example.yaml's conditioning_items. This test guards against
# a hand-edited NON-ALIGNED keyframe index sneaking into that config (T-06-01 Tampering): it drives
# the SAME shared validator (validate_conditioning_items) the SignetConfig load uses, so the staging
# script can only ever be pointed at % 8 == 0 frame indices.
# --------------------------------------------------------------------------------------------------


def _multi_frame_config() -> dict:
    return yaml.safe_load(_MULTI_FRAME_EXAMPLE.read_text(encoding="utf-8"))


def test_multi_frame_example_exists() -> None:
    assert _MULTI_FRAME_EXAMPLE.is_file(), f"missing sample config: {_MULTI_FRAME_EXAMPLE}"


def test_staging_script_target_indices_are_temporal_grid_aligned() -> None:
    """Every conditioning_items[].frame_index the ref-staging script targets must be % 8 == 0."""
    cfg = _multi_frame_config()
    items = cfg["conditioning"]["conditioning_items"]
    assert items, "sample config must declare conditioning_items for the multi-frame grid"
    width, height, frames = cfg["training_dims"]
    # Reuse the Plan 06-01 rule (raises on any off-grid or out-of-range index); returns unchanged.
    assert validate_conditioning_items(items, training_dims=(width, height, frames)) == items
    # Explicit, human-readable restatement of the alignment invariant the script relies on.
    for item in items:
        assert item["frame_index"] % 8 == 0, (
            f"keyframe frame_index {item['frame_index']} is not % 8 == 0"
        )
