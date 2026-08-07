"""CONF-02 — fail-fast LTX validators fire at config-load, before any GPU.

The four verified LTX-2.3 constraints (RESEARCH.md Q3, from ltx_core + the native
validate_video_dims + enochiatron's batch_size==1):

    (F - 1) % 8 == 0   (equivalently F % 8 == 1)
    H % 32 == 0
    W % 32 == 0
    batch_size == 1     (single-bucket; multi-bucket has per-sample shapes)

Each must raise a Pydantic ValidationError at ``model_validate`` time with a clear,
specific message naming the offending value and the rule — never a silent pass that
reaches metered compute.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import ConditioningConfig, SignetConfig
from signet_trainer.config.validators import (
    validate_batch_size,
    validate_conditioning_items,
    validate_conditioning_mode,
    validate_frames,
    validate_height,
    validate_resolution_bucket,
    validate_resolution_buckets,
    validate_training_dims,
    validate_width,
)


# A minimal-but-valid config payload. training_dims = [W, H, F].
def _valid_payload(*, width: int = 768, height: int = 512, frames: int = 49, batch_size: int = 1) -> dict:
    return {
        "training_dims": [width, height, frames],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": batch_size},
        # training.max_steps is the one required run-specific knob (Plan 03-01 / D-RUN-1).
        "training": {"max_steps": 100},
    }


# --------------------------------------------------------------------------------------------------
# Standalone validator functions (reusable, FS-free).
# --------------------------------------------------------------------------------------------------


def test_validate_frames_accepts_valid():
    assert validate_frames(49) == 49  # (49 - 1) % 8 == 0


# --- WR-04: resolution-bucket string validation (fail-fast at config load) ---


def test_validate_resolution_bucket_accepts_valid():
    assert validate_resolution_bucket("768x352x25") == "768x352x25"  # 768%32, 352%32, (25-1)%8 ok


@pytest.mark.parametrize(
    "bad",
    [
        "768x352",  # wrong arity (2 parts)
        "768x352x25x1",  # wrong arity (4 parts)
        "768X352X25",  # capital X -> unsplit -> arity 1
        "abcx352x25",  # non-int part
        "770x352x25",  # W not %32
        "768x350x25",  # H not %32
        "768x352x24",  # (24-1)%8 != 0
        "25x352x768",  # mis-ordered FxHxW -> W=25 fails %32 (metered-garbage guard)
    ],
)
def test_validate_resolution_bucket_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_resolution_bucket(bad)


def test_validate_resolution_buckets_names_offending_index():
    with pytest.raises(ValueError, match=r"resolution_buckets\[1\]"):
        validate_resolution_buckets(["768x352x25", "770x352x25"])


def test_config_rejects_malformed_resolution_bucket_at_load():
    # WR-04: the field_validator fires at model_validate time (pre-approval), not post-approval.
    payload = _valid_payload()
    payload["data"]["resolution_buckets"] = ["25x352x768"]  # mis-ordered garbage
    with pytest.raises(ValidationError, match="resolution"):
        SignetConfig.model_validate(payload)


# (F-1)%8 must == 0; these all fail it: 32->7, 48->7, 50->1, 16->7.
@pytest.mark.parametrize("bad_f", [32, 48, 50, 16])
def test_validate_frames_rejects_invalid(bad_f):
    with pytest.raises(ValueError, match="frame"):
        validate_frames(bad_f)


def test_validate_height_accepts_multiple_of_32():
    assert validate_height(512) == 512


def test_validate_height_rejects_non_multiple():
    with pytest.raises(ValueError, match="32"):
        validate_height(510)


def test_validate_width_accepts_multiple_of_32():
    assert validate_width(768) == 768


def test_validate_width_rejects_non_multiple():
    with pytest.raises(ValueError, match="32"):
        validate_width(770)


def test_validate_batch_size_accepts_one():
    assert validate_batch_size(1) == 1


@pytest.mark.parametrize("bad_bs", [0, 2, 4])
def test_validate_batch_size_rejects_non_single(bad_bs):
    with pytest.raises(ValueError, match="single-bucket|batch_size|1"):
        validate_batch_size(bad_bs)


def test_validate_training_dims_roundtrips_valid():
    assert validate_training_dims((768, 512, 49)) == (768, 512, 49)


# --------------------------------------------------------------------------------------------------
# The same constraints, enforced at SignetConfig.model_validate (fail-fast, before GPU).
# --------------------------------------------------------------------------------------------------


def test_valid_config_loads():
    cfg = SignetConfig.model_validate(_valid_payload())
    assert tuple(cfg.training_dims) == (768, 512, 49)
    assert cfg.data.batch_size == 1


def test_load_config_from_text_matches_path(tmp_path):
    """The by-value loader (used to ship a config to a remote container) parses + validates the same
    YAML text a path load would — and rejects invalid YAML with the same fail-fast ValidationError.
    """
    import yaml

    from signet_trainer.config.load import load_config, load_config_from_text

    text = yaml.safe_dump(_valid_payload())
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")

    from_path = load_config(p)
    from_text = load_config_from_text(text)
    assert tuple(from_text.training_dims) == tuple(from_path.training_dims) == (768, 512, 49)

    with pytest.raises(ValidationError):
        load_config_from_text(yaml.safe_dump(_valid_payload(frames=32)))


def test_invalid_frames_rejected():
    # F = 32 -> (32 - 1) % 8 == 7 != 0
    with pytest.raises(ValidationError, match="frame"):
        SignetConfig.model_validate(_valid_payload(frames=32))


def test_invalid_width_rejected():
    # W = 770 -> 770 % 32 != 0
    with pytest.raises(ValidationError, match="32"):
        SignetConfig.model_validate(_valid_payload(width=770))


def test_invalid_height_rejected():
    # H = 510 -> 510 % 32 != 0
    with pytest.raises(ValidationError, match="32"):
        SignetConfig.model_validate(_valid_payload(height=510))


def test_non_single_batch_size_rejected():
    with pytest.raises(ValidationError, match="single-bucket|batch_size|1"):
        SignetConfig.model_validate(_valid_payload(batch_size=2))


# --------------------------------------------------------------------------------------------------
# CR-01 — zero / negative / degenerate dimensions must fail-fast with a clear ValueError naming the
# offending field, BEFORE the divisibility check and BEFORE any GPU/tensor build. Both 0 and -32
# satisfy ``% 32 == 0`` (and 0 satisfies ``(0-1) % 8 == ...`` indirectly), so without a positivity
# floor these slip through and only crash deep inside torch.zeros with an opaque RuntimeError.
# --------------------------------------------------------------------------------------------------


def test_validate_frames_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="frame"):
        validate_frames(0)
    with pytest.raises(ValueError, match="frame"):
        validate_frames(-32)


def test_validate_height_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="height"):
        validate_height(0)
    with pytest.raises(ValueError, match="height"):
        validate_height(-32)


def test_validate_width_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="width"):
        validate_width(0)
    with pytest.raises(ValueError, match="width"):
        validate_width(-32)


# The four exact degenerate triples called out in REVIEW.md CR-01, asserted end-to-end through the
# real schema (not just the standalone validators) — each must raise a clear ValidationError.
@pytest.mark.parametrize(
    "width, height, frames",
    [
        (49, 512, 0),     # [W=49? no] -> frames=0 degenerate (mapped from CR-01 [0,512,49] family)
        (512, 49, -32),   # negative frames
        (0, 512, 49),     # zero width
        (512, 0, 49),     # zero height
        (-32, 512, 49),   # negative width
        (512, -32, 49),   # negative height
        (512, 512, 0),    # zero frames
    ],
)
def test_non_positive_dims_rejected_at_schema(width, height, frames):
    with pytest.raises(ValidationError):
        SignetConfig.model_validate(_valid_payload(width=width, height=height, frames=frames))


# --------------------------------------------------------------------------------------------------
# Phase 5 (REF-01) — single-frame ConditioningConfig fields + mode allowlist (D-CONDP / D-STRENGTH /
# D-NOHARDCODE). The conditioning probability + strength are TUNABLE config (never hardcoded); the
# mode is a fail-fast allowlist so an unknown reference-control mode dies at load, before any GPU.
# CONTRADICTION #2: the strength field is ``first_frame_conditioning_strength`` — the phantom
# ``image_cond_noise_scale`` must NOT exist anywhere in the schema.
# --------------------------------------------------------------------------------------------------


def test_conditioning_defaults_backward_compatible():
    # Phase 1-4 configs carry only mode (or nothing) — the new fields must default cleanly.
    cfg = ConditioningConfig()
    assert cfg.mode == "none"
    assert cfg.first_frame_conditioning_p == 1.0
    assert cfg.first_frame_conditioning_strength == 1.0
    assert cfg.reference_images == []


def test_conditioning_single_frame_accepts_fields():
    cfg = ConditioningConfig(
        mode="single_frame",
        first_frame_conditioning_p=1.0,
        first_frame_conditioning_strength=1.0,
        reference_images=["reference/a.png", "reference/b.png"],
    )
    assert cfg.mode == "single_frame"
    assert cfg.first_frame_conditioning_p == 1.0
    assert cfg.reference_images == ["reference/a.png", "reference/b.png"]


@pytest.mark.parametrize("bad_p", [1.5, -0.1])
def test_conditioning_p_out_of_range_rejected(bad_p):
    with pytest.raises(ValidationError):
        ConditioningConfig(first_frame_conditioning_p=bad_p)


@pytest.mark.parametrize("bad_strength", [1.5, -0.1])
def test_conditioning_strength_out_of_range_rejected(bad_strength):
    with pytest.raises(ValidationError):
        ConditioningConfig(first_frame_conditioning_strength=bad_strength)


def test_conditioning_mode_bogus_rejected_at_model():
    with pytest.raises(ValidationError, match="mode"):
        ConditioningConfig(mode="bogus")


def test_validate_conditioning_mode_standalone():
    # Standalone fail-fast validator — accepts the allowlist, rejects unknowns naming the field.
    assert validate_conditioning_mode("none") == "none"
    assert validate_conditioning_mode("single_frame") == "single_frame"
    with pytest.raises(ValueError, match="mode"):
        validate_conditioning_mode("bogus")
    # The error must list the allowed values so a hand-edited YAML gets an actionable message.
    with pytest.raises(ValueError, match="single_frame"):
        validate_conditioning_mode("keyframe")


def test_no_phantom_image_cond_noise_scale():
    # CONTRADICTION #2 — the phantom knob must never appear as a field.
    assert "image_cond_noise_scale" not in ConditioningConfig.model_fields


def test_single_frame_example_yaml_loads():
    # The shipped example config must load into a SignetConfig with the single-frame block wired.
    from pathlib import Path

    from signet_trainer.config.load import load_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "ltx23_single_frame.example.yaml")
    assert cfg.conditioning.mode == "single_frame"
    assert cfg.conditioning.first_frame_conditioning_p == 1.0
    assert cfg.conditioning.first_frame_conditioning_strength == 1.0
    assert len(cfg.conditioning.reference_images) > 0


def test_multi_frame_example_yaml_pins_render_geometry_to_bucket():
    # WR-08: the multi-frame example documents the demo 832x480x49 bucket — the sample render
    # geometry must be pinned EXPLICITLY to match (not ride the 768x352 schema defaults, which
    # would silently center-crop the keyframe references at a different aspect ratio).
    from pathlib import Path

    from signet_trainer.config.load import load_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "ltx23_multi_frame.example.yaml")
    width, height, _frames = cfg.training_dims
    assert cfg.validation.width == width == 832
    assert cfg.validation.height == height == 480


# --------------------------------------------------------------------------------------------------
# Phase 6 (REF-02) — Seam A: multi-frame mode allowlist + validate_conditioning_items standalone
# fail-fast validator. The mode allowlist widens to include ``multi_frame``; the item validator
# mirrors ``validate_frames``' error shape (names the offender + nearest valid indices) so a
# hand-edited multi-frame YAML dies at config load, before any GPU (CONF-02 / D-6-FAILFAST).
# --------------------------------------------------------------------------------------------------


def test_validate_conditioning_mode_accepts_multi_frame():
    # Allowlist widened for Phase 6 — multi_frame is now a first-class mode.
    assert validate_conditioning_mode("multi_frame") == "multi_frame"


def test_validate_conditioning_mode_bogus_lists_multi_frame():
    # The rejection message must now enumerate multi_frame alongside none/single_frame.
    with pytest.raises(ValueError, match="multi_frame"):
        validate_conditioning_mode("bogus")


def test_validate_conditioning_items_accepts_valid():
    items = [
        {"frame_index": 0, "strength": 1.0},
        {"frame_index": 24, "strength": 0.5},
    ]
    assert validate_conditioning_items(items, training_dims=(768, 512, 49)) is items


def test_validate_conditioning_items_rejects_misaligned_frame_index():
    # 25 % 8 == 1 != 0 -> must name the item, the value 25, and the nearest valid indices 24 & 32.
    items = [{"frame_index": 25, "strength": 1.0}]
    with pytest.raises(ValueError) as exc:
        validate_conditioning_items(items, training_dims=(768, 512, 49))
    msg = str(exc.value)
    assert "25" in msg
    assert "24" in msg
    assert "32" in msg


def test_validate_conditioning_items_rejects_out_of_range_frame_index():
    # 56 % 8 == 0 (aligned) but F = 49 -> valid range is 0..48 -> must name the range.
    items = [{"frame_index": 56, "strength": 1.0}]
    with pytest.raises(ValueError, match="0..48|0\\.\\.48|48"):
        validate_conditioning_items(items, training_dims=(768, 512, 49))


@pytest.mark.parametrize("bad_strength", [1.5, -0.1])
def test_validate_conditioning_items_rejects_out_of_range_strength(bad_strength):
    items = [{"frame_index": 0, "strength": bad_strength}]
    with pytest.raises(ValueError, match="strength"):
        validate_conditioning_items(items, training_dims=(768, 512, 49))


def test_validate_conditioning_items_empty_ok():
    # No items -> nothing to validate; returns the (empty) list unchanged.
    assert validate_conditioning_items([], training_dims=(768, 512, 49)) == []


def test_validate_conditioning_items_rejects_duplicate_frame_index():
    # WR-06: build_denoise_mask writes regions in order — two items anchoring the same frame
    # would silently last-win (and later FAIL the dry-run per-region gate with an opaque
    # message). The error must name BOTH offending item indices.
    items = [
        {"frame_index": 0, "strength": 1.0},
        {"frame_index": 24, "strength": 0.5},
        {"frame_index": 24, "strength": 1.0},
    ]
    with pytest.raises(ValueError) as exc:
        validate_conditioning_items(items, training_dims=(768, 512, 49))
    msg = str(exc.value)
    assert "duplicates" in msg
    assert "conditioning_items[2]" in msg  # the offender
    assert "conditioning_items[1]" in msg  # the item it collides with


def test_multi_frame_duplicate_frame_index_rejected_at_schema():
    # WR-06 end-to-end: the duplicate check fires through SignetConfig._cross_field_checks.
    items = [
        {"image": "reference/a.png", "frame_index": 24, "strength": 0.5},
        {"image": "reference/b.png", "frame_index": 24, "strength": 1.0},
    ]
    with pytest.raises(ValidationError, match="duplicates|distinct"):
        SignetConfig.model_validate(_multi_frame_payload(items=items))


# --------------------------------------------------------------------------------------------------
# Phase 6 (REF-02) — Seam A schema: ConditioningItem sub-model + multi-frame fields on
# ConditioningConfig, with fail-fast cross-field validation. The object-list carries per-item
# {image, frame_index, strength}; max_conditioning_items / conditioning_strength_range /
# conditioning_source are bounded config; the lean field-split is enforced BIDIRECTIONALLY so
# neither the Phase-5 nor the Phase-6 conditioning block is ever silently ignored.
# --------------------------------------------------------------------------------------------------


def _multi_frame_payload(*, items=None, mode: str = "multi_frame", **cond_extra) -> dict:
    payload = _valid_payload()
    cond: dict = {"mode": mode}
    if items is not None:
        cond["conditioning_items"] = items
    cond.update(cond_extra)
    payload["conditioning"] = cond
    return payload


def test_multi_frame_config_validates():
    items = [
        {"image": "reference/a.png", "frame_index": 0, "strength": 1.0},
        {"image": "reference/b.png", "frame_index": 48, "strength": 1.0},
    ]
    cfg = SignetConfig.model_validate(_multi_frame_payload(items=items))
    assert cfg.conditioning.mode == "multi_frame"
    assert len(cfg.conditioning.conditioning_items) == 2
    assert cfg.conditioning.conditioning_items[0].frame_index == 0
    assert cfg.conditioning.conditioning_items[1].image == "reference/b.png"


def test_multi_frame_bad_frame_index_rejected_at_schema():
    # 25 % 8 != 0 -> caught by SignetConfig._cross_field_checks (needs training_dims).
    items = [{"image": "reference/a.png", "frame_index": 25, "strength": 1.0}]
    with pytest.raises(ValidationError, match="frame_index|25"):
        SignetConfig.model_validate(_multi_frame_payload(items=items))


def test_multi_frame_frame_index_beyond_sample_frame_count_rejected():
    # WR-05: frame_index 48 is legal for training_dims F=49 but the SAMPLE path renders
    # validation.frame_count frames — with frame_count=25 the keyframe would target a latent
    # frame beyond the render, discovered only on the metered GPU. Must die at config load.
    items = [{"image": "reference/a.png", "frame_index": 48, "strength": 1.0}]
    payload = _multi_frame_payload(items=items)
    payload["validation"] = {"frame_count": 25}
    with pytest.raises(ValidationError, match="frame_count|48"):
        SignetConfig.model_validate(payload)


def test_multi_frame_frame_index_within_sample_frame_count_accepted():
    # Same keyframe set is fine when validation.frame_count covers it (both bounds hold).
    items = [{"image": "reference/a.png", "frame_index": 48, "strength": 1.0}]
    payload = _multi_frame_payload(items=items)
    payload["validation"] = {"frame_count": 49}
    cfg = SignetConfig.model_validate(payload)
    assert cfg.validation.frame_count == 49
    assert cfg.conditioning.conditioning_items[0].frame_index == 48


def test_max_conditioning_items_default_and_upper_bound():
    assert ConditioningConfig().max_conditioning_items == 1
    with pytest.raises(ValidationError):
        ConditioningConfig(max_conditioning_items=9)  # le=8
    with pytest.raises(ValidationError):
        ConditioningConfig(max_conditioning_items=0)  # ge=1


def test_conditioning_strength_range_default():
    assert tuple(ConditioningConfig().conditioning_strength_range) == (0.3, 1.0)


@pytest.mark.parametrize("bad_range", [(0.9, 0.3), (0.0, 1.2), (-0.1, 0.5)])
def test_conditioning_strength_range_rejected(bad_range):
    with pytest.raises(ValidationError):
        ConditioningConfig(conditioning_strength_range=bad_range)


def test_conditioning_source_default_self():
    assert ConditioningConfig().conditioning_source == "self"


def test_conditioning_source_paired_rejected():
    # Phase-6 allowlist is Literal["self"] ONLY — "paired" must fail fast (re-widens in Phase 7).
    with pytest.raises(ValidationError):
        ConditioningConfig(conditioning_source="paired")


def test_conditioning_items_requires_multi_frame_mode():
    # Lean field-split (forward): non-empty items while mode != multi_frame is rejected.
    with pytest.raises(ValidationError, match="multi_frame|mode|conditioning_items"):
        ConditioningConfig(
            mode="single_frame",
            conditioning_items=[{"image": "a.png", "frame_index": 0, "strength": 1.0}],
        )


def test_phase5_p_field_rejected_in_multi_frame_mode():
    # Lean field-split (reverse): non-default Phase-5 field while mode == multi_frame is rejected.
    with pytest.raises(ValidationError, match="first_frame_conditioning_p|multi_frame"):
        ConditioningConfig(mode="multi_frame", first_frame_conditioning_p=0.5)


def test_phase5_strength_field_rejected_in_multi_frame_mode():
    with pytest.raises(ValidationError, match="first_frame_conditioning_strength|multi_frame"):
        ConditioningConfig(mode="multi_frame", first_frame_conditioning_strength=0.5)


def test_phase5_reference_images_rejected_in_multi_frame_mode():
    with pytest.raises(ValidationError, match="reference_images|multi_frame"):
        ConditioningConfig(mode="multi_frame", reference_images=["a.png"])


@pytest.mark.parametrize("bad_strength", [0.0, 0.5, 0.99])
def test_strength_field_rejected_in_single_frame_mode(bad_strength):
    # WR-03 lean field-split: SingleFrameStrategy is hard-clean (the knob is stored, never used)
    # and loop.py threads only the p knob — a non-default strength in single_frame mode would be
    # silently ignored (training at 1.0). Reject at load until a phase wires it.
    with pytest.raises(ValidationError, match="first_frame_conditioning_strength|single_frame"):
        ConditioningConfig(mode="single_frame", first_frame_conditioning_strength=bad_strength)


def test_single_frame_default_strength_still_allowed():
    # The 1.0 default (the value the hard-clean path actually implements) must NOT trip WR-03.
    cfg = ConditioningConfig(mode="single_frame", first_frame_conditioning_strength=1.0)
    assert cfg.first_frame_conditioning_strength == 1.0


def test_strength_field_still_free_in_none_mode():
    # mode == 'none' carries no conditioning path at all; the field stays schema-legal there
    # (backward-compatible with Phase 1-4 configs that never exercise conditioning).
    cfg = ConditioningConfig(mode="none", first_frame_conditioning_strength=0.5)
    assert cfg.first_frame_conditioning_strength == 0.5


def test_multi_frame_default_phase5_fields_allowed():
    # Default Phase-5 field values (p=1.0, strength=1.0, reference_images=[]) must NOT trip the
    # reverse field-split — otherwise multi_frame could never validate.
    cfg = ConditioningConfig(
        mode="multi_frame",
        conditioning_items=[{"image": "a.png", "frame_index": 0, "strength": 1.0}],
    )
    assert cfg.mode == "multi_frame"


def test_conditioning_extra_key_rejected():
    # extra="forbid" still holds on the widened block.
    with pytest.raises(ValidationError):
        ConditioningConfig(unknown_multi_frame_key=123)


def test_conditioning_item_strength_out_of_range_rejected():
    # ConditioningItem.strength carries its own Field(ge=0, le=1) bound.
    with pytest.raises(ValidationError):
        ConditioningConfig(
            mode="multi_frame",
            conditioning_items=[{"image": "a.png", "frame_index": 0, "strength": 1.5}],
        )


# --------------------------------------------------------------------------------------------------
# WR-09 / T-06-02 — the "Volume-relative" contract is ENFORCED at config load: reference paths are
# joined Modal-side as CHECKPOINTS_DIR / value, where pathlib join semantics let an absolute value
# REPLACE the prefix entirely and '..' segments escape it. Both POSIX and Windows forms are caught
# (configs are authored on Windows).
# --------------------------------------------------------------------------------------------------

_BAD_VOLUME_PATHS = [
    "/etc/passwd.png",              # POSIX-absolute — replaces the Volume prefix
    "C:/refs/a.png",                # Windows drive-absolute
    "C:\\refs\\a.png",              # Windows drive-absolute (backslash)
    "../escape.png",                # parent traversal
    "refs/../../escape.png",        # embedded traversal
    "..\\escape.png",               # Windows-style traversal
]


def test_validate_volume_relative_path_accepts_relative():
    from signet_trainer.config.validators import validate_volume_relative_path

    assert validate_volume_relative_path("reference/demo_start.png") == "reference/demo_start.png"
    assert validate_volume_relative_path("a.png") == "a.png"


@pytest.mark.parametrize("bad", _BAD_VOLUME_PATHS)
def test_validate_volume_relative_path_rejects_escapes(bad):
    from signet_trainer.config.validators import validate_volume_relative_path

    with pytest.raises(ValueError, match="Volume-relative"):
        validate_volume_relative_path(bad)


@pytest.mark.parametrize("bad", _BAD_VOLUME_PATHS)
def test_conditioning_item_image_must_be_volume_relative(bad):
    with pytest.raises(ValidationError, match="Volume-relative"):
        ConditioningConfig(
            mode="multi_frame",
            conditioning_items=[{"image": bad, "frame_index": 0, "strength": 1.0}],
        )


@pytest.mark.parametrize("bad", _BAD_VOLUME_PATHS)
def test_reference_images_must_be_volume_relative(bad):
    with pytest.raises(ValidationError, match="Volume-relative"):
        ConditioningConfig(mode="single_frame", reference_images=[bad])
