"""Phase 9 (INPAINT, GATE-SPEC-inpaint-a2v rev 2) — inpaint config surface: the widened mode
allowlist, the ÷64 dims rule (STRICTER than the video-wide %32 rule, inpaint-mode only), the
inpaint_mask_dir / inpaint_mask_probability field block with its bidirectional lean field-split,
and the 'mask' validation-sample condition surface (masked test renders at sample time).

CPU only — no ltx_core / modal import, no filesystem touch (Anti-Pattern 6 / CONF-03 zero-spend).
[precedent] prior-project HANDOFF-2026-06-30-NIGHT: '÷64 required for inpaint; validated by the smoke'.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.load import load_config_from_text
from signet_trainer.config.schema import (
    ConditioningConfig,
    MaskCondition,
    SignetConfig,
    ValidationSample,
)
from signet_trainer.config.validators import (
    ALLOWED_CONDITIONING_MODES,
    INPAINT_HEIGHT_SCALE,
    INPAINT_WIDTH_SCALE,
    validate_conditioning_mode,
    validate_inpaint_dims,
    validate_inpaint_resolution_bucket,
    validate_inpaint_resolution_buckets,
)


# A minimal-but-valid INPAINT payload. training_dims = [W, H, F] must itself be ÷64-clean in
# inpaint mode, as must every resolution bucket (the default 768x352x* set is ÷32-only), so the
# helper pins the precedent-validated 1280x704 res family (GATE-SPEC rev 2 primary bucket).
def _payload(*, conditioning: dict | None = None, **top) -> dict:
    payload: dict = {
        "training_dims": [1280, 704, 49],
        "data": {
            "preprocessed_data_root": "/dataset/.precomputed_campaign_inpaint",
            "batch_size": 1,
            "resolution_buckets": ["1280x704x49"],
        },
        "training": {"max_steps": 50},
    }
    if conditioning is not None:
        payload["conditioning"] = conditioning
    payload.update(top)
    return payload


def _mask_sample(**overrides) -> dict:
    sample = {
        "prompt": "the subject turns her head toward the window",
        "conditions": [
            {
                "type": "mask",
                "video": "dataset/holdout/clip_00.mp4",
                "mask": "dataset/holdout/clip_00_mask.mp4",
            }
        ],
    }
    sample.update(overrides)
    return sample


# --------------------------------------------------------------------------------------------------
# Allowlist — inpaint is now first-class (widened from {none, single_frame, multi_frame, ic_lora}).
# --------------------------------------------------------------------------------------------------


def test_inpaint_in_allowlist():
    assert "inpaint" in ALLOWED_CONDITIONING_MODES


def test_validate_conditioning_mode_accepts_inpaint():
    assert validate_conditioning_mode("inpaint") == "inpaint"


def test_conditioning_mode_inpaint_accepted_at_model():
    cfg = ConditioningConfig(mode="inpaint")
    assert cfg.mode == "inpaint"


def test_existing_modes_still_allowed():
    # Widening the allowlist must not disturb the existing modes.
    for mode in ("none", "single_frame", "multi_frame", "ic_lora"):
        assert validate_conditioning_mode(mode) == mode


# --------------------------------------------------------------------------------------------------
# The ÷64 rule (validator level) — [precedent] prior-project HANDOFF-2026-06-30-NIGHT.
# 736x416 is the canary: ÷32-clean (736/32=23, 416/32=13) but NOT ÷64 (both leave remainder 32).
# --------------------------------------------------------------------------------------------------


def test_inpaint_scales_are_64():
    assert INPAINT_HEIGHT_SCALE == 64
    assert INPAINT_WIDTH_SCALE == 64


def test_inpaint_bucket_rejects_div32_clean_736x416():
    with pytest.raises(ValueError, match="64|inpaint"):
        validate_inpaint_resolution_bucket("736x416x25")


def test_inpaint_bucket_accepts_1280x704():
    # The precedent-validated primary bucket (res-matched to r5's family).
    assert validate_inpaint_resolution_bucket("1280x704x49") == "1280x704x49"


def test_inpaint_bucket_accepts_512x384():
    # The precedent-validated cheap-smoke bucket.
    assert validate_inpaint_resolution_bucket("512x384x25") == "512x384x25"


def test_inpaint_bucket_still_enforces_base_frame_rule():
    # frames % 8 == 1 is the SAME rule as every other mode (validate_frames) — a ÷64-clean bucket
    # with an off-grid frame count still dies with the base frame message.
    with pytest.raises(ValueError, match="frame"):
        validate_inpaint_resolution_bucket("1280x704x48")


def test_inpaint_bucket_still_enforces_base_32_rule():
    # A non-%32 dim dies via the base validator first (inpaint composes, not replaces).
    with pytest.raises(ValueError, match="32"):
        validate_inpaint_resolution_bucket("1280x720x49")


def test_inpaint_buckets_list_names_offending_index():
    with pytest.raises(ValueError, match=r"resolution_buckets\[1\]"):
        validate_inpaint_resolution_buckets(["1280x704x49", "736x416x25"])


def test_inpaint_dims_rejects_736x416_and_accepts_valid():
    with pytest.raises(ValueError, match="64|inpaint"):
        validate_inpaint_dims((736, 416, 25))
    assert validate_inpaint_dims((1280, 704, 49)) == (1280, 704, 49)
    assert validate_inpaint_dims((512, 384, 25)) == (512, 384, 25)


# --------------------------------------------------------------------------------------------------
# The ÷64 rule (schema level, cross-field): fires ONLY when conditioning.mode == 'inpaint'.
# --------------------------------------------------------------------------------------------------


def test_inpaint_mode_rejects_div32_only_bucket_at_schema():
    payload = _payload(conditioning={"mode": "inpaint"})
    payload["data"]["resolution_buckets"] = ["736x416x25"]
    payload["training_dims"] = [1280, 704, 49]
    with pytest.raises(ValidationError, match="64|inpaint"):
        SignetConfig.model_validate(payload)


def test_inpaint_mode_rejects_div32_only_training_dims_at_schema():
    payload = _payload(conditioning={"mode": "inpaint"})
    payload["training_dims"] = [736, 416, 25]
    with pytest.raises(ValidationError, match="64|inpaint"):
        SignetConfig.model_validate(payload)


def test_inpaint_mode_accepts_div64_buckets_at_schema():
    payload = _payload(conditioning={"mode": "inpaint"})
    payload["data"]["resolution_buckets"] = ["1280x704x49", "512x384x25"]
    # inpaint also cross-checks the VALIDATION render dims (pre-build-audit gap): the schema
    # default height 352 is %64-illegal, so an inpaint fixture must carry a legal one.
    payload["validation"] = {"height": 384}
    cfg = SignetConfig.model_validate(payload)
    assert cfg.conditioning.mode == "inpaint"
    assert cfg.data.resolution_buckets == ["1280x704x49", "512x384x25"]


def test_div32_rule_unchanged_for_other_modes():
    # The video-wide %32 rule stays the law outside inpaint mode: 736x416x25 (÷32-clean, NOT ÷64)
    # must keep loading under mode 'none' exactly as before.
    payload = _payload(conditioning={"mode": "none"})
    payload["training_dims"] = [736, 416, 25]
    payload["data"]["resolution_buckets"] = ["736x416x25"]
    cfg = SignetConfig.model_validate(payload)
    assert tuple(cfg.training_dims) == (736, 416, 25)


# --------------------------------------------------------------------------------------------------
# Inpaint field block — defaults, bounds, and the bidirectional lean field-split (rule 6d/6e,
# same shape as the ic_lora rule 6/6c guards).
# --------------------------------------------------------------------------------------------------


def test_inpaint_field_defaults():
    cfg = ConditioningConfig(mode="inpaint")
    assert cfg.inpaint_mask_dir == "video_masks"
    assert cfg.inpaint_mask_probability == 1.0


def test_inpaint_mask_probability_bounds():
    with pytest.raises(ValidationError):
        ConditioningConfig(mode="inpaint", inpaint_mask_probability=1.5)  # le=1
    with pytest.raises(ValidationError):
        ConditioningConfig(mode="inpaint", inpaint_mask_probability=-0.1)  # ge=0


def test_inpaint_mode_accepts_explicit_inpaint_fields():
    cfg = ConditioningConfig(
        mode="inpaint", inpaint_mask_dir="custom_masks", inpaint_mask_probability=0.7
    )
    assert cfg.inpaint_mask_dir == "custom_masks"
    assert cfg.inpaint_mask_probability == 0.7


def test_inpaint_mask_probability_rejected_in_none_mode():
    with pytest.raises(ValidationError, match="inpaint"):
        ConditioningConfig(mode="none", inpaint_mask_probability=0.5)


def test_inpaint_mask_dir_rejected_in_ic_lora_mode():
    with pytest.raises(ValidationError, match="inpaint"):
        ConditioningConfig(
            mode="ic_lora", conditioning_source="paired", inpaint_mask_dir="other_masks"
        )


def test_inpaint_fields_at_default_allowed_in_other_modes():
    # A DEFAULT value on an inpaint field must NOT trip the reverse split (else non-inpaint modes
    # could never validate) — only NON-default values are rejected.
    cfg = ConditioningConfig(mode="single_frame", inpaint_mask_dir="video_masks")
    assert cfg.inpaint_mask_dir == "video_masks"
    assert cfg.inpaint_mask_probability == 1.0


def test_single_frame_knobs_rejected_in_inpaint_mode():
    # 6e (mirrors the ic_lora 6c guard): the inpaint path derives conditioning from the spatial
    # mask — non-default first-frame knobs / reference_images would be silently ignored.
    with pytest.raises(ValidationError, match="first_frame_conditioning_p"):
        ConditioningConfig(mode="inpaint", first_frame_conditioning_p=0.5)
    with pytest.raises(ValidationError, match="first_frame_conditioning_strength"):
        ConditioningConfig(mode="inpaint", first_frame_conditioning_strength=0.5)
    with pytest.raises(ValidationError, match="reference_images"):
        ConditioningConfig(mode="inpaint", reference_images=["reference/x.png"])


def test_ic_lora_fields_rejected_in_inpaint_mode():
    # The existing rule-6 reverse guard covers inpaint mode too (mode != 'ic_lora').
    with pytest.raises(ValidationError, match="ic_lora"):
        ConditioningConfig(mode="inpaint", reference_downscale_factor=2)


# --------------------------------------------------------------------------------------------------
# 'mask' validation-sample condition — schema surface + round-trip (the sampler branch that
# consumes it is a separate build item; here only the config surface is proven).
# --------------------------------------------------------------------------------------------------


def test_mask_condition_round_trips_through_schema():
    payload = _payload(conditioning={"mode": "inpaint"})
    # height 384: the default 352 fails the inpaint validation-dims %64 cross-check (audit gap).
    payload["validation"] = {"samples": [_mask_sample()], "height": 384}
    cfg = SignetConfig.model_validate(payload)
    assert len(cfg.validation.samples) == 1
    sample = cfg.validation.samples[0]
    assert sample.prompt == "the subject turns her head toward the window"
    cond = sample.conditions[0]
    assert cond.type == "mask"
    assert cond.video == "dataset/holdout/clip_00.mp4"
    assert cond.mask == "dataset/holdout/clip_00_mask.mp4"
    # Full round-trip: dump -> re-validate -> identical surface.
    redumped = SignetConfig.model_validate(cfg.model_dump())
    assert redumped.validation.samples == cfg.validation.samples


def test_mask_condition_round_trips_through_yaml_loader():
    # The SAME loader path the tier-1 dry-run gate uses (CONF-01) — YAML text in, validated out.
    yaml_text = """
training_dims: [1280, 704, 49]
data:
  preprocessed_data_root: /dataset/.precomputed_campaign_inpaint
  batch_size: 1
  resolution_buckets: ["1280x704x49"]
training:
  max_steps: 50
conditioning:
  mode: inpaint
validation:
  height: 384
  samples:
    - prompt: the subject turns her head toward the window
      conditions:
        - type: mask
          video: dataset/holdout/clip_00.mp4
          mask: dataset/holdout/clip_00_mask.mp4
"""
    cfg = load_config_from_text(yaml_text)
    assert cfg.conditioning.mode == "inpaint"
    assert cfg.conditioning.inpaint_mask_dir == "video_masks"
    assert cfg.validation.samples[0].conditions[0].mask == "dataset/holdout/clip_00_mask.mp4"


def test_mask_condition_type_discriminator_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        MaskCondition(type="image", video="dataset/x.mp4", mask="dataset/x_mask.mp4")


def test_mask_condition_requires_video_and_mask():
    with pytest.raises(ValidationError):
        MaskCondition(video="dataset/x.mp4")  # mask missing
    with pytest.raises(ValidationError):
        MaskCondition(mask="dataset/x_mask.mp4")  # video missing


def test_validation_sample_requires_at_least_one_condition():
    # A condition-less render request belongs in validation.prompts — an empty conditions list is
    # an ambiguous duplicate of that surface (lean field-split).
    with pytest.raises(ValidationError):
        ValidationSample(prompt="p", conditions=[])


def test_mask_condition_paths_must_be_volume_relative():
    payload = _payload(conditioning={"mode": "inpaint"})
    payload["validation"] = {
        "samples": [_mask_sample(conditions=[{"video": "/etc/evil.mp4", "mask": "dataset/m.mp4"}])]
    }
    with pytest.raises(ValidationError, match="Volume-relative"):
        SignetConfig.model_validate(payload)
    payload["validation"] = {
        "samples": [
            _mask_sample(conditions=[{"video": "dataset/x.mp4", "mask": "dataset/../../escape.mp4"}])
        ]
    }
    with pytest.raises(ValidationError, match="Volume-relative"):
        SignetConfig.model_validate(payload)


def test_mask_sample_rejected_outside_inpaint_mode():
    # Cross-block lean field-split: a 'mask' condition under any other conditioning.mode would be
    # silently ignored on a metered render — it dies at config load instead.
    payload = _payload(conditioning={"mode": "none"})
    payload["training_dims"] = [768, 352, 25]
    payload["data"]["resolution_buckets"] = ["768x352x25"]
    payload["validation"] = {"samples": [_mask_sample()]}
    with pytest.raises(ValidationError, match="mask|inpaint"):
        SignetConfig.model_validate(payload)


def test_validation_samples_default_empty_backward_compatible():
    # Every existing config (no samples key) must keep loading unchanged.
    cfg = SignetConfig.model_validate(_payload())
    assert cfg.validation.samples == []
