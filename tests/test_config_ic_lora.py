"""Phase 7 (REF-03) — IC-LoRA config surface: allowlist + reference/palette/frozen fields, the
bidirectional lean field-split, and the D-7-BASEVAR distilled-base fail-fast.

CPU only — no ltx_core / modal import, no filesystem touch beyond reading the shipped example YAMLs
through the SAME loader the tier-1 dry-run gate uses (CONF-01 / D-NOHARDCODE). Every IC-LoRA tunable
is a documented, fail-fast config field BEFORE any GPU is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from signet_trainer.config.load import load_config
from signet_trainer.config.schema import ConditioningConfig, SignetConfig
from signet_trainer.config.validators import (
    ALLOWED_CONDITIONING_MODES,
    validate_conditioning_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IC_LORA_YAML = REPO_ROOT / "configs" / "ltx23_ic_lora.example.yaml"
IC_LORA_OVERFIT_YAML = REPO_ROOT / "configs" / "ltx23_ic_lora_overfit.example.yaml"


# A minimal-but-valid payload. training_dims = [W, H, F]; training.max_steps is the one required knob.
def _payload(*, conditioning: dict | None = None, **top) -> dict:
    payload: dict = {
        "training_dims": [832, 480, 25],
        "data": {"preprocessed_data_root": "/dataset/.precomputed_demo_seg", "batch_size": 1},
        "training": {"max_steps": 50},
    }
    if conditioning is not None:
        payload["conditioning"] = conditioning
    payload.update(top)
    return payload


# --------------------------------------------------------------------------------------------------
# Allowlist — ic_lora is now first-class (widened from the Phase-6 {none, single_frame, multi_frame}).
# --------------------------------------------------------------------------------------------------


def test_ic_lora_in_allowlist():
    assert "ic_lora" in ALLOWED_CONDITIONING_MODES


def test_validate_conditioning_mode_accepts_ic_lora():
    assert validate_conditioning_mode("ic_lora") == "ic_lora"


def test_conditioning_mode_ic_lora_accepted_at_model():
    cfg = ConditioningConfig(mode="ic_lora")
    assert cfg.mode == "ic_lora"


# --------------------------------------------------------------------------------------------------
# The shipped example YAMLs load clean through the dry-run gate's loader (a).
# --------------------------------------------------------------------------------------------------


def test_ic_lora_example_yaml_loads():
    cfg = load_config(IC_LORA_YAML)
    assert cfg.conditioning.mode == "ic_lora"
    assert cfg.conditioning.conditioning_source == "paired"
    assert cfg.conditioning.reference_latents_dir == "reference_latents"
    assert cfg.conditioning.reference_downscale_factor == 1
    assert cfg.conditioning.reference_column == "reference_path"
    assert cfg.conditioning.seg_palette_name == "compact_driving_v1"
    # 25f test bucket + explicitly-pinned render geometry (D-7-25F / WR-08).
    assert tuple(cfg.training_dims) == (832, 480, 25)
    assert cfg.validation.width == 832
    assert cfg.validation.height == 480


def test_ic_lora_overfit_example_yaml_loads():
    cfg = load_config(IC_LORA_OVERFIT_YAML)
    assert cfg.conditioning.mode == "ic_lora"
    assert cfg.conditioning.seg_palette_name == "compact_driving_v1"
    # Tiny overfit budget with a mid-run checkpoint.
    assert cfg.training.max_steps == 50
    assert cfg.training.checkpoint_every < cfg.training.max_steps


# --------------------------------------------------------------------------------------------------
# Reverse lean field-split (b): an ic_lora field authored under a non-ic_lora mode fails fast.
# --------------------------------------------------------------------------------------------------


def test_reference_field_rejected_in_single_frame_mode():
    with pytest.raises(ValidationError, match="ic_lora|reference_downscale_factor"):
        ConditioningConfig(mode="single_frame", reference_downscale_factor=2)


def test_seg_palette_name_rejected_in_none_mode():
    with pytest.raises(ValidationError, match="ic_lora|seg_palette_name"):
        ConditioningConfig(mode="none", seg_palette_name="other_palette")


def test_frozen_adapter_rejected_in_multi_frame_mode():
    with pytest.raises(ValidationError, match="ic_lora|frozen_adapter"):
        ConditioningConfig(mode="multi_frame", frozen_adapter_path="reference/frozen.safetensors")


def test_paired_source_rejected_outside_ic_lora():
    # conditioning_source == 'paired' is only meaningful in ic_lora mode (multi_frame is self-only).
    with pytest.raises(ValidationError, match="ic_lora|conditioning_source"):
        ConditioningConfig(mode="multi_frame", conditioning_source="paired")


def test_reference_field_at_default_allowed_in_other_modes():
    # A DEFAULT value on an ic_lora field must NOT trip the reverse split (else non-ic_lora modes
    # could never validate) — only NON-default values are rejected.
    cfg = ConditioningConfig(mode="single_frame", reference_downscale_factor=1)
    assert cfg.reference_downscale_factor == 1


# --------------------------------------------------------------------------------------------------
# WR-01 — reference_latents_dir is unwired: a non-default value is rejected in ic_lora mode.
# --------------------------------------------------------------------------------------------------


def test_reference_latents_dir_nondefault_rejected_in_ic_lora():
    # WR-01: nothing reads reference_latents_dir (the dataset dir name is hardcoded), so a non-default
    # value in ic_lora mode is silently ignored — reject it until the field is wired.
    with pytest.raises(ValidationError, match="reference_latents_dir|WR-01|silently ignored"):
        ConditioningConfig(mode="ic_lora", reference_latents_dir="my_refs")


def test_reference_latents_dir_default_allowed_in_ic_lora():
    cfg = ConditioningConfig(mode="ic_lora", reference_latents_dir="reference_latents")
    assert cfg.reference_latents_dir == "reference_latents"


# --------------------------------------------------------------------------------------------------
# WR-02 — first_frame_conditioning_p / _strength are unwired in ic_lora mode: reject non-default.
# --------------------------------------------------------------------------------------------------


def test_first_frame_conditioning_p_nondefault_rejected_in_ic_lora():
    # WR-02: loop.py threads only reference_downscale_factor for ic_lora (CR-01); the ICLoraStrategy
    # keeps p=0.0. A non-default first_frame_conditioning_p would silently train at 0.0 — reject it.
    with pytest.raises(ValidationError, match="first_frame_conditioning_p|ic_lora|WR-02"):
        ConditioningConfig(mode="ic_lora", first_frame_conditioning_p=0.5)


def test_first_frame_conditioning_strength_nondefault_rejected_in_ic_lora():
    # WR-02: strength is consumed by NO ic_lora path — reject a non-default value.
    with pytest.raises(ValidationError, match="first_frame_conditioning_strength|ic_lora|WR-02"):
        ConditioningConfig(mode="ic_lora", first_frame_conditioning_strength=0.5)


def test_ic_lora_default_conditioning_fields_allowed():
    # The defaults (p=1.0, strength=1.0) must NOT trip the WR-02 guards.
    cfg = ConditioningConfig(mode="ic_lora", reference_downscale_factor=1)
    assert cfg.first_frame_conditioning_p == 1.0
    assert cfg.first_frame_conditioning_strength == 1.0


# --------------------------------------------------------------------------------------------------
# Forward: the ic_lora fields ARE accepted (with defaults + explicit values) in ic_lora mode.
# --------------------------------------------------------------------------------------------------


def test_ic_lora_mode_accepts_reference_and_frozen_fields():
    cfg = ConditioningConfig(
        mode="ic_lora",
        conditioning_source="paired",
        reference_downscale_factor=2,
        seg_palette_name="compact_driving_v1",
        frozen_adapter_path="reference/frozen.safetensors",
        frozen_adapter_format="peft",
    )
    assert cfg.mode == "ic_lora"
    assert cfg.reference_downscale_factor == 2
    assert cfg.frozen_adapter_path == "reference/frozen.safetensors"


# --------------------------------------------------------------------------------------------------
# Field types / defaults (c, d).
# --------------------------------------------------------------------------------------------------


def test_reference_downscale_factor_is_int_default_one():
    cfg = ConditioningConfig()
    assert cfg.reference_downscale_factor == 1
    assert isinstance(cfg.reference_downscale_factor, int)


def test_reference_downscale_factor_rejects_below_one():
    with pytest.raises(ValidationError):
        ConditioningConfig(mode="ic_lora", reference_downscale_factor=0)  # ge=1


def test_seg_palette_name_default_and_roundtrip():
    assert ConditioningConfig().seg_palette_name == "compact_driving_v1"
    cfg = ConditioningConfig(mode="ic_lora", seg_palette_name="compact_driving_v1")
    assert cfg.seg_palette_name == "compact_driving_v1"


def test_frozen_adapter_format_allowlist():
    assert ConditioningConfig().frozen_adapter_format == "peft"
    with pytest.raises(ValidationError):
        ConditioningConfig(mode="ic_lora", frozen_adapter_format="bogus")


def test_frozen_adapter_path_must_be_volume_relative():
    # T-07-02: the frozen adapter path is joined under the checkpoints Volume prefix Modal-side —
    # an absolute path or '..' escape must die at config load.
    with pytest.raises(ValidationError, match="Volume-relative"):
        ConditioningConfig(mode="ic_lora", frozen_adapter_path="/etc/evil.safetensors")


# --------------------------------------------------------------------------------------------------
# D-7-BASEVAR (e): the distilled base variant must pair with the distilled inference path.
# --------------------------------------------------------------------------------------------------


def test_distilled_base_with_dev_single_stage_rejected():
    # distilled model_id + validation.two_stage_upscale False (the dev single-stage sample path) is a
    # fail-fast mismatch — the distilled base renders only through the two-stage distilled path.
    payload = _payload(conditioning={"mode": "ic_lora", "conditioning_source": "paired"})
    payload["model"] = {"model_id": "ltx-2.3-22b-distilled-1.1.safetensors"}
    payload["validation"] = {"two_stage_upscale": False}
    with pytest.raises(ValidationError, match="distilled|D-7-BASEVAR|two_stage_upscale"):
        SignetConfig.model_validate(payload)


def test_distilled_base_with_two_stage_path_accepted():
    # Same distilled base paired WITH the distilled two-stage inference path is fine.
    payload = _payload(conditioning={"mode": "ic_lora", "conditioning_source": "paired"})
    payload["model"] = {"model_id": "ltx-2.3-22b-distilled-1.1.safetensors"}
    payload["validation"] = {"two_stage_upscale": True}
    cfg = SignetConfig.model_validate(payload)
    assert "distilled" in cfg.model.model_id
    assert cfg.validation.two_stage_upscale is True


def test_dev_base_single_stage_is_fine():
    # The dev default base + single-stage path (both defaults) must NOT trip D-7-BASEVAR.
    cfg = SignetConfig.model_validate(
        _payload(conditioning={"mode": "ic_lora", "conditioning_source": "paired"})
    )
    assert "distilled" not in cfg.model.model_id
    assert cfg.validation.two_stage_upscale is False


# --------------------------------------------------------------------------------------------------
# 07-15 GAP-2 (T-07-15-05) — original_videos (col-1 ORIGINAL-clip list) reverse guard + the
# Volume-relative path contract. Same "no silently-ignored config block" shape as the
# reference_images-under-multi_frame reverse guard above.
# --------------------------------------------------------------------------------------------------


def test_original_videos_nonempty_rejected_outside_ic_lora_mode():
    # T-07-15-05: a non-empty original_videos under a non-ic_lora mode (e.g. multi_frame) would be
    # silently ignored on a metered render — reject it at config load (lean field-split, reverse).
    with pytest.raises(ValidationError, match="original_videos|ic_lora"):
        ConditioningConfig(mode="multi_frame", original_videos=["reference/original/x.mp4"])


def test_original_videos_nonempty_accepted_in_ic_lora_mode():
    # The forward direction: valid Volume-relative original_videos load cleanly under mode=ic_lora.
    cfg = ConditioningConfig(
        mode="ic_lora",
        conditioning_source="paired",
        original_videos=[
            "reference/original/demo_holdout_00.mp4",
            "reference/original/external_00.mp4",
        ],
    )
    assert cfg.original_videos == [
        "reference/original/demo_holdout_00.mp4",
        "reference/original/external_00.mp4",
    ]


def test_original_videos_rejects_absolute_path():
    # T-06-02/WR-09-shaped contract: original_videos joins the SAME validate_volume_relative_path
    # guard as reference_images/frozen_adapter_path — an absolute path must die at config load.
    with pytest.raises(ValidationError, match="Volume-relative"):
        ConditioningConfig(mode="ic_lora", original_videos=["/etc/evil.mp4"])


def test_original_videos_rejects_dotdot_escaping_path():
    with pytest.raises(ValidationError, match="Volume-relative"):
        ConditioningConfig(mode="ic_lora", original_videos=["reference/../../escape.mp4"])
