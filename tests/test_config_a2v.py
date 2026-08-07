"""Schema/validator proof for the audio-to-video (a2v) config surface (Phase 9, GATE-SPEC rev 2).

Covers: the ``audio_to_video`` mode allowlist, the audio modality block, the cross-modal LoRA-target
fail-fast ("the #1 silent a2v failure"), the with_audio requirement, the audio-block lean field-split
(reverse guard), and the AudioCondition validation surface — all CPU / zero-GPU / install-free.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import SignetConfig
from signet_trainer.config.validators import (
    ALLOWED_CONDITIONING_MODES,
    A2V_CROSS_MODAL_ATTN_TARGETS,
)

_A2V_TARGETS = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    *A2V_CROSS_MODAL_ATTN_TARGETS,
]


def _payload(**overrides):
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/dataset/.precomputed_a2v"},
        "training": {"max_steps": 600},
    }
    payload.update(overrides)
    return payload


def _a2v_payload(**overrides):
    base = _payload(
        conditioning={"mode": "audio_to_video"},
        lora={"target_modules": list(_A2V_TARGETS)},
        audio={"with_audio": True},
    )
    base.update(overrides)
    return base


def test_audio_to_video_in_allowlist() -> None:
    assert "audio_to_video" in ALLOWED_CONDITIONING_MODES


def test_valid_a2v_config_loads() -> None:
    cfg = SignetConfig.model_validate(_a2v_payload())
    assert cfg.conditioning.mode == "audio_to_video"
    assert cfg.audio.with_audio is True
    assert cfg.audio.is_generated is False
    assert cfg.audio.generate_audio is False


def test_a2v_missing_cross_modal_targets_fails_loud() -> None:
    payload = _a2v_payload(lora={"target_modules": ["attn1.to_q", "ff.net.2"]})
    with pytest.raises(ValidationError, match="cross-modal attention LoRA target"):
        SignetConfig.model_validate(payload)


def test_a2v_requires_with_audio() -> None:
    payload = _a2v_payload(audio={"with_audio": False})
    with pytest.raises(ValidationError, match="audio.with_audio is False"):
        SignetConfig.model_validate(payload)


def test_a2v_qualified_cross_modal_targets_accepted() -> None:
    # PEFT suffix semantics: the fully-qualified audio_transformer_blocks.* path satisfies the guard.
    targets = [
        *[t for t in _A2V_TARGETS if not t.startswith("audio_to_video_attn")],
        "audio_transformer_blocks.audio_to_video_attn.to_q",
        "audio_transformer_blocks.audio_to_video_attn.to_k",
        "audio_transformer_blocks.audio_to_video_attn.to_v",
        "audio_transformer_blocks.audio_to_video_attn.to_out.0",
    ]
    cfg = SignetConfig.model_validate(_a2v_payload(lora={"target_modules": targets}))
    assert cfg.conditioning.mode == "audio_to_video"


def test_audio_block_rejected_under_non_a2v_mode() -> None:
    # with_audio: true under mode 'none' is a silently-ignored block — reject at load.
    payload = _payload(audio={"with_audio": True})
    with pytest.raises(ValidationError, match="only valid when"):
        SignetConfig.model_validate(payload)


def test_generate_audio_default_off_under_none_ok() -> None:
    # audio block left at defaults under a non-a2v mode loads clean (backward-compat).
    cfg = SignetConfig.model_validate(_payload())
    assert cfg.audio.with_audio is False
    assert cfg.audio.generate_audio is False


def test_a2v_rejects_first_frame_conditioning() -> None:
    payload = _a2v_payload(
        conditioning={
            "mode": "audio_to_video",
            "first_frame_conditioning_p": 0.5,
        }
    )
    with pytest.raises(ValidationError, match="first_frame_conditioning_p is non-default"):
        SignetConfig.model_validate(payload)


def test_a2v_rejects_reference_images() -> None:
    payload = _a2v_payload(
        conditioning={"mode": "audio_to_video", "reference_images": ["ref.png"]}
    )
    with pytest.raises(ValidationError, match="reference_images is non-empty"):
        SignetConfig.model_validate(payload)


def test_audio_condition_only_valid_in_a2v_mode() -> None:
    # An 'audio' validation-sample condition under inpaint mode is silently ignored -> reject.
    # (inpaint dims are ÷64: 768x512x49 is clean, so the mode-mismatch guard is what fires.)
    payload = _payload(
        training_dims=[768, 512, 49],
        data={"preprocessed_data_root": "/dataset/.p", "resolution_buckets": ["768x512x49"]},
        conditioning={"mode": "inpaint"},
        validation={
            "width": 768,
            "height": 512,
            "frame_count": 49,
            "samples": [
                {"prompt": "x", "conditions": [{"type": "audio", "audio": "dataset/a.wav"}]}
            ],
        },
    )
    with pytest.raises(ValidationError, match="'audio' condition kind"):
        SignetConfig.model_validate(payload)


def test_audio_condition_accepted_in_a2v_mode() -> None:
    payload = _a2v_payload(
        validation={
            "samples": [
                {"prompt": "x", "conditions": [{"type": "audio", "audio": "dataset/a.wav"}]}
            ]
        }
    )
    cfg = SignetConfig.model_validate(payload)
    assert cfg.validation.samples[0].conditions[0].audio == "dataset/a.wav"


def test_audio_condition_path_must_be_volume_relative() -> None:
    payload = _a2v_payload(
        validation={
            "samples": [
                {"prompt": "x", "conditions": [{"type": "audio", "audio": "/etc/evil.wav"}]}
            ]
        }
    )
    with pytest.raises(ValidationError, match="Volume-relative"):
        SignetConfig.model_validate(payload)


def test_campaign_a2v_config_loads_and_carries_a2v_posture() -> None:
    from pathlib import Path

    from signet_trainer.config.load import load_config

    repo = Path(__file__).resolve().parents[1]
    c = load_config(str(repo / "configs" / "campaign_a2v.example.yaml"))
    assert c.conditioning.mode == "audio_to_video"
    assert c.audio.with_audio is True
    assert c.audio.generate_audio is False
    # dev base (NOT fused, NOT distilled).
    assert c.model.model_id == "ltx-2.3-22b-dev.safetensors"
    assert "distilled" not in c.model.model_id
    # likeness continuation from r5.
    assert c.training.init_adapter_path == "outputs/campaign_r5/checkpoint-step-03000-loss-0.1318"
    # cross-modal targets present (schema would have rejected otherwise, but assert explicitly).
    for t in A2V_CROSS_MODAL_ATTN_TARGETS:
        assert t in c.lora.target_modules
    # r5 continuation rank/alpha, small overfit, proven posture.
    assert c.lora.rank == 42 and c.lora.alpha == 42
    assert c.training.max_steps == 600
    assert c.training.checkpoint_every == 200
    assert c.offload.blocks_to_swap == 24
    # eval spec.
    assert c.validation.frame_count == 81
    assert c.validation.num_inference_steps == 40
    assert c.validation.guidance_scale == 4.0
    assert c.validation.two_stage_upscale is False
    # 1280x704 multi-F family.
    assert c.training_dims == (1280, 704, 81)
    assert "1280x704x81" in c.data.resolution_buckets
    assert c.data.preprocessed_data_root.endswith(".precomputed_campaign_a2v")


def test_mask_condition_still_disambiguates_without_type() -> None:
    # backward-compat: a {video, mask} dict (no explicit type) still parses as a MaskCondition.
    payload = _payload(
        conditioning={"mode": "inpaint"},
        training_dims=[768, 512, 49],
        data={"preprocessed_data_root": "/dataset/.p", "resolution_buckets": ["768x512x49"]},
        validation={
            "width": 768,
            "height": 512,
            "frame_count": 49,
            "samples": [
                {
                    "prompt": "x",
                    "conditions": [{"video": "dataset/clip.mp4", "mask": "dataset/m.mp4"}],
                }
            ],
        },
    )
    cfg = SignetConfig.model_validate(payload)
    assert cfg.validation.samples[0].conditions[0].type == "mask"
