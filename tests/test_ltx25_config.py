"""LTX-2.5 Stage 1 (issue #53) config layer — the CPU gate that must hold before any GPU dollar.

Mirrors the qwen_edit/H3 family-config test shape (test_qwen_edit_config.py /
test_h3_config_schema.py): every existing (``ltx_generation`` unset) config must load
byte-identically, the ``ltx25`` block is bidirectionally lean-field-split, and the split-layout /
in-loop-sampling / two-stage-upscale guards fire at config LOAD time, before any GPU dollar.

CPU only. No torch model, no Modal, no downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from signet_trainer.config.load import load_config
from signet_trainer.config.schema import Ltx25Config, ModelConfig, SignetConfig

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ltx25_train.example.yaml"


def _ltx_payload(**over) -> dict:
    """A minimal valid LTX config — the byte-identical-load control."""
    payload: dict = {
        "training_dims": [768, 352, 25],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return payload


def _ltx25_payload(**over) -> dict:
    """A minimal valid LTX-2.5 config: same shape, ``ltx_generation: '2.5'`` set."""
    payload = _ltx_payload(model={"ltx_generation": "2.5"})
    payload.update(over)
    return payload


# ======================================================================================
# Byte-identity — every existing config loads to ltx_generation == "2.3", ltx25 all-default.
# ======================================================================================


def test_default_generation_is_23() -> None:
    """Every existing config (no ``model.ltx_generation`` key) parses to ``'2.3'``."""
    cfg = SignetConfig(**_ltx_payload())
    assert cfg.model.ltx_generation == "2.3"
    assert cfg.ltx25 == Ltx25Config()  # every ltx25 field at its pristine default


def test_model_config_ltx_generation_field_default() -> None:
    assert ModelConfig().ltx_generation == "2.3"


def test_byte_identity_regression_dump_is_unaffected_by_the_new_fields() -> None:
    """Regression proof (LTX25_STAGE1_DESIGN.md byte-identity requirement): a loaded LTX-2.3
    config's dump carries the new fields ONLY at their all-default values — nothing about the
    pre-existing dump shape changed, and every new key is inert for a '2.3' config."""
    cfg = SignetConfig(**_ltx_payload())
    dump = cfg.model_dump()
    assert dump["model"]["ltx_generation"] == "2.3"
    assert dump["ltx25"] == {
        "checkpoint_layout": "monolith",
        "video_vae_path": None,
        "audio_vae_path": None,
    }


# ======================================================================================
# Reverse guards (bidirectional lean field-split, pristine-instance technique).
# ======================================================================================


def test_ltx25_block_reverse_guard_fires_naming_the_offending_field() -> None:
    with pytest.raises(ValidationError, match="ltx25 field"):
        SignetConfig(**_ltx_payload(ltx25={"checkpoint_layout": "split", "video_vae_path": "x.safetensors"}))


def test_ltx25_block_all_default_is_legal_under_generation_23() -> None:
    """An all-default ``ltx25`` block never trips the reverse guard, even under gen 2.3 (the
    pristine-instance comparison must not false-positive on a config that never touched it)."""
    SignetConfig(**_ltx_payload(ltx25={}))


def test_ltx_generation_reverse_guard_fires_outside_family_ltx() -> None:
    with pytest.raises(ValidationError, match="ltx_generation"):
        SignetConfig(
            **_ltx_payload(
                model={"family": "h3", "ltx_generation": "2.5"},
                h3={},
            )
        )


def test_ltx_generation_25_requires_family_ltx_implicitly() -> None:
    """family defaults to 'ltx', so setting ltx_generation: '2.5' alone is legal."""
    cfg = SignetConfig(**_ltx25_payload())
    assert cfg.model.family == "ltx"
    assert cfg.model.ltx_generation == "2.5"


# ======================================================================================
# Split-layout fail-fast.
# ======================================================================================


def test_split_layout_requires_video_vae_path() -> None:
    with pytest.raises(ValidationError, match="video_vae_path"):
        SignetConfig(**_ltx25_payload(ltx25={"checkpoint_layout": "split"}))


def test_split_layout_with_video_vae_path_is_legal() -> None:
    cfg = SignetConfig(
        **_ltx25_payload(ltx25={"checkpoint_layout": "split", "video_vae_path": "ltx25_vae.safetensors"})
    )
    assert cfg.ltx25.checkpoint_layout == "split"
    assert cfg.ltx25.video_vae_path == "ltx25_vae.safetensors"


def test_monolith_layout_with_video_vae_path_set_is_legal_too() -> None:
    """The design forward-declares video_vae_path for Stage 2/3 use — nothing refuses setting it
    under 'monolith' (only the REQUIREDNESS under 'split' is asserted)."""
    cfg = SignetConfig(
        **_ltx25_payload(ltx25={"checkpoint_layout": "monolith", "video_vae_path": "x.safetensors"})
    )
    assert cfg.ltx25.checkpoint_layout == "monolith"


# ======================================================================================
# Stage-2-scope refusals (in_loop_sampling / two_stage_upscale) for ltx_generation == "2.5".
# ======================================================================================


def test_in_loop_sampling_refused_on_25() -> None:
    with pytest.raises(ValidationError, match="ValidationSampler"):
        SignetConfig(
            **_ltx25_payload(
                validation={"in_loop_sampling": True, "prompts": ["a cat"]},
            )
        )


def test_in_loop_sampling_false_is_the_default_and_legal_on_25() -> None:
    cfg = SignetConfig(**_ltx25_payload())
    assert cfg.validation.in_loop_sampling is False


def test_two_stage_upscale_refused_on_25() -> None:
    with pytest.raises(ValidationError, match="two_stage_upscale"):
        SignetConfig(**_ltx25_payload(validation={"two_stage_upscale": True}))


def test_two_stage_upscale_true_still_legal_on_23() -> None:
    """Byte-identity: the NEW guard must not touch a 2.3 config that legitimately pairs a
    distilled base with the two-stage path (D-7-BASEVAR)."""
    cfg = SignetConfig(
        **_ltx_payload(
            model={"model_id": "ltx-2.3-22b-distilled.safetensors"},
            validation={"two_stage_upscale": True},
        )
    )
    assert cfg.validation.two_stage_upscale is True


def test_distilled_pairing_check_is_scoped_to_generation_23_not_25() -> None:
    """The D-7-BASEVAR distilled/two_stage_upscale pairing check must not fire for a '2.5'
    config carrying a 'distilled'-named model_id — that pairing is a 2.3-specific fact, and the
    two_stage_upscale-refused-on-25 guard above is the one that speaks for 2.5."""
    with pytest.raises(ValidationError, match="two_stage_upscale"):
        # two_stage_upscale defaults False, and gen 2.5 refuses True -- so a 'distilled'-named
        # model_id under gen 2.5 hits ONLY the 2.5 guard, never the (scoped-out) 2.3 pairing check.
        SignetConfig(
            **_ltx25_payload(
                model={"model_id": "ltx-2.5-22b-distilled.safetensors", "ltx_generation": "2.5"},
                validation={"two_stage_upscale": True},
            )
        )
    # And with two_stage_upscale left at its default False, the SAME distilled-named model_id
    # loads clean under gen 2.5 (proving the 2.3-only pairing check truly never fires for it) —
    # it would raise under gen 2.3 (test_distilled_still_refused_on_23 below).
    cfg = SignetConfig(
        **_ltx25_payload(model={"model_id": "ltx-2.5-22b-distilled.safetensors", "ltx_generation": "2.5"})
    )
    assert cfg.model.model_id == "ltx-2.5-22b-distilled.safetensors"


def test_distilled_still_refused_on_23_without_two_stage_upscale() -> None:
    """Regression: the pre-existing D-7-BASEVAR check keeps its exact 2.3 behavior."""
    with pytest.raises(ValidationError, match="DISTILLED"):
        SignetConfig(**_ltx_payload(model={"model_id": "ltx-2.3-22b-distilled.safetensors"}))


# ======================================================================================
# The shipped example config loads clean.
# ======================================================================================


def test_example_config_loads() -> None:
    cfg = load_config(str(EXAMPLE_CONFIG))
    assert cfg.model.ltx_generation == "2.5"
    assert cfg.model.family == "ltx"
    assert cfg.validation.in_loop_sampling is False
    assert cfg.validation.two_stage_upscale is False
    assert cfg.conditioning.mode == "none"
