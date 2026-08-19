"""#20 (config-layer honesty gaps) + #4 (H3 renderable-band notice) + the GC Tier-1 knob.

Seven findings, all the same shape: the config layer CERTIFIED something it did not enforce. Each
test here pins the refusal (or notice) that now fires at config load, $0, before any metered
container — and cites the file:line the finding named. CPU-only, no ltx_core / modal / torch
import (Anti-Pattern 6 / CONF-03 zero-spend).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from signet_trainer.config.load import load_config_from_text
from signet_trainer.config.schema import DataConfig, SignetConfig
from signet_trainer.config.validators import A2V_CROSS_MODAL_ATTN_TARGETS

REPO = Path(__file__).resolve().parents[1]
FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"

# The operator's real Phase-10 reference corpus is NOT needed here — every H3 payload below sets
# references_per_sample: 0 (NO-REFERENCE, ALPHA) so the budget machinery stays out of the way of
# the geometry/render checks under test.
_H3_LEGAL_TRAINING_DIMS = [1344, 768, 22]  # 17n+5, matches target_aspect [16, 9]'s canvas exactly


def _ltx_payload(**over) -> dict:
    payload: dict = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return payload


def _h3_payload(**over) -> dict:
    payload: dict = {
        "training_dims": list(_H3_LEGAL_TRAINING_DIMS),
        "data": {
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x22"],
        },
        "model": {"family": "h3"},
        "training": {"max_steps": 100},
        "h3": {"references_per_sample": 0},
        "validation": {"frame_count": 124},  # inside the renderable band
    }
    payload.update(over)
    return payload


# ======================================================================================
# #20 checklist item 1 (the only `major`) — validation.width/height/frame_count unguarded
# outside inpaint mode (schema.py: validation width/height/frame_count fields).
# ======================================================================================


def test_ltx_render_triple_illegal_frame_count_is_refused_outside_inpaint() -> None:
    """The issue's own probe: `frame_count: 50` ((50-1)%8 == 1) loaded clean on main."""
    payload = _ltx_payload(validation={"frame_count": 50})
    with pytest.raises(ValidationError, match="frame"):
        SignetConfig.model_validate(payload)


def test_ltx_render_triple_illegal_height_is_refused_outside_inpaint() -> None:
    """The issue's own probe: `height: 350` (350 % 32 == 30) loaded clean on main."""
    payload = _ltx_payload(validation={"height": 350})
    with pytest.raises(ValidationError, match="32"):
        SignetConfig.model_validate(payload)


def test_ltx_render_triple_legal_values_still_load() -> None:
    """Byte-identical-load control: every shipped LTX example's render triple stays legal."""
    cfg = SignetConfig.model_validate(_ltx_payload(validation={"width": 832, "height": 480, "frame_count": 25}))
    assert cfg.validation.frame_count == 25


def test_h3_render_triple_off_band_frame_count_is_refused_without_waiver() -> None:
    """H3's render frame law is the RENDERABILITY band, not 17n+5 — 90 is a legal training bucket
    but is outside the 120-360 renderable band and h3_sample would die inside
    MiniMaxH3Ref2VASetupStep on a metered A100 (after both 61.7 GiB loads)."""
    payload = _h3_payload(validation={"frame_count": 90})
    with pytest.raises(ValidationError, match="eval-design decision"):
        SignetConfig.model_validate(payload)


def test_h3_render_triple_off_band_frame_count_is_accepted_with_waiver() -> None:
    """validation.allow_offband_frame_count (merged PR #5) is the documented render-time waiver."""
    payload = _h3_payload(validation={"frame_count": 90, "allow_offband_frame_count": True})
    cfg = SignetConfig.model_validate(payload)
    assert cfg.validation.frame_count == 90


def test_h3_render_triple_below_decode_floor_is_refused_even_with_waiver() -> None:
    """The 22-frame VAE decode floor is arithmetic, not policy — no waiver reaches it."""
    payload = _h3_payload(validation={"frame_count": 5, "allow_offband_frame_count": True})
    with pytest.raises(ValidationError, match="DECODE FLOOR"):
        SignetConfig.model_validate(payload)


def test_h3_render_triple_illegal_width_is_refused() -> None:
    payload = _h3_payload(validation={"frame_count": 124, "width": 750})
    with pytest.raises(ValidationError, match="32"):
        SignetConfig.model_validate(payload)


def test_qwen_edit_render_triple_illegal_width_is_refused() -> None:
    payload = {
        "training_dims": [1024, 1024, 1],
        "data": {
            "preprocessed_data_root": "/data/qwen_edit",
            "batch_size": 1,
            "resolution_buckets": ["1024x1024x1"],
        },
        "model": {"family": "qwen_edit"},
        "training": {"max_steps": 100},
        "lora": {"rank": 42, "alpha": 42},
        "validation": {"frame_count": 1, "width": 1000},
        "conditioning": {"mode": "none"},
    }
    with pytest.raises(ValidationError, match="32"):
        SignetConfig.model_validate(payload)


def test_inpaint_mode_still_takes_the_stricter_div64_path() -> None:
    """The pre-existing inpaint override is UNCHANGED — this is the regression test for it."""
    payload = _ltx_payload(
        training_dims=[1280, 704, 49],
        data={
            "preprocessed_data_root": "/data/preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1280x704x49"],
        },
        conditioning={"mode": "inpaint"},
        validation={"height": 384},  # the default 352 fails inpaint's %64 rule
    )
    cfg = SignetConfig.model_validate(payload)
    assert cfg.conditioning.mode == "inpaint"


# ======================================================================================
# #20 checklist item ("training_dims[0:2] validated-but-unconsumed under h3", schema.py:1584) —
# the h3.target_aspect canvas cross-check. Coordinates with #1 (resolution_buckets' twin finding).
# ======================================================================================


def test_h3_training_dims_disagreeing_with_target_aspect_canvas_is_refused() -> None:
    payload = _h3_payload(h3={"references_per_sample": 0, "target_aspect": [4, 3]})
    with pytest.raises(ValidationError, match="target_aspect"):
        SignetConfig.model_validate(payload)


def test_h3_training_dims_matching_target_aspect_canvas_still_loads() -> None:
    """Byte-identical-load control: the shipped h3_t2va_noref.example.yaml shape (1344x768, 16:9)."""
    cfg = SignetConfig.model_validate(_h3_payload())
    assert cfg.training_dims[0] == 1344 and cfg.training_dims[1] == 768


# ======================================================================================
# #4 — config-load NOTICE (not a refusal) when training_dims' frame count is outside the
# renderable band. Shares h3_renderable_frame_bounds() with the render-time assertion (#20 item 1).
# ======================================================================================


def test_offband_training_length_prints_a_notice_not_a_refusal(capsys) -> None:
    """The shipped Embe-shaped campaign (F=22) and a still-frame campaign (F=5) both train legally
    below the renderable band — this must load AND print, never raise."""
    cfg = SignetConfig.model_validate(_h3_payload())  # training_dims F=22
    assert cfg.training_dims[2] == 22
    captured = capsys.readouterr()
    assert "[signet] NOTE" in captured.err
    assert "renderable band" in captured.err
    assert "D-10-DEF-15" in captured.err


def test_offband_training_length_notice_mentions_the_waiver_state(capsys) -> None:
    SignetConfig.model_validate(_h3_payload(validation={"frame_count": 124, "allow_offband_frame_count": True}))
    captured = capsys.readouterr()
    assert "allow_offband_frame_count is already set" in captured.err


def test_inband_training_length_prints_no_notice(capsys) -> None:
    """training_dims F=124 (17*7+5, inside the 120-360 band) must not trip the notice at all."""
    SignetConfig.model_validate(
        _h3_payload(
            training_dims=[1344, 768, 124],
            # #77's coherence guard: the declared bucket must agree with training_dims F now that
            # buckets are load-bearing on h3 — keep the fixture coherent at 124.
            data={
                "preprocessed_data_root": "/data/h3_preprocessed",
                "batch_size": 1,
                "resolution_buckets": ["1344x768x124"],
            },
            validation={"frame_count": 124},
            # A 124f NO-REFERENCE campaign busts the measured 13,777-row ceiling (unrelated to
            # this test) — widen the budget so only the notice logic under test is exercised.
            h3={"references_per_sample": 0, "gpu_usable_gib": 200.0},
        )
    )
    captured = capsys.readouterr()
    assert "[signet] NOTE" not in captured.err


# ======================================================================================
# #20 checklist item — output_dir: the only operator path exempt from
# validate_volume_relative_path (schema.py:1617 in the issue's citation).
# ======================================================================================


def test_output_dir_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        SignetConfig.model_validate(_ltx_payload(output_dir="/outputs/run1"))


def test_output_dir_rejects_dotdot_escape() -> None:
    with pytest.raises(ValidationError, match="output_dir"):
        SignetConfig.model_validate(_ltx_payload(output_dir="../escape"))


def test_output_dir_default_and_relative_values_still_load() -> None:
    assert SignetConfig.model_validate(_ltx_payload()).output_dir == "outputs"
    cfg = SignetConfig.model_validate(_ltx_payload(output_dir="outputs/campaign_r1_mf"))
    assert cfg.output_dir == "outputs/campaign_r1_mf"


# ======================================================================================
# #20 checklist item — data.metadata_path's default was RELATIVE while the comment claimed it
# matched fns.preprocess's own ABSOLUTE default (schema.py:103 / modal/fns.py:141).
# ======================================================================================


def test_metadata_path_default_matches_fns_preprocess_exactly() -> None:
    expected = "/dataset/fresh/metadata.jsonl"
    assert DataConfig(preprocessed_data_root="/x").metadata_path == expected
    code = FNS.read_text(encoding="utf-8")
    match = re.search(
        r'def preprocess\(\s*metadata_path:\s*str\s*=\s*str\(\s*DATASET_DIR\s*/\s*'
        r'"fresh"\s*/\s*"metadata\.jsonl"\s*\)',
        code,
    )
    assert match is not None, "fns.preprocess's own default shape must still be str(DATASET_DIR / 'fresh' / 'metadata.jsonl')"
    assert 'DATASET_DIR = Path("/dataset")' in (
        REPO / "src" / "signet_trainer" / "modal" / "app.py"
    ).read_text(encoding="utf-8"), "DATASET_DIR must still resolve to /dataset for the literal above to be exact"


# ======================================================================================
# #20 checklist item — audio.audio_latents_dir is DEAD: conditioning/a2v.py:118 hardcodes the
# source-key list rather than reading it, and the reverse guard only fired OUTSIDE a2v mode.
# ======================================================================================


def _a2v_payload(**over) -> dict:
    payload: dict = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/dataset/.precomputed_a2v"},
        "training": {"max_steps": 100},
        "conditioning": {"mode": "audio_to_video"},
        "lora": {
            "target_modules": [
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
                "ff.net.0.proj", "ff.net.2",
                *A2V_CROSS_MODAL_ATTN_TARGETS,
            ]
        },
        "audio": {"with_audio": True},
    }
    payload.update(over)
    return payload


def test_audio_latents_dir_non_default_is_refused_in_a2v_mode() -> None:
    payload = _a2v_payload(audio={"with_audio": True, "audio_latents_dir": "custom_audio"})
    with pytest.raises(ValidationError, match="audio_latents_dir"):
        SignetConfig.model_validate(payload)


def test_audio_latents_dir_default_still_loads_in_a2v_mode() -> None:
    cfg = SignetConfig.model_validate(_a2v_payload())
    assert cfg.audio.audio_latents_dir == "audio_latents"


# ======================================================================================
# Load-path sanity: the NOTICE fires through the SAME seam (config/load.py::load_config_from_text)
# every real config load and every in-container revalidation go through.
# ======================================================================================


def test_notice_fires_through_load_config_from_text(capsys) -> None:
    yaml_text = """
training_dims: [1344, 768, 22]
data:
  preprocessed_data_root: /data/h3_preprocessed
  batch_size: 1
  resolution_buckets: ["1344x768x22"]
model:
  family: h3
training:
  max_steps: 100
h3:
  references_per_sample: 0
validation:
  frame_count: 124
"""
    load_config_from_text(yaml_text)
    captured = capsys.readouterr()
    assert "[signet] NOTE" in captured.err
