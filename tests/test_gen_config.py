"""INFR-01 — the pure CPU GenerationConfig mapper (build_generation_config layer).

Proves that ``inference.sampler._generation_kwargs`` maps a default ``SignetConfig`` onto
the RE-VALIDATED LTX-2.3 canonical sampling kwargs (Euler + STG, guidance 3.0, frames
8k+1, W/H %32, generate_audio=False) — NOT the Wan-tuned set. This layer is deliberately
pure/CPU so it unit-tests on Windows/CI WITHOUT ltx_trainer installed: the heavy
``GenerationConfig`` import lives inside ``build_generation_config``, never at module top
(Anti-Pattern 6, mirrors ``models/loader.py``).
"""

from __future__ import annotations

import sys

import pytest

from signet_trainer.config.schema import SignetConfig
from signet_trainer.inference.multi_condition import (
    plan_conditioning_items,
    plan_multi_frame_columns,
)
from signet_trainer.inference.sampler import _generation_kwargs


def _default_config() -> SignetConfig:
    """A minimal-but-valid SignetConfig (mirrors tests/test_config_validators._valid_payload)."""
    return SignetConfig.model_validate(
        {
            "training_dims": [768, 512, 49],
            "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
            "training": {"max_steps": 100},
        }
    )


def _multi_frame_config() -> SignetConfig:
    """A valid multi_frame SignetConfig with 3 %8-aligned keyframes (mirrors Plan 06-01 shape)."""
    return SignetConfig.model_validate(
        {
            "training_dims": [768, 512, 49],
            "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
            "training": {"max_steps": 100},
            "conditioning": {
                "mode": "multi_frame",
                "conditioning_items": [
                    {"image": "kf_a.png", "frame_index": 0, "strength": 0.8},
                    {"image": "kf_b.png", "frame_index": 8, "strength": 0.6},
                    {"image": "kf_c.png", "frame_index": 16, "strength": 0.4},
                ],
                "max_conditioning_items": 3,
                "conditioning_strength_range": [0.3, 1.0],
            },
        }
    )


def test_generation_kwargs_are_the_ltx_canonical_set() -> None:
    cfg = _default_config()
    kw = _generation_kwargs(cfg, prompt="hello", seed=42)

    assert kw["prompt"] == "hello"
    assert kw["seed"] == 42
    assert kw["guidance_scale"] == 3.0
    assert kw["num_inference_steps"] == 30
    assert kw["stg_scale"] == 1.0
    assert kw["stg_blocks"] == [29]
    assert kw["stg_mode"] == "stg_v"
    assert kw["generate_audio"] is False
    assert kw["frame_rate"] == 25.0
    assert kw["width"] == 768
    assert kw["height"] == 352
    assert kw["negative_prompt"] == ""
    # num_frames rides the validation frame_count (default 49 — an 8k+1 value).
    assert kw["num_frames"] == cfg.validation.frame_count == 49


def test_generation_kwargs_has_no_wan_params() -> None:
    """No Wan token (shift / decode_timestep / UniPC) leaks into the mapped kwargs."""
    kw = _generation_kwargs(_default_config(), prompt="p", seed=1)
    for forbidden in ("shift", "decode_timestep", "scheduler", "sampler"):
        assert forbidden not in kw


def test_condition_image_defaults_to_none_ref_off_control() -> None:
    """Omitting condition_image yields the ref-OFF control default (None) — SC#2."""
    kw = _generation_kwargs(_default_config(), prompt="p", seed=1)
    assert "condition_image" in kw
    assert kw["condition_image"] is None


def test_condition_image_passthrough_ref_on() -> None:
    """A supplied condition_image ([C,H,W] in [0,1], here a sentinel) rides the kwargs unchanged."""
    sentinel = object()  # stands in for a torch.Tensor — mapper is pure/CPU, never touches it
    kw = _generation_kwargs(_default_config(), prompt="p", seed=1, condition_image=sentinel)
    assert kw["condition_image"] is sentinel


def test_condition_image_none_reproduces_ref_off_kwargs() -> None:
    """condition_image=None must reproduce the exact ref-OFF control kwargs (no drift)."""
    base = _generation_kwargs(_default_config(), prompt="p", seed=1)
    with_none = _generation_kwargs(_default_config(), prompt="p", seed=1, condition_image=None)
    assert base == with_none


def test_sampler_module_is_import_confined_on_cpu() -> None:
    """Importing the sampler module must NOT pull ltx_trainer/ltx_core/modal on CPU/CI."""
    for mod in ("modal", "ltx_core", "ltx_trainer"):
        sys.modules.pop(mod, None)

    import signet_trainer.inference.sampler  # noqa: F401

    for mod in ("modal", "ltx_core", "ltx_trainer"):
        assert mod not in sys.modules, (
            f"import-confinement violation: inference.sampler transitively imported {mod!r}"
        )


# --------------------------------------------------------------------------------------------------
# REF-02 (Seam E) — pure planners: plan_conditioning_items + plan_multi_frame_columns
# --------------------------------------------------------------------------------------------------


def test_plan_conditioning_items_maps_pixel_frame_to_latent_index() -> None:
    """frame_index // 8 mapping with strength carried verbatim (Pitfall 3, dict form)."""
    assert plan_conditioning_items([{"frame_index": 24, "strength": 0.5}]) == [(3, 0.5)]


def test_plan_conditioning_items_maps_multiple_ordered() -> None:
    items = [
        {"frame_index": 0, "strength": 0.8},
        {"frame_index": 8, "strength": 0.6},
        {"frame_index": 16, "strength": 0.4},
    ]
    assert plan_conditioning_items(items) == [(0, 0.8), (1, 0.6), (2, 0.4)]


def test_plan_conditioning_items_rejects_unaligned_frame_index() -> None:
    """Defense-in-depth: a non-%8 frame_index raises ValueError (config already validated)."""
    with pytest.raises(ValueError, match="frame_index"):
        plan_conditioning_items([{"frame_index": 25, "strength": 1.0}])


def test_plan_conditioning_items_accepts_conditioning_item_objects() -> None:
    """Accepts attribute-style items (the real ConditioningItem sub-model), not just dicts."""
    cfg = _multi_frame_config()
    planned = plan_conditioning_items(cfg.conditioning.conditioning_items)
    assert planned == [(0, 0.8), (1, 0.6), (2, 0.4)]


def test_plan_multi_frame_columns_has_the_sc3_columns_in_order() -> None:
    """Ordered N=2, N=3, strength-lo, strength-hi, no-reference column plan (SC#3 grid)."""
    cols = plan_multi_frame_columns(_multi_frame_config())
    assert [c.row_key for c in cols] == [
        "n2_mp4",
        "n3_mp4",
        "strength_lo_mp4",
        "strength_hi_mp4",
        "no_ref_mp4",
    ]


def test_plan_multi_frame_columns_no_reference_column_is_empty() -> None:
    cols = {c.row_key: c for c in plan_multi_frame_columns(_multi_frame_config())}
    assert cols["no_ref_mp4"].items == []


def test_plan_multi_frame_columns_n2_and_n3_take_first_items() -> None:
    cols = {c.row_key: c for c in plan_multi_frame_columns(_multi_frame_config())}
    assert cols["n2_mp4"].items == [(0, 0.8), (1, 0.6)]
    assert cols["n3_mp4"].items == [(0, 0.8), (1, 0.6), (2, 0.4)]


def test_plan_multi_frame_columns_strength_sweep_overrides_mid_item() -> None:
    """strength-lo/hi override the MIDDLE item's strength to the config range lo/hi; distinct."""
    cols = {c.row_key: c for c in plan_multi_frame_columns(_multi_frame_config())}
    # 3 items, middle index = 1; range = (0.3, 1.0).
    assert cols["strength_lo_mp4"].items == [(0, 0.8), (1, 0.3), (2, 0.4)]
    assert cols["strength_hi_mp4"].items == [(0, 0.8), (1, 1.0), (2, 0.4)]
    assert cols["strength_lo_mp4"].items != cols["strength_hi_mp4"].items


def test_plan_multi_frame_columns_are_labelled() -> None:
    cols = plan_multi_frame_columns(_multi_frame_config())
    labels = [c.label for c in cols]
    assert all(isinstance(lbl, str) and lbl for lbl in labels)
    assert len(set(labels)) == len(labels)  # distinct labels


@pytest.mark.parametrize("keep", [1, 2])
def test_plan_multi_frame_columns_rejects_fewer_than_three_items(keep: int) -> None:
    """WR-07: with 1-2 keyframes the N=3 column would silently duplicate N=2 (byte-identical
    videos under different labels) — the planner must fail fast, before any render spend."""
    cfg = _multi_frame_config()
    cfg.conditioning.conditioning_items = cfg.conditioning.conditioning_items[:keep]
    with pytest.raises(ValueError, match=">= 3"):
        plan_multi_frame_columns(cfg)


def test_plan_multi_frame_columns_rejects_empty_items() -> None:
    cfg = _multi_frame_config()
    cfg.conditioning.conditioning_items = []
    with pytest.raises(ValueError, match=">= 3"):
        plan_multi_frame_columns(cfg)
