"""MASK-SPEC.yaml loader tests (D-03). Pure CPU — NO modal, NO GPU, NO network.

Locks the fail-LOUD ``load_mask_spec`` contract: the mask spec is house law, so its absence raises
``FileNotFoundError`` and a half-written spec raises ``ValueError`` naming the file + the missing
keys (mirrors ``harness_state.load_house_spec`` / ``load_tier_taxonomy``). The per-class D-02
auto-extend dilation config is shape-checked at load — a hair-heavy band missing its dilation
sub-block is a half-written spec, never "no auto-extend".

House test rules honored: no metered dispatch, no modal CLI, no network — tmp-file fixtures plus a
read-only load of the committed MASK-SPEC.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from signet_trainer import harness_data
from signet_trainer.prep.spec import load_mask_spec

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The PACKAGED spec (signet_trainer.harness_data) — the same resolution production uses.
_REAL_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)

_REQUIRED = ("hex", "chroma", "thresholds", "polarity", "coverage_bands", "dims")


def _valid_spec() -> dict:
    """A minimal-but-complete mask spec mirroring the committed MASK-SPEC.yaml shape."""
    return {
        "hex": "#458070",
        "rgb": [69, 128, 112],
        "chroma": {
            "method": "channel_relationship",
            "alternate_method": "distance",
            "tolerance": 40,
            "channel_rules": {"b_min": 90, "g_min": 90, "g_minus_r_min": 30, "b_minus_r_min": 20},
        },
        "thresholds": {"seed_load": 127, "encode_binarize": 0.5},
        "polarity": {
            "chain": [
                "paint_region=white_255",
                "stage_negate",
                "mask_mp4_region=black_0",
                "encode_gt_0.5",
                "tensor_keep=1.0",
                "tensor_generate=0.0",
            ]
        },
        "coverage_bands": {
            "face": {"target": 0.08, "warn_below": 0.04,
                     "dilation": {"max_margin_px": 12, "grow_to_target": True}},
            "face_hair": {"target": 0.14, "warn_below": 0.08,
                          "dilation": {"max_margin_px": 28, "grow_to_target": True}},
            "full_body": {"target": 0.14, "warn_below": 0.08,
                          "dilation": {"max_margin_px": 32, "grow_to_target": True}},
        },
        "dims": {"spatial_divisor": 64, "frame_rule": "8n+1"},
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "MASK-SPEC.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_valid_spec_loads_and_carries_required_keys(tmp_path: Path) -> None:
    spec = load_mask_spec(_write(tmp_path, _valid_spec()))
    assert spec["hex"] == "#458070"
    assert set(_REQUIRED) <= set(spec)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mask_spec(tmp_path / "absent.yaml")


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "MASK-SPEC.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_mask_spec(p)


@pytest.mark.parametrize("drop", _REQUIRED)
def test_missing_required_key_names_file_and_key(tmp_path: Path, drop: str) -> None:
    data = _valid_spec()
    del data[drop]
    p = _write(tmp_path, data)
    with pytest.raises(ValueError) as exc:
        load_mask_spec(p)
    msg = str(exc.value)
    assert p.name in msg and drop in msg


def test_hair_heavy_band_missing_dilation_fails_loud(tmp_path: Path) -> None:
    data = _valid_spec()
    del data["coverage_bands"]["face_hair"]["dilation"]
    p = _write(tmp_path, data)
    with pytest.raises(ValueError, match="dilation"):
        load_mask_spec(p)


def test_band_missing_target_fails_loud(tmp_path: Path) -> None:
    data = _valid_spec()
    del data["coverage_bands"]["face"]["target"]
    p = _write(tmp_path, data)
    with pytest.raises(ValueError, match="target"):
        load_mask_spec(p)


def test_real_committed_spec_canonizes_d01_and_d02() -> None:
    spec = load_mask_spec(_REAL_SPEC)
    assert spec["hex"] == "#458070"
    assert set(_REQUIRED) <= set(spec)
    # D-02 auto-extend knob present on the hair-heavy class.
    assert "dilation" in spec["coverage_bands"]["face_hair"]


def test_loader_is_import_confined() -> None:
    import signet_trainer.prep.spec as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "import cv2", "import modal", "from torch ", "from modal "):
        assert banned not in src, f"{banned!r} leaked into prep.spec (must stay stdlib + yaml)"
