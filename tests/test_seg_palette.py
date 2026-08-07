"""CPU proof of the Cityscapes-19 → compact-driving-palette remap (07-06 Task 1, REF-03, D-7-PALETTE).

numpy-only, zero-GPU. Proves ``data/seg_palette.py`` is a pure, deterministic, config-selectable
NAMED registry: correct colors for known Cityscapes train-ids, unknown ids → 'other' (never raises),
named-registry lookup works, and an unknown palette NAME raises. The palette NAME is the config-visible
handle (``ConditioningConfig.seg_palette_name``, added in 07-02) — this test guards D-NOHARDCODE by
asserting the active palette is selected by name, not an anonymous constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from signet_trainer.data.seg_palette import (
    CITYSCAPES_TO_PALETTE,
    DEFAULT_PALETTE_NAME,
    PALETTE_VERSION,
    PALETTES,
    remap_to_palette,
)

# Documented compact-driving-palette colors (07-RESEARCH lines 96-104).
ROAD = (64, 64, 64)  # #404040
BUILDING = (255, 0, 0)  # #FF0000
VEGETATION = (0, 255, 0)  # #00FF00
SKY = (0, 0, 255)  # #0000FF
PERSON = (255, 255, 0)  # #FFFF00
VEHICLE = (0, 255, 255)  # #00FFFF
OTHER = (0, 0, 0)  # #000000


def test_named_registry_shape() -> None:
    """The palette is exposed as a NAMED registry with a documented default + version (D-7-PALETTE)."""
    assert DEFAULT_PALETTE_NAME == "compact_driving_v1"
    assert DEFAULT_PALETTE_NAME in PALETTES
    assert isinstance(PALETTE_VERSION, str) and PALETTE_VERSION
    # The default table carries exactly the 7 documented palette classes.
    table = PALETTES[DEFAULT_PALETTE_NAME]
    assert set(table) == {"road", "building", "vegetation", "sky", "person", "vehicle", "other"}


def test_documented_palette_colors() -> None:
    """The 7 palette colors match the documented hex values (07-RESEARCH lines 96-104)."""
    table = PALETTES[DEFAULT_PALETTE_NAME]
    assert tuple(table["road"]) == ROAD
    assert tuple(table["building"]) == BUILDING
    assert tuple(table["vegetation"]) == VEGETATION
    assert tuple(table["sky"]) == SKY
    assert tuple(table["person"]) == PERSON
    assert tuple(table["vehicle"]) == VEHICLE
    assert tuple(table["other"]) == OTHER


def test_cityscapes_mapping_covers_19_trainids() -> None:
    """CITYSCAPES_TO_PALETTE maps each of the 19 Cityscapes train-ids to a valid palette class."""
    assert set(CITYSCAPES_TO_PALETTE) == set(range(19))
    valid_classes = set(PALETTES[DEFAULT_PALETTE_NAME])
    assert all(cls in valid_classes for cls in CITYSCAPES_TO_PALETTE.values())


def test_remap_known_ids_to_correct_colors() -> None:
    """A few known Cityscapes train-ids remap to their documented palette color."""
    # 0=road, 2=building, 8=vegetation, 10=sky, 11=person, 13=car(vehicle).
    label_map = np.array([[0, 2, 8], [10, 11, 13]], dtype=np.int64)
    out = remap_to_palette(label_map)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], ROAD)
    np.testing.assert_array_equal(out[0, 1], BUILDING)
    np.testing.assert_array_equal(out[0, 2], VEGETATION)
    np.testing.assert_array_equal(out[1, 0], SKY)
    np.testing.assert_array_equal(out[1, 1], PERSON)
    np.testing.assert_array_equal(out[1, 2], VEHICLE)


def test_unknown_id_maps_to_other_without_raising() -> None:
    """An unknown/background class id maps to the 'other' color and never raises (ignore=255)."""
    label_map = np.array([[255, 99, -1]], dtype=np.int64)
    out = remap_to_palette(label_map)
    for px in out.reshape(-1, 3):
        np.testing.assert_array_equal(px, OTHER)


def test_deterministic() -> None:
    """Same input + name → byte-identical output."""
    label_map = np.random.default_rng(0).integers(0, 19, size=(16, 24), dtype=np.int64)
    a = remap_to_palette(label_map, DEFAULT_PALETTE_NAME)
    b = remap_to_palette(label_map, DEFAULT_PALETTE_NAME)
    np.testing.assert_array_equal(a, b)


def test_unknown_palette_name_raises() -> None:
    """An unknown palette_name raises a clear KeyError (named-registry lookup, D-NOHARDCODE)."""
    with pytest.raises(KeyError):
        remap_to_palette(np.zeros((2, 2), dtype=np.int64), palette_name="does_not_exist")
