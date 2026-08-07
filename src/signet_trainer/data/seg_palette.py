"""data.seg_palette — pure Cityscapes-19 → compact-driving-palette remap (REF-03, D-7-PALETTE).

The genuine net-new preprocessing primitive for the IC-LoRA paired (seg-map-video → video) dataset:
a deterministic, numpy-only lookup that turns a per-frame Cityscapes class-id label map into a
3-channel high-contrast palette RGB frame. The staging script (``scripts/_stage_demo_seg_dataset.py``)
calls ``remap_to_palette`` on each segmenter frame to build the aligned ``<stem>_reference.mp4`` clips.

D-7-PALETTE / D-NOHARDCODE — the palette is exposed as a NAMED registry (``PALETTES``), not an anonymous
module constant. The active palette is selected by NAME via the config-visible handle
``ConditioningConfig.seg_palette_name`` (added in 07-02, defaults to ``"compact_driving_v1"``). Adding a
new palette = adding a named entry to ``PALETTES`` + naming it in config — never editing hardcoded colors.

Cityscapes 19 train-id → ~7 palette-class mapping (07-RESEARCH lines 93-104):

    | palette class | Cityscapes source classes                            | color            |
    |---------------|------------------------------------------------------|------------------|
    | road          | road(0), sidewalk(1)                                 | dark grey #404040|
    | building      | building(2), wall(3), fence(4)                       | red     #FF0000  |
    | vegetation    | vegetation(8), terrain(9)                            | green   #00FF00  |
    | sky           | sky(10)                                              | blue    #0000FF  |
    | person        | person(11), rider(12)                                | yellow  #FFFF00  |
    | vehicle       | car(13), truck(14), bus(15), train(16), motorcycle(17), bicycle(18) | cyan #00FFFF |
    | other         | pole(5), traffic light(6), traffic sign(7), background | black #000000  |

Import confinement: stdlib + numpy ONLY. No heavy ML, video-decode, or Modal-SDK imports — this module
is CPU/CI-cheap and must stay importable without the segmentation stack (the segmenter is loaded
function-locally in the staging script; MODL-04 keeps it off metered compute entirely).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: Bump when the class mapping or default colors change (lets downstream artifacts record provenance).
PALETTE_VERSION = "1.0.0"

#: The config-visible default palette handle (mirrors ``ConditioningConfig.seg_palette_name`` default).
DEFAULT_PALETTE_NAME = "compact_driving_v1"

#: Cityscapes 19 train-id → compact palette class. Standard Cityscapes ``trainId`` ordering (0-18) as
#: emitted by ``facebook/mask2former-swin-large-cityscapes-semantic``. Unlisted ids (e.g. the 255
#: ignore label / background) fall through to 'other' in :func:`remap_to_palette` and never raise.
CITYSCAPES_TO_PALETTE: Dict[int, str] = {
    0: "road",        # road
    1: "road",        # sidewalk
    2: "building",    # building
    3: "building",    # wall
    4: "building",    # fence
    5: "other",       # pole
    6: "other",       # traffic light
    7: "other",       # traffic sign
    8: "vegetation",  # vegetation
    9: "vegetation",  # terrain
    10: "sky",        # sky
    11: "person",     # person
    12: "person",     # rider
    13: "vehicle",    # car
    14: "vehicle",    # truck
    15: "vehicle",    # bus
    16: "vehicle",    # train
    17: "vehicle",    # motorcycle
    18: "vehicle",    # bicycle
}

RGB = Tuple[int, int, int]

#: NAMED palette registry ({name: {palette-class: RGB}}). The active table is chosen by NAME (the
#: config-visible ``seg_palette_name`` handle) — D-7-PALETTE / D-NOHARDCODE. Every table MUST define an
#: 'other' entry (the fallback color for unmapped/background class ids).
PALETTES: Dict[str, Dict[str, RGB]] = {
    "compact_driving_v1": {
        "road": (0x40, 0x40, 0x40),        # #404040 dark grey
        "building": (0xFF, 0x00, 0x00),    # #FF0000 red
        "vegetation": (0x00, 0xFF, 0x00),  # #00FF00 green
        "sky": (0x00, 0x00, 0xFF),         # #0000FF blue
        "person": (0xFF, 0xFF, 0x00),      # #FFFF00 yellow
        "vehicle": (0x00, 0xFF, 0xFF),     # #00FFFF cyan
        "other": (0x00, 0x00, 0x00),       # #000000 black
    },
}


def _build_lut(palette_name: str) -> np.ndarray:
    """Build a (256, 3) uint8 lookup table: class-id → palette RGB, unmapped ids → 'other'.

    Resolves the color table by NAME from :data:`PALETTES` (raises ``KeyError`` on an unknown name —
    the config-selected-not-hardcoded contract). Ids without an explicit Cityscapes mapping default to
    the table's 'other' color, so background / ignore (255) labels are handled without a raise.
    """
    table = PALETTES[palette_name]  # KeyError on unknown name — clear, intentional (D-NOHARDCODE).
    other = np.asarray(table["other"], dtype=np.uint8)
    lut = np.tile(other, (256, 1))  # default every id to 'other'
    for train_id, cls in CITYSCAPES_TO_PALETTE.items():
        lut[train_id] = np.asarray(table[cls], dtype=np.uint8)
    return lut


def remap_to_palette(
    label_map: np.ndarray, palette_name: str = DEFAULT_PALETTE_NAME
) -> np.ndarray:
    """Remap an HxW Cityscapes class-id label map to an HxWx3 uint8 palette RGB frame.

    Deterministic: same ``label_map`` + ``palette_name`` → byte-identical output. Ids outside the
    documented Cityscapes mapping (background, the 255 ignore label, or any out-of-range value) map to
    the palette's 'other' color and never raise. An unknown ``palette_name`` raises ``KeyError``.

    Args:
        label_map: HxW integer array of Cityscapes train-ids (0-18); other values → 'other'.
        palette_name: Which named table in :data:`PALETTES` to use (the config-visible handle).

    Returns:
        HxWx3 ``uint8`` RGB array, one palette color per pixel.
    """
    lm = np.asarray(label_map)
    if lm.ndim != 2:
        raise ValueError(f"label_map must be a 2-D HxW array, got shape {lm.shape!r}")

    lut = _build_lut(palette_name)  # (256, 3); raises on unknown palette_name

    # Fill with 'other' by default, then apply the LUT only where the id is in [0, 256). Negative /
    # out-of-range ids stay 'other' — keeps the remap total (never raises on a stray label value).
    out = np.tile(lut[  # 'other' is the id that no Cityscapes class maps to; but use table explicitly:
        # index-safe 'other' color = LUT row for an id guaranteed unmapped (255 is the ignore label).
        255
    ], (*lm.shape, 1)).astype(np.uint8)
    valid = (lm >= 0) & (lm < lut.shape[0])
    ids = lm[valid].astype(np.intp)
    out[valid] = lut[ids]
    return out
