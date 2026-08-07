"""MASK-SPEC.yaml loader (D-03). Fail-LOUD, import-confined (stdlib + yaml only).

The mask spec is house law: its absence is never "no constraint" (raises ``FileNotFoundError``),
and a half-written spec raises ``ValueError`` naming the file + the missing keys — a partial spec
must never render a plausible-looking convention value (mirrors
``harness_state.load_house_spec`` / ``load_tier_taxonomy`` doctrine). This is the single loader the
prep package's chroma / propagate / stage / preflight modules and the masking app read, so hex,
chroma tolerance, thresholds, polarity, coverage bands, and dims rules live in ONE typed source and
are never hardcoded in code.

Import-confined (Anti-Pattern 6): stdlib + ``yaml`` only — NO torch, NO cv2, NO modal — so the
module PARSES and unit-tests run free on Windows/CI without a GPU env.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Every top-level convention block the spec MUST carry. A half-written spec missing any of these
# must abort loudly rather than proceed with a wrong/partial convention.
_REQUIRED_KEYS = ("hex", "chroma", "thresholds", "polarity", "coverage_bands", "dims")

# Hair-heavy subject classes MUST carry the D-02 auto-extend dilation config: exact-SAM masks
# (~4.8% coverage) HURT likeness on these; extended (~14%) won (r1 finding). A hair-heavy band
# present WITHOUT a dilation sub-block is a half-written spec, never "no auto-extend".
_HAIR_HEAVY_CLASSES = ("face_hair", "full_body")


def load_mask_spec(path: str | Path) -> dict:
    """Load + shape-check the typed mask spec (``MASK-SPEC.yaml``). Fails LOUD.

    Raises ``FileNotFoundError`` on absence (the spec is house law, not "no constraint") and
    ``ValueError`` naming the file when: the top level is not a mapping; any of
    ``{hex, chroma, thresholds, polarity, coverage_bands, dims}`` is missing; a coverage band is not
    a mapping carrying a ``target``; or a hair-heavy band omits its ``dilation`` auto-extend config.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mask spec file not found: {path} (the mask spec is house law)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"mask spec {path}: top level must be a mapping, got {type(data).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"mask spec {path}: missing required keys {missing}; a half-written mask spec must "
            f"never render a plausible convention block"
        )

    bands = data["coverage_bands"]
    if not isinstance(bands, dict) or not bands:
        raise ValueError(
            f"mask spec {path}: 'coverage_bands' must be a non-empty mapping of subject-class bands"
        )
    for name, band in bands.items():
        if not isinstance(band, dict) or "target" not in band:
            raise ValueError(
                f"mask spec {path}: coverage band {name!r} must be a mapping carrying a 'target' "
                f"coverage fraction"
            )
    for cls in _HAIR_HEAVY_CLASSES:
        band = bands.get(cls)
        if isinstance(band, dict) and "dilation" not in band:
            raise ValueError(
                f"mask spec {path}: hair-heavy coverage band {cls!r} is missing its 'dilation' "
                f"auto-extend config (D-02: exact-SAM masks HURT likeness; the per-class dilation "
                f"margin + target is required, never hardcoded in code)"
            )
    return data
