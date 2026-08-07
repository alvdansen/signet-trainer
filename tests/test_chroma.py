"""Paintover -> binary teal-seed chroma-extraction tests (D-10). Pure CPU — NO modal, NO GPU, NO network.

Locks the reverse-engineered r1 contract (the extraction script is gone; its outputs pinned the
contract): an external iPad/Procreate teal ``#458070`` paintover becomes a binary WHITE(255) seed the
SAM propagator consumes, and an ``img_match.json``-shaped record (``match``/``paint_pct``/``resid``/
``bbox``/``paint_rgb``) is emitted — including an empty entry (``paint_rgb`` null) when no paint is
found. ALL chroma constants (teal hex/rgb, channel rules, tolerance) are read from ``MASK-SPEC.yaml``
via the passed spec dict — never hardcoded in the extractor (D-03/config-first).

Two always-run assertions (synthetic teal array built in-memory; no binary fixture needed) plus a
skip-when-absent round-trip against a surviving ``_inpaint_prep/masks_frame0/`` seed — which asserts
IoU >= 0.9 in any checkout that still carries the r1 outputs and skips gracefully in bare CI.

House test rules honored: no metered dispatch, no modal CLI, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signet_trainer import harness_data
from signet_trainer.prep.chroma import extract_chroma_mask
from signet_trainer.prep.spec import load_mask_spec

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The PACKAGED spec (signet_trainer.harness_data) — the same resolution production uses.
_REAL_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)
_PREP_DIR = _REPO_ROOT / "_inpaint_prep"


def _spec() -> dict:
    """The committed MASK-SPEC.yaml — the extractor's single source of chroma truth."""
    return load_mask_spec(_REAL_SPEC)


def _teal_bgr(spec: dict) -> np.ndarray:
    """Canonical teal as a BGR pixel, derived from the spec rgb (never a code literal)."""
    r, g, b = spec["rgb"]
    return np.array([b, g, r], dtype=np.uint8)


def _synthetic_paintover(spec: dict, h: int = 64, w: int = 80) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A content-ish BGR canvas with a teal rectangle painted in. Returns (img, (x0, y0, x1, y1))."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark non-teal "content"
    # a reddish patch to prove non-teal content is NOT matched
    img[5:15, 5:20] = np.array([40, 40, 120], dtype=np.uint8)  # BGR reddish
    x0, y0, x1, y1 = 20, 24, 60, 52  # inclusive-max box painted teal
    img[y0 : y1 + 1, x0 : x1 + 1] = _teal_bgr(spec)
    return img, (x0, y0, x1, y1)


def test_teal_region_extracts_to_white_seed() -> None:
    spec = _spec()
    img, (x0, y0, x1, y1) = _synthetic_paintover(spec)
    mask, record = extract_chroma_mask(img, spec)

    assert mask.dtype == np.uint8
    # region = WHITE(255) inside the painted box, 0 outside (polarity T-09.3-03-T)
    assert mask[y0 : y1 + 1, x0 : x1 + 1].min() == 255
    assert mask[0:4, 0:4].max() == 0
    # binary only — no intermediate greys
    assert set(np.unique(mask).tolist()) <= {0, 255}
    # record shape (img_match.json contract)
    assert set(record) == {"match", "paint_pct", "resid", "bbox", "paint_rgb"}
    assert record["paint_pct"] > 0
    assert record["bbox"] == [x0, y0, x1, y1]
    # auto-detected paint color reproduces the canonical teal rgb
    assert record["paint_rgb"] == list(spec["rgb"])


def test_no_paint_yields_null_record() -> None:
    spec = _spec()
    img = np.full((48, 48, 3), 30, dtype=np.uint8)  # no teal anywhere
    mask, record = extract_chroma_mask(img, spec)

    assert mask.max() == 0
    assert record["paint_pct"] == 0.0
    assert record["paint_rgb"] is None
    assert record["bbox"] is None


def test_match_field_passthrough() -> None:
    spec = _spec()
    img, _ = _synthetic_paintover(spec)
    _, record = extract_chroma_mask(img, spec, match="10_014_clip__face")
    assert record["match"] == "10_014_clip__face"


def test_distance_method_branch_extracts_teal() -> None:
    spec = _spec()
    spec = {**spec, "chroma": {**spec["chroma"], "method": spec["chroma"]["alternate_method"]}}
    assert spec["chroma"]["method"] == "distance"
    img, (x0, y0, x1, y1) = _synthetic_paintover(spec)
    mask, record = extract_chroma_mask(img, spec)
    assert mask[y0 : y1 + 1, x0 : x1 + 1].min() == 255
    assert record["paint_pct"] > 0


def test_unknown_method_fails_loud() -> None:
    spec = _spec()
    spec = {**spec, "chroma": {**spec["chroma"], "method": "nonsense"}}
    img, _ = _synthetic_paintover(spec)
    with pytest.raises(ValueError, match="method"):
        extract_chroma_mask(img, spec)


def test_corrupt_input_raises() -> None:
    """T-09.3-03-I: a None/malformed decode must raise, never a silent empty mask."""
    spec = _spec()
    with pytest.raises(ValueError):
        extract_chroma_mask(None, spec)
    with pytest.raises(ValueError):
        extract_chroma_mask(np.zeros((10, 10), dtype=np.uint8), spec)  # not HxWx3


def test_extractor_is_import_confined() -> None:
    import signet_trainer.prep.chroma as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    lines = src.splitlines()
    for banned in ("import cv2", "import numpy", "import torch", "import modal"):
        top_level = [ln for ln in lines if ln.startswith(banned)]
        assert not top_level, f"{banned!r} must be function-local, not module-scope, in prep.chroma"


# --- Real-data round-trip: asserts when the surviving r1 outputs are present, skips in bare CI. ----

_ROUNDTRIP_PAIN = _PREP_DIR / "paintovers" / "Face" / "IMG_4442.PNG"
# The matching frame-0 `__face` seed for clip 02_007 — discovered structurally (clip index +
# mask type), never by a hardcoded source filename; an absent dir/seed just skips the round-trip.
_ROUNDTRIP_SEED = next(
    iter(sorted((_PREP_DIR / "masks_frame0").glob("02_007_*__face.png"))),
    _PREP_DIR / "masks_frame0" / "_absent__face.png",
)


@pytest.mark.skipif(
    not (_ROUNDTRIP_PAIN.is_file() and _ROUNDTRIP_SEED.is_file()),
    reason="surviving _inpaint_prep r1 outputs not present in this checkout",
)
def test_roundtrip_reproduces_committed_seed() -> None:
    cv2 = pytest.importorskip("cv2")
    spec = _spec()
    paintover = cv2.imread(str(_ROUNDTRIP_PAIN))
    seed = cv2.imread(str(_ROUNDTRIP_SEED), cv2.IMREAD_GRAYSCALE)
    assert paintover is not None and seed is not None

    mask, record = extract_chroma_mask(paintover, spec)
    # the r1 seed was resized to the source-clip resolution; align NEAREST before IoU (load_seed L188).
    if mask.shape != seed.shape:
        mask = cv2.resize(mask, (seed.shape[1], seed.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = mask > 127
    b = seed > 127
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    iou = inter / union if union else 0.0
    assert iou >= 0.9, f"round-trip IoU {iou:.4f} < 0.9 (polarity or threshold drift)"
    assert record["paint_rgb"] == list(spec["rgb"])
