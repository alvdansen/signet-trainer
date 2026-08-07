"""Masking-app server-side export contract (D-09). Pure CPU — NO modal, NO GPU, NO network socket.

Locks the binary-by-construction guarantee that lives SERVER-SIDE (never trusting the browser's
anti-aliasing): an RGBA overlay with anti-aliased grey edges (alpha 0..255) is thresholded on write
to a pure binary ``{0, 255}`` PNG (``alpha > 127 -> 255``), region = WHITE(255) per the D-05 polarity
chain. Also locks the two POST-route safety mitigations (T-09.3-06-01 path traversal, T-09.3-06-02
payload cap) and the app-export -> ``read_mask_frames`` -> ``encode_mask_pixels`` round-trip to the
expected latent grid.

The helpers are called DIRECTLY (no live socket needed) — the export threshold, the traversal guard,
and the payload cap are pure functions of the app module. House test rules honored: no metered
dispatch, no modal CLI, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from signet_trainer.data.mask_encode import (
    encode_mask_pixels,
    latent_grid_for_pixels,
    read_mask_frames,
)
from signet_trainer import harness_data
from signet_trainer.prep import app as mask_app
from signet_trainer.prep.spec import load_mask_spec

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The PACKAGED spec (signet_trainer.harness_data) — the same resolution production uses, so this
# test would catch a spec that stopped shipping inside the package.
_REAL_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)


def _spec() -> dict:
    return load_mask_spec(_REAL_SPEC)


# --------------------------------------------------------------------------------------------------
# D-09 — the binary-by-construction threshold (the guarantee lives server-side).
# --------------------------------------------------------------------------------------------------


def test_aa_overlay_thresholds_to_binary_white() -> None:
    """An anti-aliased alpha ramp collapses to pure {0,255}; alpha>127 -> WHITE(255), else 0."""
    rgba = np.zeros((8, 20, 4), dtype=np.uint8)
    rgba[:, :, :3] = 40  # teal-ish content underneath is irrelevant — only alpha decides
    rgba[:, 0:5, 3] = 255   # solid paint            -> WHITE
    rgba[:, 5:10, 3] = 200  # inner AA (still > 127) -> WHITE
    rgba[:, 10:15, 3] = 128  # 128 > 127             -> WHITE
    rgba[:, 15:18, 3] = 127  # 127 is NOT > 127      -> 0
    rgba[:, 18:20, 3] = 30   # outer AA halo         -> 0

    mask = mask_app.threshold_alpha_to_binary(rgba)

    assert mask.dtype == np.uint8
    # binary ONLY — no intermediate greys survive (browser AA is discarded)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    # region = WHITE(255) polarity for every alpha>127 column
    assert mask[:, 0:15].min() == 255
    # the 127-exact and low-alpha halo are 0
    assert mask[:, 15:20].max() == 0


def test_threshold_rejects_non_rgba() -> None:
    with pytest.raises(ValueError):
        mask_app.threshold_alpha_to_binary(np.zeros((4, 4, 3), dtype=np.uint8))  # no alpha channel


# --------------------------------------------------------------------------------------------------
# T-09.3-06-01 — path traversal guard (mirrors the _serve_clip_picker.py f.parent == DIR shape).
# --------------------------------------------------------------------------------------------------


def test_traversal_target_is_rejected(tmp_path: Path) -> None:
    prep = tmp_path / "prep"
    prep.mkdir()
    # a resolved target escaping the prep dir must raise
    with pytest.raises(ValueError):
        mask_app.resolve_under_dir(prep, "../evil.png")
    with pytest.raises(ValueError):
        mask_app.resolve_under_dir(prep, "masks_frame0/../../etc/passwd.png")
    # a stem carrying separators / .. is rejected before it reaches the filesystem
    with pytest.raises(ValueError):
        mask_app.export_target_rel("../evil", "frame0", None)
    with pytest.raises(ValueError):
        mask_app.export_target_rel("a/b", "frame0", None)
    # a legitimate target resolves UNDER the prep dir
    good = mask_app.resolve_under_dir(prep, "masks_frame0/clip__face.png")
    assert prep.resolve() in good.resolve().parents


# --------------------------------------------------------------------------------------------------
# T-09.3-06-02 — POST payload cap (reject oversized bodies before buffering).
# --------------------------------------------------------------------------------------------------


def test_payload_cap_rejects_oversized() -> None:
    cap = 1024
    mask_app.check_payload_size(cap, cap)  # exactly at cap is allowed
    mask_app.check_payload_size(cap - 1, cap)
    with pytest.raises(ValueError):
        mask_app.check_payload_size(cap + 1, cap)


# --------------------------------------------------------------------------------------------------
# Export writes a binary WHITE seed in the D-05 seed/per-frame layouts.
# --------------------------------------------------------------------------------------------------


def _white_overlay(h: int, w: int) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # fully painted -> the whole frame is the region
    return rgba


def test_export_writes_binary_white_seed(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    prep = tmp_path / "prep"
    prep.mkdir()
    target, mask = mask_app.write_binary_mask(prep, "10_clip__face", _white_overlay(32, 48), kind="frame0")

    assert target == (prep / "masks_frame0" / "10_clip__face.png")
    assert target.is_file()
    disk = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(disk).tolist()) <= {0, 255}
    assert disk.min() == 255  # region = WHITE(255) polarity


def test_export_per_frame_video_layout(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    prep = tmp_path / "prep"
    prep.mkdir()
    target, _ = mask_app.write_binary_mask(prep, "10_clip__face", _white_overlay(16, 16), kind="video", frame=5)
    assert target == (prep / "masks_video" / "10_clip__face" / "00005.png")
    assert target.is_file()


# --------------------------------------------------------------------------------------------------
# D-10 — the paintover import routes through the rebuilt chroma extractor (Plan 03).
# --------------------------------------------------------------------------------------------------


def test_paintover_import_reuses_chroma(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    spec = _spec()
    r, g, b = spec["rgb"]
    img = np.full((40, 60, 3), 30, dtype=np.uint8)  # dark non-teal content (BGR)
    img[10:30, 15:45] = np.array([b, g, r], dtype=np.uint8)  # teal box, BGR order
    ok, buf = cv2.imencode(".png", img)
    assert ok
    prep = tmp_path / "prep"
    prep.mkdir()
    target, mask, record = mask_app.import_paintover(prep, "07_clip__face_hair", buf.tobytes(), spec)

    assert target == (prep / "masks_frame0" / "07_clip__face_hair.png")
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[10:30, 15:45].min() == 255  # teal region -> WHITE seed
    assert record["paint_pct"] > 0


# --------------------------------------------------------------------------------------------------
# Round-trip — an app-exported seed feeds read_mask_frames -> encode_mask_pixels to the latent grid.
# --------------------------------------------------------------------------------------------------


def test_app_export_encode_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    prep = tmp_path / "prep"
    prep.mkdir()
    h, w = 64, 64
    target, _ = mask_app.write_binary_mask(prep, "clip__face", _white_overlay(h, w), kind="frame0")

    f_lat, h_lat, w_lat = latent_grid_for_pixels(9, h, w)  # (2, 2, 2)
    pixel_f = (f_lat - 1) * 8 + 1  # 9
    mask_pixels = read_mask_frames(target, expected_frames=pixel_f)  # tile the seed across frames
    assert mask_pixels.shape == (pixel_f, h, w)

    encoded = encode_mask_pixels(mask_pixels, f_lat, h_lat, w_lat)
    assert encoded.shape == (f_lat, h_lat, w_lat)
    assert encoded.dtype == torch.float32
    assert set(torch.unique(encoded).tolist()) <= {0.0, 1.0}


# --------------------------------------------------------------------------------------------------
# Import confinement (Anti-Pattern 6): numpy/cv2 must be function-local, never module-scope.
# --------------------------------------------------------------------------------------------------


def test_app_is_import_confined() -> None:
    src = Path(mask_app.__file__).read_text(encoding="utf-8")
    lines = src.splitlines()
    for banned in ("import cv2", "import numpy", "import torch", "import modal"):
        top_level = [ln for ln in lines if ln.startswith(banned)]
        assert not top_level, f"{banned!r} must be function-local, not module-scope, in prep.app"


def test_bind_is_loopback_only() -> None:
    src = Path(mask_app.__file__).read_text(encoding="utf-8")
    assert "0.0.0.0" not in src
    assert "127.0.0.1" in src
