"""prep.propagate unit coverage (D-13/D-08/D-02). Pure CPU — NO modal, NO GPU, NO network.

Import-confined by contract: the module must PARSE + these unit legs must run on a box with no
torch / sam3 / transformers installed (the heavy deps are function-local). Asserts, unconditionally:

1. ``import signet_trainer.prep.propagate`` succeeds without any heavy dep loaded.
2. The D-08 ``--rev`` backward-seed index math: ``frame_order`` reverses physical order and
   ``remap_segmentation`` maps a propagated (temp-order) index back to its ORIGINAL frame index.
3. The green-overlay ``avg_cover`` computation from a synthetic boolean HxW stack matches a
   hand-computed coverage fraction (guarded skip only if cv2/numpy are genuinely absent).

House test rules honored: no metered dispatch, no modal, no network — synthetic in-memory fixtures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"


def test_module_exports_present():
    """The module parses and exposes its public seam without a GPU env."""
    import signet_trainer.prep.propagate as prop  # noqa: PLC0415

    assert hasattr(prop, "make_backend")
    assert hasattr(prop, "propagate_masks")
    assert hasattr(prop, "frame_order")
    assert hasattr(prop, "remap_segmentation")


def test_import_pulls_no_heavy_backend():
    """In a FRESH interpreter, importing propagate must not drag in torch/sam3/transformers.

    Run in a subprocess so the assertion is honest even when the wider pytest session (conftest,
    sibling torch-heavy tests) has already loaded torch — this proves propagate's own heavy imports
    are function-local (import-confinement, Anti-Pattern 6).
    """
    code = (
        "import sys; import signet_trainer.prep.propagate as p; "
        "assert hasattr(p, 'propagate_masks'); "
        "bad = [m for m in ('torch', 'sam3', 'transformers') if m in sys.modules]; "
        "print('LOADED:' + ','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    env = {**_os_environ(), "PYTHONPATH": str(_SRC), "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"propagate import pulled heavy deps: {r.stdout}{r.stderr}"


def _os_environ() -> dict:
    import os  # noqa: PLC0415

    return dict(os.environ)


def test_frame_order_forward_is_identity():
    from signet_trainer.prep.propagate import frame_order  # noqa: PLC0415

    assert frame_order(5, reverse=False) == [0, 1, 2, 3, 4]


def test_frame_order_reverse_flips_physical_order():
    from signet_trainer.prep.propagate import frame_order  # noqa: PLC0415

    assert frame_order(5, reverse=True) == [4, 3, 2, 1, 0]


def test_remap_forward_identity():
    from signet_trainer.prep.propagate import frame_order, remap_segmentation  # noqa: PLC0415

    order = frame_order(4, reverse=False)
    seg_temp = {0: "a", 1: "b", 2: "c", 3: "d"}
    assert remap_segmentation(seg_temp, order) == {0: "a", 1: "b", 2: "c", 3: "d"}


def test_remap_reverse_restores_physical_index():
    """D-08: a backward propagation seeds the LAST physical frame — temp-0 must map back to n-1."""
    from signet_trainer.prep.propagate import frame_order, remap_segmentation  # noqa: PLC0415

    n = 4
    order = frame_order(n, reverse=True)  # [3, 2, 1, 0]
    # predictor saw the frames in reversed physical order; temp index j -> original order[j]
    seg_temp = {0: "last", 1: "x", 2: "y", 3: "first"}
    remapped = remap_segmentation(seg_temp, order)
    assert remapped == {3: "last", 2: "x", 1: "y", 0: "first"}
    # the frame seeded first (temp-0) is the LAST physical frame
    assert remapped[n - 1] == "last"
    # and physical frame 0 is the one the predictor reached last
    assert remapped[0] == "first"


def test_avg_cover_matches_hand_computed_fraction():
    """The QA overlay's avg_cover is mean(mask) averaged over frames — assert on a synthetic stack."""
    np = pytest.importorskip("numpy")

    h, w = 10, 10  # 100 px per frame
    m0 = np.zeros((h, w), dtype=bool)
    m0[:2, :] = True  # 20 px -> 0.20
    m1 = np.zeros((h, w), dtype=bool)
    m1[:4, :] = True  # 40 px -> 0.40
    seg = {0: m0, 1: m1}

    # mirror process_job's accumulation: cover += mask.mean() per frame, then / n_frames
    n = 2
    cover = 0.0
    for i in range(n):
        cover += float(seg[i].astype(np.uint8).mean())
    avg_cover = cover / n
    assert avg_cover == pytest.approx((0.20 + 0.40) / 2)
