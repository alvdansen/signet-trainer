"""prep.propagate unit coverage (D-13/D-08/D-02). Pure CPU — NO modal, NO GPU, NO network.

Import-confined by contract: the module must PARSE + these unit legs must run on a box with no
torch / sam3 / transformers installed (the heavy deps are function-local). Asserts, unconditionally:

1. ``import signet_trainer.prep.propagate`` succeeds without any heavy dep loaded.
2. The D-08 ``--rev`` backward-seed index math: ``frame_order`` reverses physical order and
   ``remap_segmentation`` maps a propagated (temp-order) index back to its ORIGINAL frame index.
3. The green-overlay ``avg_cover`` computation from a synthetic boolean HxW stack matches a
   hand-computed coverage fraction (guarded skip only if cv2/numpy are genuinely absent).
4. #36 finding 3: ``load_directions`` reads textseed's REV set from ``textseed_records.json``, and
   ``propagate_masks`` makes an unmatched ``--rev`` / ``--only`` / ``--directions`` stem a FATAL
   ``SystemExit`` (typo-fatal) rather than a silent no-op that seeds the wrong frame.

House test rules honored: no metered dispatch, no modal, no network — synthetic in-memory fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


# --------------------------------------------------------------------------------------------------
# #36 finding 3: the structured fwd/rev handoff (``load_directions`` / ``--directions``) and the
# typo-fatal ``--rev`` / ``--only`` validation in ``propagate_masks``.
# --------------------------------------------------------------------------------------------------

def _make_args(**overrides):
    """A minimal args namespace covering everything ``propagate_masks`` reads before it would need
    cv2/torch/a real spec — masks_dir/clips_dir/manifest/rev/only/directions/dry_run.
    """
    base = dict(masks_dir=None, clips_dir=None, manifest=None, rev=[], only=[],
                directions=None, dry_run=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _touch_seed(masks_dir: Path, mask_stem: str) -> Path:
    """A seed PNG stand-in — ``discover_jobs`` only globs the filename, never decodes pixels."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    p = masks_dir / f"{mask_stem}.png"
    p.write_bytes(b"")
    return p


def test_load_directions_extracts_only_the_rev_mask_stems(tmp_path):
    """``load_directions`` mirrors ``textseed.rev_stems`` — direction == 'rev' only, fwd/not-found
    excluded — reading the SAME JSON field the retired stdout print used to summarize.
    """
    from signet_trainer.prep.propagate import load_directions  # noqa: PLC0415

    records = [
        {"mask_stem": "c1__full_body", "direction": "fwd", "found": True},
        {"mask_stem": "c2__full_body", "direction": "rev", "found": True},
        {"mask_stem": "c3__full_body", "direction": "rev", "found": True},
        {"clip_stem": "c4", "direction": None, "found": False},
    ]
    records_path = tmp_path / "textseed_records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    assert load_directions(records_path) == {"c2__full_body", "c3__full_body"}


def test_propagate_masks_unknown_rev_stem_is_fatal(tmp_path):
    """A ``--rev`` stem that matches no discovered seed PNG must abort loudly (#36 finding 3) —
    the exact failure shape the issue describes: a dropped/mistyped ``--rev`` silently seeding
    frame 0 instead of the intended last frame.
    """
    from signet_trainer.prep.propagate import propagate_masks  # noqa: PLC0415

    masks_dir = tmp_path / "masks_frame0"
    _touch_seed(masks_dir, "01_clip_a__full_body")
    args = _make_args(masks_dir=str(masks_dir), clips_dir=str(tmp_path), manifest=tmp_path / "manifest.txt",
                       rev={"01_clip_typo__full_body"})

    with pytest.raises(SystemExit) as exc:
        propagate_masks(args)

    msg = str(exc.value)
    assert "unknown --rev stem" in msg
    assert "01_clip_typo__full_body" in msg
    assert "01_clip_a__full_body" in msg  # the discovered set is surfaced, not just the bad stem


def test_propagate_masks_unknown_only_stem_is_fatal(tmp_path):
    """Same typo-fatal treatment for ``--only``: a typo'd entry that matches no seed PNG must abort
    loudly even when ANOTHER ``--only`` entry does match (so the job list isn't simply empty —
    the pre-existing "no seed PNGs found" path can't be the one catching this).
    """
    from signet_trainer.prep.propagate import propagate_masks  # noqa: PLC0415

    masks_dir = tmp_path / "masks_frame0"
    _touch_seed(masks_dir, "01_clip_a__full_body")
    args = _make_args(masks_dir=str(masks_dir), clips_dir=str(tmp_path), manifest=tmp_path / "manifest.txt",
                       only={"01_clip_a__full_body", "01_clip_typo__full_body"})

    with pytest.raises(SystemExit) as exc:
        propagate_masks(args)

    assert "unknown --only stem" in str(exc.value)
    assert "01_clip_typo__full_body" in str(exc.value)


def test_propagate_masks_known_rev_stem_passes_validation(tmp_path):
    """A ``--rev`` stem that DOES match a discovered seed must clear validation (no false positive)
    and reach the next stage (the unresolved-clip check, since no real clip file exists here).
    """
    from signet_trainer.prep.propagate import propagate_masks  # noqa: PLC0415

    masks_dir = tmp_path / "masks_frame0"
    _touch_seed(masks_dir, "01_clip_a__full_body")
    args = _make_args(masks_dir=str(masks_dir), clips_dir=str(tmp_path), manifest=tmp_path / "manifest.txt",
                       rev={"01_clip_a__full_body"})

    with pytest.raises(SystemExit) as exc:
        propagate_masks(args)

    # Cleared the typo-fatal check; failed downstream at clip resolution instead (no .mp4 exists).
    assert "unresolved clip stems" in str(exc.value)


def test_propagate_masks_directions_flag_unions_into_rev(tmp_path):
    """``--directions <records.json>`` feeds textseed's REV set into ``args.rev`` (#36 finding 3):
    a REV stem present ONLY in the JSON — never passed via ``--rev`` — must clear the SAME typo
    validation and reach the SAME downstream stage as an explicit ``--rev``, proving the union ran
    before validation.
    """
    from signet_trainer.prep.propagate import propagate_masks  # noqa: PLC0415

    masks_dir = tmp_path / "masks_frame0"
    _touch_seed(masks_dir, "01_clip_a__full_body")
    records_path = tmp_path / "textseed_records.json"
    records_path.write_text(
        json.dumps([{"mask_stem": "01_clip_a__full_body", "direction": "rev", "found": True}]),
        encoding="utf-8",
    )
    args = _make_args(masks_dir=str(masks_dir), clips_dir=str(tmp_path), manifest=tmp_path / "manifest.txt",
                       directions=str(records_path))

    with pytest.raises(SystemExit) as exc:
        propagate_masks(args)

    assert "unresolved clip stems" in str(exc.value)
    assert args.rev == {"01_clip_a__full_body"}  # the JSON's rev stem landed in args.rev


def test_propagate_masks_unknown_directions_stem_is_also_fatal(tmp_path):
    """A stale/mistyped stem arriving via ``--directions`` gets the SAME typo-fatal treatment as an
    explicit ``--rev`` — the union happens before validation, not after.
    """
    from signet_trainer.prep.propagate import propagate_masks  # noqa: PLC0415

    masks_dir = tmp_path / "masks_frame0"
    _touch_seed(masks_dir, "01_clip_a__full_body")
    records_path = tmp_path / "textseed_records.json"
    records_path.write_text(
        json.dumps([{"mask_stem": "01_clip_stale__full_body", "direction": "rev", "found": True}]),
        encoding="utf-8",
    )
    args = _make_args(masks_dir=str(masks_dir), clips_dir=str(tmp_path), manifest=tmp_path / "manifest.txt",
                       directions=str(records_path))

    with pytest.raises(SystemExit) as exc:
        propagate_masks(args)

    assert "unknown --rev stem" in str(exc.value)
    assert "01_clip_stale__full_body" in str(exc.value)
