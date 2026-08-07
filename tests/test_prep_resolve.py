"""prep.resolve tests (stem-resolution dedup source). Pure CPU — NO modal, NO GPU, NO network.

Locks the single-sourced stem-resolution primitives lifted verbatim from
``_sam_propagate_masks.py`` L96-140 (byte-dup of ``_stage_campaign_inpaint_dataset.py`` L64-90): the
non-alnum-collapse sanitizer, the ``__<type>`` splitter, and the three-tier clip resolver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signet_trainer.prep.resolve import resolve_clip, sanitize_stem, split_mask_stem


def test_sanitize_stem_collapses_runs_and_strips_edges() -> None:
    assert sanitize_stem("A b--c!!d") == "A_b_c_d"
    assert sanitize_stem("__lead and trail__") == "lead_and_trail"
    assert sanitize_stem("clip (2024) scene#1") == "clip_2024_scene_1"


def test_split_mask_stem_splits_on_last_dunder() -> None:
    assert split_mask_stem("clip__face") == ("clip", "face")
    assert split_mask_stem("01_my__clip__full_body") == ("01_my__clip", "full_body")


def test_split_mask_stem_without_dunder_raises() -> None:
    with pytest.raises(ValueError):
        split_mask_stem("nounderscoretype")


def test_resolve_clip_direct_match(tmp_path: Path) -> None:
    (tmp_path / "clipA.mp4").write_bytes(b"x")
    assert resolve_clip("clipA", {}, tmp_path) == tmp_path / "clipA.mp4"


def test_resolve_clip_manifest_mapped(tmp_path: Path) -> None:
    src = tmp_path / "real source.mp4"
    src.write_bytes(b"x")
    manifest_map = {"01_stem": src}
    assert resolve_clip("01_stem", manifest_map, tmp_path) == src


def test_resolve_clip_nn_prefix_stripped_sanitize_match(tmp_path: Path) -> None:
    clip = tmp_path / "my clip.mp4"
    clip.write_bytes(b"x")
    # mask stem carries the NN_ photo-index prefix + the sanitized clip name
    assert resolve_clip("07_my_clip", {}, tmp_path) == clip


def test_resolve_clip_unresolvable_returns_none(tmp_path: Path) -> None:
    assert resolve_clip("99_nope", {}, tmp_path) is None


def test_resolve_is_import_confined() -> None:
    import signet_trainer.prep.resolve as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "import cv2", "import modal", "from torch ", "from modal "):
        assert banned not in src, f"{banned!r} leaked into prep.resolve (must stay stdlib only)"
