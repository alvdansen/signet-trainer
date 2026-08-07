"""Stem-resolution primitives — the SINGLE source (D-13 dedup target).

Lifted verbatim from ``scripts/_sam_propagate_masks.py`` L96-140, which was byte-identical to the
copy in ``scripts/_stage_campaign_inpaint_dataset.py`` L64-90. The r1 divergent-copy drift is the
failure mode this module closes: both scripts become thin CLIs importing these once.

Import-confined: stdlib ``re`` + ``pathlib`` only — no torch/cv2/modal — so the module PARSES and
unit-tests run free on Windows/CI.
"""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_stem(stem: str) -> str:
    """Collapse every non-alphanumeric run to '_' (matches the first-frame extraction naming)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def split_mask_stem(mask_stem: str) -> tuple[str, str]:
    """'NN_clipstem__type' -> ('NN_clipstem', 'type'). Splits on the LAST '__'."""
    if "__" not in mask_stem:
        raise ValueError(f"mask stem has no '__<type>' suffix: {mask_stem}")
    clip_part, mask_type = mask_stem.rsplit("__", 1)
    return clip_part, mask_type


def load_manifest_map(manifest: Path) -> dict[str, Path]:
    """Parse ``_inpaint_prep/manifest.txt`` -> {'NN_<sanitized clipstem>': <source clip Path>}."""
    mapping: dict[str, Path] = {}
    if not manifest.exists():
        return mapping
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        png_stem = Path(parts[1]).stem  # 'NN_<sanitized clipstem>'
        mapping[png_stem] = Path(parts[2])
    return mapping


def resolve_clip(clip_part: str, manifest_map: dict[str, Path], clips_dir: Path) -> Path | None:
    """Resolve the clip half of a mask stem to a real clip file.

    Order: (1) exact '<clip_part>.mp4' in clips_dir (test-video case, stems match verbatim);
    (2) manifest.txt mapping (authoritative for the campaign corpus — handles the unicode/paren
    filenames); (3) sanitize-match after stripping the 'NN_' photo-index prefix. Exact full-stem
    equality everywhere — the '..._scene001_scene001' re-export dupe can never match.
    """
    direct = clips_dir / f"{clip_part}.mp4"
    if direct.exists():
        return direct
    if clip_part in manifest_map and manifest_map[clip_part].exists():
        return manifest_map[clip_part]
    bare = re.sub(r"^\d{2}_", "", clip_part)  # strip the NN_ photo-index prefix
    for clip in sorted(clips_dir.glob("*.mp4")):
        if sanitize_stem(clip.stem) == bare:
            return clip
    return None
