"""Thin CLI: external teal paintover directory -> binary WHITE(255) frame-0 seeds + img_match.json.

Rebuilt r1 step (D-10): wraps ``signet_trainer.prep.chroma.extract_chroma_mask`` (the spec-aware
paintover->seed extractor) with argparse + I/O only — NO new extraction logic. Each iPad/Procreate
paintover (teal ``#458070`` painted over the region to regenerate on frame 0) becomes a binary seed
named ``masks_frame0/<NN_clipstem>__<type>.png`` (region=WHITE(255), the layout
``_sam_propagate_masks.py`` consumes as seeds — D-05/D-09), plus an ``img_match.json``-shaped record
file for QA. App-drawn masks (Plan 06) are binary-by-construction and skip this path.

The (clip_part, type) of each seed is read from the paintover layout, NOT re-derived by image
matching (that heavy img->clip matcher is out of scope — this CLI wires the chroma step only):
  * ``<clipstem>__<type>.png``  (flat, the mask-stem convention)  -> split on the last ``__``
  * ``<type-subdir>/<clipstem>.png``  (r1's paintovers/ layout)   -> parent folder names the type
The ``type`` taxonomy is {face, face_hair, full_body}; folder spellings normalize to it.

ALL chroma constants (teal hex/rgb, channel rules, tolerance) come from ``MASK-SPEC.yaml`` via the
loaded spec dict — never hardcoded (D-03/config-first).

NO Modal import, NO metered dispatch — local/CPU only. Never deletes outputs (existing seeds are
skipped unless ``--force``). Prints ``CHROMA_EXTRACT_DONE: <n> seeds`` on completion.

Usage:
  python scripts/prep_inpaint_chroma.py --dry-run                       # resolve + plan, no writes
  python scripts/prep_inpaint_chroma.py                                  # extract with committed spec
  python scripts/prep_inpaint_chroma.py --paintovers-dir _inpaint_prep/paintovers --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = REPO_ROOT / "_inpaint_prep"

sys.path.insert(0, str(REPO_ROOT / "src"))

from signet_trainer import harness_data  # noqa: E402  (after bootstrap)
from signet_trainer.prep.resolve import sanitize_stem, split_mask_stem  # noqa: E402  (after bootstrap)
from signet_trainer.prep.spec import load_mask_spec  # noqa: E402

DEFAULT_PAINTOVERS_DIR = PREP_DIR / "paintovers"
DEFAULT_MASKS_OUT = PREP_DIR / "masks_frame0"
DEFAULT_RECORD_OUT = PREP_DIR / "img_match.json"
# PACKAGED spec (signet_trainer.harness_data), resolved via importlib.resources — NOT a
# ".planning/harness/" path built from __file__, which only exists in a source checkout.
DEFAULT_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)

# canonical mask-type taxonomy (D-02 per-class coverage band); folder spellings normalize here
_VALID_TYPES = ("face", "face_hair", "full_body")
_FOLDER_TO_TYPE = {
    "face": "face",
    "face_and_hair": "face_hair",
    "face_hair": "face_hair",
    "facehair": "face_hair",
    "full_body": "full_body",
    "fullbody": "full_body",
}


def _need_cv2():
    """Import cv2 lazily with a clear hint (the script must PARSE + --help without it)."""
    try:
        import cv2  # noqa: PLC0415  (deliberate function-local import — script parses without cv2)

        return cv2
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"[prep-chroma] missing dependency 'cv2': {exc}\n    install hint: pip install opencv-python"
        ) from exc


def _resolve_type(paintover: Path, paintovers_dir: Path) -> tuple[str, str] | None:
    """(clip_part, type) from a paintover path. None when the type cannot be named (fail loud upstream)."""
    stem = paintover.stem
    if "__" in stem:
        clip_part, mask_type = split_mask_stem(stem)
        return (clip_part, mask_type) if mask_type in _VALID_TYPES else None
    # else the immediate parent folder names the type
    folder = sanitize_stem(paintover.parent.name).lower()
    mask_type = _FOLDER_TO_TYPE.get(folder)
    if mask_type is None:
        return None
    return sanitize_stem(stem), mask_type


def _discover(paintovers_dir: Path) -> list[Path]:
    """All PNG paintovers under the dir (recursive), case-insensitive extension, sorted."""
    seen: dict[str, Path] = {}
    for p in paintovers_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".png":
            seen[str(p).lower()] = p
    return sorted(seen.values())


def dry_run(paintovers: list[Path], args) -> int:
    """Resolve every paintover -> (clip_part, type, output seed) WITHOUT extraction or writes."""
    masks_out = Path(args.masks_out)
    print(f"[prep-chroma] DRY-RUN: {len(paintovers)} paintover(s) under {args.paintovers_dir}")
    bad = 0
    for p in paintovers:
        rel = p.relative_to(args.paintovers_dir)
        resolved = _resolve_type(p, Path(args.paintovers_dir))
        if resolved is None:
            print(f"  FAIL {rel}: cannot name (type) — expected '<stem>__<type>.png' or a type-subdir")
            bad += 1
            continue
        clip_part, mask_type = resolved
        out = masks_out / f"{clip_part}__{mask_type}.png"
        exists = " (exists — skipped unless --force)" if out.exists() else ""
        print(f"  ok   {rel}: -> {out.name}{exists}")
    print(f"[prep-chroma] DRY-RUN done: {len(paintovers) - bad} ok, {bad} problem(s). No spec loaded, no writes.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paintovers-dir", default=str(DEFAULT_PAINTOVERS_DIR),
                    help="Dir of external teal paintover PNGs (flat '<stem>__<type>.png' or type-subdirs).")
    ap.add_argument("--masks-out", default=str(DEFAULT_MASKS_OUT),
                    help="Output dir for binary WHITE(255) frame-0 seed PNGs (masks_frame0/).")
    ap.add_argument("--record-out", default=str(DEFAULT_RECORD_OUT),
                    help="Output path for the img_match.json-shaped QA record.")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC),
                    help="MASK-SPEC.yaml path (source of teal hex + chroma tolerance/channel rules).")
    ap.add_argument("--force", action="store_true", help="Re-extract even if a seed PNG already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve inputs/outputs + naming only; no spec load, no extraction, no writes.")
    args = ap.parse_args()

    paintovers_dir = Path(args.paintovers_dir)
    if not paintovers_dir.is_dir():
        raise SystemExit(f"[prep-chroma] paintovers dir not found: {paintovers_dir}")
    paintovers = _discover(paintovers_dir)
    if not paintovers:
        raise SystemExit(f"[prep-chroma] no PNG paintovers under {paintovers_dir}")

    if args.dry_run:
        return dry_run(paintovers, args)

    # real run — heavy deps + spec load happen only past the dry-run gate
    from signet_trainer.prep.chroma import extract_chroma_mask  # noqa: PLC0415  (import-confined)

    cv2 = _need_cv2()
    spec = load_mask_spec(args.spec)
    masks_out = Path(args.masks_out)
    masks_out.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    written = 0
    for p in paintovers:
        rel = p.relative_to(paintovers_dir).as_posix()
        resolved = _resolve_type(p, paintovers_dir)
        if resolved is None:
            print(f"[prep-chroma] SKIP {rel}: cannot name (type)")
            continue
        clip_part, mask_type = resolved
        out = masks_out / f"{clip_part}__{mask_type}.png"

        img = cv2.imread(str(p))  # BGR; extract_chroma_mask fails loud on a None/corrupt decode
        mask, record = extract_chroma_mask(img, spec, match=clip_part)
        record["type"] = mask_type
        records[rel] = record

        if out.exists() and not args.force:
            print(f"[prep-chroma] {rel}: record only (seed exists — --force to overwrite)")
        else:
            cv2.imwrite(str(out), mask)  # region = WHITE(255)
            written += 1
            print(f"[prep-chroma] {rel}: -> {out.name} paint_pct={record['paint_pct']} "
                  f"resid={record['resid']} paint_rgb={record['paint_rgb']}")

    record_out = Path(args.record_out)
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"[prep-chroma] record -> {record_out}")
    print(f"CHROMA_EXTRACT_DONE: {written} seeds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
