"""Thin inpaint-staging CLI — wraps ``signet_trainer.prep.stage.stage_inpaint_dataset`` (D-13).

Assembles ``<dest>/{muxed,masks}/`` + ``metadata.jsonl`` from the propagated per-frame mask PNGs and
their source clips: muxes each clip (trimmed to the largest 8n+1, NEVER padded), renders the polarity
mask mp4 via the committed ``data.mask_encode.render_mask_video`` seam, surfaces the ÷64/8n+1
preflight per clip, and emits the precedent-shaped rows the canonical encode consumes.

NO Modal import, NO metered dispatch — pure local file prep (ffprobe + ffmpeg required to stage). The
gated GPU step (canonical preprocess through the single gate) comes AFTER, separately. Nothing here
launches a metered run.

Usage:
  # self-check only (resolves every clip<->mask pairing; runs NO ffmpeg; writes nothing):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_stage.py --dry-run \
      --mask-manifest _inpaint_prep/mask_manifest.json --masks-video _inpaint_prep/masks_video \
      --clips-dir <clips> --dest _staging_campaign_inpaint

  # stage:
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_stage.py \
      --mask-manifest _inpaint_prep/mask_manifest.json --masks-video _inpaint_prep/masks_video \
      --clips-dir <clips> --dest _staging_campaign_inpaint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from signet_trainer import harness_data
from signet_trainer.prep.stage import stage_inpaint_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREP_DIR = REPO_ROOT / "_inpaint_prep"
# PACKAGED spec (signet_trainer.harness_data), resolved via importlib.resources — NOT a
# ".planning/harness/" path built from __file__, which only exists in a source checkout.
DEFAULT_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)
DEFAULT_DEST = REPO_ROOT / "_staging_campaign_inpaint"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mask-manifest", default=str(DEFAULT_PREP_DIR / "mask_manifest.json"),
                    help="JSON {mask_stem: {...}} — one entry per MASK (from the propagation step).")
    ap.add_argument("--stem-manifest", default=str(DEFAULT_PREP_DIR / "manifest.txt"),
                    help="'NN | png | clip' lines mapping mask stems -> source clips.")
    ap.add_argument("--masks-video", default=str(DEFAULT_PREP_DIR / "masks_video"),
                    help="Root of propagated per-frame mask PNGs (<masks-video>/<mask_stem>/*.png).")
    ap.add_argument("--seeds-dir", default=str(DEFAULT_PREP_DIR / "masks_frame0"),
                    help="Per-mask seed PNGs (dims cross-check only).")
    ap.add_argument("--captions", default=str(DEFAULT_PREP_DIR / "inpaint_captions.json"),
                    help="JSON {mask_stem: caption}; missing keys/file -> '[PENDING: operator]'.")
    ap.add_argument("--clips-dir", default=str(DEFAULT_PREP_DIR),
                    help="Directory of source .mp4 clips (exact-stem resolution).")
    ap.add_argument("--spec", default=str(DEFAULT_SPEC),
                    help="MASK-SPEC.yaml — the ÷64/8n+1 dims rules come FROM here (D-03).")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="Staging output directory.")
    ap.add_argument("--force", action="store_true", help="Re-stage rows whose outputs already exist.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Self-check only: resolve every pairing + print the trim/preflight plan; "
                         "runs NO ffmpeg and writes nothing.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    result = stage_inpaint_dataset(
        mask_manifest=args.mask_manifest,
        stem_manifest=args.stem_manifest,
        masks_video_dir=args.masks_video,
        clips_dir=args.clips_dir,
        dest=args.dest,
        spec=args.spec,
        seeds_dir=args.seeds_dir,
        captions=args.captions,
        force=args.force,
        dry_run=args.dry_run,
    )

    n = len(result["rows"])
    if args.dry_run:
        print(f"STAGE_DONE: 0 rows (dry-run — {n} planned, {result['fails']} failing, "
              f"{result['pending']} awaiting propagation; nothing written)")
        return 1 if result["fails"] else 0

    print(f"STAGE_DONE: {n} rows -> {result['manifest']}")
    print("[stage-inpaint] next (GATED GPU): upload this staging dir and run the canonical "
          "preprocess through the single gate (--mode preprocess). This script does NO metered dispatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
