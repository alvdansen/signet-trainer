"""Thin inpaint-propagation CLI — wraps ``signet_trainer.prep.propagate`` (D-13/D-14/D-08/D-02).

Seeds each frame-0 binary mask (``masks_frame0/<NN_clipstem>__<type>.png``, region=WHITE(255)) into
SAM and propagates it across the matching source clip, then DILATES the exact-SAM masks per subject
class toward the spec's coverage target (D-02 auto-extend) BEFORE writing the per-frame WHITE PNGs,
the green-overlay QA mp4, and the ``avg_cover`` print — so the QA gate judges the EXTENDED mask.

Backend default is ``sam3`` (auto-resolves the Plan-02 install: standalone facebookresearch/sam3 on
Linux, else the transformers native tracker-video path); ``sam21`` is the last-ditch fallback (D-14).
When SAM3 is unavailable this surfaces the D-14 FIX-IT-FIRST path (HF gated access -> install
transformers -> weights auto-pull) — NOT an immediate SAM2.1 fallback.

``--rev <mask_stem>`` re-seeds a drifted clip on its LAST frame and propagates backward (D-08).
``--directions <textseed_records.json>`` reads textseed's fwd/rev decision straight from its JSON
output and unions the REV stems into ``--rev`` — the structured handoff (#36 finding 3); textseed's
stdout print of the same list is a convenience only, never the transport.

NO Modal import, NO metered dispatch — local/CPU-or-local-GPU only. The gated GPU staging + canonical
preprocess come AFTER, separately. Nothing here launches a metered run.

Usage:
  # self-check only (resolves clips + seed dims; loads NO SAM model; writes nothing):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py --dry-run

  # propagate (default sam3, real GPU box):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py --device cuda

  # re-seed a drifted clip backward (manual override):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py \
      --rev 07_012_source_film_part_11_scene003__face_hair

  # propagate using textseed's decided fwd/rev set (structured handoff, no copy-paste):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py \
      --directions _inpaint_prep/textseed_records.json --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from signet_trainer.prep import propagate as prop  # noqa: E402  (after path bootstrap)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["sam3", "sam21"], default="sam3",
                    help="sam3 = SAM 3 (DEFAULT; auto-resolves standalone->transformers per Plan 02). "
                         "sam21 = SAM 2.1 last-ditch fallback (needs --checkpoint).")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint path. REQUIRED for sam21 (sam2.1_hiera_large.pt); optional for "
                         "sam3 (default: HF auto-download, gated).")
    ap.add_argument("--model-cfg", default=prop.SAM21_CFG_DEFAULT,
                    help=f"sam21 model config (default {prop.SAM21_CFG_DEFAULT}).")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--storage-device", default="cpu",
                    help="transformers session video_storage_device (default cpu — offload for big clips).")
    ap.add_argument("--state-device", default="cpu",
                    help="transformers session inference_state_device (default cpu — offload for big clips).")
    ap.add_argument("--masks-dir", default=str(prop.DEFAULT_MASKS_DIR), help="Dir of frame-0 binary seed PNGs.")
    ap.add_argument("--clips-dir", default=str(prop.DEFAULT_CLIPS_DIR), help="Dir of source clips.")
    ap.add_argument("--manifest", type=Path, default=prop.DEFAULT_MANIFEST,
                    help="NN|png|clip manifest for stem resolution (optional; sanitize fallback otherwise).")
    ap.add_argument("--masks-out", default=str(prop.DEFAULT_MASKS_OUT), help="Per-frame mask PNG output root.")
    ap.add_argument("--overlays-out", default=str(prop.DEFAULT_OVERLAYS_OUT), help="QA overlay mp4 output dir.")
    ap.add_argument("--spec", default=str(prop.default_spec()),
                    help="MASK-SPEC.yaml — seed threshold + per-class D-02 dilation targets come FROM here (D-03).")
    ap.add_argument("--rev", action="append", default=[],
                    help="Mask stem(s) to seed on the LAST frame + propagate backward (D-08, repeatable). "
                         "Manual override/re-seed path — for the textseed-decided set, use --directions.")
    ap.add_argument("--directions", default=None,
                    help="textseed_records.json — REV mask stems are read from its 'direction' field "
                         "(the structured fwd/rev handoff, #36 finding 3). Unioned with --rev; any stem "
                         "here that matches no discovered seed PNG is a fatal error, not a silent no-op.")
    ap.add_argument("--only", action="append", default=[], help="Process only these mask stem(s) (repeatable).")
    ap.add_argument("--force", action="store_true", help="Re-propagate even if output PNGs already exist.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve stems + verify seeds/clips; loads NO SAM model, writes nothing.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return prop.propagate_masks(args)


if __name__ == "__main__":
    sys.exit(main())
