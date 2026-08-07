"""Thin held-out test-video CLI — wraps ``signet_trainer.prep.testvideos`` (D-13, GATE-SPEC-inpaint).

Pulls the held-out grid renders (``<grid-dir>/<prompt>/step_<N>.mp4``), copies them with clean stems,
runs the ÷64/8n+1 inpaint dims preflight (single-sourced through ``prep.preflight`` reading
``MASK-SPEC.yaml``, D-13), detects the frame-0 face (mediapipe -> haar crop-confirm), and writes a
binary WHITE(255) frame-0 seed + a QA overlay per video, plus ``testvideos_manifest.json``.

The seeds are then ready for the SAME propagate step:
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py \
      --masks-dir _inpaint_prep/testvideos_seeds --clips-dir _inpaint_prep/testvideos \
      --masks-out _inpaint_prep/testvideos_masks_video --overlays-out _inpaint_prep/testvideos_overlays

NO Modal import, NO metered dispatch. cv2 required for a real run; mediapipe optional (better face
boxes). Never deletes anything; --force re-writes existing outputs.

Usage:
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_testvideos.py              # copy + probe + seed all 3
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_testvideos.py --dry-run    # probe + report only
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_testvideos.py --step 3000 --prompt <extra_prompt_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from signet_trainer import harness_data  # noqa: E402  (after path bootstrap)
from signet_trainer.prep import testvideos as tv  # noqa: E402  (after path bootstrap)

PREP_DIR = REPO_ROOT / "_inpaint_prep"
DEFAULT_GRID_DIR = REPO_ROOT / "_grid_campaign_r5"
DEFAULT_STEP = 3000
# The three held-out prompts the operator picked (different prompts than the training captions).
DEFAULT_PROMPTS = [
    "a_cinematic_film_still_of_the_subject_in_evening_wear",
    "In_a_sunlit_kitchen_the_subject_laughs_while_flipping_a",
    "the_subject_walks_briskly_toward_the_camera_down_a_rain",
]
DEFAULT_VIDEOS_OUT = PREP_DIR / "testvideos"
DEFAULT_SEEDS_OUT = PREP_DIR / "testvideos_seeds"
# PACKAGED spec (signet_trainer.harness_data), resolved via importlib.resources — NOT a
# ".planning/harness/" path built from __file__, which only exists in a source checkout.
DEFAULT_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    ap.add_argument("--step", type=int, default=DEFAULT_STEP, help="Checkpoint render to pull (step_<N>.mp4).")
    ap.add_argument("--prompt", action="append", default=None,
                    help=f"Prompt dir(s) to pull (repeatable). Default: the 3 held-out prompts {DEFAULT_PROMPTS}")
    ap.add_argument("--videos-out", type=Path, default=DEFAULT_VIDEOS_OUT)
    ap.add_argument("--seeds-out", type=Path, default=DEFAULT_SEEDS_OUT)
    ap.add_argument("--spec", default=str(DEFAULT_SPEC),
                    help="MASK-SPEC.yaml — the ÷64/8n+1 dims rules come FROM here (D-03).")
    ap.add_argument("--force", action="store_true", help="Re-write outputs that already exist.")
    ap.add_argument("--dry-run", action="store_true", help="Probe + report only; write nothing.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.prompts = DEFAULT_PROMPTS  # the default set prep_testvideos falls back to when --prompt is unset
    return tv.prep_testvideos(args)


if __name__ == "__main__":
    sys.exit(main())
