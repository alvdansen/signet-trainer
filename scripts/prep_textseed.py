"""Thin text-prompt auto-seeding CLI — wraps ``signet_trainer.prep.textseed``.

Produces the frame-0 / last-frame binary seed PNGs that ``prep_inpaint_propagate.py`` consumes,
WITHOUT any hand-drawing: SAM 3's concept head turns a natural-language prompt into the subject mask,
several evenly-spaced frames are probed (both endpoints always), and the fwd/rev propagation
direction is decided per clip (D-08 seam reuse).

Outputs, all into ``--seeds-out`` / ``--records-out`` / ``--manifest-out``:
  * ``<clip_stem>__<mask_type>.png`` — binary seed, region WHITE(255) (propagate's `load_seed` shape);
  * ``textseed_records.json`` — per-clip {seed_frame_idx, direction, score, coverage, probes};
  * ``manifest.txt`` — propagate's ``NN | <clip_stem>.png | <clip>`` stem->clip resolution map.

NO Modal import, NO metered dispatch — local GPU only. Nothing here launches a metered run.

Usage:
  # self-check (counts frames, prints probe plan; loads NO model, writes nothing):
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_textseed.py --clips-dir <dir> --dry-run

  # real auto-seed:
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_textseed.py \
      --clips-dir <dir> --seeds-out <dir> --prompt "<concept prompt>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from signet_trainer.prep import textseed as ts  # noqa: E402  (after path bootstrap)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clips-dir", required=True, help="Directory of source clips (.mov/.mp4/.mkv).")
    ap.add_argument("--seeds-out", default=None, help="Seed PNG output dir (default <clips-dir>/../seeds).")
    ap.add_argument("--records-out", default=None, help="Per-clip JSON records (default <seeds-out>/../textseed_records.json).")
    ap.add_argument("--manifest-out", default=None, help="propagate manifest.txt (default <seeds-out>/../manifest.txt).")
    ap.add_argument("--config", default=str(ts.DEFAULT_CONFIG),
                    help="prep_textseed.yaml — every threshold comes FROM here (config-first).")
    ap.add_argument("--prompt", default=None,
                    help="Concept prompt (campaign material — pass at the CLI, never tracked in config).")
    ap.add_argument("--mask-type", default=None, help="Override seed.mask_type (stem suffix).")
    ap.add_argument("--n-probes", type=int, default=None, help="Override seed.n_probes.")
    ap.add_argument("--min-score", type=float, default=None, help="Override seed.min_score.")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Override model.device.")
    ap.add_argument("--only", action="append", default=[], help="Process only these clip stem(s) (repeatable).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve clips + frame counts + probe plan; loads NO model, writes nothing.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clips_dir = Path(args.clips_dir)
    root = clips_dir.parent
    args.seeds_out = args.seeds_out or str(root / "seeds")
    base = Path(args.seeds_out).parent
    args.records_out = args.records_out or str(base / "textseed_records.json")
    args.manifest_out = args.manifest_out or str(base / "manifest.txt")
    return ts.run(args)


if __name__ == "__main__":
    sys.exit(main())
