"""Thin launcher for the signet masking app (D-06..D-11) — the hand-draw inpaint-mask surface.

Serves the ``signet_trainer.prep.app`` web-canvas tool on ``127.0.0.1`` (house rule — loopback ONLY;
tailnet exposure for iPad/Apple-Pencil use goes through ``scripts/_tailnet_relay.py``, NEVER a
wildcard bind). Users draw binary teal masks by hand over video frames; the binary-by-construction
guarantee lives server-side (the server thresholds every exported pixel to ``{0,255}``, region=WHITE).

    PYTHONPATH=src PYTHONUTF8=1 python scripts/serve_mask_app.py <clips_dir> [--output <prep_dir>] \\
        [--spec <MASK-SPEC.yaml>] [--port 8030]

No metered dispatch, no Modal — pure local file prep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from signet_trainer import harness_data
from signet_trainer.prep.app import serve

# PACKAGED spec (signet_trainer.harness_data), resolved via importlib.resources — NOT a
# ".planning/harness/" path, which only exists in a source checkout (a fresh install has none).
_DEFAULT_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the signet hand-draw inpaint masking app (127.0.0.1 only).")
    ap.add_argument("clips_dir", type=Path, help="directory of source clips (*.mp4) to mask")
    ap.add_argument("--output", "--prep-dir", dest="output_dir", type=Path, default=Path("_inpaint_prep"),
                    help="prep output dir where masks_frame0/ + masks_video/ are written (default: _inpaint_prep)")
    ap.add_argument("--spec", dest="spec_path", type=Path, default=_DEFAULT_SPEC,
                    help=f"MASK-SPEC.yaml path (brush hex + coverage bands; default: {_DEFAULT_SPEC})")
    ap.add_argument("--port", type=int, default=8030, help="localhost port (default: 8030)")
    args = ap.parse_args()

    # serve() binds 127.0.0.1 ALWAYS and refuses any non-loopback host (house rule).
    serve(
        host="127.0.0.1",
        port=args.port,
        clips_dir=args.clips_dir,
        prep_dir=args.output_dir,
        spec_path=args.spec_path,
    )


if __name__ == "__main__":
    main()
