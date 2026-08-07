"""Machine-check the masking-app client contract (no browser needed).

Renders the served page HTML via ``prep.app.render_page`` and asserts the 09.3-UI-SPEC Copywriting
Contract + interaction hooks are present (exact CTA strings, both empty states, the SAM-unavailable
fix-it-first message, the ÷64/8n+1 dims error copy, the Clear-all inline confirm) plus the
load-bearing brush hex + canvas attributes. Exits non-zero if ANY required string is missing — so the
client contract stays enforceable in CI without a live browser.

    PYTHONPATH=src PYTHONUTF8=1 python scripts/check_mask_app_client.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from signet_trainer import harness_data
from signet_trainer.prep.app import AppConfig, render_page
from signet_trainer.prep.spec import load_mask_spec

# PACKAGED spec (signet_trainer.harness_data), resolved via importlib.resources — a CWD-relative
# ".planning/harness/" path only resolved when run from the maintainer's checkout.
_SPEC = harness_data.spec_path(harness_data.MASK_SPEC)

# Every required token in the served client HTML. Grouped by contract dimension for readable failures.
REQUIRED = [
    # Copywriting Contract — CTAs + tool labels
    "Export masks", "Propagate from frame 0", "Re-seed clip", "Approve & stage", "Load paintover",
    "Brush", "Eraser", "Zoom", "Undo", "Redo", "Clear frame", "Clear all",
    # Copywriting Contract — empty states
    "No clips loaded", "Point the masking app at a clips directory to start",
    "Paint the region to regenerate",
    "The teal brush is the inpaint mask.",
    # Copywriting Contract — SAM-unavailable fix-it-first (D-14)
    "SAM3 not available", "grant HF access to", "install SAM3 into",
    "SAM 2.1 is a last-ditch fallback only",
    # Copywriting Contract — dims error (÷64 / 8n+1) + Clear-all inline confirm
    "inpaint requires ÷64 spatial", "Clear all masks for this clip?",
    # Interaction hooks — binary brush + touch + crisp zoom
    "imageSmoothingEnabled", "touch-action", "image-rendering:pixelated",
    # Review / QA overlay mode
    "green = region to REGENERATE",
]


def main() -> int:
    spec = load_mask_spec(_SPEC)
    config = AppConfig(clips_dir=Path("."), prep_dir=Path("_inpaint_prep"), spec=spec, spec_path=_SPEC)
    html = render_page(config)

    missing = [tok for tok in REQUIRED if tok not in html]
    # the load-bearing brush hex must be injected from the spec (never a client literal)
    if spec["hex"] not in html:
        missing.append(f"brush hex {spec['hex']} (spec-injected)")

    if missing:
        print("[check-mask-app-client] FAIL — missing required contract strings:", file=sys.stderr)
        for m in missing:
            print(f"  - {m!r}", file=sys.stderr)
        return 1
    print(f"[check-mask-app-client] OK — all {len(REQUIRED)} contract strings + brush hex {spec['hex']} present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
