"""data.dataset_file — the fresh-clip JSONL manifest writer (Plan 02-01 Task 1).

Writes the ``{caption, media_path}`` JSONL manifest that the canonical ``process_dataset.py``
consumes (RESEARCH.md Pattern 2). ``media_path`` is RELATIVE to the manifest's parent dir.

Mirrors ``config/load.py``'s thin-importable I/O style (explicit ``encoding="utf-8"``).
Kept torch-free / encoder-free so it never breaks the loop-purity scan (TRAIN-03 / D-08).
"""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(rows: list[dict], out_path: str | Path) -> Path:
    """Write a ``{caption, media_path}`` JSONL manifest, one JSON object per line (utf-8).

    Each row is ``{"caption": str, "media_path": str}`` where ``media_path`` is RELATIVE to
    the manifest's parent dir (resolved by ``process_dataset.py`` against that parent).

    Args:
        rows: The manifest rows (preserved verbatim).
        out_path: Destination path for the ``.jsonl`` manifest.

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:  # utf-8 explicit (Phase-1 carry-forward)
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out
