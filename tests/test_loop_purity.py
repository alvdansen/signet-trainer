"""Structural loop-purity gate (TRAIN-03 / D-08) — the CI enforcer for the data path.

Two complementary assertions on ``signet_trainer.data.precomputed``:

  (a) Transitive-import proof: pop ``modal`` / ``ltx_core`` / ``ltx_trainer`` from
      ``sys.modules``, import the module, then assert none reappeared (mirrors
      ``tests/test_schema_compose.py``'s sys.modules-absence style).

  (b) Text-scan offender check: read the module SOURCE, strip comments + docstrings (so
      the docstring's "VAE/Gemma" prose does not false-positive), and assert no encoder /
      modal import statement appears in the real code (mirrors ``tests/test_no_warm_gpu.py``).

If either fails, Phase 3's training loop could silently pull an encoder into the data path,
breaking the D-08 boundary. This is the CI gate the plan's loop-purity must_have hangs on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The module under guard (path used for the text scan; never imported as text-only).
_PRECOMPUTED_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "data"
    / "precomputed.py"
)

# Heavy modules the loop-purity boundary forbids in the data path.
_FORBIDDEN_MODULES = ("modal", "ltx_core", "ltx_trainer")


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments and triple-quoted strings (copied from test_no_warm_gpu.py).

    The module DOCUMENTS its no-encoder contract (mentions VAE/Gemma/ltx_core in prose); we
    must match only real import statements, not the warnings about them.
    """
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_data_precomputed_imports_no_encoder() -> None:
    """(a) import signet_trainer.data.precomputed must NOT transitively pull a heavy encoder."""
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    import signet_trainer.data.precomputed  # noqa: F401

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"loop-purity violation (TRAIN-03 / D-08): data.precomputed imported {mod!r}"
        )


def test_data_precomputed_source_has_no_encoder_import() -> None:
    """(b) the real source carries no encoder / modal import statement (docstring prose excluded)."""
    code = _strip_comments_and_docstrings(_PRECOMPUTED_SRC.read_text(encoding="utf-8"))

    # Top-level offender import statements (module-load surface). The legacy einops import is
    # lazy + inside a function, which is fine; encoders/modal must appear NOWHERE.
    offenders = [
        r"^\s*import\s+modal\b",
        r"^\s*from\s+modal\b",
        r"^\s*import\s+ltx_core\b",
        r"^\s*from\s+ltx_core\b",
        r"^\s*import\s+ltx_trainer\b",
        r"^\s*from\s+ltx_trainer\b",
        r"models\.loader\b",
        r"\bVAE\b",
        r"\bGemma\b",
    ]
    hits = [pat for pat in offenders if re.search(pat, code, re.MULTILINE)]
    assert not hits, (
        "loop-purity violation (TRAIN-03 / D-08): encoder/modal reference in data/precomputed.py "
        f"real code: {hits}"
    )
