"""INFR-01 Wave-0 gate — no Wan params + Anti-Pattern-6 confinement across ``inference/``.

Two source-scan assertions over every ``*.py`` under ``src/signet_trainer/inference/`` (so it
naturally covers modules added by later Phase-4 plans once they land):

  (a) NO Wan-tuned sampling token (``UniPC`` / ``shift`` / ``frames=33`` / ``guidance_scale=5``
      / ``num_inference_steps=50`` / ``shift=4.0``) appears in real code. The LTX re-validated
      set (Euler + STG, guidance 3.0, steps 30) is the only one allowed (T-04-01).
  (b) Anti-Pattern-6: no MODULE-LEVEL ``import modal`` / ``ltx_core`` / ``ltx_trainer`` in
      ``sampler.py`` (function-local + ``TYPE_CHECKING`` imports are allowed — they never run
      on CPU/CI).

The comment/docstring stripper (ported from tests/test_lora.py) means the Wan tokens are only
flagged in executable code — the docstrings here and in sampler.py may legitimately NAME the
forbidden Wan set to explain why it is excluded.
"""

from __future__ import annotations

import re
from pathlib import Path

_INFERENCE_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "inference"
)
_SAMPLER_SRC = _INFERENCE_DIR / "sampler.py"
_MULTI_CONDITION_SRC = _INFERENCE_DIR / "multi_condition.py"
_REFERENCE_VIDEO_SRC = _INFERENCE_DIR / "reference_video.py"

# Wan-tuned tokens that must never appear in an LTX code path (RESEARCH §Canonical Params).
_WAN_TOKENS = ("UniPC", "shift", "frames=33", "guidance_scale=5", "num_inference_steps=50", "shift=4.0")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _inference_sources() -> list[Path]:
    return sorted(_INFERENCE_DIR.glob("*.py"))


# --------------------------------------------------------------------------------------------------
# (a) no Wan tokens in executable code under inference/
# --------------------------------------------------------------------------------------------------


def test_no_wan_params_in_inference_code() -> None:
    offenders: list[str] = []
    for path in _inference_sources():
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        for token in _WAN_TOKENS:
            if token in code:
                offenders.append(f"{path.name}: {token!r}")
    assert not offenders, f"Wan sampling token(s) leaked into inference/ code: {offenders}"


def test_scanner_would_catch_a_wan_token() -> None:
    """RED self-check: the scanner must flag a synthetic UniPC/shift=4.0 fixture."""
    fixture = 'scheduler = "UniPC"\nshift = 4.0  # shift=4.0\n'
    stripped = _strip_comments_and_docstrings(fixture)
    hits = [tok for tok in _WAN_TOKENS if tok in stripped]
    assert "UniPC" in hits and "shift" in hits, (
        "scanner failed to flag a known-bad Wan fixture — the gate is not actually protecting"
    )


def test_a_wan_token_only_in_a_docstring_is_not_flagged() -> None:
    """The stripper must let a docstring NAME the Wan set (as sampler.py's docstring does)."""
    fixture = '"""not the Wan UniPC / shift=4.0 / frames=33 set."""\nx = 1\n'
    stripped = _strip_comments_and_docstrings(fixture)
    assert not any(tok in stripped for tok in _WAN_TOKENS)


# --------------------------------------------------------------------------------------------------
# (b) Anti-Pattern-6 module-level import confinement in sampler.py
# --------------------------------------------------------------------------------------------------

# Column-0 anchored: only MODULE-LEVEL imports match. Indented function-local and
# ``if TYPE_CHECKING:`` imports (which never execute on CPU/CI) are deliberately allowed.
_MODULE_LEVEL_HEAVY_IMPORT = re.compile(
    r"^(?:import|from)\s+(?:modal|ltx_core|ltx_trainer)\b", re.MULTILINE
)


def test_sampler_has_no_module_level_heavy_import() -> None:
    code = _strip_comments_and_docstrings(_SAMPLER_SRC.read_text(encoding="utf-8"))
    hits = _MODULE_LEVEL_HEAVY_IMPORT.findall(code)
    assert not hits, f"Anti-Pattern-6 violation: module-level heavy import in sampler.py: {hits}"


def test_confinement_regex_would_catch_a_top_level_import() -> None:
    """RED self-check: the confinement regex must flag a column-0 heavy import."""
    assert _MODULE_LEVEL_HEAVY_IMPORT.search("from ltx_trainer.x import Y\n")
    # ...but NOT an indented (function-local / TYPE_CHECKING) one.
    assert not _MODULE_LEVEL_HEAVY_IMPORT.search("    from ltx_trainer.x import Y\n")


# --------------------------------------------------------------------------------------------------
# REF-02 (Seam E) — multi_condition.py: no Wan tokens + Anti-Pattern-6 confinement
# --------------------------------------------------------------------------------------------------


def test_multi_condition_is_covered_by_the_inference_scan() -> None:
    """The Seam-E module must be one of the scanned inference sources (not silently skipped)."""
    assert _MULTI_CONDITION_SRC in _inference_sources()


def test_multi_condition_has_no_wan_params() -> None:
    """The multi-condition inference path introduces none of the banned Wan sampling tokens."""
    code = _strip_comments_and_docstrings(_MULTI_CONDITION_SRC.read_text(encoding="utf-8"))
    offenders = [token for token in _WAN_TOKENS if token in code]
    assert not offenders, f"Wan sampling token(s) leaked into multi_condition.py: {offenders}"


def test_multi_condition_has_no_module_level_heavy_import() -> None:
    """Anti-Pattern-6: ltx_core/ltx_trainer/modal imports in multi_condition.py are function-local."""
    code = _strip_comments_and_docstrings(_MULTI_CONDITION_SRC.read_text(encoding="utf-8"))
    hits = _MODULE_LEVEL_HEAVY_IMPORT.findall(code)
    assert not hits, f"Anti-Pattern-6 violation: module-level heavy import in multi_condition.py: {hits}"


# --------------------------------------------------------------------------------------------------
# REF-03 (D-7-BASEVAR) — reference_video.py: no Wan tokens + Anti-Pattern-6 confinement
# --------------------------------------------------------------------------------------------------


def test_reference_video_is_covered_by_the_inference_scan() -> None:
    """The Phase-7 V2V module must be one of the scanned inference sources (not silently skipped)."""
    assert _REFERENCE_VIDEO_SRC in _inference_sources()


def test_reference_video_has_no_wan_params() -> None:
    """The V2V dev single-stage render path introduces none of the banned Wan sampling tokens."""
    code = _strip_comments_and_docstrings(_REFERENCE_VIDEO_SRC.read_text(encoding="utf-8"))
    offenders = [token for token in _WAN_TOKENS if token in code]
    assert not offenders, f"Wan sampling token(s) leaked into reference_video.py: {offenders}"


def test_reference_video_has_no_module_level_heavy_import() -> None:
    """Anti-Pattern-6: ltx_core/ltx_trainer/modal imports in reference_video.py are function-local."""
    code = _strip_comments_and_docstrings(_REFERENCE_VIDEO_SRC.read_text(encoding="utf-8"))
    hits = _MODULE_LEVEL_HEAVY_IMPORT.findall(code)
    assert not hits, f"Anti-Pattern-6 violation: module-level heavy import in reference_video.py: {hits}"
