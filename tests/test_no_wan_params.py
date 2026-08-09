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

# --------------------------------------------------------------------------------------------------
# FAMILY SCOPING (2026-08-09) — the bare ``shift`` ban is an LTX-PATH rule, not a repo-wide one.
#
# This gate was written when ``inference/`` was LTX-only, so "the word shift must not appear" was a
# clean proxy for "no Wan sampler contamination". It is no longer clean. TWO families REQUIRE a
# static shift as a locked recipe parameter:
#
#   qwen_edit : METHOD §8 — Qwen-Image-Edit-2511 static shift 3.0, base Qwen-Image static 7.0.
#               Getting this wrong is the documented "training samples look fine but renders are
#               muddy" failure, and the scheduler LIES about it — set_shift() under
#               use_dynamic_shifting=True reports the value you asked for and leaves the sigmas
#               unchanged (measured 2026-08-09: max |sigma delta| = 0.000e+00).
#   wan       : musubi-tuner --discrete_flow_shift 7.0, in the house Wan 2.1 recipe.
#
# So a directory-scoped ban would either block those families or force their sampling parameters to
# live somewhere the gate cannot see — which is exactly what happened by accident:
# ``models/qwen_edit_pipeline.py`` holds the Qwen scheduler pin and is OUTSIDE this scan, while
# H3's sibling (``inference/h3_pipeline_source.py``) is inside it. The gate passed for a reason that
# had nothing to do with correctness.
#
# The fix is to scope by FAMILY rather than by directory, and to make the non-LTX rule STRONGER
# rather than weaker: instead of "the word must not appear", assert "the VALUE must be the family's
# own, and never a Wan value". A family file may say shift; it may not say Wan's shift.
# --------------------------------------------------------------------------------------------------

#: Filename prefixes that mark a module as owned by a non-LTX family.
_FAMILY_PREFIXES = ("h3_", "qwen_edit_", "wan_")

#: Wan tokens that stay banned EVERYWHERE, including in family-scoped modules. The bare ``shift``
#: is deliberately absent: it is the token this scoping exists to release.
_WAN_TOKENS_ALWAYS = ("UniPC", "frames=33", "guidance_scale=5", "num_inference_steps=50", "shift=4.0")

#: Static shift values a family may legitimately pin, by family prefix. A literal outside its
#: family's set is refused — including Wan's 4.0, which is what the original token guarded against.
_ALLOWED_SHIFT_VALUES = {
    "qwen_edit_": {"3.0", "7.0"},   # METHOD §8: edit 3.0, base Qwen-Image 7.0
    "wan_": {"7.0"},                # musubi --discrete_flow_shift 7.0
    "h3_": set(),                   # H3 pins no static shift; any literal here is a surprise
}

#: Modules OUTSIDE inference/ that carry sampling parameters and must be scanned anyway. Added
#: because the Qwen scheduler pin lives in models/, and a gate that misses the file holding the
#: parameter is decorative.
_EXTRA_SCANNED = ("src/signet_trainer/models/qwen_edit_pipeline.py",)


def _family_of(name: str) -> str | None:
    """The family prefix owning ``name``, or None for an LTX-path module."""
    for prefix in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def _shift_literals(code: str) -> set[str]:
    r"""Every numeric literal bound to a shift-ish name in executable code.

    Three forms, all of which these modules genuinely use, and each of which an earlier version of
    this pattern missed — every one caught by its own RED self-check rather than by review:

        shift = 3.0                                  bare assignment
        scheduler.shift = 3.0                        attribute
        {"shift": 3.0}                               dict key (quote before the separator)
        discrete_flow_shift = 7.0                    PREFIXED name (underscore is a word char, so
                                                     an anchored [^\w] boundary excluded it)
        QWEN_EDIT_RENDER_SCHEDULER_SHIFT: float = 3.0   UPPERCASE annotated constant — and this is
                                                     the form that holds the REAL pin, so a
                                                     case-sensitive scan reported a clean gate over
                                                     the one line that matters

    Case-insensitive for the last reason. A scanner that silently under-matches is worse than no
    scanner: it reports a clean gate over an unread file.
    """
    pattern = re.compile(
        r"""[\w.]*shift\w*["']?\s*(?::\s*[\w\[\]]+\s*)?=\s*([0-9]+\.?[0-9]*)"""
        r"""|[\w.]*shift\w*["']\s*:\s*([0-9]+\.?[0-9]*)""",
        re.IGNORECASE,
    )
    found: set[str] = set()
    for a, b in pattern.findall(code):
        if a:
            found.add(a)
        if b:
            found.add(b)
    return found


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
    """LTX-path modules keep the total ban; family modules keep a stricter, value-level one."""
    offenders: list[str] = []
    for path in _inference_sources():
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        family = _family_of(path.name)
        tokens = _WAN_TOKENS if family is None else _WAN_TOKENS_ALWAYS
        for token in tokens:
            if token in code:
                offenders.append(f"{path.name}: {token!r}")
    assert not offenders, f"Wan sampling token(s) leaked into inference/ code: {offenders}"


def test_family_modules_pin_only_their_own_shift_values() -> None:
    """A family may say ``shift``; it may NOT say a value that is not its own.

    Strictly stronger than the bare-token ban it replaces. The old rule could only answer "is the
    word present"; this answers "is the VALUE right", which is the question that actually matters —
    Wan's 4.0 in a Qwen sampler is a real defect, and Qwen's own 3.0 is a locked requirement. Both
    were indistinguishable to the token scan.
    """
    scanned: list[str] = []
    offenders: list[str] = []
    paths = list(_inference_sources()) + [
        Path(__file__).resolve().parents[1] / rel for rel in _EXTRA_SCANNED
    ]
    for path in paths:
        if not path.exists():
            continue
        family = _family_of(path.name)
        if family is None:
            continue
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        allowed = _ALLOWED_SHIFT_VALUES.get(family, set())
        for value in _shift_literals(code):
            scanned.append(f"{path.name}:{value}")
            if value not in allowed:
                offenders.append(
                    f"{path.name}: shift literal {value} not in {sorted(allowed) or '{}'} for "
                    f"family {family!r}"
                )
    assert not offenders, (
        "family module pinned a shift value that is not its own — Wan's 4.0 and a stray default are "
        f"both caught here: {offenders}"
    )


def test_the_qwen_pipeline_module_is_actually_scanned() -> None:
    """The file holding the Qwen scheduler pin must be IN scope.

    It lives under models/, not inference/, so the directory-scoped scan missed it entirely — the
    gate passed for a reason unrelated to correctness. A gate that cannot see the file holding the
    parameter it guards is decorative, so the module is named explicitly in ``_EXTRA_SCANNED``.
    """
    target = Path(__file__).resolve().parents[1] / "src/signet_trainer/models/qwen_edit_pipeline.py"
    assert target.exists(), "models/qwen_edit_pipeline.py is gone — update _EXTRA_SCANNED"
    assert "src/signet_trainer/models/qwen_edit_pipeline.py" in _EXTRA_SCANNED
    code = _strip_comments_and_docstrings(target.read_text(encoding="utf-8"))
    assert "shift" in code, (
        "the Qwen pipeline no longer pins a shift in executable code. Either the pin moved (point "
        "_EXTRA_SCANNED at its new home) or it was lost — and losing it is the documented "
        "muddy-render failure, which looks like a model problem rather than a settings one."
    )


def test_a_wan_shift_value_in_a_family_module_is_still_refused() -> None:
    """RED self-check: family scoping must not have opened a hole for Wan's 4.0."""
    code = _strip_comments_and_docstrings('shift = 4.0\n')
    assert "shift=4.0" not in code  # the literal-token form is not what saves us here
    assert _shift_literals(code) == {"4.0"}
    assert "4.0" not in _ALLOWED_SHIFT_VALUES["qwen_edit_"], (
        "4.0 became allowed for qwen_edit — that is Wan's value and the whole point of this gate"
    )
    assert "4.0" not in _ALLOWED_SHIFT_VALUES["wan_"]


def test_the_shift_literal_scanner_reads_real_forms() -> None:
    """RED self-check: the extractor must catch the forms these modules actually use."""
    for fixture, expected in (
        ("shift = 3.0", {"3.0"}),
        ("scheduler.shift = 3.0", {"3.0"}),
        ("set_shift(x)\nshift=7.0", {"7.0"}),
        ('cfg = {"shift": 3.0}', {"3.0"}),
        ("discrete_flow_shift = 7.0", {"7.0"}),
        ("no_shift_here()", set()),
    ):
        assert _shift_literals(fixture) == expected, f"extractor missed {fixture!r}"


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
