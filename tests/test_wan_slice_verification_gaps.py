"""Verifier-added gaps for the multi-source / Wan slices. PURE stdlib + pydantic — no ``modal``.

Three properties that the two slices ASSERT IN PROSE but pin nowhere, each found by reproducing
the slices rather than by reading them. Deliberately free of ``modal`` imports so this file runs on
the 3.10 bench AND on a bare 3.13 interpreter, which is where the Wan stage's own test file cannot
go (``src/signet_trainer/modal/app.py`` imports ``modal`` at module level).

  1. The ``training_dims`` PRE-SCREEN widening opened no hole. Slice A widened the field-level
     screen with a third (wan) arm and argued in a code comment that this is "provably hole-free
     ... checked rather than assumed". The validator-level halves are pinned
     (``test_the_examples_geometry_is_rejected_by_every_NATIVE_family``), but the END-TO-END claim —
     that a whole ``SignetConfig`` carrying wan-legal / native-illegal geometry on a NATIVE family
     is still refused, with its ORIGINAL message — was pinned nowhere. That is the exact thing the
     widening could break, and it breaks silently: the config loads and the run costs money.

  2. ``runners/musubi_toml.py`` sits outside EVERY Wan-token scan. Slice B added its sibling
     ``runners/wan_musubi.py`` to ``test_no_wan_params._EXTRA_SCANNED`` because that file holds the
     ``discrete_flow_shift`` pin. The sibling was not added, is not ``wan_``-prefixed, and is not
     under ``inference/`` — so it is in no scan at all. It is clean TODAY (verified before writing
     this); nothing keeps it clean. Same defect ``test_the_qwen_pipeline_module_is_actually_scanned``
     records, one directory over.

  3. Exactly ONE module produces musubi dataset-TOML text. The byte-exact oracle proof is a proof
     about ``runners/musubi_toml``; a second writer anywhere would fork it, and the in-container
     stage would ship bytes no oracle ever saw. The shipped check is a source-SUBSTRING assertion
     that the entrypoint still calls ``render_from_config(cfg)`` — it cannot see a second producer.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from signet_trainer.config.load import load_config_from_text

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "signet_trainer"
_RENDERER = _SRC / "runners" / "musubi_toml.py"
_FNS = _SRC / "modal" / "fns.py"
#: A real, shipped, LTX config — the template every case below mutates. Using a shipped config
#: rather than a hand-rolled minimum keeps these cases honest about required fields: an earlier
#: draft of this file omitted ``training.max_steps`` and every case "passed" by rejecting for the
#: wrong reason.
_TEMPLATE = _REPO / "configs" / "ltx23_lora.example.yaml"


def _strip(src: str) -> str:
    """Comments and docstrings out — the same stripper shape ``test_no_wan_params`` uses.

    These files legitimately NAME musubi keys and Wan parameters in prose to explain them; only
    executable code is in scope.
    """
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _config_text(*, family: str, dims: list[int]) -> str:
    payload = json.loads(json.dumps(yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))))
    payload.setdefault("model", {})["family"] = family
    payload["training_dims"] = dims
    return yaml.safe_dump(payload)


# ==================================================================================================
# 1. The pre-screen widening opened no hole
# ==================================================================================================

#: Triples that are LEGAL under the wan law Slice A added to the pre-screen and ILLEGAL under both
#: native laws — so each is a live candidate to fall through the widened screen onto a family that
#: cannot render it. 1280x720x21 is the reference method's own geometry (720 % 32 == 16, so it fails
#: LTX/H3 SPATIALLY); the two 640-wide triples are spatially legal everywhere and fail on the FRAME
#: law alone, which exercises the other axis of the widening.
#:
#: ⚠ 640x360x45 — the method's second video view, and the obvious case to reach for — is NOT here.
#: It is not wan-legal either (360 % 16 == 8; musubi would bucket it to 352), so it dies under all
#: three laws and would have proved nothing. Caught by
#: ``test_the_hole_cases_are_genuinely_wan_legal_so_the_test_above_is_not_vacuous``, which is why
#: that check is in this file.
#:
#: The expected substring NAMES THE FAMILY'S OWN LAW rather than the offending number: the claim is
#: not merely that the config was refused, it is that the refusal still comes from the native law
#: with its original wording.
_HOLE_CASES = [
    ("ltx", [1280, 720, 21], "LTX-2.3"),
    ("ltx", [640, 352, 45], "LTX-2.3"),
    ("ltx", [640, 480, 21], "LTX-2.3"),
    ("h3", [1280, 720, 21], "MiniMax-H3"),
    ("h3", [640, 352, 45], "MiniMax-H3"),
    ("h3", [640, 480, 21], "MiniMax-H3"),
    ("qwen_edit", [1280, 720, 21], "Qwen-Image-Edit-2511"),
    ("qwen_edit", [640, 352, 45], "Qwen-Image-Edit-2511"),
    ("qwen_edit", [640, 480, 21], "Qwen-Image-Edit-2511"),
]


@pytest.mark.parametrize(("family", "dims", "expected"), _HOLE_CASES)
def test_wan_legal_geometry_is_still_refused_on_every_native_family(
    family: str, dims: list[int], expected: str
) -> None:
    """A widened screen must not let wan geometry reach a native family — and must still SAY WHY.

    The message is asserted, not just the raise. The widening's whole safety argument is that the
    ``else:`` arm of ``_cross_field_checks`` re-asserts the native law "verbatim ... the identical
    verdict and the identical message — only LATER". A refusal that arrived with a DIFFERENT message
    would satisfy a bare ``pytest.raises`` while proving the argument false.
    """
    with pytest.raises(ValueError, match=re.escape(expected)):
        load_config_from_text(_config_text(family=family, dims=dims))


def test_the_hole_cases_are_genuinely_wan_legal_so_the_test_above_is_not_vacuous() -> None:
    """Non-vacuity: if these triples were illegal EVERYWHERE the test above would prove nothing.

    It would pass identically against an unwidened screen, a screen with no wan arm, or a screen
    that rejected all three families for unrelated reasons.
    """
    from signet_trainer.config import validators as v

    for _family, dims, _expected in _HOLE_CASES:
        assert v.validate_wan_training_dims(tuple(dims)) == tuple(dims), (
            f"{dims} is not accepted by the wan law, so it cannot exercise the widened pre-screen"
        )


def test_the_widened_screen_still_ADMITS_the_geometry_on_the_wan_family() -> None:
    """The other half: the widening must actually work, or case 1 passes for the wrong reason.

    Asserted through the shipped example rather than the validator, so this fails if the wan arm is
    reachable in ``validators`` but unreachable through a real config load.
    """
    cfg = load_config_from_text(
        (_REPO / "configs" / "wan21_kaboom.example.yaml").read_text(encoding="utf-8")
    )
    assert cfg.model.family == "wan"
    assert list(cfg.training_dims) == [1280, 720, 21]


# ==================================================================================================
# 2. No runners/ module escapes the Wan-token scan
# ==================================================================================================

def test_every_runners_module_is_either_scanned_or_provably_clean() -> None:
    """``runners/musubi_toml.py`` is in no Wan-token scan. Pin its cleanliness rather than hope.

    ``test_no_wan_params`` scans ``inference/*.py`` plus a hand-maintained ``_EXTRA_SCANNED``
    allowlist. A ``runners/`` module reaches that scan only by being named in the allowlist, and
    only ``wan_musubi.py`` is. This asserts the property the allowlist exists to protect, for the
    whole directory, so a new runner cannot arrive unscanned.
    """
    from tests.test_no_wan_params import (  # noqa: PLC0415
        _ALLOWED_SHIFT_VALUES,
        _EXTRA_SCANNED,
        _WAN_TOKENS_ALWAYS,
        _family_of,
        _shift_literals,
    )

    runners = sorted(p for p in (_SRC / "runners").glob("*.py"))
    assert runners, "runners/ has no modules — this scan has silently stopped covering anything"

    offenders: list[str] = []
    for path in runners:
        code = _strip(path.read_text(encoding="utf-8"))
        for token in _WAN_TOKENS_ALWAYS:
            if token in code:
                offenders.append(f"{path.name}: banned token {token!r}")
        family = _family_of(path.name)
        allowed = _ALLOWED_SHIFT_VALUES.get(family, set()) if family else set()
        for literal in _shift_literals(code) - allowed:
            rel = f"src/signet_trainer/runners/{path.name}"
            scanned = rel in _EXTRA_SCANNED
            offenders.append(
                f"{path.name}: shift literal {literal!r} outside its family's allowed set "
                f"{sorted(allowed)} (family={family!r}, in _EXTRA_SCANNED={scanned})"
            )
    assert not offenders, (
        "a runners/ module carries a Wan sampling parameter that no gate is watching: "
        f"{offenders}. Either give it a family prefix AND add it to "
        "tests/test_no_wan_params._EXTRA_SCANNED, or remove the parameter."
    )


def test_the_renderer_is_now_scanned_and_the_gap_is_closed() -> None:
    """RETIRED-AND-INVERTED 2026-08-09. Its predecessor said so itself.

    The original asserted musubi_toml.py was NOT in _EXTRA_SCANNED, and its docstring named its own
    successor condition: "If musubi_toml.py is ever added to _EXTRA_SCANNED (the better fix), this
    test starts failing and should be deleted." The better fix landed, so this is the inverted form
    that keeps the property rather than the gap.

    Why it stays a test instead of a deletion: the renderer is the module that WRITES the sampling
    config, which makes it the single most valuable place for a Wan token to land unnoticed. It was
    outside every scan for a structural reason — not under inference/, no wan_ prefix — so nothing
    would have surfaced a regression except someone re-deriving the blind spot.
    """
    from tests.test_no_wan_params import _EXTRA_SCANNED  # noqa: PLC0415

    assert _RENDERER.exists()
    assert "src/signet_trainer/runners/musubi_toml.py" in _EXTRA_SCANNED, (
        "the renderer left _EXTRA_SCANNED. It has no family prefix, so _family_of returns None and "
        "the value-level shift rule does not reach it either — removing it from the extra scan "
        "puts the TOML writer back outside every gate."
    )


# ==================================================================================================
# 3. One producer of musubi dataset-TOML text
# ==================================================================================================

#: A musubi TABLE HEADER, emitted bare. The renderer appends these as whole lines
#: (``lines.append("[[datasets]]")``); prose that merely NAMES a marker always embeds it in a longer
#: sentence, so an exact-equality test separates construction from description.
_TOML_MARKERS = frozenset({"[[datasets]]", "[general]"})


def _emits_bare_marker(fn: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Constant) and n.value in _TOML_MARKERS for n in ast.walk(fn)
    )


def _emits_key_equals_value(fn: ast.AST) -> bool:
    """An f-string whose literal part carries ``' = '`` — a TOML key/value emitter.

    This is the half that separates BUILDING a table from mentioning one. ``fns.wan_train`` holds a
    bare ``'[[datasets]]'`` (it is the argument to ``dataset_toml.count(...)`` in its provenance
    print) and would be a false positive on the marker alone; it emits no key/value pairs, so it is
    correctly not a producer.
    """
    return any(
        isinstance(n, ast.JoinedStr)
        and any(isinstance(v, ast.Constant) and " = " in str(v.value) for v in n.values)
        for n in ast.walk(fn)
    )


def _toml_producers() -> list[str]:
    """``module::function`` for every function that both emits a bare marker AND key/value pairs."""
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _emits_bare_marker(fn) and _emits_key_equals_value(fn):
                found.append(f"{path.relative_to(_SRC).as_posix()}::{fn.name}")
    return sorted(found)


def test_exactly_one_function_in_the_package_produces_musubi_dataset_toml() -> None:
    """A second TOML writer would fork the byte-exact oracle proof. There must not be one.

    The oracle proof (``render_musubi_toml`` reproduces the production TOML byte-for-byte, sha256
    ``e4ae5eb9…``) is a proof about ONE function. If any other function assembled a ``[[datasets]]``
    table, the bytes a metered container writes could diverge from the bytes any test ever compared
    — invisibly, because both files would still parse and both would still train something.

    This is the assertion the shipped
    ``test_the_dataset_toml_is_rendered_at_dispatch_not_baked_into_the_image`` cannot make: that one
    checks the entrypoint still CALLS ``render_from_config(cfg)``, which stays true even if a second
    producer is added beside it.
    """
    assert _toml_producers() == ["runners/musubi_toml.py::render_musubi_toml"], (
        "the set of musubi dataset-TOML producers changed: "
        f"{_toml_producers()}. Exactly one function may build this document — the one the tracked "
        "oracle in docs/source-methods/musubi-wan21/ is compared against."
    )


def test_the_producer_detector_is_not_vacuous() -> None:
    """RED self-check on BOTH halves of the detector, since either alone would misclassify.

    Without the marker half every f-string in the package matches; without the key/value half
    ``fns.wan_train`` matches on its ``.count('[[datasets]]')`` argument. The detector is only
    meaningful if each half is independently doing work — asserted here rather than assumed.
    """
    renderer = ast.parse(_RENDERER.read_text(encoding="utf-8"))
    render_fn = next(
        n
        for n in ast.walk(renderer)
        if isinstance(n, ast.FunctionDef) and n.name == "render_musubi_toml"
    )
    assert _emits_bare_marker(render_fn), "the marker half matches nothing even in the renderer"
    assert _emits_key_equals_value(render_fn), "the key/value half matches nothing in the renderer"

    stage = ast.parse(_FNS.read_text(encoding="utf-8"))
    wan_train = next(
        n
        for n in ast.walk(stage)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "wan_train"
    )
    assert _emits_bare_marker(wan_train), (
        "wan_train no longer holds a bare marker — this near-miss is what proves the key/value "
        "half is load-bearing rather than decorative; find another or drop this assertion"
    )
    assert not _emits_key_equals_value(wan_train), (
        "wan_train has started emitting TOML key/value pairs — it is becoming a second producer"
    )


def test_the_stage_writes_the_toml_it_was_HANDED_and_does_not_rebuild_it() -> None:
    """``wan_train`` must persist its ``dataset_toml`` parameter verbatim.

    The complement of the scan above: no second producer exists, AND the one consumer does not
    re-derive. If the stage wrote anything but the threaded string, the dry-run gate would have
    parsed one document and the round would have trained on another.
    """
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "wan_train"
    )
    assert "dataset_toml" in {a.arg for a in fn.args.args + fn.args.kwonlyargs}, (
        "wan_train no longer takes dataset_toml as a parameter — the TOML is being obtained "
        "some other way, and the dispatch-time render is no longer the source of the bytes"
    )
    writes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
    ]
    assert len(writes) == 1, f"expected exactly one write_text in wan_train, found {len(writes)}"
    first = writes[0].args[0]
    assert isinstance(first, ast.Name) and first.id == "dataset_toml", (
        "wan_train writes something other than its dataset_toml parameter "
        f"({ast.unparse(first)!r}) — the bytes on the Volume would no longer be the bytes the "
        "renderer produced and the dry-run gate parsed"
    )
