"""VERIFIER ADDENDA — two things the three landing slices left unasserted.

Written during the cross-slice verification pass, not by the slice that shipped the code, and
scoped to exactly what that pass found uncovered:

  (1) ``models/qwen_edit_loader.quantize_qwen_edit`` grew a PRINTED recipe-lock banner. The change
      is real and load-bearing (Modal shows stdout at WARNING, so the previous ``logger.info`` left
      a metered run with no evidence that the qfloat8 lock had been applied) — but it arrived
      unclaimed by any slice report and with no test. An unasserted print is exactly the kind of
      line a later refactor deletes as noise.

  (2) ``modal/fns.qwen_edit_sample`` must compose its per-cell output path from
      ``inference/qwen_edit_layout.qwen_edit_cell_relpath`` and from NOTHING ELSE. That is this
      family's own documented failure class: the HTML surface was once proved against a path shape
      the render never produced, and every tile would have fallen back to "generation failed" on
      renders that succeeded. The single-transcription rule is asserted for the 2x2 pack
      (``test_qwen_edit_verifier_gaps``) but was not asserted for the column -> file join.

Both are CPU-pure. No GPU, no Modal dispatch, no weight downloads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_FNS = _REPO / "src" / "signet_trainer" / "modal" / "fns.py"
_LOADER = _REPO / "src" / "signet_trainer" / "models" / "qwen_edit_loader.py"


# --------------------------------------------------------------------------------------------------
# (1) The quantize recipe-lock banner.
# --------------------------------------------------------------------------------------------------


def _quantize_fn() -> ast.FunctionDef:
    tree = ast.parse(_LOADER.read_text(encoding="utf-8"))
    fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "quantize_qwen_edit"
    ]
    assert len(fns) == 1, f"expected exactly one quantize_qwen_edit, found {len(fns)}"
    return fns[0]


def test_the_quantize_recipe_lock_is_printed_not_only_logged() -> None:
    """``print``, not just ``logger.info`` — Modal's default stdout level is WARNING.

    The 250-step run on live weights completed and committed a correct-looking adapter while leaving
    no evidence either way that the qfloat8 lock had been applied. It had been; the report was at
    INFO and was never shown. This asserts the fix stays a print, so the banner is an artifact an
    operator can diff rather than a line that only exists in a debug log nobody enabled.
    """
    fn = _quantize_fn()
    prints = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"
    ]
    assert prints, (
        "quantize_qwen_edit no longer PRINTS its recipe-lock banner. qfloat8 is a locked recipe "
        "parameter that QwenEditConfig deliberately refuses to expose as a field; a locked "
        "parameter whose application leaves no trace in a metered run's log is unauditable. "
        "logger.info alone is not enough — Modal shows stdout at WARNING."
    )


def test_the_quantize_banner_carries_numbers_rather_than_an_adjective() -> None:
    """A silently-skipped pass must still print, and print a ZERO.

    The banner interpolates the block count and the converted-leaf count. If it were reworded to
    "quantized successfully", a pass that converted nothing would be indistinguishable from one that
    converted the whole model — which is the failure the print exists to make visible.
    """
    src = ast.get_source_segment(_LOADER.read_text(encoding="utf-8"), _quantize_fn())
    assert src is not None
    assert "converted" in src and "blocks_note" in src, (
        "the quantize banner no longer interpolates the converted-module count and the block "
        "count. Numbers, not an adjective: a skipped pass must print a zero, not a cheerful string."
    )


def test_the_banner_survives_a_component_with_no_block_list() -> None:
    """``blocks`` is bound unconditionally, so the text-encoder path cannot NameError.

    The text encoder comes through the same function and has no ``transformer_blocks``. The banner
    reads ``blocks`` after the block loop; if that name were bound only inside ``if blocks is not
    None:`` the second component through this function would raise NameError AFTER the weights were
    loaded and quantized — the most expensive possible place for a typo.
    """
    fn = _quantize_fn()
    body_assigns = [
        n
        for n in fn.body  # TOP level of the function body only — not nested in any branch
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "blocks" for t in n.targets)
    ]
    assert body_assigns, (
        "`blocks` is no longer assigned at the top level of quantize_qwen_edit's body. The banner "
        "reads it after the block loop; a conditional binding makes the text-encoder pass "
        "(no transformer_blocks) raise NameError after the component is already loaded."
    )


def test_the_quantize_banner_actually_prints_on_a_blockless_component(capsys) -> None:
    """Executed, not merely parsed — the blockless branch really renders a banner.

    Drives the real function with ``optimum.quanto`` stubbed at the two names it imports
    function-locally, so this stays CPU-pure and does not need a 40 GiB component.
    """
    quanto = pytest.importorskip("optimum.quanto")
    from signet_trainer.models import qwen_edit_loader as L

    class Blockless:
        """No ``transformer_blocks`` — the text-encoder shape."""

    captured: dict[str, object] = {}

    def fake_leaves(model, weights, *, what):
        captured["what"] = what
        return 0  # a pass that converted NOTHING must still report, and report zero

    monkey_freeze_called: list[object] = []
    orig_leaves = L._quantize_leaves
    orig_assert = L.assert_qwen_edit_not_peft_wrapped
    try:
        L._quantize_leaves = fake_leaves
        L.assert_qwen_edit_not_peft_wrapped = lambda model, *, what: None
        quanto_freeze = quanto.freeze
        quanto.freeze = lambda m: monkey_freeze_called.append(m)
        try:
            L.quantize_qwen_edit(Blockless(), what="text encoder", qtype="qfloat8")
        finally:
            quanto.freeze = quanto_freeze
    finally:
        L._quantize_leaves = orig_leaves
        L.assert_qwen_edit_not_peft_wrapped = orig_assert

    out = capsys.readouterr().out
    assert "[qwen-edit-quantize]" in out, f"no banner on stdout; got {out!r}"
    assert "text encoder" in out, "the banner must name WHICH component — both come through here"
    assert "qfloat8" in out, "the banner must name the locked qtype"
    assert "0 extra module(s)" in out, (
        f"a zero-conversion pass must print a ZERO so it is distinguishable from a full one; "
        f"got {out!r}"
    )
    assert monkey_freeze_called, "freeze() was not called — the component was left unmaterialised"


# --------------------------------------------------------------------------------------------------
# (2) The column -> file join has exactly ONE transcription.
# --------------------------------------------------------------------------------------------------


def _qwen_edit_sample_fn() -> ast.FunctionDef:
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    fns = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qwen_edit_sample"
    ]
    assert len(fns) == 1, f"expected exactly one qwen_edit_sample in fns.py, found {len(fns)}"
    return fns[0]


def test_the_sample_stage_composes_its_cell_path_only_from_qwen_edit_cell_relpath() -> None:
    """The render's output path is built by ``qwen_edit_cell_relpath`` and by nothing else.

    ``inference/grid._qwen_edit_block`` reads ``row[column.row_key]``; the sampler writes under
    ``column.render_subdir``. Nothing in ``src/`` joined those two until ``qwen_edit_cell_relpath``
    landed, and the HTML surface was proved against a path shape the render never produces. A SECOND
    spelling inside the stage re-opens exactly that hole, and it fails in the most expensive way
    available: every tile falls back to "generation failed" on renders that actually succeeded, on
    the metered box, after the A100 time is already spent.
    """
    fn = _qwen_edit_sample_fn()
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "qwen_edit_cell_relpath"
    ]
    assert len(calls) == 1, (
        f"expected exactly ONE qwen_edit_cell_relpath call in qwen_edit_sample, found "
        f"{len(calls)} at lines {[c.lineno for c in calls]}. One is the join; two is a fork."
    )


def test_the_sample_stage_does_not_hand_build_a_cell_path_from_render_subdir() -> None:
    """No f-string or ``/``-join in the stage reassembles a cell path out of the column's parts.

    The failure this blocks is not a crash — it is a plausible path that disagrees with the one the
    grid reads. Any expression naming ``render_subdir`` outside the single relpath call is that
    fork appearing again.
    """
    fn = _qwen_edit_sample_fn()
    relpath_lines = {
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "qwen_edit_cell_relpath"
    }
    strays = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Attribute)
        and n.attr == "render_subdir"
        and n.lineno not in relpath_lines
    ]
    assert not strays, (
        f"`render_subdir` is referenced outside the single qwen_edit_cell_relpath call at line(s) "
        f"{strays}. The column -> file join has exactly one transcription; a second one is how the "
        f"grid ends up pointing at files the render never wrote."
    )


def test_the_sample_stage_never_spells_guidance_scale() -> None:
    """``true_cfg_scale``, never ``guidance_scale`` — the latter renders at effective CFG 1.0.

    ``guidance_embeds=False`` on this transformer means the pipeline logs "ignored... not
    guidance-distilled", sets ``guidance=None``, and renders with CFG off: the negative encode never
    runs, the reference stops feeding both encode nodes, the CFG-norm rescale never fires, and the
    output is the muddy render METHOD §8 says reads as a bad adapter.
    """
    fn = _qwen_edit_sample_fn()
    # AST-precise, deliberately. A naive line scan flags the function's own DOCSTRING, which spends
    # three lines explaining why validation.guidance_scale is not read — prose that argues for the
    # rule is not a violation of it. (This assertion was written the naive way first and fired on
    # exactly those three lines; the finding was the test's, not the code's.)
    offenders: list[int] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "guidance_scale":
            offenders.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "guidance_scale":
            offenders.append(node.lineno)
        elif isinstance(node, ast.keyword) and node.arg == "guidance_scale":
            offenders.append(getattr(node.value, "lineno", fn.lineno))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # a string LITERAL naming the field is a config read by another name — but the
            # docstring is itself a Constant, so exclude the docstring node specifically.
            if node.value == "guidance_scale":
                offenders.append(node.lineno)
    assert not offenders, (
        f"qwen_edit_sample names `guidance_scale` in executable code at line(s) {offenders}. "
        f"The render path takes true_cfg_scale; guidance_scale disables CFG."
    )


# --------------------------------------------------------------------------------------------------
# (3) The scheduler pin is reached on EVERY render, not on a branch.
# --------------------------------------------------------------------------------------------------


def test_the_scheduler_gate_precedes_the_pipeline_call_unconditionally() -> None:
    """``assert_qwen_edit_scheduler_pinned`` runs at the top level of the render, before ``pipeline(``.

    This is THE documented trap on this architecture: an unpinned scheduler produces a plausible
    IMAGE rather than an exception, and an image gets judged. If the gate ever moved inside a branch
    — or after the call — a muddy band would be indistinguishable from a bad adapter, and the
    diagnosis would condemn 5000 steps of A100 time for a settings bug.
    """
    pipeline_src = (
        _REPO / "src" / "signet_trainer" / "models" / "qwen_edit_pipeline.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(pipeline_src)
    gen = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "qwen_edit_generate"
    ]
    assert len(gen) == 1, "expected exactly one qwen_edit_generate"
    fn = gen[0]

    gate_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "assert_qwen_edit_scheduler_pinned"
    ]
    assert gate_lines, "the scheduler pin assertion is GONE from qwen_edit_generate"

    # the gate must be a statement in the function's own body, not nested in an if/try/loop
    top_level = {
        n.value.lineno
        for n in fn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "assert_qwen_edit_scheduler_pinned"
    }
    assert top_level, (
        "the scheduler gate is no longer an unconditional top-level statement in "
        "qwen_edit_generate. A gate on a branch is not a gate."
    )

    render_lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "pipeline"
    ]
    assert render_lines, "no `pipeline(...)` call found in qwen_edit_generate"
    assert min(top_level) < min(render_lines), (
        f"the scheduler gate (line {min(top_level)}) does not precede the render call "
        f"(line {min(render_lines)}). It must run BEFORE the first sigma is drawn."
    )
