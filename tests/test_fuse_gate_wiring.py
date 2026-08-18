"""PR-6 — fused-base gate wiring + fuse-dispatch honesty (static source-scan, zero modal import).

Same discipline as ``tests/test_backup_restore_fns.py``: these tests READ ``modal/fns.py`` source
and assert the money-safety invariants WITHOUT importing it (importing pulls ``modal`` and would
build the app graph).

Asserted invariants (audit findings gap-fuse-1 / gap-fuse-2 / gap-fuse-3):
  * ``train`` and ``sample`` both run the ``verify_fused_metadata`` pre-load gate — and run it
    STRICTLY BEFORE ``load_ltxv_components`` — so a damaged/truncated/unfused artifact under a
    ``-fused`` model_id dies as a named RuntimeError in the container's first seconds, never a
    raw SafetensorError after the 44GB mount (pre-fix: ZERO consuming-path call sites existed).
  * ``fuse`` passes ``overwrite=config.fuse.allow_overwrite`` (the config-gated house-rule-6
    opt-in) and no longer prints the hardcoded ``strength=1.0`` literal — the log interpolates
    the same value the call passes.
  * ``FuseConfig.allow_overwrite`` defaults False (every existing YAML loads unchanged).

Gate-wiring verification note (post-audit fix): the ORIGINAL version of the gate-wiring test below
did ``"verify_fused_metadata" in body`` on the raw source TEXT of the function. That substring
check is satisfied by the doc-comment above the gate block (which names ``verify_fused_metadata``
in prose) even after the actual ``if is_fused_base_filename(...): ... verify_fused_metadata(...)``
statements are deleted — a comment is part of the source text ``ast.get_source_segment`` returns,
even though it is not part of the AST. The rewrite below walks the parsed AST for a REAL ``Call``
node (never a comment, docstring, or bare import) invoking ``verify_fused_metadata``, confirms it
sits inside an ``if`` block actually gated on a ``is_fused_base_filename(...)`` call, and orders it
against the real ``load_ltxv_components`` ``Call`` node by line number — deleting the gate now
fails the test (see the mutation check recorded in the PR-6 commit message).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from signet_trainer.config.schema import FuseConfig, SignetConfig

FNS_PATH = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"


def _fn_source(name: str) -> str:
    source = FNS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"function {name!r} not found in {FNS_PATH}")


def _fn_node(name: str) -> ast.FunctionDef:
    source = FNS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {FNS_PATH}")


def _call_name(node: ast.AST) -> str | None:
    """The bare/attribute name an ``ast.Call`` node invokes (``foo(...)`` or ``mod.foo(...)`` both

    resolve to ``'foo'``); ``None`` for anything that is not a Call at all — imports, comments,
    and docstrings never produce a Call node, so they can never satisfy the checks below (the
    exact gap that let the pre-fix text-substring version pass with the gate deleted).
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls_named(root: ast.AST, target: str) -> list[ast.Call]:
    """Every real ``Call`` node under ``root`` invoking ``target`` — AST-level, not text search."""
    return [n for n in ast.walk(root) if _call_name(n) == target]


def _if_blocks_guarded_on(fn: ast.FunctionDef, guard_call_name: str) -> list[ast.If]:
    """Every ``if`` node in ``fn`` whose CONDITION itself calls ``guard_call_name``."""
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.If) and _calls_named(n.test, guard_call_name)
    ]


@pytest.mark.parametrize("fn_name", ["train", "sample"])
def test_consuming_fns_run_the_fused_base_gate_before_model_load(fn_name: str) -> None:
    fn = _fn_node(fn_name)

    gate_calls = _calls_named(fn, "verify_fused_metadata")
    assert gate_calls, (
        f"{fn_name}() no longer CALLS verify_fused_metadata — a comment or import mentioning the "
        "name does not count. The advertised pre-dispatch gate would again have zero "
        "consuming-path call sites (gap-fuse-1)."
    )

    guard_ifs = _if_blocks_guarded_on(fn, "is_fused_base_filename")
    assert guard_ifs, (
        f"{fn_name}() must gate the verify behind an `if is_fused_base_filename(...):` block — "
        "the dev base has no fused marker and would fail a blanket verify."
    )

    # The gate call must sit INSIDE one of those guarded blocks (by line span), not merely
    # coexist somewhere else in the function alongside an unrelated `if`.
    assert any(
        guard.lineno <= call.lineno <= (guard.end_lineno or guard.lineno)
        for call in gate_calls
        for guard in guard_ifs
    ), (
        f"{fn_name}() calls verify_fused_metadata OUTSIDE the is_fused_base_filename guard — an "
        "unguarded verify would raise on every plain (non-fused) dev-base load."
    )

    load_calls = _calls_named(fn, "load_ltxv_components")
    assert load_calls, f"{fn_name}() no longer calls load_ltxv_components — precondition broken."

    gate_at = min(c.lineno for c in gate_calls)
    load_at = min(c.lineno for c in load_calls)
    assert gate_at < load_at, (
        f"{fn_name}() runs the fused-base gate AFTER load_ltxv_components — the whole point is "
        "to fail on the header BEFORE the 44GB model load."
    )


def test_fuse_dispatch_passes_config_gated_overwrite() -> None:
    body = _fn_source("fuse")
    assert "overwrite=config.fuse.allow_overwrite" in body, (
        "fuse() must gate scaffold replacement on the config knob (house rule 6, gap-fuse-2) — "
        "an ungated fuse_inoutpaint call silently clobbers the frozen scaffold in place."
    )


def test_fuse_dispatch_print_interpolates_the_real_strength() -> None:
    body = _fn_source("fuse")
    assert "strength=1.0" not in body, (
        "fuse() prints a hardcoded 'strength=1.0' literal — if DEFAULT_FUSE_STRENGTH ever "
        "changes, the log lies about what the call passed (gap-fuse-3)."
    )
    assert "strength={strength}" in body
    assert "strength=strength" in body  # the call passes the SAME value the print shows


def test_fuse_config_defaults_off_and_rides_signet_config() -> None:
    assert FuseConfig().allow_overwrite is False
    assert SignetConfig.model_fields["fuse"].default_factory is FuseConfig
    with pytest.raises(Exception):  # noqa: B017 — extra='forbid' contract, exact type is pydantic's
        FuseConfig(unknown_knob=True)
