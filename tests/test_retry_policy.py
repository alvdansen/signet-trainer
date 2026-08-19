"""issue #45 PR-2 — the cost gate must price what the launch AUTHORIZES: retries x bounded hours.

Pre-PR-2 the only spend quantity computed anywhere was ``hourly_rate_usd * est_hours`` — one
container life — while the dispatched decorators granted server-side retries (LTX ``train``:
``max_retries=10`` behind a config-derived shell = 11 lives x bounded hours x $1.64, $54.12
authorized behind a single-life print). Three welds close it, each pinned here:

  1. ``retry_policy.ARM_MAX_RETRIES`` mirrors every shipped ``@app.function`` retry budget —
     AST-welded below, so a decorator edit that forgets the table fails the suite, never the print;
  2. ``resolve_arm`` mirrors the entrypoint's mode/family dispatch ladder (all four families: ltx,
     h3, qwen_edit, wan);
  3. every arm the entrypoint can actually dispatch is present in the table (a missing key would
     ``KeyError`` the gate instead of pricing it).

The genuinely BEHAVIORAL half of this fix — that the entrypoint actually derives ``lives`` from this
table and prices the SAME ``bounded_hours`` the dispatched arm's ``.with_options(timeout=...)`` uses
— is tested by actually calling ``main()`` in ``test_entrypoint_gate_behavioral.py``, not by a source
regex here. A verifier previously found that gap real: a regex only proves the literal expression
appears in ``entrypoint.py``, not that the printed/guardrailed number and the dispatched timeout stay
welded together at runtime.

Pure CPU source/AST scans — NO modal import (Anti-Pattern 6), zero spend. Mirrors the
``_strip_comments_and_docstrings`` convention of test_entrypoint_gate.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

from signet_trainer.modal.retry_policy import ARM_MAX_RETRIES, resolve_arm

_ROOT = Path(__file__).resolve().parents[1]
_FNS = _ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _decorated_retry_budgets() -> dict[str, int]:
    """``{function_name: max_retries}`` for every ``@app.function``-decorated def in fns.py.

    Parsed via ``ast`` so the weld reads the number each decorator will actually construct. A
    decorator with no ``retries=`` kwarg budgets 0 (one container life).
    """
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    budgets: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("app.function")):
                continue
            budget = 0
            for kw in dec.keywords:
                if kw.arg != "retries":
                    continue
                for inner in ast.walk(kw.value):
                    if (
                        isinstance(inner, ast.keyword)
                        and inner.arg == "max_retries"
                        and isinstance(inner.value, ast.Constant)
                    ):
                        budget = int(inner.value.value)
            budgets[node.name] = budget
    return budgets


def test_arm_table_is_welded_to_every_shipped_decorator() -> None:
    """Every arm in ARM_MAX_RETRIES must carry EXACTLY that budget in its shipped decorator.

    This is the weld that lets the cost gate read the table instead of introspecting Modal
    function objects (no stable public surface): a decorator budget that drifts from the table
    fails HERE, so the printed estimate can never silently under-price a retry grant.
    """
    budgets = _decorated_retry_budgets()
    for arm, expected in ARM_MAX_RETRIES.items():
        assert arm in budgets, f"ARM_MAX_RETRIES names {arm!r} but fns.py ships no such @app.function"
        assert budgets[arm] == expected, (
            f"{arm}'s decorator budgets max_retries={budgets[arm]} but ARM_MAX_RETRIES says "
            f"{expected} — the cost gate would price the WRONG number of container lives (PR-2)"
        )


def test_every_dispatched_gpu_arm_is_in_the_table() -> None:
    """Every gated GPU dispatch across all four families must be priceable — an absent key would
    ``KeyError`` the gate instead of pricing it."""
    for arm in (
        "train",
        "sample",
        "preprocess",
        "h3_train",
        "h3_sample",
        "h3_preprocess",
        "qwen_edit_train",
        "qwen_edit_sample",
        "qwen_edit_preprocess",
        "wan_train",
    ):
        assert arm in ARM_MAX_RETRIES, f"{arm} missing from ARM_MAX_RETRIES — the cost gate cannot price it"


def test_resolve_arm_mirrors_the_dispatch_ladder() -> None:
    """(mode, family) -> function name, exactly as the entrypoint routes — all four families."""
    assert resolve_arm("train", "ltx") == "train"
    assert resolve_arm("train", "h3") == "h3_train"
    assert resolve_arm("train", "qwen_edit") == "qwen_edit_train"
    assert resolve_arm("train", "wan") == "wan_train"
    assert resolve_arm("sample", "ltx") == "sample"
    assert resolve_arm("sample", "h3") == "h3_sample"
    assert resolve_arm("sample", "qwen_edit") == "qwen_edit_sample"
    assert resolve_arm("preprocess", "ltx") == "preprocess"
    assert resolve_arm("preprocess", "h3") == "h3_preprocess"
    assert resolve_arm("preprocess", "qwen_edit") == "qwen_edit_preprocess"
    assert resolve_arm("fuse", "ltx") == "fuse"
    assert resolve_arm("restore", "ltx") == "restore"
    assert resolve_arm("backup", "ltx") == "backup_sync"
    # fuse/restore/backup route to their single CPU arm regardless of family.
    assert resolve_arm("fuse", "h3") == "fuse"
    assert resolve_arm("restore", "wan") == "restore"
    assert resolve_arm("backup", "qwen_edit") == "backup_sync"
    # wan serves exactly one supported dispatch (mode == "train"); every other mode is refused
    # before any spawn, but the cost print that runs BEFORE that refusal still needs an answer.
    assert resolve_arm("sample", "wan") == "wan_train"
    assert resolve_arm("preprocess", "wan") == "wan_train"
    # every resolvable (mode, family) pair must be priceable.
    for mode in ("train", "sample", "preprocess", "fuse", "restore", "backup"):
        for family in ("ltx", "h3", "qwen_edit", "wan"):
            assert resolve_arm(mode, family) in ARM_MAX_RETRIES


def test_wan_train_and_h3_sample_and_qwen_edit_sample_carry_no_retries() -> None:
    """Named directly — these three are the arms whose ZERO is deliberate, not an oversight.

    ``h3_sample`` / ``qwen_edit_sample``: re-dispatch resumes in-dir instead (D-10 resume).
    ``wan_train``: musubi's resume semantics are unverified here, so a retry over an unverified
    resume would not be a safety feature (fns.py's own decorator comment).
    """
    assert ARM_MAX_RETRIES["h3_sample"] == 0
    assert ARM_MAX_RETRIES["qwen_edit_sample"] == 0
    assert ARM_MAX_RETRIES["wan_train"] == 0


def test_lives_is_max_retries_plus_one() -> None:
    """The quantity the cost gate actually multiplies by — spelled out, not just implied."""
    for arm, max_retries in ARM_MAX_RETRIES.items():
        lives = max_retries + 1
        assert lives >= 1
        if max_retries == 10:
            assert lives == 11  # the train-shaped arms' worst-case container count.
