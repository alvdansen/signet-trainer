"""Issue #13 step 2 (verifier round 2) — every ``h3_draw_timesteps`` call site inside the H3 Modal
stages must pass ``std=config.training.timestep_std``, not the library default.

A mutation reverting that threading (dropping the ``std=`` keyword, or hardcoding ``std=1.0``, at
the per-step draw in ``h3_train`` and at ``validate_gate._make_schedule``) left every prior test
green: the schema tests only prove the field is *accepted* (``tests/test_h3_config_schema.py``),
the ``FlowMatchingSchedule`` tests only prove the *constructor default* is byte-identical
(``tests/test_flow_match.py``), and nothing exercised the wiring between the two. This file closes
that gap statically — parsing ``modal/fns.py`` without importing it (importing it builds the Modal
app graph and eagerly resolves every ``Secret.from_name``) — for the production call sites; the
sibling functional test for ``validate_gate._make_schedule`` lives in
``tests/test_validate_gate.py``.

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"

#: Every top-level function whose body (nested functions included — ``ast.walk`` descends into
#: ``h3_train``'s local ``_h3_step``) must thread ``std=config.training.timestep_std`` into any
#: ``h3_draw_timesteps`` call it makes.
_H3_DRAW_SITES = ("h3_train", "_h3_fixed_delta_batch")


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {path.name}")


def _draw_timesteps_calls(node: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        func = candidate.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name == "h3_draw_timesteps":
            calls.append(candidate)
    return calls


def _std_keyword_expr(call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg == "std":
            return ast.unparse(kw.value)
    return None


# --------------------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------------------


def test_every_h3_draw_timesteps_call_threads_config_timestep_std() -> None:
    """Every call site in the listed functions must pass ``std=config.training.timestep_std``.

    Dropping the keyword (falling back to the library default) or hardcoding a literal must FAIL
    this test — both are exactly the class of mutation that reverted issue #13's threading.
    """
    violations: list[str] = []
    for fn_name in _H3_DRAW_SITES:
        node = _function_node(_FNS, fn_name)
        calls = _draw_timesteps_calls(node)
        assert calls, f"{fn_name}() no longer calls h3_draw_timesteps — re-target this guard"
        for call in calls:
            expr = _std_keyword_expr(call)
            if expr != "config.training.timestep_std":
                violations.append(
                    f"{fn_name}() line {call.lineno}: std={expr!r}, "
                    "expected 'config.training.timestep_std'"
                )
    assert not violations, (
        "h3_draw_timesteps call site(s) do not thread config.training.timestep_std:\n  "
        + "\n  ".join(violations)
    )


def test_the_collector_finds_every_known_draw_site() -> None:
    """Non-vacuity: the known call-site count must not silently drop (D-10-DEF-1's failure shape)."""
    counts = {fn: len(_draw_timesteps_calls(_function_node(_FNS, fn))) for fn in _H3_DRAW_SITES}
    assert counts["h3_train"] == 3, (
        f"h3_train() should have 3 h3_draw_timesteps call sites (CPU preflight, per-step draw, "
        f"acceptance-signal draw); found {counts['h3_train']}. A guard over an empty/shrunk set "
        "is a guard over nothing."
    )
    assert counts["_h3_fixed_delta_batch"] == 1


# --------------------------------------------------------------------------------------------------
# Negative control — the guard is PROVEN to bite the reverted threading, not assumed to.
# --------------------------------------------------------------------------------------------------


def test_the_guard_would_go_red_on_a_reverted_per_step_draw() -> None:
    """Drive the identical detection over a MUTATED copy of ``h3_train``'s per-step draw.

    This reproduces exactly the reverted edit the verifier found survives: the per-step
    ``h3_draw_timesteps`` call inside ``h3_train``'s nested ``_h3_step`` loses its ``std=`` keyword
    (falling back to the unthreaded default), leaving the other two call sites in ``h3_train``
    untouched.
    """
    source = _FNS.read_text(encoding="utf-8")
    anchor = (
        'batch["t_video"], batch["t_audio"] = h3_draw_timesteps(\n'
        "            rng_, uniform_prob=config.training.uniform_prob, std=config.training.timestep_std\n"
        "        )"
    )
    assert anchor in source, "the mutation anchor no longer matches — re-target it"
    mutated = source.replace(
        anchor,
        'batch["t_video"], batch["t_audio"] = h3_draw_timesteps(\n'
        "            rng_, uniform_prob=config.training.uniform_prob\n"
        "        )",
        1,
    )
    assert mutated != source

    node = next(
        n
        for n in ast.walk(ast.parse(mutated))
        if isinstance(n, ast.FunctionDef) and n.name == "h3_train"
    )
    calls = _draw_timesteps_calls(node)
    violations = [c for c in calls if _std_keyword_expr(c) != "config.training.timestep_std"]
    assert violations, "the mutated (reverted) per-step draw MUST be reported — the guard is decorative otherwise"
