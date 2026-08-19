"""Plan 10-12 — the H3 DISPATCH SEAM: source-order gate scan + the self-deriving params guard.

Two questions, both answered against the real sources and never against a remembered answer:

1. **Does every H3 dispatch sit behind the single gate?** ``h3_train`` / ``h3_sample`` /
   ``h3_preprocess`` each cost real A100 minutes, so each ``.spawn(`` must appear strictly AFTER
   ``guardrail_check`` -> ``format_cost_line`` -> the blocking ``_require_approval`` pause. The
   accepted ``--mode`` set must stay at its six existing values: H3 routes on ``model.family``
   INSIDE the existing arms, adding no mode. That last claim is asserted HERE as well as in
   ``tests/test_skill_entrypoint_coverage.py``, deliberately — a future mode addition should fail in
   TWO places, because the second place is the one that explains *why* the set is closed.

2. **Does ``_h3_encode_params`` supply every required parameter of ``h3_preprocess``?** This is the
   ``download_image`` import-closure lesson (``modal/app.py``'s INVARIANT banner: *"a passing local
   gate proves nothing about the container"*) applied to the DISPATCH seam. The answer is
   **re-derived from both real signatures** and diffed — never written down twice — in the spirit of
   ``tests/test_backup_restore_fns.py``'s self-deriving closure gate, and a NEGATIVE CONTROL proves
   the checker actually fails when a parameter is missing (without it, a silently broken AST walk
   passes forever, which is the same class of defect the whole file exists to catch).

``signet_trainer.modal.fns`` cannot be imported without the ``modal`` SDK (and importing it builds the
app graph and eagerly resolves every ``Secret.from_name``), so BOTH sides are recovered by
AST-parsing the source as TEXT — the same static technique ``test_skill_entrypoint_coverage.py`` uses
to read the entrypoint's CLI surface.

Pure CPU: no ``modal``, no GPU, no network, no filesystem beyond reading two source files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _REPO / "src" / "signet_trainer" / "modal" / "entrypoint.py"
_FNS = _REPO / "src" / "signet_trainer" / "modal" / "fns.py"

#: The three H3 Modal stages the entrypoint routes to by ``model.family``.
_H3_DISPATCHES = ("h3_train", "h3_sample", "h3_preprocess")

#: The lean-threading helper that turns the validated config into ``h3_preprocess``'s kwargs.
_H3_PARAMS_HELPER = "_h3_encode_params"

#: The six ``--mode`` values that existed BEFORE Phase 10 and must still be the whole set after it.
_EXPECTED_MODES = frozenset({"train", "sample", "preprocess", "fuse", "restore", "backup"})


# --------------------------------------------------------------------------------------------------
# Source access
# --------------------------------------------------------------------------------------------------


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose cannot satisfy a source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _entrypoint_code() -> str:
    return _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))


def _find_function(src: str, name: str) -> ast.FunctionDef:
    """The ``FunctionDef`` named ``name``, parsed out of ``src``. Fails loud when it is gone."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"no function named {name!r} in the parsed source — it was renamed or deleted; this gate "
        f"describes nothing until the name is updated."
    )


def _function_source(src: str, name: str) -> str:
    """The comment/docstring-stripped source of ONE function, sliced by its AST line span."""
    node = _find_function(src, name)
    lines = src.splitlines()
    end = node.end_lineno or node.lineno
    return _strip_comments_and_docstrings("\n".join(lines[node.lineno - 1 : end]))


def _dispatch_re(fn: str) -> re.Pattern[str]:
    """``fn[.with_options(...)].spawn(`` anchored so ``h3_train`` never matches as ``train``.

    ``tests/test_entrypoint_gate.py``'s original pattern is substring-prone: ``h3_train.spawn(``
    literally contains ``train.spawn(``, so the three LTX checks would silently pass on the H3
    dispatches (and vice versa) and the six checks would not be six distinct claims. The
    ``(?<![\\w.])`` lookbehind makes each one genuinely about its own function. The
    ``\\(\\s*(?!\\))`` tail is the second half: a real dispatch passes an argument, while the arm's
    APPROVED print writes ``<fn>.spawn()`` with EMPTY parens — and string literals survive the
    comment strip, so without it a deleted dispatch is satisfied by its own log message.

    D-10-DEF-17 swapped the verb ``.remote`` -> ``.spawn`` (SYNC -> ASYNC). Everything this pattern
    guards is about anchoring and arguments, not about which verb it is.
    """
    return re.compile(rf"(?<![\w.]){re.escape(fn)}(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*(?!\))")


def _last_index(code: str, pattern: str) -> int:
    matches = list(re.finditer(pattern, code))
    assert matches, f"entrypoint.py must contain {pattern!r}"
    return matches[-1].start()


# --------------------------------------------------------------------------------------------------
# AST positions — because comment-stripping does NOT reach string literals
# --------------------------------------------------------------------------------------------------
#
# ⚠ Found by the mutation probe run against this file before it was committed: DELETING the
# ``h3_train...spawn(config_text)`` dispatch entirely left every text-based order assertion GREEN.
# The reason is the arm's own APPROVED print, which contains the sentence "Dispatching
# h3_train.spawn() (gated, family: h3)" — ``_strip_comments_and_docstrings`` removes comments and
# triple-quoted blocks but NOT ordinary string literals, so the log message satisfied the scan with
# the call gone. Same class 10-11 hit ("a log message naming a helper satisfies a text scan with the
# call deleted"), and exactly the failure this file exists to prevent: a gate that describes a
# dispatch that is not there.
#
# So the three order claims below run on the AST. A string can no longer stand in for a call.


#: Every verb that hands work to a Modal container. ``remote`` is SYNC
#: (``FUNCTION_CALL_INVOCATION_TYPE_SYNC``) and ``spawn`` is ASYNC
#: (``FUNCTION_CALL_INVOCATION_TYPE_ASYNC``); D-10-DEF-17 moved this repo from the first to the
#: second. The GATE claims below are verb-agnostic ON PURPOSE — MODL-02/03 must hold for ANY
#: dispatch verb, so the receiver-blind scan matches both rather than tracking whichever one is
#: current. (Which verb the arms must actually USE is pinned separately, in
#: ``tests/test_dispatch_is_spawned.py``.)
_DISPATCH_VERBS = frozenset({"remote", "spawn"})


def _root_name(node: ast.expr) -> str | None:
    """The base Name of an attribute/call chain: ``h3_train.with_options(...).spawn`` -> h3_train."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def _remote_call_pos(name: str) -> tuple[int, int] | None:
    """``(lineno, col_offset)`` of the dispatch CALL whose receiver chain roots at ``name``."""
    tree = ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))
    positions = [
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _DISPATCH_VERBS
        and _root_name(node.func.value) == name
    ]
    return min(positions) if positions else None


def _gate_marker_pos(name: str) -> tuple[int, int]:
    """``(lineno, col_offset)`` of the LAST call to ``name`` — the gate step every dispatch follows."""
    tree = ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))
    positions = [
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _root_name(node.func) == name
    ]
    assert positions, f"entrypoint.py must CALL {name}() — not merely mention it"
    return max(positions)


# --------------------------------------------------------------------------------------------------
# (1) The gate — every H3 dispatch is behind the cost print and the blocking approval
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", _H3_DISPATCHES)
def test_h3_dispatch_exists_as_a_real_call(fn: str) -> None:
    """Each H3 stage is REALLY dispatched — an AST call, not a sentence in a log line."""
    assert _remote_call_pos(fn) is not None, (
        f"entrypoint.py contains no {fn}.spawn() CALL. H3-07 routes preprocess/train/sample by "
        f"model.family inside the EXISTING arms, so all three H3 stages must be reachable from the "
        f"gate. (Checked on the AST: the arm's APPROVED print mentions '{fn}.spawn()' in prose and "
        f"would satisfy a text scan with the call deleted.)"
    )


@pytest.mark.parametrize("fn", _H3_DISPATCHES)
def test_h3_dispatch_follows_the_blocking_approval(fn: str) -> None:
    """MODL-02: an H3 ``.spawn(`` may never precede ``_require_approval`` — no auto-launch."""
    dispatch = _remote_call_pos(fn)
    approval = _gate_marker_pos("_require_approval")
    assert dispatch is not None and dispatch > approval, (
        f"{fn}.spawn() must appear AFTER the _require_approval gate (MODL-02: a metered run can "
        f"never auto-launch) — approval@{approval} vs {fn}.spawn@{dispatch}"
    )


@pytest.mark.parametrize("fn", _H3_DISPATCHES)
def test_h3_dispatch_follows_the_cost_print(fn: str) -> None:
    """MODL-03: the operator sees the priced line before an H3 container is ever requested."""
    dispatch = _remote_call_pos(fn)
    cost = _gate_marker_pos("format_cost_line")
    assert dispatch is not None and dispatch > cost, (
        f"format_cost_line must be called BEFORE {fn}.spawn() (MODL-03) — an H3 run reuses the "
        f"same ledger and the same session cap as every other metered dispatch."
    )


@pytest.mark.parametrize("fn", _H3_DISPATCHES)
def test_h3_dispatch_follows_the_guardrail(fn: str) -> None:
    """The cost/guardrail computation precedes every H3 dispatch (MODL-03)."""
    dispatch = _remote_call_pos(fn)
    guardrail = _gate_marker_pos("guardrail_check")
    assert dispatch is not None and dispatch > guardrail, (
        f"guardrail_check must run before {fn}.spawn() (MODL-03)"
    )


def test_NO_remote_call_of_any_receiver_precedes_the_approval() -> None:
    """MODL-02, stated name-agnostically: **nothing** dispatches before the approval pause.

    The per-name checks above each answer "is THIS stage gated?". This one answers the question that
    actually matters — "can ANYTHING spend before the operator says yes?" — and it is strictly
    stronger, because a per-name scan is blind to a dispatch made through an alias
    (``from ... import h3_sample as _early; _early.spawn(...)``). The mutation probe for this plan
    found exactly that hole; a name roster can never close it, so the roster is not the assertion.

    It is also VERB-blind (``_DISPATCH_VERBS``): D-10-DEF-17 moved every arm from ``.remote`` to
    ``.spawn``, and a guard that tracked only the current verb would have been silently defeated by
    exactly that change. MODL-02 must hold for ANY way of handing work to a container.
    """
    tree = ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))
    approval = _gate_marker_pos("_require_approval")
    early = [
        (node.lineno, node.col_offset, ast.unparse(node.func))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _DISPATCH_VERBS
        and (node.lineno, node.col_offset) < approval
    ]
    assert not early, (
        f"a dispatch ({'/'.join(sorted(_DISPATCH_VERBS))}) appears BEFORE the blocking "
        f"_require_approval pause (approval@{approval}): {early}. MODL-02 is absolute, "
        f"receiver-blind AND verb-blind — a metered run can never auto-launch, whatever the "
        f"dispatching name happens to be bound to or which verb hands it over."
    )


def test_the_six_h3_and_ltx_dispatches_are_distinct_matches() -> None:
    """The anchored pattern must not let ``train`` and ``h3_train`` match each other's dispatch.

    Without the ``(?<![\\w.])`` lookbehind this whole file is a false green: ``h3_train.spawn(``
    contains ``train.spawn(``, so a missing LTX arm would be "found" by the H3 one.
    """
    code = _entrypoint_code()
    starts = {}
    for fn in ("train", "sample", "preprocess", *_H3_DISPATCHES):
        match = _dispatch_re(fn).search(code)
        assert match is not None, f"no anchored dispatch found for {fn}"
        starts[fn] = match.start()
    assert len(set(starts.values())) == 6, (
        f"the six dispatches must be six DISTINCT source positions, got {starts} — the anchored "
        f"pattern is matching a substring (h3_train.spawn( contains train.spawn()."
    )


def test_the_mode_surface_is_unchanged_by_the_h3_leg() -> None:
    """H3 adds NO ``--mode`` value: it routes on ``model.family`` inside the existing arms.

    Asserted here as well as in ``tests/test_skill_entrypoint_coverage.py`` on purpose. That file
    guards skill<->code drift; THIS one records the H3-specific reason the set is closed —
    ``test_real_entrypoint_modes_are_all_documented`` requires every real mode to be documented in a
    lifecycle skill, so an ``h3_*`` mode would break the coverage guard as well as this one.
    """
    main_fn = _find_function(_ENTRYPOINT.read_text(encoding="utf-8"), "main")
    modes: set[str] = set()
    for node in ast.walk(main_fn):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "mode"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.comparators[0], ast.Tuple | ast.List | ast.Set)
        ):
            for elt in node.comparators[0].elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    modes.add(elt.value)
    assert modes == set(_EXPECTED_MODES), (
        f"the accepted --mode set is {sorted(modes)}, expected {sorted(_EXPECTED_MODES)}. H3 must "
        f"NOT add a mode value: tests/test_skill_entrypoint_coverage.py pins this exact set AND "
        f"requires every real mode to be documented in a lifecycle skill, and H3-07 specifies "
        f"--mode preprocess|train|sample. Route on model.family inside the existing arms."
    )


def test_the_params_helper_is_called_before_the_dispatch_not_inside_it() -> None:
    """``_h3_encode_params(cfg)`` is evaluated BEFORE ``.spawn(``, never as an inline argument.

    The burned-gate class this prevents: a field read that raises ``AttributeError`` (``extra=
    "forbid"``) evaluated inside the dispatch expression is indistinguishable, at a glance, from one
    evaluated before it — but only the second can be reasoned about, and only the second keeps the
    failure a $0 abort with an actionable message.
    """
    code = _entrypoint_code()
    call_idx = _last_index(code, rf"{_H3_PARAMS_HELPER}\s*\(\s*cfg\s*\)")
    preprocess_idx = _dispatch_re("h3_preprocess").search(code)
    assert preprocess_idx is not None, "no h3_preprocess dispatch found"
    assert call_idx < preprocess_idx.start(), (
        f"{_H3_PARAMS_HELPER}(cfg) must be evaluated before h3_preprocess.spawn(), not inside it"
    )
    dispatch_line = code[preprocess_idx.start() : preprocess_idx.start() + 400]
    assert _H3_PARAMS_HELPER not in dispatch_line.split("\n")[0], (
        f"{_H3_PARAMS_HELPER} appears inline in the h3_preprocess dispatch expression — bind it to "
        f"a local first so a field-read failure is a readable abort rather than a traceback out of "
        f"a .spawn() call"
    )


def test_the_ltx_dispatch_arms_are_untouched() -> None:
    """The three LTX arms keep their exact dispatch forms — H3 widened nothing about them."""
    code = _entrypoint_code()
    assert re.search(r"(?<![\w.])train\.spawn\s*\(\s*config_text\s*\)", code), (
        "the LTX train arm must still be a bare train.spawn(config_text)"
    )
    assert re.search(
        r"(?<![\w.])sample\.with_options\(timeout=sample_timeout_s\)\.spawn\s*\(\s*config_text\s*\)",
        code,
    ), "the LTX sample arm must still dispatch via sample.with_options(timeout=...).spawn(config_text)"
    assert re.search(r"(?<![\w.])preprocess\.with_options\(timeout=preprocess_timeout_s\)\.spawn", code), (
        "the LTX preprocess arm must still dispatch via preprocess.with_options(timeout=...).spawn"
    )


# --------------------------------------------------------------------------------------------------
# (2) The SELF-DERIVING params-completeness guard
# --------------------------------------------------------------------------------------------------


def _required_params(fn: ast.FunctionDef) -> set[str]:
    """Every parameter of ``fn`` that has NO default — derived from the signature, never listed."""
    positional = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args]]
    n_defaults = len(fn.args.defaults)
    required = set(positional[: len(positional) - n_defaults] if n_defaults else positional)
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True):
        if default is None:
            required.add(arg.arg)
    return required - {"self", "cls"}


def _supplied_keys(entrypoint_src: str, helper_name: str) -> set[str]:
    """Every string KEY of every dict literal the params helper returns — again, derived."""
    helper = _find_function(entrypoint_src, helper_name)
    keys: set[str] = set()
    for node in ast.walk(helper):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _missing_params(
    fns_src: str, entrypoint_src: str, fn_name: str, helper_name: str
) -> list[str]:
    """Required parameters of ``fn_name`` that ``helper_name`` does not supply. The whole check.

    Factored out so the SAME code path runs against the real sources (expecting empty) and against a
    synthetic signature carrying an extra required parameter (expecting that name back). A checker
    that is only ever driven with the passing input is not a checker.
    """
    required = _required_params(_find_function(fns_src, fn_name))
    supplied = _supplied_keys(entrypoint_src, helper_name)
    return sorted(required - supplied)


def test_h3_preprocess_required_params_are_all_supplied_by_the_helper() -> None:
    """Every required ``h3_preprocess`` parameter appears as a key in ``_h3_encode_params``."""
    missing = _missing_params(
        _FNS.read_text(encoding="utf-8"),
        _ENTRYPOINT.read_text(encoding="utf-8"),
        "h3_preprocess",
        _H3_PARAMS_HELPER,
    )
    assert not missing, (
        f"{_H3_PARAMS_HELPER} does not supply required h3_preprocess parameter(s) {missing}. An "
        f"unsupplied required parameter is a TypeError at dispatch (best case) or a silently wrong "
        f"default (worst case, if someone adds a default to make the error go away) — and either "
        f"way it is discovered in a PAID container after every local gate has already passed."
    )


_SYNTHETIC_FNS_WITH_AN_EXTRA_REQUIRED_PARAM = '''
@app.function(gpu="A100-80GB", image=h3_gpu_image)
def h3_preprocess(
    metadata_path: str,
    output_dir: str,
    a_brand_new_required_knob: int,
) -> str:
    """A synthetic stand-in: same shape, one parameter nobody threaded."""
    return ""
'''


def test_negative_control_an_unthreaded_required_param_is_caught() -> None:
    """The checker must FAIL when a required parameter is missing — proven, not assumed.

    Drives ``_missing_params`` (the real comparison, not a copy of it) with a synthetic ``fns.py``
    fragment that grew a required parameter the helper knows nothing about. Without this, a silently
    broken AST walk would return an empty set forever and the test above would be a rubber stamp.
    """
    missing = _missing_params(
        _SYNTHETIC_FNS_WITH_AN_EXTRA_REQUIRED_PARAM,
        _ENTRYPOINT.read_text(encoding="utf-8"),
        "h3_preprocess",
        _H3_PARAMS_HELPER,
    )
    assert missing == ["a_brand_new_required_knob"], (
        f"the params-completeness checker did not catch an unthreaded required parameter "
        f"(returned {missing}) — the AST walk is broken, so the real assertion above is vacuous."
    )


def test_max_packed_rows_is_supplied_and_COMPUTED_not_a_literal() -> None:
    """``max_packed_rows`` must be threaded, and derived from the config's budget triple.

    Called out by name because it arms ``build_h3_packed_batch``'s runtime ceiling assertion — the
    THIRD layer of the budget defense, after the config-load worst-pair check and the dry-run
    refusal. Unthreaded it defaults to ``None`` and that guard is dead code while every other test
    in the suite still passes.
    """
    entrypoint_src = _ENTRYPOINT.read_text(encoding="utf-8")
    supplied = _supplied_keys(entrypoint_src, _H3_PARAMS_HELPER)
    assert "max_packed_rows" in supplied, (
        f"{_H3_PARAMS_HELPER} does not supply max_packed_rows — build_h3_packed_batch's runtime "
        f"ceiling assertion then compares against None and never fires (dead code, green suite)."
    )

    helper = _find_function(entrypoint_src, _H3_PARAMS_HELPER)
    value_nodes = [
        value
        for node in ast.walk(helper)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "max_packed_rows"
    ]
    assert value_nodes, "no max_packed_rows entry found in the helper's returned dict"
    for value in value_nodes:
        assert isinstance(value, ast.Call), (
            "max_packed_rows must be COMPUTED, never a literal — a hardcoded ceiling stops "
            "tracking the config's gpu_usable_gib / resident_gib / mib_per_packed_row triple the "
            "moment an H200 escalation edits the YAML (D-NOHARDCODE)."
        )
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        assert name == "max_packed_rows_for_budget", (
            f"max_packed_rows is computed by {name!r}; conditioning/h3_geometry."
            f"max_packed_rows_for_budget is the single authority for that arithmetic."
        )

    body = _function_source(entrypoint_src, _H3_PARAMS_HELPER)
    assert "max_packed_rows_for_budget" in body, (
        "the helper body must call max_packed_rows_for_budget (the plan's one-liner check)"
    )


# --------------------------------------------------------------------------------------------------
# (3) The two config-text stages — the params-helper bug class cannot exist for them
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ("h3_train", "h3_sample"))
def test_config_text_stages_take_exactly_config_yaml(fn: str) -> None:
    """``h3_train`` / ``h3_sample`` take ``(config_yaml: str)`` and re-parse in-container.

    Pinned rather than assumed: by taking the recipe BY VALUE they have no params helper, so the
    omission bug the guard above exists for is impossible for them BY CONSTRUCTION — but only while
    the signature stays this shape. The day one of them grows a second parameter, it needs the same
    threading discipline and this test says so.
    """
    node = _find_function(_FNS.read_text(encoding="utf-8"), fn)
    args = [a.arg for a in [*node.args.posonlyargs, *node.args.args]]
    assert args == ["config_yaml"], (
        f"{fn} takes {args}, expected exactly ['config_yaml'] — a second parameter means the "
        f"entrypoint must thread it, which reintroduces the params-completeness bug class."
    )
    assert not node.args.kwonlyargs, f"{fn} grew keyword-only parameters: {node.args.kwonlyargs}"
    assert node.args.vararg is None and node.args.kwarg is None, (
        f"{fn} grew *args/**kwargs — the by-value contract is what makes it threading-proof"
    )


@pytest.mark.parametrize("fn", ("h3_train", "h3_sample"))
def test_config_text_stages_compute_their_own_ceiling(fn: str) -> None:
    """Both by-value stages derive ``max_packed_rows`` in-container rather than receiving one.

    ``h3_train`` needs it because ``build_h3_packed_batch``'s runtime ceiling assertion is the third
    layer of the budget defense. ``h3_sample`` needs it for the same reason on the fixed batch it
    builds to measure the adapter delta — a batch built without a ceiling can OOM the render before
    a single frame is produced, and the failure would look like a render problem.

    Asserted on the AST, not on text: 10-11's mutation probe established that a comment-stripped
    text scan still matches a helper NAME inside a log message, so a scan can go green with the call
    deleted. (This test was ADDED because the mutation probe for this plan replaced ``h3_sample``'s
    budget call and nothing went red — h3_train was pinned, h3_sample was not.)
    """
    node = _find_function(_FNS.read_text(encoding="utf-8"), fn)
    calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "max_packed_rows_for_budget")
            or getattr(n.func, "attr", "") == "max_packed_rows_for_budget"
        )
    ]
    assert calls, (
        f"{fn} must CALL max_packed_rows_for_budget on its re-validated config: it receives the "
        f"recipe by value, so it owns its own ceiling rather than trusting one shipped from the "
        f"local gate. Unset, the ceiling defaults to None and the guard is dead code."
    )


# --------------------------------------------------------------------------------------------------
# (4) Purity
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# (5) #39 finding 2 — pipeline_root_id required, pre-approval, for family: h3 --mode sample
# --------------------------------------------------------------------------------------------------


def test_h3_sample_refuses_a_missing_pipeline_root_id_pre_approval() -> None:
    """The ONLY prior enforcement was ``fns.py:h3_sample``'s in-container RuntimeError, AFTER the
    ~61.7 GiB arch gate — a dispatch paid for a refusal that costs nothing locally. This asserts the
    free, load-time twin exists and fires strictly before ``_require_approval``."""
    source = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "cfg.model.pipeline_root_id" in source and "not cfg.model.pipeline_root_id" in source, (
        "entrypoint.py must check model.pipeline_root_id before dispatch — the field is documented "
        "(config/schema.py) as 'Required by --mode sample on a family: h3 config' but nothing at "
        "load asked for it"
    )
    refusal = source.index("config.model.pipeline_root_id is unset")
    # issue #37's session-cap ask-first downgrade (already merged) renamed the argument passed
    # here to `approve_for_gate` — match the CALL site ("not _require_approval(...)") generically
    # rather than pinning the argument name, so this test does not confuse the call with the
    # `def _require_approval(approve: bool)` definition earlier in the file.
    approval = source.index("not _require_approval(")
    assert refusal < approval, (
        "the pipeline_root_id refusal must fire BEFORE the approval gate (zero-spend), exactly "
        "like the sibling no-reference --mode sample refusal it sits beside"
    )


def _compact(source: str) -> str:
    """Collapse adjacent string-literal concatenation + whitespace, so a message split across
    several ``"..."`` lines can be matched as the single sentence it becomes at runtime — the same
    technique ``test_the_alpha_banner_is_printed_at_the_front_of_a_no_ref_dispatch`` uses."""
    return re.sub(r"\s+", " ", source.replace('"', ""))


def test_h3_pipeline_root_id_gate_message_matches_h3_sample_verbatim() -> None:
    """#39's proposed direction: 'reuse fns.py:4513's message text verbatim so the operator sees the
    same instruction either way' — pinned as a shared sentence across both refusals."""
    entry_compact = _compact(_ENTRYPOINT.read_text(encoding="utf-8"))
    fns_compact = _compact(_FNS.read_text(encoding="utf-8"))
    shared = (
        "config.model.pipeline_root_id is unset. The render needs the pipeline "
        "ROOT (the dir holding model_index.json and every component partition); "
        "`model.model_id` names the transformer PARTITION inside it and means something "
        "different to h3_train / h3_loader, so it is NOT reused here (D-10-DEF-14)."
    )
    assert shared in entry_compact, "the entrypoint's pre-approval refusal text has drifted from fns.py"
    assert shared in fns_compact, "fns.py's in-container RuntimeError text has drifted from the entrypoint"


def test_h3_pipeline_root_id_gate_is_scoped_to_sample_and_h3_only() -> None:
    """The gate must not fire for train/preprocess (pipeline_root_id is unused there, per
    schema.py's own field doc) nor for any other family — only mode == 'sample' and family == 'h3'."""
    main_fn = _find_function(_ENTRYPOINT.read_text(encoding="utf-8"), "main")
    guarded = [
        node
        for node in ast.walk(main_fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "config.model.pipeline_root_id is unset" in n.value
            for n in ast.walk(node)
        )
    ]
    assert guarded, "no `if` guarding the pipeline_root_id refusal was found"
    condition_src = ast.unparse(guarded[0].test)
    assert "mode ==" in condition_src and "'sample'" in condition_src.replace('"', "'"), (
        f"the guard condition ({condition_src!r}) must be scoped to mode == 'sample'"
    )
    assert "family" in condition_src and "h3" in condition_src, (
        f"the guard condition ({condition_src!r}) must be scoped to model.family == 'h3'"
    )


def test_this_gate_imports_no_modal() -> None:
    """The whole file must stay CPU-pure — importing modal would break the invariant it asserts."""
    src = Path(__file__).read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        assert not re.match(r"\s*(import\s+modal\b|from\s+modal\b)", line), (
            f"line {lineno} imports modal — this gate is a source scan and must never need the SDK"
        )
