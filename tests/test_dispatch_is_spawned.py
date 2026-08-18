"""D-10-DEF-17 — the dispatch VERB pinned as STRUCTURE, derived from the real source and the real SDK.

**The defect.** Every gated run in this repo was handed to Modal with a blocking ``.remote()``,
which is ``FUNCTION_CALL_INVOCATION_TYPE_SYNC``. The invocation type is sent to the SERVER at call
creation, and Modal cancels in-flight SYNC inputs whose owning client disappears. On
2026-08-07T01:00-01:01Z that killed two healthy runs — a training round at step 300 and a
measurement render at denoise step 13 — at the moment the dispatching subagent returned.

**Why a print-wording assertion was not enough last time.** The repo already KNEW:
``.planning/research/PITFALLS.md:172`` prescribed ``.spawn(...)`` + ``modal run --detach`` in Phase
0, and a test docstring in ``test_entrypoint_timeout_and_prints.py`` even stated that "``.remote()``
is not itself detached" — and then pinned the PRINT WORDING instead of the PROPERTY. A knowledge
that lives in prose at a site does not survive; a knowledge encoded as structure does. So this file
pins the PROPERTY, and pins it by DERIVATION rather than restatement:

  * the expected dispatch COUNT is derived from the lazy ``from signet_trainer.modal.fns import X``
    imports inside ``main`` (one per arm) — never a hand-typed 9;
  * the set of spawned receiver names is derived and DIFFED against the set of imported names, so a
    new arm that forgets to spawn fails here automatically;
  * ``spawn``-is-ASYNC / ``remote``-is-SYNC is read off the INSTALLED SDK's own source, not off
    this file's prose.

Pure CPU: no GPU, no network, no Modal at import (the one test that needs the SDK imports it INSIDE
its own body, mirroring ``test_single_use_containers_is_a_real_modal_sdk_kwarg``).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"

#: The module the lazy per-arm imports name. One ``from <this> import <fn>`` per dispatch arm is
#: what makes the expected spawn count DERIVABLE instead of a literal somebody has to maintain.
_FNS_MODULE = "signet_trainer.modal.fns"

#: The config field carrying the bounded post-dispatch watch window (D-NOHARDCODE).
_WATCH_FIELD = "dispatch_watch_seconds"


# --------------------------------------------------------------------------------------------------
# Source access
# --------------------------------------------------------------------------------------------------


def _entrypoint_tree() -> ast.Module:
    return ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))


def _main() -> ast.FunctionDef:
    """The ``main`` FunctionDef. Fails LOUD if it was renamed — this file then describes nothing."""
    for node in ast.walk(_entrypoint_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError(
        "no function named 'main' in entrypoint.py — the gated launch seam was renamed or deleted, "
        "and every assertion in this file is vacuous until the name is updated."
    )


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


def _dispatch_calls(main: ast.FunctionDef, verb: str) -> list[ast.Call]:
    """Every ``<receiver>.<verb>(...)`` CALL inside ``main``, found on the AST (never on text).

    AST, not regex, on purpose: ``_strip_comments_and_docstrings`` (the convention the sibling gate
    scans use) removes comments and triple-quoted blocks but NOT ordinary string literals, and every
    arm's APPROVED print contains the sentence ``Dispatching <fn>.spawn() (gated…)``. A text scan can
    therefore go green with the call deleted. A ``Call`` node cannot be faked by a log message.
    """
    return [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == verb
    ]


def _lazily_imported_fns(main: ast.FunctionDef) -> list[str]:
    """Every name lazily imported from ``signet_trainer.modal.fns`` INSIDE ``main`` — one per arm."""
    names: list[str] = []
    for node in ast.walk(main):
        if isinstance(node, ast.ImportFrom) and node.module == _FNS_MODULE:
            names.extend(alias.asname or alias.name for alias in node.names)
    return names


# --------------------------------------------------------------------------------------------------
# (1) The verb
# --------------------------------------------------------------------------------------------------


def test_no_dispatch_arm_uses_blocking_remote() -> None:
    """⛔ ZERO ``.remote(`` calls in ``main`` — the SYNC verb is what killed two healthy runs."""
    offenders = [
        (node.lineno, ast.unparse(node.func)) for node in _dispatch_calls(_main(), "remote")
    ]
    assert not offenders, (
        f"a blocking .remote() dispatch is back in main(): {offenders}. `_Function.remote` "
        f"delegates to `_call_function`, which carries FUNCTION_CALL_INVOCATION_TYPE_SYNC, and the "
        f"invocation type is sent to the SERVER at call creation — Modal CANCELS in-flight SYNC "
        f"inputs whose owning client disappears. That is D-10-DEF-17: it killed the h3-embe training "
        f"round (ap-EnKczw0E1VpXKOfn4uSAzi, at step 300) and the measurement render "
        f"(ap-5m9xcvntkylahK4GNEUHmq, at denoise step 13) at 2026-08-07T01:00-01:01Z, the moment the "
        f"dispatching agent returned. Use .spawn(...) and watch the returned FunctionCall."
    )


def test_every_dispatch_arm_spawns_exactly_once() -> None:
    """The spawn COUNT and the spawn NAMES are both DERIVED, then diffed — never restated.

    A hand-typed "9" rots the first time an arm is added or removed. Deriving the expectation from
    the lazy per-arm imports means a NEW arm that forgets to spawn (or spawns twice) fails here
    automatically, with no test edit and nobody having to remember.
    """
    main = _main()
    imported = _lazily_imported_fns(main)
    spawns = _dispatch_calls(main, "spawn")

    assert imported, (
        f"main() lazily imports nothing from {_FNS_MODULE} — the derivation this test rests on is "
        f"gone, so the count below would be vacuous. Re-derive it before trusting this file."
    )
    assert len(spawns) == len(imported), (
        f"main() lazily imports {len(imported)} fn(s) from {_FNS_MODULE} ({sorted(imported)}) but "
        f"makes {len(spawns)} .spawn() call(s). Every gated arm imports its fn and dispatches it "
        f"exactly once; a mismatch means an arm was added without a dispatch, or one arm dispatches "
        f"twice."
    )

    spawned_names = {_root_name(node.func.value) for node in spawns}
    assert spawned_names == set(imported), (
        f"the SPAWNED receivers {sorted(n for n in spawned_names if n)} do not match the LAZILY "
        f"IMPORTED fns {sorted(set(imported))}. Both sides are derived from the real source, so a "
        f"difference is a real arm that imports a function and then dispatches something else — or "
        f"does not dispatch at all."
    )


def test_every_spawn_follows_the_approval_gate() -> None:
    """MODL-02 re-pinned on the NEW verb: every ``.spawn(`` is strictly after ``_require_approval``.

    On AST positions, not regex indices — a string literal cannot move a lineno.
    """
    main = _main()
    approvals = [
        (node.lineno, node.col_offset)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and _root_name(node.func) == "_require_approval"
    ]
    assert approvals, "main() must CALL _require_approval — the single gate (MODL-02)"
    gate = max(approvals)

    early = [
        (node.lineno, ast.unparse(node.func))
        for node in _dispatch_calls(main, "spawn")
        if (node.lineno, node.col_offset) < gate
    ]
    assert not early, (
        f"a .spawn() dispatch appears BEFORE the blocking _require_approval pause (gate@{gate}): "
        f"{early}. Changing the verb must not move the ordering — MODL-02 is absolute, a metered run "
        f"can never auto-launch."
    )


# --------------------------------------------------------------------------------------------------
# (2) The bounded window — config-first, both halves derived
# --------------------------------------------------------------------------------------------------


def test_the_watch_window_is_config_first() -> None:
    """The ``get(timeout=...)`` window is an ATTRIBUTE ACCESS on the config, never a literal.

    Both halves are derived: the value node's TYPE is read off the AST, and the field's existence is
    read off the REAL ``ModalConfig.model_fields``. A hardcoded window would stop tracking the config
    the moment the fast-abort profile changes (D-NOHARDCODE), and a config field nobody reads is
    dead documentation.
    """
    timeout_values = [
        keyword.value
        for node in ast.walk(_entrypoint_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        for keyword in node.keywords
        if keyword.arg == "timeout"
    ]
    assert timeout_values, (
        "entrypoint.py must call FunctionCall.get(timeout=...) — the BOUNDED synchronous watch that "
        "keeps cheap early aborts (arch-gate FAIL, CPU preflight, config refusal, import death) "
        "surfacing at the console after an ASYNC dispatch."
    )
    for value in timeout_values:
        assert not isinstance(value, ast.Constant), (
            f"the watch window is a hardcoded literal ({ast.unparse(value)}). It must be an "
            f"attribute access on the config (cfg.modal.{_WATCH_FIELD}, threaded to the helper) — "
            f"D-NOHARDCODE. The 300 s default is DERIVED from measured fast-abort latencies and its "
            f"derivation is documented at the field, where it is auditable."
        )
        assert isinstance(value, ast.Attribute | ast.Name), (
            f"the watch window value is a {type(value).__name__}; expected a config attribute access "
            f"(or the local it was bound to)."
        )

    from signet_trainer.config.schema import ModalConfig  # noqa: PLC0415 — no modal SDK needed

    assert _WATCH_FIELD in ModalConfig.model_fields, (
        f"ModalConfig has no {_WATCH_FIELD!r} field — the entrypoint reads a knob the schema does "
        f"not declare, so every shipped config would fail (or silently inherit nothing)."
    )
    field = ModalConfig.model_fields[_WATCH_FIELD]
    assert field.default is not None and field.default > 0, (
        f"{_WATCH_FIELD} must carry a positive DEFAULT so every existing config keeps loading "
        f"unchanged; got {field.default!r}."
    )
    assert field.description, (
        f"{_WATCH_FIELD} must carry a description documenting the derivation of its default — the "
        f"whole point is that the number is auditable AT THE SITE, not only in a plan file."
    )


# --------------------------------------------------------------------------------------------------
# (3) The SDK premise — derived from the installed SDK, never from this file's prose
# --------------------------------------------------------------------------------------------------


def test_spawn_is_async_and_remote_is_sync_in_the_installed_sdk() -> None:
    """``spawn`` -> ASYNC and ``remote`` -> SYNC, read off the INSTALLED modal's OWN source.

    This is the premise the entire fix rests on, so it is DERIVED rather than asserted from memory.
    Confirmed True at modal 1.5.0 during planning.

    ⛔ Fails LOUDLY rather than skipping when the source is unavailable. An SDK that reorganises this
    is precisely the moment a human must re-derive the premise — a silent skip would leave the repo
    believing something it can no longer check.

    ⚠ ``sys.modules`` is restored afterwards. ``test_dryrun_*.py::test_*_does_not_import_modal_*``
    assert ``"modal" not in sys.modules`` GLOBALLY, so any module importing the SDK earlier in the
    alphabet silently breaks them — and ``test_dispatch_is_spawned`` sorts before ``test_dryrun_*``
    while the pre-existing SDK-importing modules (``test_entrypoint_*``, ``test_h3_*``) all sort
    after it. Undoing our own import keeps that (order-dependent) purity guard meaningful instead of
    quietly weakening someone else's assertion to make room for ours.
    """
    import inspect  # noqa: PLC0415 — local so this module's import stays SDK-free
    import sys  # noqa: PLC0415

    preexisting = set(sys.modules)
    try:
        import modal  # noqa: PLC0415
        import modal._functions  # noqa: PLC0415

        version = modal.__version__
        spawn_src = inspect.getsource(modal._functions._Function.spawn)
        # `remote` delegates to `_call_function`; the invocation type lives there, not on the wrapper.
        call_src = inspect.getsource(modal._functions._Function._call_function)
    finally:
        for name in [n for n in sys.modules if n == "modal" or n.startswith("modal.")]:
            if name not in preexisting:
                del sys.modules[name]

    assert "FUNCTION_CALL_INVOCATION_TYPE_ASYNC" in spawn_src, (
        f"modal {version}: _Function.spawn no longer carries FUNCTION_CALL_INVOCATION_TYPE_ASYNC. "
        f"The whole D-10-DEF-17 fix rests on spawn being the ASYNC verb (the invocation type is sent "
        f"to the server at call creation, and only SYNC inputs are cancelled on client death). "
        f"Re-derive before trusting the dispatch path."
    )
    assert "FUNCTION_CALL_INVOCATION_TYPE_SYNC" in call_src, (
        f"modal {version}: _Function._call_function (which .remote delegates to) no longer carries "
        f"FUNCTION_CALL_INVOCATION_TYPE_SYNC. The reason .remote() is BANNED here has changed — "
        f"re-derive it rather than keeping a ban whose justification no longer holds."
    )


# --------------------------------------------------------------------------------------------------
# (4) The two things the operator actually needs at the console
# --------------------------------------------------------------------------------------------------


def test_the_function_call_id_is_printed() -> None:
    """The FunctionCall id — the re-attach handle that OUTLIVES this client — can never be dropped.

    ``poll_function`` uses ``clear_on_success=False`` and ``FunctionCall.from_id(...)`` re-hydrates a
    handle from an id, so a process that did NOT dispatch the call can still collect its output.
    That is the entire operational value of an async dispatch: without the id printed, a surviving
    run is unreachable by the next agent.
    """
    src = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "object_id" in src, (
        "entrypoint.py never references FunctionCall.object_id — the spawned run survives this "
        "client but nothing prints the handle needed to re-attach to it, which throws away the "
        "benefit of the async dispatch."
    )
    assert "FunctionCall.from_id" in src, (
        "entrypoint.py must print the re-attach RECIPE (modal.FunctionCall.from_id('<id>')), not "
        "merely the bare id — the next agent should not have to know the API."
    )


def test_every_documented_entrypoint_launch_block_has_detach() -> None:
    """issue #19 item 4 — every fenced ``modal run ... signet_trainer.modal.entrypoint`` block in a
    TRACKED ``.md`` file must contain ``--detach``, extending this file's AST-derived pin from the
    CODE side (the entrypoint's own runtime advisory, `test_the_detach_advisory_exists` below) to the
    DOCS side.

    The regression this closes: the canonical launch block six docs pointed operators at all omitted
    ``--detach`` — a copy-pasted multi-hour ``--mode preprocess``/``--mode sample`` dispatch survives
    the async ``.spawn()`` input but the non-detached ephemeral APP shell still dies with the local
    client, killing the encode/render partway with nothing committed. `entrypoint.py`'s own
    post-spawn warning claims "every documented launch form in this repo carries --detach" — this
    test is what makes that claim true instead of aspirational.

    ``git ls-files`` (not a glob) so a NEW doc under any directory is covered automatically, and a
    generated/gitignored ``.md`` (e.g. a frozen grid page) can never be scanned by accident. Skips
    (rather than fails) when git itself is unavailable — this is a documentation pin, not a git
    dependency for the whole suite.
    """
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=str(_ROOT), capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git ls-files unavailable — cannot enumerate tracked .md files")

    fence_re = re.compile(r"```(?:[^\n]*)\n([\s\S]*?)```")
    offenders: list[str] = []
    for rel_path in out.stdout.splitlines():
        if not rel_path:
            continue
        text = (_ROOT / rel_path).read_text(encoding="utf-8")
        for block in fence_re.findall(text):
            if "signet_trainer.modal.entrypoint" not in block or "modal run" not in block:
                continue  # a fenced block that mentions the entrypoint in prose, not a launch line
            if "--detach" not in block:
                offenders.append(f"{rel_path}: {block.strip()[:80]!r}...")
    assert not offenders, (
        "these fenced entrypoint launch blocks omit --detach (a copy-paste of any of them can lose "
        f"a multi-hour metered run to a closed laptop): {offenders}"
    )


def test_the_detach_advisory_exists() -> None:
    """AP-DETACH as STRUCTURE, not doc prose: the entrypoint checks ``sys.argv`` for ``--detach``.

    ``.spawn()`` fixes the INPUT; ``--detach`` keeps the ephemeral app SHELL alive. Both are
    required and neither alone is sufficient — a spawned call on a NON-detached ephemeral app is
    still stopped when the client exits. ``--detach`` is a CLI-level flag the entrypoint cannot
    control, so the honest handling is an ADVISORY (never a hard failure), and the advisory is
    pinned here because "documented in a skill" is exactly the failure mode this defect came from.
    """
    src = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "sys.argv" in src and "--detach" in src, (
        "entrypoint.py must check sys.argv for --detach and warn when it is missing. .spawn() alone "
        "does NOT make a run survive client death on an ephemeral app: the app shell is torn down at "
        "client exit unless --detach was passed. Encoding this as a runtime check is the point — the "
        "rule already existed as prose in PITFALLS.md and in the skills, and prose did not hold."
    )
