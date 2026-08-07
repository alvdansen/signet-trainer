"""Structural no-warm-GPU guard (SC#4 / MODL-03 / T-01-MD3).

Scans the Modal source files as TEXT (does NOT import the ``modal``-decorated modules — keeps the
test runnable on Windows/CI with zero Modal spend) and fails if any ``@app.function`` ever sets
``keep_warm`` or ``min_containers`` to a warm value. Warm containers are opt-in on Modal; SC#4 is
satisfied by NEVER opting in, so the only safe count of these PARAMETERS is zero.

This also asserts the cost-print-before-launch ordering in entrypoint.py (MODL-03): the cost
estimate must be computed/printed before any ``.spawn()`` dispatch in the launch path. (D-10-DEF-17
swapped the dispatch verb ``.remote`` -> ``.spawn``; the ORDERING claim is unchanged.)
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo-root-relative path to the modal package source.
_MODAL_DIR = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal"

#: A REAL dispatch — ``.spawn(<something>``. The ``\(\s*(?!\))`` tail is load-bearing and is the
#: repo's established convention (see ``test_entrypoint_gate.py::_dispatch_pattern``):
#: ``_strip_comments_and_docstrings`` removes comments and triple-quoted blocks but NOT ordinary
#: string literals, and ``_watch_dispatch``'s ``--detach`` advisory legitimately mentions
#: ``.spawn()`` with EMPTY parens near the TOP of the module — before the cost print. Without the
#: lookahead that advisory string satisfies the scan and the MODL-03 ordering claim reports on a
#: log message instead of on a dispatch.
_DISPATCH_RE = r"\.spawn\s*\(\s*(?!\))"


def _modal_source_files() -> list[Path]:
    files = sorted(_MODAL_DIR.glob("*.py"))
    assert files, f"expected modal source files under {_MODAL_DIR}"
    return files


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments and triple-quoted strings so doc mentions don't trip the scan.

    The plan's modules DOCUMENT 'no keep_warm/min_containers' in prose; we must match only real
    keyword-argument usage, not the warnings about it.
    """
    # Drop triple-quoted blocks (both quote styles), then line comments.
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_no_keep_warm_or_min_containers_in_modal_package() -> None:
    # Match an actual keyword-arg assignment: ``keep_warm=`` / ``min_containers=``.
    offender = re.compile(r"\b(keep_warm|min_containers)\s*=")
    hits: list[str] = []
    for path in _modal_source_files():
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        for m in offender.finditer(code):
            hits.append(f"{path.name}: {m.group(0)!r}")
    assert not hits, (
        "warm-GPU parameters found in modal/ — SC#4 requires zero keep_warm/min_containers: "
        + "; ".join(hits)
    )


def test_import_modal_confined_to_modal_package() -> None:
    """``import modal`` may appear ONLY under src/signet_trainer/modal/ (Anti-Pattern 6)."""
    pkg_root = Path(__file__).resolve().parents[1] / "src" / "signet_trainer"
    offenders: list[str] = []
    for path in pkg_root.rglob("*.py"):
        if _MODAL_DIR in path.parents or path.parent == _MODAL_DIR:
            continue
        code = _strip_comments_and_docstrings(path.read_text(encoding="utf-8"))
        if re.search(r"^\s*import\s+modal\b", code, re.MULTILINE) or re.search(
            r"^\s*from\s+modal\b", code, re.MULTILINE
        ):
            offenders.append(str(path))
    assert not offenders, f"import modal must be confined to modal/: {offenders}"


def test_cost_estimate_printed_before_remote_in_entrypoint() -> None:
    """MODL-03 ordering: the cost estimate is computed/printed BEFORE any ``.spawn()`` dispatch.

    Source-level ordering check (no Modal run): in entrypoint.py the cost-estimate call must
    appear earlier than the first ``.spawn(`` call in the launch path.

    ⚠ Both matches are now asserted NON-None. Previously the dispatch match was wrapped in
    ``if remote_match is not None``, so renaming the verb (which D-10-DEF-17 did) would have turned
    this into a silent PASS asserting nothing.
    """
    entrypoint = _MODAL_DIR / "entrypoint.py"
    code = _strip_comments_and_docstrings(entrypoint.read_text(encoding="utf-8"))

    cost_match = re.search(r"estimate_cost\s*\(|guardrail_check\s*\(", code)
    remote_match = re.search(_DISPATCH_RE, code)

    assert cost_match is not None, "entrypoint.py must compute a cost estimate (MODL-03)"
    assert remote_match is not None, (
        "entrypoint.py must contain a .spawn( dispatch — if the verb changed again, re-derive this "
        "scan rather than letting it pass vacuously (D-10-DEF-17)"
    )
    assert cost_match.start() < remote_match.start(), (
        "cost estimate must be computed before any .spawn() dispatch (MODL-03)"
    )


def test_secret_names_resolved_via_env_override() -> None:
    """app.py builds the app graph at import time, so secret NAMES come from env-var overrides.

    Modal eagerly resolves every ``Secret.from_name`` in the app graph, so the names must be
    config-driven (defaulting to the account's ``my-*`` secrets) rather than hard-coded. The
    env-var seam (``SIGNET_HUGGINGFACE_SECRET_NAME`` / ``SIGNET_WANDB_SECRET_NAME``) lets the
    Phase-2 entrypoint export the loaded config's names before invoking remote functions, with
    defaults kept aligned to ``SignetConfig.modal.{huggingface,wandb}_secret_name``.
    """
    app_src = (_MODAL_DIR / "app.py").read_text(encoding="utf-8")
    for env_var, default in (
        ("SIGNET_HUGGINGFACE_SECRET_NAME", "my-huggingface-secret"),
        ("SIGNET_WANDB_SECRET_NAME", "my-wandb-secret"),
    ):
        assert env_var in app_src, f"app.py must read {env_var} for the secret-name override"
        assert default in app_src, f"app.py default for {env_var} must be {default!r}"


def test_entrypoint_runs_dryrun_gate_before_remote() -> None:
    """The dry-run gate must run before any remote dispatch (gated-launch seam)."""
    entrypoint = _MODAL_DIR / "entrypoint.py"
    code = _strip_comments_and_docstrings(entrypoint.read_text(encoding="utf-8"))

    # The entrypoint imports/calls the dryrun gate (signet_trainer.dryrun.shapes.main).
    assert re.search(r"dryrun", code), "entrypoint.py must invoke the dry-run gate before launch"
    remote_match = re.search(_DISPATCH_RE, code)
    dryrun_match = re.search(r"dryrun", code)
    assert remote_match is not None, (
        "entrypoint.py must contain a .spawn( dispatch (D-10-DEF-17) — asserted rather than skipped "
        "so a verb rename cannot make this gate vacuous"
    )
    assert dryrun_match.start() < remote_match.start(), (
        "the dry-run gate must run before any .spawn() dispatch"
    )
