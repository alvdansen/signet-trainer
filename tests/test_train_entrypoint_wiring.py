"""MODL-02 structural gate — ``train.spawn()`` is wired STRICTLY behind the approval pause.

The single invariant all metered runs depend on: the ``train.spawn()`` dispatch in
``modal/entrypoint.py`` must appear AFTER the ``_require_approval`` gate in source order, so an
unapproved run can never auto-launch (T-03-61). This is proven by a CPU source-order scan (no
Modal run, zero spend) — mirroring ``tests/test_no_warm_gpu.py``'s text-scan convention — plus a
behavioral assert that ``_require_approval`` returns False when no approval is given (EOFError /
declined), so a declined run aborts with NO dispatch.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_MODAL_DIR = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal"
_ENTRYPOINT = _MODAL_DIR / "entrypoint.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_train_remote_dispatched_after_approval_gate() -> None:
    """MODL-02 / T-03-61: the ``train.spawn(`` dispatch must follow ``_require_approval`` in source.

    A source-order index comparison on the comment/docstring-stripped entrypoint proves the
    dispatch can never precede the blocking approval pause — no auto-launch of a metered run.
    (D-10-DEF-17 swapped the verb ``.remote`` -> ``.spawn``; the ORDERING invariant is unchanged.)
    """
    code = _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))

    remote_match = re.search(r"train\.spawn\s*\(", code)
    assert remote_match is not None, "entrypoint.py must dispatch train.spawn() (Phase-3 wiring)"

    # The gate's CALL-site in main() (not the def) — the last occurrence is the call inside main().
    approval_calls = [m for m in re.finditer(r"_require_approval\s*\(", code)]
    assert approval_calls, "entrypoint.py must call _require_approval before dispatch (MODL-02)"
    approval_call_idx = approval_calls[-1].start()

    assert approval_call_idx < remote_match.start(), (
        "train.spawn() must appear AFTER the _require_approval gate (MODL-02: no auto-launch) — "
        f"approval@{approval_call_idx} vs spawn@{remote_match.start()}"
    )


def test_entrypoint_registers_app_functions_at_import() -> None:
    """``train.spawn()`` needs the fns ``@app.function`` stubs registered when ``modal run`` builds
    the app graph. Without a MODULE-LEVEL import of ``modal.fns`` the functions are never registered
    and the real dispatch fails to hydrate ("Function has not been hydrated"). Proven by a source
    scan: the registering import must appear before the first ``def`` (i.e. at module scope).
    """
    code = _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))
    first_def = re.search(r"\ndef\s", code)
    head = code[: first_def.start()] if first_def else code
    assert re.search(
        r"import\s+signet_trainer\.modal\.fns"
        r"|from\s+signet_trainer\.modal\s+import\s+fns"
        r"|from\s+signet_trainer\.modal\.fns\s+import",
        head,
    ), (
        "entrypoint.py must import modal.fns at module level so the @app.function stubs register on "
        "the app graph (else train.spawn() fails to hydrate on the real launch)"
    )


def test_train_remote_passes_config_text() -> None:
    """The container re-parses + revalidates the config by value: the dispatch forwards the YAML text.

    The ``configs/`` dir is not shipped into the image, so a path would not resolve remotely — the
    dispatch must pass the config CONTENTS (``config_text``), which the train body re-validates.
    """
    code = _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))
    assert re.search(r"train\.spawn\s*\(\s*config_text\s*\)", code), (
        "train.spawn(config_text) must forward the config YAML text so the container re-runs the "
        "gate without needing the file on its filesystem"
    )


def test_require_approval_returns_false_when_not_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_require_approval`` returns False on a non-interactive stdin (EOFError) with no flag.

    Declining (or a piped/CI stdin) must abort with NO dispatch — the behavioral half of MODL-02.
    """
    modal = pytest.importorskip("modal")  # entrypoint imports the modal app graph at module load.
    assert modal is not None
    from signet_trainer.modal.entrypoint import _require_approval

    # --approve flag set -> authorized.
    assert _require_approval(True) is True

    # No flag + non-interactive stdin (EOFError) -> never auto-spend.
    def _raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert _require_approval(False) is False

    # No flag + an explicit non-"approved" answer -> declined.
    monkeypatch.setattr("builtins.input", lambda _prompt="": "no")
    assert _require_approval(False) is False
