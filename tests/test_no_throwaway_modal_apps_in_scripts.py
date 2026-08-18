"""Issue #40 finding 4 — the two throwaway scripts that defined their own Modal apps and dispatched
via `.remote()` outside the single gate are gone, per their own docstrings ("Delete after Phase 2").

**The defect.** `scripts/_verify_gpu_image.py` and `scripts/_probe_hf_access.py` each declared a
``modal.App(...)`` and called ``<fn>.remote()`` from a ``@app.local_entrypoint()`` — a path
CONTRIBUTING.md's "single gate" house rule (`.spawn()`/`.remote()` only through
`signet_trainer.modal.entrypoint`) forbids, and one every guard in `tests/test_no_warm_gpu.py` /
`test_dispatch_is_spawned.py` is scoped away from (they only read
`src/signet_trainer/modal/entrypoint.py`). `python -m modal run scripts/_verify_gpu_image.py` built
the full GPU image (git+ffmpeg, cu129 torch, an LTX-2 clone, three editable installs) with no cost
print, guardrail check, ledger entry, or approval pause — unbounded, unlogged spend with zero test
coverage.

This pins the deletion, and adds a source-level regression scan so any FUTURE script under
`scripts/` that defines its own ``modal.App(`` can't reintroduce an unguarded dispatch surface
silently — matching the AST-vs-regex convention `tests/test_dispatch_is_spawned.py` documents
(a call site can't be faked by a docstring mention).
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT / "scripts"


def test_the_two_throwaway_scripts_are_gone() -> None:
    for name in ("_verify_gpu_image.py", "_probe_hf_access.py"):
        assert not (_SCRIPTS_DIR / name).exists(), (
            f"scripts/{name} still exists — its own docstring says 'Delete after Phase 2', and it "
            f"defined a modal.App(...) that dispatched via .remote() with no cost print, guardrail, "
            f"or ledger entry (issue #40 finding 4)"
        )


def _scripts_defining_modal_app() -> dict[str, list[int]]:
    """Every ``modal.App(`` CALL, by file, found on the AST (never on text/docstring mentions)."""
    hits: dict[str, list[int]] = {}
    for path in sorted(_SCRIPTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "App"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "modal"
        ]
        if lines:
            hits[path.name] = lines
    return hits


def test_no_script_defines_its_own_modal_app() -> None:
    """Regression guard: no file under scripts/ may declare a ``modal.App(...)`` of its own.

    Every gated Modal function this repo runs lives in ``src/signet_trainer/modal/`` and is
    dispatched exclusively through ``signet_trainer.modal.entrypoint`` (the single gate). A script
    defining its own app is exactly the shape both throwaway scripts had — no cost print, no
    guardrail, no ledger entry, and no guard scoped to see it.
    """
    hits = _scripts_defining_modal_app()
    assert not hits, (
        f"scripts/ file(s) define their own modal.App(...): {hits}. Every metered Modal dispatch "
        f"must go through signet_trainer.modal.entrypoint (CONTRIBUTING.md 'single gate'); a script "
        f"with its own App is unguarded, unmetered, and invisible to every dispatch-discipline test "
        f"(issue #40 finding 4)."
    )
