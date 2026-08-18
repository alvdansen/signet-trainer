"""#36 finding 1 regression guard — every ``scripts/prep_*.py`` CLI's ``build_parser()`` must
actually construct, ``--help`` included.

The packaged-spec refactor turned ``propagate.DEFAULT_SPEC`` (a module-level constant) into
``propagate.default_spec()`` (a function) but ``scripts/prep_inpaint_propagate.py:65`` kept reading
the old constant as an ``argparse`` default. Because that default is evaluated while
``add_argument`` runs — i.e. INSIDE ``build_parser()`` — the AttributeError fired before any
parsing happened: ``--help`` and ``--dry-run`` died exactly like a real invocation. Nothing in the
suite imported the script, so this stayed green through the whole rename.

This test loads every standalone ``scripts/prep_*.py`` module that exposes a ``build_parser``
function (the house CLI-wrapper shape) and asserts ``build_parser().parse_args([])`` never raises
anything other than argparse's OWN missing-required-argument ``SystemExit`` — never an
``AttributeError`` (or any other exception) from a stale module-level reference. The next rename
that breaks a CLI's parser construction fails HERE, not in the operator's terminal.

Standalone-script loading follows the ``test_prep_segdataset.py`` / ``test_qa_overlay_h264.py``
importlib precedent. NO modal, NO GPU, NO network — pure argparse construction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _scripts_with_build_parser() -> list[Path]:
    """``scripts/prep_*.py`` files that define a module-level ``build_parser`` (house CLI shape)."""
    out = []
    for path in sorted(SCRIPTS_DIR.glob("prep_*.py")):
        if "def build_parser(" in path.read_text(encoding="utf-8"):
            out.append(path)
    return out


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SCRIPTS = _scripts_with_build_parser()
assert _SCRIPTS, "expected at least one scripts/prep_*.py with build_parser() — glob/pattern broke"


@pytest.mark.parametrize("script_path", _SCRIPTS, ids=lambda p: p.name)
def test_build_parser_constructs_and_parses_empty_args(script_path):
    """``build_parser()`` must not raise, and ``parse_args([])`` must not raise anything but the
    argparse-native SystemExit(2) a script with a genuinely required flag emits on its own.

    Reproduces the exact #36 finding-1 failure mode: ``prop.DEFAULT_SPEC`` no longer exists (renamed
    to ``default_spec()``), so ``ap.add_argument("--spec", default=str(prop.DEFAULT_SPEC), ...)``
    raised ``AttributeError`` while ``build_parser()`` was still assembling the parser — before any
    argument was ever parsed.
    """
    mod = _load(script_path)

    import argparse

    parser = mod.build_parser()
    assert isinstance(parser, argparse.ArgumentParser), (
        f"{script_path.name}: build_parser() must return an ArgumentParser"
    )

    try:
        parser.parse_args([])
    except SystemExit as exc:
        # A script with a genuinely required flag (e.g. --clips-dir) exits 2 here — that is
        # argparse's OWN validation, not the bug. Anything else (a raw AttributeError propagating
        # out of add_argument, etc.) must fail this test.
        assert exc.code == 2, (
            f"{script_path.name}: parse_args([]) raised SystemExit({exc.code}), expected the "
            f"argparse missing-required-argument exit code 2"
        )
